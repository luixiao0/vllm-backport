# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare candidate-score paths on an isolated GPU, including full selection."""

import argparse
import importlib.metadata
import json
import statistics
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import envs
from vllm.model_executor.kernels.attention.dsa import candidate_blocks as candidates
from vllm.triton_utils import triton

FLAG = "VLLM_DSV4_LENGTH_AWARE_CANDIDATE_SCORES"


def make_stages(logits, ends, out, scores, values, indices, block_size, topk, enabled):
    rows, width = logits.shape
    nblocks, k = scores.shape[1], values.shape[1]

    def score():
        candidates._block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
            logits,
            None,
            ends,
            scores,
            *logits.stride(),
            0,
            ends.stride(0),
            width,
            nblocks,
            block_size,
            False,
            1,
            128,
            LENGTH_AWARE=enabled,
        )

    def top_k():
        torch.topk(scores, k, dim=-1, out=(values, indices))

    def store():
        candidates._store_candidates_kernel[(rows, triton.cdiv(topk, 256))](
            values, indices, out, *out.stride(), k, topk, 256
        )

    def selector():
        candidates.select_candidate_blocks(logits, None, ends, topk, block_size, out)

    score()
    top_k()
    store()
    return [
        ("scores", score),
        ("topk", top_k),
        ("store", store),
        ("selector", selector),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 48])
    parser.add_argument("--width", type=int, default=278528)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[8192, 32768, 131072]
    )
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            [
                *args.rows,
                args.width,
                *args.contexts,
                args.block_size,
                args.topk,
                args.rounds,
                args.repeat_iters,
            ]
        )
        <= 0
    ):
        parser.error("shapes and iteration counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("An isolated CUDA GPU is required.")

    torch.manual_seed(42)
    records = []
    for rows in args.rows:
        for context in sorted(set([*args.contexts, args.width])):
            logits = torch.randn(rows, args.width, device="cuda", dtype=torch.float32)
            ends = torch.full(
                (rows,), min(context, args.width), device="cuda", dtype=torch.int32
            )
            if rows > 1:
                ends[-1] = 0
            out = torch.empty(rows, args.topk, device="cuda", dtype=torch.int32)
            reference = torch.empty_like(out)
            nblocks = triton.cdiv(args.width, args.block_size)
            scores = logits.new_empty(rows, nblocks)
            k = min(args.topk, nblocks)
            values = logits.new_empty(rows, k)
            indices = torch.empty(rows, k, device="cuda", dtype=torch.int64)

            with patch.object(envs, FLAG, False):
                candidates.select_candidate_blocks(
                    logits, None, ends, args.topk, args.block_size, reference
                )
            with patch.object(envs, FLAG, True):
                candidates.select_candidate_blocks(
                    logits, None, ends, args.topk, args.block_size, out
                )
            torch.testing.assert_close(out, reference, rtol=0, atol=0)

            for round_id in range(args.rounds):
                order = [False, True] if round_id % 2 == 0 else [True, False]
                for enabled in order:
                    with patch.object(envs, FLAG, enabled):
                        stages = make_stages(
                            logits,
                            ends,
                            out,
                            scores,
                            values,
                            indices,
                            args.block_size,
                            args.topk,
                            enabled,
                        )
                        for stage, fn in stages:
                            for _ in range(5):
                                fn()
                            torch.accelerator.synchronize()
                            samples = bench_gpu_time_with_cupti(
                                fn,
                                dry_run_iters=5,
                                repeat_iters=args.repeat_iters,
                                use_cuda_graph=True,
                                cold_l2_cache=True,
                            )
                            record = {
                                "rows": rows,
                                "width": args.width,
                                "context": min(context, args.width),
                                "topk": args.topk,
                                "block_size": args.block_size,
                                "round": round_id,
                                "length_aware": enabled,
                                "stage": stage,
                                "median_us": statistics.median(samples) * 1000,
                            }
                            if stage == "scores":
                                active_rows = rows - 1 if rows > 1 else rows
                                record["score_bytes"] = (
                                    active_rows * min(context, args.width) * 4
                                    + rows * nblocks * 4
                                )
                                record["score_gbps"] = record["score_bytes"] / (
                                    record["median_us"] * 1000
                                )
                            records.append(record)
                            print(json.dumps(record), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.output.write_text(
        json.dumps(
            {
                "revision": revision,
                "dirty": bool(
                    subprocess.check_output(["git", "status", "--porcelain"])
                ),
                "device": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "versions": {
                    p: importlib.metadata.version(p)
                    for p in ["torch", "triton", "flashinfer-python"]
                },
                "timing": (
                    "FlashInfer CUPTI helper, CUDA graph, cold L2; helper may fall back"
                ),
                "dtype": "float32",
                "records": records,
                "scope": "kernel replay only; not serving latency or model quality",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
