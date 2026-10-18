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
from vllm.model_executor.kernels.attention.dsa import candidate_radix as radix
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


def make_radix_stages(logits, ends, out, scores, lengths, block_size, topk):
    rows, width = logits.shape
    nblocks = scores.shape[1]
    shards = triton.cdiv(nblocks, radix._HIST_TILE)
    histograms = torch.empty(
        (rows, shards, 512), device=logits.device, dtype=torch.int32
    )
    state = torch.empty((rows, 5), device=logits.device, dtype=torch.int64)
    boundary = torch.empty((rows, nblocks), device=logits.device, dtype=torch.int32)
    selection = torch.empty((rows, 2), device=logits.device, dtype=torch.int64)
    tiles = triton.cdiv(nblocks, 1024)
    counts = torch.empty((rows, tiles, 2), device=logits.device, dtype=torch.int32)
    offsets = torch.empty_like(counts)

    def score():
        radix._ragged_scores_kernel[(rows, min(triton.cdiv(nblocks, 128), 32))](
            logits,
            None,
            ends,
            scores,
            lengths,
            *logits.stride(),
            0,
            ends.stride(0),
            width,
            nblocks,
            block_size,
            False,
            1,
            128,
        )

    def coarse_histogram():
        radix._coarse_histogram_kernel[(rows, shards)](
            scores,
            lengths,
            histograms,
            nblocks,
            shards,
            radix._HIST_TILE,
            radix._LONG_ROW,
        )

    def coarse():
        radix._coarse_threshold_kernel[(rows,)](
            histograms, lengths, state, topk, shards, radix._LONG_ROW
        )

    def compact():
        # Reset the reservation counter on every timed invocation.
        coarse()
        radix._compact_boundary_kernel[(rows, triton.cdiv(nblocks, 1024))](
            scores, lengths, state, boundary, nblocks, 1024, radix._LONG_ROW
        )

    def select():
        radix._radix_candidates_kernel[(rows,)](
            scores,
            lengths,
            out,
            nblocks,
            *out.stride(),
            state,
            boundary,
            selection,
            topk,
            1024,
            radix._LONG_ROW,
            num_warps=4,
        )

    def count():
        radix._selection_counts_kernel[(rows, tiles)](
            scores,
            lengths,
            selection,
            counts,
            nblocks,
            topk,
            tiles,
            1024,
            radix._LONG_ROW,
        )

    def prefix():
        radix._selection_prefix_kernel[(rows,)](
            lengths, selection, counts, offsets, topk, tiles, 1024, radix._LONG_ROW
        )

    def scatter():
        radix._selection_scatter_kernel[(rows, tiles)](
            scores,
            lengths,
            selection,
            offsets,
            out,
            nblocks,
            *out.stride(),
            topk,
            tiles,
            1024,
            radix._LONG_ROW,
        )

    def selector():
        candidates.select_candidate_blocks(logits, None, ends, topk, block_size, out)

    score()
    coarse_histogram()
    coarse()
    compact()
    select()
    count()
    prefix()
    scatter()
    return [
        ("scores", score),
        ("coarse_histogram", coarse_histogram),
        ("coarse", coarse),
        ("coarse_and_compact", compact),
        ("select", select),
        ("count", count),
        ("prefix", prefix),
        ("scatter", scatter),
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
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["torch", "score_only", "triton_radix"],
        default=["torch", "triton_radix"],
    )
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
            lengths = torch.empty(rows, device="cuda", dtype=torch.int32)

            with (
                patch.object(envs, FLAG, False),
                patch.object(envs, "VLLM_DSV4_CANDIDATE_SELECTION_BACKEND", "torch"),
            ):
                candidates.select_candidate_blocks(
                    logits, None, ends, args.topk, args.block_size, reference
                )
            for backend in args.backends:
                with (
                    patch.object(envs, FLAG, backend == "score_only"),
                    patch.object(
                        envs,
                        "VLLM_DSV4_CANDIDATE_SELECTION_BACKEND",
                        "triton_radix" if backend == "triton_radix" else "torch",
                    ),
                ):
                    candidates.select_candidate_blocks(
                        logits, None, ends, args.topk, args.block_size, out
                    )
                torch.testing.assert_close(
                    out.sort(1).values, reference.sort(1).values, rtol=0, atol=0
                )

            for round_id in range(args.rounds):
                order = args.backends if round_id % 2 == 0 else args.backends[::-1]
                for backend in order:
                    with (
                        patch.object(envs, FLAG, backend == "score_only"),
                        patch.object(
                            envs,
                            "VLLM_DSV4_CANDIDATE_SELECTION_BACKEND",
                            "triton_radix" if backend == "triton_radix" else "torch",
                        ),
                    ):
                        if backend == "triton_radix":
                            stages = make_radix_stages(
                                logits,
                                ends,
                                out,
                                scores,
                                lengths,
                                args.block_size,
                                args.topk,
                            )
                        else:
                            stages = make_stages(
                                logits,
                                ends,
                                out,
                                scores,
                                values,
                                indices,
                                args.block_size,
                                args.topk,
                                backend == "score_only",
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
                                "backend": backend,
                                "stage": stage,
                                "median_us": statistics.median(samples) * 1000,
                            }
                            if stage == "scores":
                                active_rows = rows - 1 if rows > 1 else rows
                                written_scores = (
                                    active_rows
                                    * triton.cdiv(
                                        min(context, args.width), args.block_size
                                    )
                                    if backend == "triton_radix"
                                    else rows * nblocks
                                )
                                record["score_bytes"] = (
                                    active_rows * min(context, args.width) * 4
                                    + written_scores * 4
                                )
                                if backend == "triton_radix":
                                    record["score_bytes"] += rows * 4
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
