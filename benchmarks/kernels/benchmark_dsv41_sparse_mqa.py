# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded shared-A100 screening of the complete FP8 consumer-indexer pipeline.

CUDA-event graph timing deliberately replaces CUPTI/cold-L2 profiling on the
production-shared device. Baseline autotuning is pinned to its existing legal
4-warp/2-stage configuration to avoid Triton's 256MiB cache-flush allocation.
This process never loads model weights, the serving engine, or production KV.
"""

import argparse
import gc
import hashlib
import inspect
import json
import pathlib
import statistics
import time

import torch

from vllm.model_executor.kernels.attention.dsa import candidate_mqa
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops import mqa_logits_triton as dense_mqa


def make_inputs(rows, next_n, length):
    batch = rows // next_n
    pages = triton.cdiv(length, 128)
    generator = torch.Generator().manual_seed(111 + rows + length)
    # Each request owns distinct physical pages: no cross-request cache reuse.
    cache = torch.randint(
        0, 256, (batch * pages, 128 * 132), dtype=torch.uint8, generator=generator
    )
    values = cache[:, : 128 * 128]
    values[(values & 127) == 127] = 0
    scales = torch.full((batch * pages, 128), 0.03125, dtype=torch.float32)
    cache[:, 128 * 128 :] = scales.view(torch.uint8)
    cache = cache.view(batch * pages, 128, 1, 132).to("cuda")
    table = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
    table = table.flip(1).to("cuda")
    q = torch.randn(batch, next_n, 32, 128, generator=generator)
    q = q.to(torch.float8_e4m3fn).to("cuda")
    weights = torch.rand(rows, 32, generator=generator).to("cuda")
    lens = torch.full((batch, next_n), length, dtype=torch.int32)
    lens -= torch.arange(next_n - 1, -1, -1, dtype=torch.int32)
    lens = lens.to("cuda")
    nblocks = triton.cdiv(length, 8)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32)
    for row in range(rows):
        chosen = torch.randperm(nblocks, generator=generator)[:2048]
        candidates[row, : chosen.numel()] = chosen
    return q, cache, weights, lens, table, candidates.to("cuda")


def run(args):
    started = time.monotonic()
    assert args.shared_screening and current_platform.get_device_capability() == (8, 0)
    assert torch.accelerator.device_count() == 1
    torch.set_num_threads(1)
    total = current_platform.get_device_total_memory(0)
    torch.cuda.set_per_process_memory_fraction(512 * 1024**2 / total)
    dense_mqa._fp8_paged_mqa_logits_kernel.configs = [
        triton.Config({}, num_warps=4, num_stages=2)
    ]
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    results = []
    for rows in args.rows:
        next_n = 1 if rows == 1 else 6
        for length in args.lengths:
            if time.monotonic() - started > 150:
                raise RuntimeError("Shared-device screening budget exceeded")
            q, cache, weights, lens, table, candidates = make_inputs(
                rows, next_n, length
            )
            expected = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
            actual = torch.empty_like(expected)

            def baseline(
                q=q,
                cache=cache,
                weights=weights,
                lens=lens,
                table=table,
                length=length,
                candidates=candidates,
                expected=expected,
            ):
                logits = dense_mqa.fp8_paged_mqa_logits_triton(
                    q, cache, weights, lens, table, length, clean_logits=False
                )
                apply_candidate_mask(logits, None, lens.reshape(-1), candidates, 8)
                torch.ops._C.persistent_topk(
                    logits, lens.reshape(-1), expected, workspace, 512, length
                )

            def compact(
                q=q,
                cache=cache,
                weights=weights,
                lens=lens,
                table=table,
                length=length,
                candidates=candidates,
                actual=actual,
            ):
                candidate_mqa.fp8_candidate_paged_mqa_topk(
                    q,
                    cache,
                    weights,
                    lens,
                    table,
                    candidates,
                    8,
                    actual,
                    length,
                    workspace,
                )

            baseline()
            compact()
            torch.accelerator.synchronize()
            assert torch.equal(expected, actual), (rows, length, expected, actual)
            graphs = []
            for function in [baseline, compact]:
                for _ in range(2):
                    function()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    function()
                graph.replay()
                graphs.append(graph)
            torch.accelerator.synchronize()
            samples = [[], []]
            for round_id in range(4):
                for which in [0, 1] if round_id % 2 == 0 else [1, 0]:
                    torch.accelerator.synchronize()
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graphs[which].replay()
                    end.record()
                    end.synchronize()
                    samples[which].append(begin.elapsed_time(end) * 1000)
            old, new = (statistics.median(s) for s in samples)
            result = {
                "rows": rows,
                "next_n": next_n,
                "requests": rows // next_n,
                "compressed_key_positions": length,
                "raw_context_if_ratio1": length,
                "raw_context_if_ratio2": 2 * length,
                "baseline_us": old,
                "compact_us": new,
                "compact_over_baseline": new / old,
                "samples_us": samples,
                "v2_host_gate_eligible": rows >= 6 and length >= 131072,
                "candidate_positions_bound": 16384,
                "cache_page_size": 128,
                "compact_logits_width_with_tie_fillers": 16896,
                "distinct_physical_kv_bytes": cache.numel(),
                "exact_ordered_topk": True,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            del graphs, graph, function, baseline, compact
            del q, cache, weights, lens, table, candidates, expected, actual
            gc.collect()
            torch.accelerator.empty_cache()
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "source_sha256": hashlib.sha256(
            pathlib.Path(inspect.getfile(candidate_mqa)).read_bytes()
        ).hexdigest(),
        "baseline_autotune": "fixed num_warps=4 num_stages=2 in probe only",
        "timing": "CUDA graph events; four interleaved samples; shared GPU",
        "pipeline": "candidate sort/dedup + FP8 scoring + native topk + remap",
        "scope": "forced compact wrapper comparison; flag remains off by default",
        "peak_allocated": torch.accelerator.max_memory_allocated(),
        "peak_reserved": torch.accelerator.max_memory_reserved(),
        "cap_bytes": 512 * 1024**2,
        "elapsed_s": time.monotonic() - started,
        "results": results,
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-screening", action="store_true", required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[8192, 32768, 131072, 278528]
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 48])
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    assert len(arguments.lengths) <= 4
    assert all(0 < n <= 278528 for n in arguments.lengths)
    assert all(m in (1, 6, 12, 24, 48) for m in arguments.rows)
    run(arguments)
