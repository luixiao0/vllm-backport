# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 WO_A: cached BF16 inverse-RoPE/einsum versus compact SM80 projection.

Default timing uses CUPTI, CUDA graphs and cold L2 on an isolated GPU. The
explicit --shared-screening mode uses four event-timed graph replays per path,
256 MiB allocator cap, and no cache flush or profiler. Limit external runtime
and expose exactly one GPU when running on a serving host.
"""

import argparse
import gc
import json
import statistics
import time

import torch

from vllm.model_executor.kernels.linear.mxfp8.sm80 import sm80_mxfp8_bmm
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-screening", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 7, 32, 129])
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    start = time.monotonic()
    if args.shared_screening:
        assert torch.accelerator.device_count() == 1
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(256 * 1024**2 / total)
        assert len(args.tokens) * len(args.groups) <= 8
        assert all(0 < m <= 129 for m in args.tokens)
        assert all(g in (1, 2) for g in args.groups)
    else:
        from flashinfer.testing import bench_gpu_time_with_cupti

    torch.manual_seed(19)
    for g in args.groups:
        n, k, head, rope = 1024, 4096, 512, 64
        raw = torch.randint(0, 256, (g, n, k), device="cuda", dtype=torch.uint8)
        raw[(raw & 127) == 127] = 0
        weight = raw.view(torch.float8_e4m3fn)
        scale = torch.full((g, n, k // 32), 119, device="cuda", dtype=torch.uint8)
        wb = dequant_mxfp8_to_bf16(weight, scale)
        angle = torch.randn(257, rope // 2, device="cuda")
        cache = torch.cat((angle.cos(), angle.sin()), -1)
        for m in args.tokens:
            if args.shared_screening and time.monotonic() - start > 120:
                raise TimeoutError("Shared screening exceeded 120 seconds")
            o = torch.randn(m, g * 8 + 3, head, device="cuda", dtype=torch.bfloat16)
            o = o[:, : g * 8]
            positions = torch.arange(m, device="cuda") % 257

            def baseline(
                o=o, positions=positions, cache=cache, rope=rope, m=m, g=g, k=k, wb=wb
            ):
                rotated = _fused_inverse_rope_gptj(o, positions, cache, rope)
                return torch.einsum("tgk,gnk->tgn", rotated.reshape(m, g, k), wb)

            def port(
                o=o,
                weight=weight,
                scale=scale,
                positions=positions,
                cache=cache,
                rope=rope,
            ):
                return sm80_mxfp8_bmm(
                    o,
                    weight,
                    scale,
                    positions=positions,
                    cos_sin_cache=cache,
                    rope_dim=rope,
                )

            functions = {"bf16": baseline, "sm80": port}
            rotated = _fused_inverse_rope_gptj(o, positions, cache, rope).reshape(
                m, g, k
            )
            ref = torch.einsum("tgk,gnk->tgn", rotated.float(), wb.float())
            errors = {
                name: ((fn().float() - ref).norm() / ref.norm()).item()
                for name, fn in functions.items()
            }
            assert all(error < 0.005 for error in errors.values()), errors
            if args.shared_screening:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graphs = {}
                for name, fn in functions.items():
                    with torch.cuda.stream(stream):
                        fn()
                        fn()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        result = fn()
                    graphs[name] = (graph, result)
                first = torch.cuda.Event(enable_timing=True)
                last = torch.cuda.Event(enable_timing=True)
                samples = {name: [] for name in functions}
                first.record()
                last.record()
                last.synchronize()
                for round_id in range(4):
                    order = ["bf16", "sm80"] if round_id % 2 == 0 else ["sm80", "bf16"]
                    for name in order:
                        first.record()
                        graphs[name][0].replay()
                        last.record()
                        last.synchronize()
                        samples[name].append(first.elapsed_time(last) * 1000)
                del graphs, result
            else:
                samples = {
                    name: [
                        ms * 1000
                        for ms in bench_gpu_time_with_cupti(
                            fn, use_cuda_graph=True, cold_l2_cache=True
                        )
                    ]
                    for name, fn in functions.items()
                }
            medians = {name: statistics.median(v) for name, v in samples.items()}
            # FLOPs = 2*T*G*N*K; bytes is a one-read weight+scale lower bound.
            matmul_settings = torch.backends.cuda.matmul
            print(
                json.dumps(
                    {
                        "m": m,
                        "groups": g,
                        "n": n,
                        "k": k,
                        "mode": "shared_events"
                        if args.shared_screening
                        else "isolated_cupti",
                        "device": torch.cuda.get_device_name(),
                        "median_us": medians,
                        "relative_l2_to_fp32": errors,
                        "bf16_reduced_precision_reduction": (
                            matmul_settings.allow_bf16_reduced_precision_reduction
                        ),
                        "samples_us": samples,
                        "tflops": {
                            name: 2 * m * g * n * k / us / 1e6
                            for name, us in medians.items()
                        },
                        "sm80_weight_read_gb_s_lower_bound": g
                        * n
                        * k
                        * 33
                        / 32
                        / medians["sm80"]
                        / 1000,
                        "resident_weight_bytes": {
                            "bf16": wb.numel() * 2 + scale.numel(),
                            "sm80": weight.numel() + scale.numel(),
                        },
                    }
                )
            )
            gc.collect()
        del raw, weight, scale, wb, angle, cache, o, positions
        gc.collect()


if __name__ == "__main__":
    main()
