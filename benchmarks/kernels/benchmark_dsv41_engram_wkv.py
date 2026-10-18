# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 Engram WKV: replicated versus TP-sharded MXFP8 projection.

Use torchrun with an isolated allocation for extended measurements. The explicit
--shared-screening mode allows only short event-timed graph comparisons, caps
PyTorch allocation to 512 MiB/rank, and uses no profiler or cache flush. It
requires external free-VRAM admission and a container runtime limit. No model
weights or KV cache are loaded. Embedding lookup and the downstream Engram gate
are common to both paths and excluded; the new path includes its all-gather.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import graph_capture, set_custom_all_reduce
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.models.deepseek_v4_1.common import engram
from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


def make_layer(kind, weight, scales):
    n, k = weight.shape
    quant = DeepseekV4FP8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
        }
    )
    kwargs = {"gather_output": True} if kind is ColumnParallelLinear else {}
    with torch.device("cuda"):
        layer = kind(
            k,
            n,
            bias=False,
            quant_config=quant,
            return_bias=False,
            params_dtype=torch.bfloat16,
            prefix="model.layers.1.engram.wkv",
            **kwargs,
        )
    layer.weight.weight_loader(layer.weight, weight)
    layer.weight_scale.weight_loader(layer.weight_scale, scales)
    width = layer.output_size_per_partition if kind is ColumnParallelLinear else n
    start = dist.get_rank() * width if kind is ColumnParallelLinear else 0
    torch.testing.assert_close(
        layer.weight.cpu().view(torch.uint8),
        weight[start : start + width].view(torch.uint8),
        rtol=0,
        atol=0,
    )
    expected_scale = scales.view(torch.uint8).repeat_interleave(32, dim=0)
    torch.testing.assert_close(
        layer.weight_scale.cpu(), expected_scale[start : start + width], rtol=0, atol=0
    )
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.quant_method.kernel.__class__.__name__ == "MarlinMxfp8LinearKernel"
    return layer


def run(args):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    torch.set_num_threads(1)
    start_time = time.monotonic()
    if args.shared_screening:
        total = current_platform.get_device_total_memory(local_rank)
        torch.cuda.set_per_process_memory_fraction(512 * 1024**2 / total)
        assert world == 8 and torch.accelerator.device_count() == 8
        assert len(args.tokens) <= 5 and all(0 < m <= 128 for m in args.tokens)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world,
            disable_custom_all_reduce=True,
        )
    )
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        set_custom_all_reduce(False)
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            timeout=timedelta(seconds=60),
        )
        ensure_model_parallel_initialized(world, 1)
        torch.manual_seed(17)
        n, k = 25600, 6144
        raw = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cpu")
        raw[(raw & 127) == 127] = 0
        weight = raw.view(torch.float8_e4m3fn)
        scales = torch.randint(
            119, 125, (n // 32, k // 32), dtype=torch.uint8, device="cpu"
        ).view(torch.float8_e8m0fnu)
        baseline = make_layer(ReplicatedLinear, weight, scales)
        candidate = make_layer(ColumnParallelLinear, weight, scales)
        del raw, weight, scales
        records = []
        for m in args.tokens:
            if args.shared_screening and time.monotonic() - start_time > 150:
                raise TimeoutError("Shared comparison exceeded 150 seconds")
            # The same token matrix reaches every TP rank (DP=1, no SP).
            torch.manual_seed(m)
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            expected, actual = baseline(x), candidate(x)
            error = (
                (actual.float() - expected.float()).norm() / expected.float().norm()
            ).item()
            assert error < 0.005, error
            graphs = {}
            for name, layer in [("replicated", baseline), ("tp_sharded", candidate)]:
                for _ in range(2):
                    layer(x)
                torch.accelerator.synchronize()
                dist.barrier(device_ids=[local_rank])
                graph = torch.cuda.CUDAGraph()
                with (
                    graph_capture(device=torch.device("cuda", local_rank)) as ctx,
                    torch.cuda.graph(graph, stream=ctx.stream),
                ):
                    output = layer(x)
                graphs[name] = (graph, output)
            for graph, _ in graphs.values():
                graph.replay()
            torch.accelerator.synchronize()
            first, last = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            first.record()
            last.record()
            last.synchronize()
            samples = {name: [] for name in graphs}
            for trial in range(4):
                order = (
                    ["replicated", "tp_sharded"]
                    if trial % 2 == 0
                    else ["tp_sharded", "replicated"]
                )
                for name in order:
                    dist.barrier(device_ids=[local_rank])
                    torch.accelerator.synchronize()
                    first.record()
                    graphs[name][0].replay()
                    last.record()
                    last.synchronize()
                    elapsed = torch.tensor(
                        first.elapsed_time(last) * 1000,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    samples[name].append(elapsed.item())
            row = {
                "tokens": m,
                "n": n,
                "k": k,
                "max_rank_median_us": {
                    name: statistics.median(v) for name, v in samples.items()
                },
                "samples_max_rank_us": samples,
                "relative_l2_error": error,
            }
            records.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
            for graph, _ in graphs.values():
                graph.reset()
            del graphs, graph, expected, actual, output, x
            gc.collect()
        report = {
            "rank": rank,
            "world_size": world,
            "device": torch.cuda.get_device_name(),
            "records": records,
            "allocator_cap_mib": 512 if args.shared_screening else None,
            "peak_allocated": torch.accelerator.max_memory_allocated(),
            "peak_reserved": torch.accelerator.max_memory_reserved(),
            "source_sha256": hashlib.sha256(
                Path(engram.__file__).read_bytes()
            ).hexdigest(),
            "elapsed_s": time.monotonic() - start_time,
            "mode": "shared_events_warm_cache" if args.shared_screening else "events",
            "weight_bytes": {
                "replicated": sum(
                    p.numel() * p.element_size() for p in baseline.parameters()
                ),
                "tp_sharded": sum(
                    p.numel() * p.element_size() for p in candidate.parameters()
                ),
            },
        }
        Path(args.output).mkdir(parents=True, exist_ok=True)
        Path(args.output, f"rank-{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        del baseline, candidate, layer, first, last, elapsed, ctx
        gc.collect()
        dist.barrier(device_ids=[local_rank])
        torch.accelerator.synchronize()
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-screening", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 6, 12, 48, 128])
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
