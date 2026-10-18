# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen native SM80 mHC scheduling through a synthetic decoder block.

The producer, post, coefficient math and stream joins are actual model code.
Consumers preserve TP8 attention projection dimensions and one 2304-wide expert,
with BF16 attention and either BF16 or native MXFP4 Marlin MoE. EP
collectives and global routing are omitted. These timings are a scheduling proxy,
not model tokens/s. The timed
boundary includes two pre kernels, attention, post, FFN, both stream joins, and
final post consumption. Shared screening uses events without profiling or cache
flush, and requires external free-VRAM admission and a container runtime limit.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.deepseek_v4_1.nvidia import mhc_overlap
from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4DecoderLayer
from vllm.platforms import current_platform


def make_decoder(ffn_consumer):
    decoder = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
    nn.Module.__init__(decoder)
    decoder.hc_mult, decoder.hc_sinkhorn_iters = 4, 20
    decoder.hc_eps, decoder.rms_norm_eps, decoder.hc_post_alpha = 1e-6, 1e-20, 2.0
    decoder.use_sequence_parallel, decoder.engram = False, None
    decoder.mhc_stream = None
    decoder.attn_norm = RMSNorm(5120, 1e-20).to(device="cuda", dtype=torch.bfloat16)
    decoder.ffn_norm = RMSNorm(5120, 1e-20).to(device="cuda", dtype=torch.bfloat16)
    for name in ("attn", "ffn"):
        setattr(decoder, f"hc_{name}_fn", torch.randn(24, 20480, device="cuda") * 0.02)
        setattr(decoder, f"hc_{name}_scale", torch.ones(3, device="cuda"))
        setattr(decoder, f"hc_{name}_base", torch.randn(24, device="cuda"))

    def weight(k, n):
        return torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.01

    wqa, wqb, woa, wob = (
        weight(5120, 1280),
        weight(1280, 4096),
        weight(4096, 1024),
        weight(1024, 5120),
    )
    key, value = weight(512, 512), weight(512, 512)
    if ffn_consumer == "bf16":
        gate_up, down = weight(5120, 4608), weight(2304, 5120)
    else:
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            fused_marlin_moe,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
            marlin_permute_scales,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            mxfp4_marlin_process_scales,
        )
        from vllm.scalar_type import scalar_types

        def packed_weight(k, n):
            raw = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
            packed = ops.gptq_marlin_repack(
                raw.view(torch.int32).T.contiguous(),
                size_k=k,
                size_n=n,
                num_bits=4,
                is_a_8bit=False,
            )
            scales = torch.randint(
                119, 123, (n, k // 32), device="cuda", dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            scales = marlin_permute_scales(scales.T.to(torch.bfloat16), k, n, 32)
            scales = mxfp4_marlin_process_scales(scales).to(torch.float8_e8m0fnu)
            return packed.unsqueeze(0), scales.unsqueeze(0)

        w1, s1 = packed_weight(5120, 4608)
        w2, s2 = packed_weight(2304, 5120)
        workspace = marlin_make_workspace_new(torch.device("cuda"), 4)

    def attention(positions, x, ignored):
        q = (x @ wqa @ wqb).view(-1, 8, 512).transpose(0, 1)
        scores = q @ key.T / math.sqrt(512)
        probabilities = scores.float().softmax(-1).to(torch.bfloat16)
        attended = (probabilities @ value).transpose(0, 1).reshape(-1, 4096)
        return attended @ woa @ wob

    def ffn(x, input_ids):
        if ffn_consumer == "bf16":
            gate, up = (x @ gate_up).chunk(2, dim=-1)
            return (F.silu(gate) * up) @ down
        return fused_marlin_moe(
            x,
            w1,
            w2,
            None,
            None,
            s1,
            s2,
            torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32),
            torch.zeros(x.shape[0], 1, device=x.device, dtype=torch.int32),
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=1,
            workspace=workspace,
        )

    decoder.attn, decoder.ffn = attention, ffn
    return decoder


def run(args):
    assert args.shared_screening, "Only bounded shared-screening is implemented"
    assert current_platform.is_device_capability(80)
    assert torch.accelerator.device_count() == 1
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        256 * 1024**2 / current_platform.get_device_total_memory(0)
    )
    torch.manual_seed(57)
    started = time.monotonic()
    rows = []
    config = VllmConfig()
    with set_current_vllm_config(config), torch.inference_mode():
        decoder = make_decoder(args.ffn_consumer)
        side = torch.cuda.Stream()
        # Compile split modes through the exact helper before any graph capture.
        sample = torch.randn(1, 4, 5120, device="cuda", dtype=torch.bfloat16)
        mhc_overlap.mhc_pre_delayed_overlap(
            sample,
            decoder.hc_attn_fn,
            decoder.hc_attn_scale,
            decoder.hc_attn_base,
            1e-20,
            1e-6,
            1e-6,
            2.0,
            20,
            pre_mix=torch.ones(1, 4, device="cuda"),
            norm_weight=decoder.attn_norm.weight,
            norm_eps=1e-20,
            stream=side,
        )
        torch.cuda.current_stream().wait_stream(side)
        for tokens in (1, 6, 12, 24, 48):
            assert time.monotonic() - started < 150
            x = torch.randn(tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
            pre = torch.rand(tokens, 4, device="cuda")
            positions = torch.arange(tokens, device="cuda")

            def pipeline(x=x, positions=positions, pre=pre):
                out = decoder(x, positions, None, pre_mix=pre)
                return mhc_post_tilelang(out[0], out[1], out[2], out[3]), out[4]

            graphs = {}
            with set_forward_context(
                None, config, cudagraph_runtime_mode=CUDAGraphMode.FULL
            ):
                for name in ("native_serial", "overlap_opt_in"):
                    decoder.mhc_stream = side if name == "overlap_opt_in" else None
                    for _ in range(2):
                        pipeline()
                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = pipeline()
                    graphs[name] = graph, output
            for graph, output in graphs.values():
                graph.replay()
            torch.accelerator.synchronize()
            for baseline, candidate in zip(
                graphs["native_serial"][1], graphs["overlap_opt_in"][1], strict=True
            ):
                torch.testing.assert_close(candidate, baseline, atol=0, rtol=0)
            samples = {name: [] for name in graphs}
            first, last = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for trial in range(4):
                order = list(graphs) if trial % 2 == 0 else list(reversed(graphs))
                for name in order:
                    torch.accelerator.synchronize()
                    first.record()
                    graphs[name][0].replay()
                    last.record()
                    last.synchronize()
                    samples[name].append(first.elapsed_time(last) * 1000)
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            row = dict(
                tokens=tokens,
                overlap_active=tokens <= mhc_overlap.MHC_OVERLAP_MAX_TOKENS,
                median_us=medians,
                samples_us=samples,
                candidate_over_baseline=medians["overlap_opt_in"]
                / medians["native_serial"],
                output_bitwise_equal=True,
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
            del graphs, graph, output, x, pre, positions
            gc.collect()
        result = dict(
            gpu=torch.cuda.get_device_name(),
            elapsed_s=time.monotonic() - started,
            peak_allocated=torch.accelerator.memory.max_memory_allocated(),
            peak_reserved=torch.accelerator.memory.max_memory_reserved(),
            cap_bytes=256 * 1024**2,
            rows=rows,
            consumer=(
                f"BF16 TP8-sized attention and one 2304-wide {args.ffn_consumer} expert"
            ),
            official_checkpoint_loaded=False,
            serving_tokens_per_second_measured=False,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            flags={
                name: os.environ.get(name)
                for name in ("VLLM_DSV41_MHC_OVERLAP", "VLLM_DSV41_SPARSE_MQA")
            },
            helper_sha256=hashlib.sha256(
                Path(mhc_overlap.__file__).read_bytes()
            ).hexdigest(),
        )
        Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-screening", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ffn-consumer", choices=("bf16", "marlin_mxfp4"), default="marlin_mxfp4"
    )
    run(parser.parse_args())
