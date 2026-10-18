# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen radix/source MQA + consumer MQA + mHC in the same FULL graph.

The default uses BF16 proxy attention and a native MXFP4 single expert.
--native-consumers replaces them with Ampere FP8 MLA, native MXFP8 projections
and 384-route/top6 native MXFP4 with eight materialized experts. Both include
the actual sparse-indexer bodies and native mHC. There are no TP/EP collectives,
production weights, serving-engine calls, profiling, or cache flushes.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from benchmark_dsv41_mhc_overlap import make_decoder
from benchmark_dsv41_sparse_mqa import make_inputs

import vllm.envs as envs
import vllm.model_executor.layers.sparse_attn_indexer as indexer
import vllm.models.deepseek_v4_1.nvidia.model as model_module
import vllm.v1.worker.workspace as workspace_module
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.kernels.attention.dsa import candidate_mqa, candidate_radix
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.models.deepseek_v4_1.nvidia.mhc_overlap import mhc_pre_delayed_overlap
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops import mqa_logits_triton as mqa

PROFILES = {
    "serial_dense": ("torch", False, False),
    "general": ("triton_radix", True, False),
    "long_context": ("triton_radix", True, True),
}


def run(args):
    assert args.shared_screening and current_platform.is_device_capability(80)
    assert torch.accelerator.device_count() == 1
    started = time.monotonic()
    native_consumers = getattr(args, "native_consumers", False)
    cap_bytes = (512 if native_consumers else 256) * 1024**2
    # The external harness checks fresh free VRAM before starting the container.
    total = current_platform.get_device_total_memory(0)
    torch.cuda.set_per_process_memory_fraction(cap_bytes / total)
    torch.set_num_threads(1)
    torch.manual_seed(113)
    # Avoid the baseline autotuner's 256 MiB cold-L2 allocation on shared GPUs.
    mqa._fp8_paged_mqa_logits_kernel.configs = [
        triton.Config({}, num_warps=4, num_stages=2)
    ]
    rows, length = 6, 131072
    _, cache, _, lens, table, _ = make_inputs(rows, rows, length)
    cache = cache.squeeze(-2)
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=table,
        seq_lens=lens,
        max_seq_len=length,
        decode_lens=torch.tensor([rows], dtype=torch.int32, device="cuda"),
        requires_padding=False,
        schedule_metadata=torch.empty(0, dtype=torch.int32, device="cuda"),
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=lens[:, -1],
        max_seq_len=length,
        slot_mapping=torch.arange(rows, dtype=torch.int64, device="cuda"),
        num_decodes=1,
        num_decode_tokens=rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    context = ForwardContext(
        {}, {"index": metadata}, {}, cudagraph_runtime_mode=CUDAGraphMode.FULL
    )
    workspace_module._manager = workspace_module.WorkspaceManager(torch.device("cuda"))
    candidates = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
    source_topk = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    consumer_topk = torch.empty_like(source_topk)
    counters = dict(radix=0, compact=0, overlap=0)

    def count(name, function):
        def wrapped(*args, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                counters[name] += 1
            return function(*args, **kwargs)

        return wrapped

    candidate_radix.select_candidate_blocks_radix = count(
        "radix", candidate_radix.select_candidate_blocks_radix
    )
    candidate_mqa.fp8_candidate_paged_mqa_topk = count(
        "compact", candidate_mqa.fp8_candidate_paged_mqa_topk
    )
    model_module.mhc_pre_delayed_overlap = count(
        "overlap", model_module.mhc_pre_delayed_overlap
    )
    config = VllmConfig()
    results, graphs, outputs, capture_dispatch = [], {}, {}, {}
    with set_current_vllm_config(config), torch.inference_mode():
        consumer_hashes = None
        consumer_helper_sha256 = None
        consumer_references = {}
        if native_consumers:
            import benchmark_dsv41_native_consumers as native
            from benchmark_dsv41_native_consumers import (
                NativeAttention,
                NativeMoE,
                source_hashes,
            )
            from benchmark_dsv41_native_consumers import (
                make_decoder as make_native_decoder,
            )

            from vllm.utils.torch_utils import set_default_torch_dtype

            with set_default_torch_dtype(torch.bfloat16):
                native_attention = NativeAttention(length)
                native_attention.configure(rows)
                decoder = make_native_decoder(native_attention, NativeMoE())
                consumer_references["moe"] = decoder.ffn.check_reference(
                    torch.randn(2, 5120, device="cuda", dtype=torch.bfloat16)
                )
            native_attention.kernel.topk_indices_buffer = consumer_topk
            context.attn_metadata.update(native_attention.metadata)
            consumer_hashes = source_hashes()
            consumer_helper_sha256 = hashlib.sha256(
                Path(native.__file__).read_bytes()
            ).hexdigest()
        else:
            decoder = make_decoder("marlin_mxfp4")
        attention_proxy = decoder.attn
        side = torch.cuda.Stream()
        x = torch.randn(rows, 4, 5120, dtype=torch.bfloat16, device="cuda")
        original = x.clone()
        pre = torch.rand(rows, 4, device="cuda")
        positions = torch.arange(rows, device="cuda")

        def attention(positions, hidden, ignored):
            query = hidden[:, :4096].reshape(rows, 32, 128)
            weights = hidden[:, 4096:4128].float().abs() + 0.1
            for write, query_in, out in (
                (True, query, source_topk),
                (False, query.roll(1, -1), consumer_topk),
            ):
                indexer.sparse_attn_indexer(
                    hidden,
                    "index",
                    cache,
                    query_in.to(torch.float8_e4m3fn),
                    None,
                    None,
                    weights,
                    128,
                    "ue8m0",
                    512,
                    128,
                    length,
                    length,
                    out,
                    True,
                    False,
                    "",
                    candidate_blocks=candidates,
                    candidate_block_size=8,
                    candidate_write=write,
                )
            selected = consumer_topk.float().mean(-1, keepdim=True) * 1e-7
            return attention_proxy(positions, hidden, ignored) + selected.to(
                hidden.dtype
            )

        class IndexedAttention(torch.nn.Module):
            def forward(self, positions, hidden, ignored):
                return attention(positions, hidden, ignored)

        decoder.attn = IndexedAttention()

        def pipeline():
            out = decoder(x, positions, None, pre_mix=pre)
            return (
                mhc_post_tilelang(out[0], out[1], out[2], out[3]),
                out[4],
                source_topk.clone(),
                consumer_topk.clone(),
                candidates.clone(),
            )

        # Compile both coefficient epilogues before joining the capture.
        mhc_pre_delayed_overlap(
            x,
            decoder.hc_attn_fn,
            decoder.hc_attn_scale,
            decoder.hc_attn_base,
            1e-20,
            1e-6,
            1e-6,
            2.0,
            20,
            pre_mix=pre,
            norm_weight=decoder.attn_norm.weight,
            norm_eps=1e-20,
            stream=side,
        )
        torch.cuda.current_stream().wait_stream(side)
        with override_forward_context(context):
            for name, (backend, overlap, compact) in PROFILES.items():
                os.environ["VLLM_DSV4_CANDIDATE_SELECTION_BACKEND"] = backend
                os.environ["VLLM_DSV41_MHC_OVERLAP"] = str(int(overlap))
                os.environ["VLLM_DSV41_SPARSE_MQA"] = str(int(compact))
                assert compact == envs.VLLM_DSV41_SPARSE_MQA
                decoder.mhc_stream = side if overlap else None
                pipeline()
                torch.accelerator.synchronize()
                if native_consumers and name == "serial_dense":
                    consumer_references["attention"] = (
                        native_attention.check_reference()
                    )
                counters.update(radix=0, compact=0, overlap=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name] = pipeline()
                graphs[name] = graph
                capture_dispatch[name] = counters.copy()
                assert counters == dict(
                    radix=int(backend == "triton_radix"),
                    compact=int(compact),
                    overlap=2 * int(overlap),
                ), counters

        initial = {}
        for replay_id, (device_length, factor) in enumerate(
            (
                (length, 1.0),
                (32768, -0.75),
                (8192, 0.5),
                (length, 1.0),
            )
        ):
            assert time.monotonic() - started < 210
            lens.copy_(
                torch.arange(device_length - 5, device_length + 1, device="cuda")
            )
            x.copy_(original * factor)
            for graph in graphs.values():
                graph.replay()
            torch.accelerator.synchronize()
            baseline = outputs["serial_dense"]
            for name in ("general", "long_context"):
                for got, want in zip(outputs[name][:4], baseline[:4], strict=True):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
                assert torch.equal(
                    outputs[name][4].sort(-1).values, baseline[4].sort(-1).values
                )
            for name, output in outputs.items():
                assert ((output[3] >= 0) & (output[3] < lens.reshape(-1, 1))).all()
                if replay_id == 0:
                    initial[name] = [tensor.clone() for tensor in output]
                elif replay_id == 3:
                    for got, want in zip(output, initial[name], strict=True):
                        torch.testing.assert_close(got, want, atol=0, rtol=0)
                else:
                    assert not torch.equal(output[0], initial[name][0])
            samples = {name: [] for name in graphs}
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for trial in range(4):
                for name in list(graphs)[:: 1 if trial % 2 == 0 else -1]:
                    torch.accelerator.synchronize()
                    begin.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(begin.elapsed_time(end) * 1000)
            row = dict(
                device_keys=device_length,
                input_factor=factor,
                median_us={name: statistics.median(v) for name, v in samples.items()},
                samples_us=samples,
                outputs_bitwise_equal=True,
                candidate_sets_equal=True,
                indices_within_device_lengths=True,
                restored_initial_output=(replay_id == 3),
            )
            results.append(row)
            print(json.dumps(row), flush=True)
    # Release captured allocations before returning or any distributed teardown.
    for graph in graphs.values():
        graph.reset()
    graphs.clear()
    outputs.clear()
    gc.collect()
    result = dict(
        gpu=torch.cuda.get_device_name(),
        rows=rows,
        capture_keys=length,
        capture_dispatch=capture_dispatch,
        results=results,
        elapsed_s=time.monotonic() - started,
        peak_allocated=torch.accelerator.memory.max_memory_allocated(),
        peak_reserved=torch.accelerator.memory.max_memory_reserved(),
        cap_bytes=cap_bytes,
        source_revision=json.loads(Path("/opt/sm80-series.json").read_text())[
            "source_revision"
        ],
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        boundary=(
            "synthetic decoder + source indexer/radix + consumer indexer + "
            "native mHC + Ampere fp8_ds_mla + Marlin MXFP8 projections + "
            "native MXFP4 384-route/top6/8-active-expert FFN"
            if native_consumers
            else "synthetic decoder + source indexer/radix + consumer indexer + "
            "native mHC/MXFP4 expert + BF16 attention proxy"
        ),
        consumer_source_hashes=consumer_hashes,
        consumer_helper_sha256=consumer_helper_sha256,
        consumer_references=consumer_references,
        official_weights_loaded=False,
        engram_tp_collectives=False,
        model_tokens_per_second_measured=False,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-screening", action="store_true")
    parser.add_argument("--native-consumers", action="store_true")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
