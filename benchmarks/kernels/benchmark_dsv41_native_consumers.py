# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic official-format consumers; no production/model checkpoint access."""

import hashlib
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    mxfp4_marlin_process_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_mxfp8_marlin_linear,
    prepare_mxfp8_layer_for_marlin,
)
from vllm.models.deepseek_v4_1.ampere.ampere_sparse import DeepseekV41AmpereMLAAttention
from vllm.models.deepseek_v4_1.common.ops.cache_utils import quantize_and_insert_k_cache
from vllm.models.deepseek_v4_1.nvidia import mhc_overlap
from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4DecoderLayer
from vllm.scalar_type import scalar_types
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum

ACTIVE_GLOBAL = torch.arange(0, 384, 48, device="cpu", dtype=torch.long)


def source_hashes():
    names = [
        "vllm.models.deepseek_v4_1.ampere.ampere_sparse",
        "vllm.models.deepseek_v4_1.amd.rocm",
        "vllm.v1.attention.ops.rocm_aiter_mla_sparse",
        "vllm.models.deepseek_v4_1.common.ops.cache_utils",
        "vllm.model_executor.layers.fused_moe.experts.marlin_moe",
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
        "vllm.model_executor.layers.fused_moe.router.gate_linear",
        "vllm.model_executor.kernels.linear.gemv_triton",
        "vllm.model_executor.layers.quantization.utils.marlin_utils_fp8",
        "vllm.models.deepseek_v4_1.nvidia.mhc_overlap",
        "vllm.models.deepseek_v4_1.nvidia.model",
        "vllm.model_executor.kernels.mhc.tilelang",
    ]
    import importlib

    return {
        name: hashlib.sha256(
            Path(importlib.import_module(name).__file__).read_bytes()
        ).hexdigest()
        for name in names
    }


class Mxfp8(nn.Module):
    def __init__(self, k, n, seed):
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        raw = torch.randn(n, k, generator=gen, dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        scales = torch.randint(
            119, 122, (n // 32, k // 32), generator=gen, dtype=torch.uint8
        ).repeat_interleave(32, dim=0)
        self.weight = nn.Parameter(raw.to("cuda"), requires_grad=False)
        self.weight_scale = nn.Parameter(scales.to("cuda"), requires_grad=False)
        prepare_mxfp8_layer_for_marlin(self)

    def forward(self, x):
        return apply_mxfp8_marlin_linear(
            x,
            self.weight,
            self.weight_scale,
            self.workspace,
            self.output_size_per_partition,
            self.input_size_per_partition,
        )


def make_cache(tokens):
    bs = 128
    stride = math.ceil(bs * 584 / 576) * 576
    blocks = math.ceil(tokens / bs)
    cache2d = torch.zeros(blocks, stride, device="cuda", dtype=torch.uint8)
    for start in range(0, tokens, 4096):
        length = min(4096, tokens - start)
        k = torch.randn(length, 512, device="cuda", dtype=torch.bfloat16)
        slots = torch.arange(start, start + length, device="cuda", dtype=torch.int64)
        quantize_and_insert_k_cache(k, cache2d, slots, block_size=bs, use_fnuz=False)
    cache3d = cache2d.as_strided((blocks, bs, 576), (stride, 576, 1))
    assert cache3d.stride(0) == stride
    return cache2d, cache3d


def dequant_selected(cache2d, ids):
    ids = ids.cpu().long()
    blocks = ids // 128
    offsets = ids % 128
    base = blocks * cache2d.stride(0)
    data_index = (base + offsets * 576)[:, None] + torch.arange(576, dtype=torch.long)[
        None, :
    ]
    scale_index = (base + 128 * 576 + offsets * 8)[:, None] + torch.arange(
        7, dtype=torch.long
    )[None, :]
    assert int(data_index.max()) < cache2d.numel() and int(data_index.min()) >= 0
    flat = cache2d.reshape(-1)
    data = (
        flat.index_select(0, data_index.flatten().to("cuda"))
        .cpu()
        .reshape(-1, 576)
        .contiguous()
    )
    scales = (
        flat.index_select(0, scale_index.flatten().to("cuda"))
        .cpu()
        .reshape(-1, 7)
        .contiguous()
    )
    no_pe = (
        data[:, :448].contiguous().view(torch.float8_e4m3fn).float().reshape(-1, 7, 64)
    )
    no_pe *= torch.pow(2.0, scales.float() - 127).unsqueeze(-1)
    rope = data[:, 448:576].contiguous().view(torch.bfloat16).float()
    return torch.cat([no_pe.reshape(-1, 448), rope], dim=-1)


class NativeAttention(nn.Module):
    def __init__(self, compressed_tokens=131072, rank=0, world=1):
        super().__init__()
        self.rank, self.world = rank, world
        self.q_a = Mxfp8(5120, 1280, 100)
        self.q_b = Mxfp8(1280, 4096, 101 + rank)
        self.wo_b = Mxfp8(1024, 5120, 102 + rank)
        self.q_norm = RMSNorm(1280, 1e-20).to(device="cuda", dtype=torch.bfloat16)
        self.wo_a = nn.Module()
        self.wo_a.weight = nn.Parameter(
            torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16) * 0.01,
            requires_grad=False,
        )
        self.rotary = SimpleNamespace(
            cos_sin_cache=torch.cat(
                [
                    torch.ones(64, 32, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(64, 32, device="cuda", dtype=torch.bfloat16),
                ],
                dim=-1,
            )
        )
        self.comp2d, self.comp3d = make_cache(compressed_tokens)
        self.swa2d, self.swa3d = make_cache(128)
        self.kernel = DeepseekV41AmpereMLAAttention.__new__(
            DeepseekV41AmpereMLAAttention
        )
        nn.Module.__init__(self.kernel)
        self.kernel.compress_ratio = 1
        self.kernel.compressed_cache_prefix = "probe.compressed"
        self.kernel.swa_cache_layer = SimpleNamespace(
            prefix="probe.swa", kv_cache=self.swa3d
        )
        self.kernel._compressed_kv_cache = lambda: self.comp3d
        self.kernel.attn_sink = torch.linspace(
            -0.5, 0.5, 8, device="cuda", dtype=torch.float32
        )
        self.kernel.scale = 512**-0.5
        self.kernel.head_dim = 512
        self.kernel.nope_head_dim = 448
        self.kernel.rope_head_dim = 64
        self.kernel.kv_cache_dtype = "fp8_ds_mla"
        self.kernel._has_kv_transfer = True
        self.comp_tokens = compressed_tokens
        self.last_q = None
        self.last_attn = None

    def configure(self, tokens):
        self.tokens = tokens
        nreq = max(1, math.ceil(tokens / 6))
        dense_cpu = (
            torch.arange(512, device="cpu", dtype=torch.int64)
            * (self.comp_tokens - 1)
            // 511
        )
        assert (
            int(dense_cpu.min()) == 0 and int(dense_cpu.max()) == self.comp_tokens - 1
        )
        dense = dense_cpu.to(device="cuda", dtype=torch.int32)
        self.kernel.topk_indices_buffer = dense[None, :].repeat(tokens, 1)
        swa_ids = torch.arange(128, device="cuda", dtype=torch.int32)[None, :].repeat(
            tokens, 1
        )
        self.metadata = {
            "probe.compressed": SimpleNamespace(
                block_size=128,
                block_table=torch.arange(
                    self.comp_tokens // 128, device="cuda", dtype=torch.int32
                )[None, :].repeat(nreq, 1),
            ),
            "probe.swa": SimpleNamespace(
                num_decodes=nreq,
                num_decode_tokens=tokens,
                num_prefills=0,
                is_valid_token=torch.ones(tokens, device="cuda", dtype=torch.bool),
                token_to_req_indices=torch.arange(
                    tokens, device="cuda", dtype=torch.int32
                )
                // 6,
                decode_swa_indices=swa_ids,
                decode_swa_lens=torch.full(
                    (tokens,), 128, device="cuda", dtype=torch.int32
                ),
                decode_swa_ragged_indices=None,
                decode_swa_ragged_indptr=None,
            ),
        }

    def forward(self, positions, x, ignored):
        q = self.q_b(self.q_norm(self.q_a(x))).view(-1, 8, 512)
        out = torch.empty_like(q)
        self.kernel.forward_mqa(q, None, positions, out)
        z = rocm_inv_rope_einsum(self.rotary, out, positions, 64, 1, 1024, self.wo_a)
        result = self.wo_b(z.flatten(1))
        if self.world > 1:
            import torch.distributed as dist

            dist.all_reduce(result)
        self.last_q = q
        self.last_attn = out
        return result

    def check_reference(self):
        q = self.last_q[:1].detach().float().cpu()[0]
        comp = dequant_selected(self.comp2d, self.kernel.topk_indices_buffer[0].cpu())
        swa = dequant_selected(self.swa2d, torch.arange(128))
        kv = torch.cat([swa, comp], dim=0)
        logits = q @ kv.T * self.kernel.scale
        logits = torch.cat([logits, self.kernel.attn_sink.cpu()[:, None]], dim=-1)
        expected = logits.softmax(-1)[:, :-1] @ kv
        actual = self.last_attn[0].float().cpu()
        error = float((actual - expected).norm() / expected.norm())
        assert error < 0.015, error
        torch.testing.assert_close(actual, expected, atol=0.007, rtol=0.04)
        return dict(
            relative_l2=error,
            selected_positions=640,
            sink=True,
            format="fp8_ds_mla padded block stride",
            path=(
                "Ampere.forward_mqa -> inherited _forward_decode -> "
                "rocm_sparse_attn_decode"
            ),
            scope=(
                "Core native decode plus Marlin qA/qB/woB and native "
                "inverse-RoPE/BF16 woA; query RoPE identity and KV/indexer "
                "producer excluded"
            ),
        )


class NativeMoE(nn.Module):
    def __init__(self, local_expert_rank=None):
        super().__init__()
        self.local_expert_rank = local_expert_rank
        self.global_ids = (
            ACTIVE_GLOBAL.tolist()
            if local_expert_rank is None
            else [local_expert_rank * 48]
        )
        self.raw = []
        self.packed = []
        self.scales = []
        for k, n in [(5120, 4608), (2304, 5120)]:
            raws = []
            sraws = []
            packed = []
            scales = []
            for eid in self.global_ids:
                gen = torch.Generator(device="cpu").manual_seed(2000 + eid + k)
                raw = torch.randint(
                    0, 256, (n, k // 2), generator=gen, dtype=torch.uint8
                )
                sr = torch.randint(
                    119, 122, (n, k // 32), generator=gen, dtype=torch.uint8
                )
                q = ops.gptq_marlin_repack(
                    raw.to("cuda").view(torch.int32).T.contiguous(),
                    size_k=k,
                    size_n=n,
                    num_bits=4,
                    is_a_8bit=False,
                )
                s = marlin_permute_scales(
                    sr.to("cuda").view(torch.float8_e8m0fnu).to(torch.bfloat16).T,
                    k,
                    n,
                    32,
                )
                s = mxfp4_marlin_process_scales(s).to(torch.float8_e8m0fnu)
                raws.append(raw)
                sraws.append(sr)
                packed.append(q)
                scales.append(s)
            self.raw.append((raws, sraws))
            self.packed.append(torch.stack(packed))
            self.scales.append(torch.stack(scales))
            del packed, scales, q, s
        self.expert_map = torch.full((384,), -1, device="cuda", dtype=torch.int32)
        self.expert_map[torch.tensor(self.global_ids, device="cuda")] = torch.arange(
            len(self.global_ids), device="cuda", dtype=torch.int32
        )
        self.workspace = marlin_make_workspace_new(torch.device("cuda"), 4)
        # Bind the native SM80 dispatcher without constructing TP groups.
        self.gate = GateLinear.__new__(GateLinear)
        nn.Module.__init__(self.gate)
        self.gate.allow_ll_bf16_gemm = False
        self.gate.allow_fp32_router_gemm = False
        self.gate.allow_bf16x3_router_gemm = False
        self.gate.allow_specialized_router_gemm = False
        self.gate._router_gemm_no_bias = True
        self.gate.allow_cublas_router_gemm = True
        self.gate.out_dtype = torch.float32
        self.gate.weight = nn.Parameter(
            torch.randn(384, 5120, device="cuda", dtype=torch.bfloat16) * 0.01,
            requires_grad=False,
        )
        self.mask = torch.zeros(384, device="cuda", dtype=torch.bool)
        self.mask[ACTIVE_GLOBAL.to("cuda")] = True
        self.bias = torch.zeros(384, device="cuda", dtype=torch.float32)
        self.last_route = None
        self.manager = None

    def route(self, x):
        logits, _ = self.gate(x)
        assert logits.dtype == torch.float32
        logits = torch.where(self.mask[None, :], logits, -100.0)
        weights, ids = fused_topk_bias(
            x,
            logits,
            "sqrtsoftplus",
            self.bias,
            6,
            True,
            indices_type=torch.int32,
            routed_scaling_factor=1.5,
        )
        self.last_route = (weights, ids)
        return weights, ids

    def experts(self, x, weights, ids):
        return fused_marlin_moe(
            x,
            self.packed[0],
            self.packed[1],
            None,
            None,
            self.scales[0],
            self.scales[1],
            weights,
            ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=384,
            expert_map=self.expert_map,
            workspace=self.workspace,
            activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
        )

    def forward(self, x, input_ids):
        weights, ids = self.route(x)
        if self.manager is None:
            return self.experts(x, weights, ids)
        recv = self.manager.moe_alltoall.dispatch(
            token_selected_experts=ids,
            input_payloads=[x, ids, weights],
            runtime_max_tokens_per_rank=x.shape[0],
            invalid_token_expert_id=-1,
            expert_id_payload_index=1,
        )
        rx, ri, rw = recv
        out = self.experts(rx.reshape(-1, 5120), rw.reshape(-1, 6), ri.reshape(-1, 6))
        target = torch.empty_like(x)
        self.manager.combine_into(out.view(8, x.shape[0], 5120), x.shape[0], target)
        return target

    def decoded_weight(self, which, index):
        raw, sr = self.raw[which]
        r = raw[index]
        s = sr[index]
        table = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        codes = torch.stack([r & 15, r >> 4], dim=-1).reshape(r.shape[0], -1).long()
        return table[codes] * torch.pow(2.0, s.float() - 127).repeat_interleave(32, 1)

    def check_reference(self, x):
        with torch.inference_mode():
            logits, _ = self.gate(x)
            gate_reference = F.linear(x.float().cpu(), self.gate.weight.float().cpu())
            torch.testing.assert_close(
                logits.cpu(), gate_reference, atol=2e-5, rtol=1e-4
            )
            weights, ids = self.route(x)
            actual = self.experts(x, weights, ids)
            xc = x.float().cpu()
            wc = weights.cpu()
            ic = ids.cpu()
            expected = torch.zeros_like(xc)
            # Dequantize one expert at a time on CPU; no full floating GPU copy.
            for local, eid in enumerate(self.global_ids):
                w1 = self.decoded_weight(0, local)
                up = F.linear(xc, w1).to(torch.bfloat16).float()
                del w1
                gate, up = up.chunk(2, -1)
                activated = (
                    (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10))
                    .to(torch.bfloat16)
                    .float()
                )
                w2 = self.decoded_weight(1, local)
                partial = F.linear(activated, w2).to(torch.bfloat16).float()
                del w2
                effective = torch.where(ic == eid, wc, 0.0).sum(1)
                expected += partial * effective[:, None]
            actualc = actual.float().cpu()
            err = float((actualc - expected).norm() / expected.norm())
            assert err < 0.03, err
            assert bool(torch.isin(ids, ACTIVE_GLOBAL.to(ids.device).int()).all())
            torch.testing.assert_close(
                weights.sum(1),
                torch.full_like(weights[:, 0], 1.5),
                rtol=1e-5,
                atol=1e-5,
            )
            return dict(
                relative_l2=err,
                materialized_experts=len(self.global_ids),
                global_experts=384,
                topk=6,
                scoring="sqrtsoftplus",
                clamp_limit=10.0,
                routed_scale=1.5,
                gate_projection="native GateLinear BF16 GEMV / cuBLAS, FP32 output",
            )


def make_decoder(attention, moe):
    decoder = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
    nn.Module.__init__(decoder)
    decoder.hc_mult = 4
    decoder.hc_sinkhorn_iters = 20
    decoder.hc_eps = 1e-6
    decoder.rms_norm_eps = 1e-20
    decoder.hc_post_alpha = 2.0
    decoder.use_sequence_parallel = False
    decoder.engram = None
    decoder.mhc_stream = None
    decoder.attn_norm = RMSNorm(5120, 1e-20).to(device="cuda", dtype=torch.bfloat16)
    decoder.ffn_norm = RMSNorm(5120, 1e-20).to(device="cuda", dtype=torch.bfloat16)
    for name in ["attn", "ffn"]:
        setattr(
            decoder,
            "hc_" + name + "_fn",
            torch.randn(24, 20480, device="cuda", dtype=torch.float32) * 0.02,
        )
        setattr(
            decoder,
            "hc_" + name + "_scale",
            torch.ones(3, device="cuda", dtype=torch.float32),
        )
        setattr(
            decoder,
            "hc_" + name + "_base",
            torch.randn(24, device="cuda", dtype=torch.float32),
        )
    decoder.attn = attention
    decoder.ffn = moe
    return decoder


def helper_variants():
    original = inspect.getsource(mhc_overlap.mhc_pre_delayed_overlap)
    result = {}
    for name, first, last in [
        ("input_first", "True", "False"),
        ("stats_first", "False", "True"),
    ]:
        src = original.replace("if n <= 8:", f"if {first}:").replace(
            "if n > 8:", f"if {last}:"
        )
        ns = dict(mhc_overlap.__dict__)
        exec(compile(src, f"<mhc-{name}>", "exec"), ns)
        result[name] = ns["mhc_pre_delayed_overlap"]
    return result
