# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 port of the Blackwell MXFP8 grouped projection (recipe 1, 1, 32).

Keep E4M3/E8M0 weights resident, decode tiles to BF16, and use mma.sync with
FP32 accumulation. Activations remain BF16, matching the Ampere reference.
"""

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


@triton.jit
def _decode_e4m3(u):
    u = u.to(tl.uint32)
    mag = u & 127
    sub = (mag & 7).to(tl.float32) * (1.0 / 512.0)
    bits = tl.where(mag < 8, sub.to(tl.uint32, bitcast=True), (mag << 20) + 0x3C000000)
    bits = tl.where(mag == 127, 0x7FC00000, bits) | ((u & 128) << 24)
    return bits.to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _dequant_weight(W, S, B, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    raw = tl.load(W + i, i < SIZE, 0)
    scale = tl.load(S + i // 32, i < SIZE, 127)
    sf = (
        tl.where(scale == 0, 0x00400000, scale.to(tl.uint32) << 23)
        .to(tl.uint32)
        .to(tl.float32, bitcast=True)
    )
    tl.store(B + i, (_decode_e4m3(raw) * sf).to(tl.bfloat16), i < SIZE)


@triton.jit
def _grouped_gemv(
    X,
    W,
    S,
    Y,
    Partial,
    Positions,
    Cache,
    Bias,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    XM: tl.constexpr,
    XG: tl.constexpr,
    XH: tl.constexpr,
    CS: tl.constexpr,
    HEAD: tl.constexpr,
    ROPE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    n = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    mg = tl.program_id(1)
    group = mg % G
    m = mg // G * BM + tl.arange(0, BM)
    split = tl.program_id(2)
    k = split * BK + tl.arange(0, BK)
    if ROPE > 0:
        head, dim = k // HEAD, k % HEAD
        xp = X + m[:, None] * XM + group * XG + head[None, :] * XH
        a = tl.load(xp + dim[None, :], (m[:, None] < M) & (k[None, :] < K), 0).to(
            tl.float32
        )
        partner = tl.load(
            xp + (dim ^ 1)[None, :],
            (m[:, None] < M) & (k[None, :] < K) & (dim[None, :] >= HEAD - ROPE),
            0,
        ).to(tl.float32)
        pos = tl.load(Positions + m, m < M, 0).to(tl.int64)
        cp = Cache + pos[:, None] * CS + (dim[None, :] - (HEAD - ROPE)) // 2
        mask = (m[:, None] < M) & (dim[None, :] >= HEAD - ROPE)
        cos = tl.load(cp, mask, 0).to(tl.float32)
        sin = tl.load(cp + ROPE // 2, mask, 0).to(tl.float32)
        rotated = a * cos + tl.where((dim & 1)[None, :] == 0, partner, -partner) * sin
        a = tl.where(dim[None, :] >= HEAD - ROPE, rotated, a).to(tl.bfloat16)
    else:
        a = tl.load(
            X + m[:, None] * XM + group * XG + k[None, :],
            (m[:, None] < M) & (k[None, :] < K),
            0,
        )
    raw = tl.load(
        W + (group * N + n[:, None]) * K + k[None, :],
        (n[:, None] < N) & (k[None, :] < K),
        0,
    )
    scale = tl.load(
        S + (group * N + n[:, None]) * (K // 32) + k[None, :] // 32,
        (n[:, None] < N) & (k[None, :] < K),
        127,
    )
    sf = (
        tl.where(scale == 0, 0x00400000, scale.to(tl.uint32) << 23)
        .to(tl.uint32)
        .to(tl.float32, bitcast=True)
    )
    w = (_decode_e4m3(raw) * sf).to(tl.bfloat16)
    acc = tl.sum(a[:, None, :].to(tl.float32) * w[None, :, :].to(tl.float32), 2)
    offset = (m[:, None] * G + group) * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        value = acc.to(tl.bfloat16)
        if HAS_BIAS:
            value = value.to(tl.float32) + tl.load(
                Bias + group * N + n[None, :], n[None, :] < N, 0
            )
        tl.store(Y + offset, value, mask)
    else:
        tl.store(Partial + split * M * G * N + offset, acc, mask)


@triton.jit
def _grouped_mm(
    X,
    W,
    Y,
    Bias,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    XM: tl.constexpr,
    XG: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    group = tl.program_id(2)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        k = step * BK + rk
        a = tl.load(
            X + m[:, None] * XM + group * XG + k[None, :],
            (m[:, None] < M) & (k[None, :] < K),
            0,
        )
        b = tl.load(
            W + (group * N + n[:, None]) * K + k[None, :],
            (n[:, None] < N) & (k[None, :] < K),
            0,
        )
        acc = tl.dot(a, tl.trans(b), acc)
    value = acc.to(tl.bfloat16)
    if HAS_BIAS:
        value = value.to(tl.float32) + tl.load(
            Bias + group * N + n[None, :], n[None, :] < N, 0
        )
    offset = (m[:, None] * G + group) * N + n[None, :]
    tl.store(Y + offset, value, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce_splits(
    Partial,
    Y,
    Bias,
    SIZE: tl.constexpr,
    GN: tl.constexpr,
    SPLIT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    value = tl.full((BLOCK,), 0, tl.float32)
    for s in tl.static_range(SPLIT):
        value += tl.load(Partial + s * SIZE + i, i < SIZE, 0)
    # Match the existing BMM's BF16 output, followed by a BF16 bias add.
    value = value.to(tl.bfloat16)
    if HAS_BIAS:
        value = value.to(tl.float32) + tl.load(Bias + i % GN)
    tl.store(Y + i, value, i < SIZE)


def sm80_mxfp8_bmm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    rope_dim: int = 0,
) -> torch.Tensor:
    """Project [T,G,K], optionally applying inverse GPT-J RoPE to [T,H,D].

    Weight and scale are contiguous [G,N,K] E4M3 and [G,N,K/32] E8M0.
    Up to eight tokens use fused tile decoding and inverse RoPE. Larger batches
    use transient BF16 weights and SM80 tensor cores. No BF16 weights persist
    between calls. Split-K partials use fixed-order FP32 reduction.
    """
    assert x.dtype == torch.bfloat16 and x.ndim == 3 and x.stride(-1) == 1
    assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous()
    assert scale.dtype == torch.uint8 and scale.is_contiguous()
    g, n, k = weight.shape
    m = x.shape[0]
    assert 0 < g <= 65535 and n > 0 and k > 0
    assert k % 32 == 0 and scale.shape == (g, n, k // 32)
    if rope_dim:
        assert positions is not None and cos_sin_cache is not None
        assert 0 < rope_dim <= x.shape[2] and rope_dim % 2 == 0
        assert x.shape[1] % g == 0 and x.shape[1] * x.shape[2] == g * k
        assert positions.shape == (m,) and positions.is_contiguous()
        assert cos_sin_cache.shape[1] == rope_dim
        assert cos_sin_cache.stride(1) == 1
        head = x.shape[2]
        xg, xh = x.stride(1) * (x.shape[1] // g), x.stride(1)
        cs = cos_sin_cache.stride(0)
    else:
        assert x.shape[1:] == (g, k)
        head, xg, xh, cs = 0, x.stride(1), 0, 0
    if bias is not None:
        assert bias.numel() == g * n and bias.is_contiguous()
        assert bias.dtype == torch.bfloat16
    out = torch.empty((m, g, n), dtype=torch.bfloat16, device=x.device)
    if m == 0:
        return out
    if m <= 8:
        bm, bn, bk = triton.next_power_of_2(m), 4, min(1024, triton.next_power_of_2(k))
        split = triton.cdiv(k, bk)
        partial = (
            torch.empty((split, m, g, n), dtype=torch.float32, device=x.device)
            if split > 1
            else out
        )
        _grouped_gemv[(triton.cdiv(n, bn), triton.cdiv(m, bm) * g, split)](
            x,
            weight.view(torch.uint8),
            scale,
            out,
            partial,
            positions,
            cos_sin_cache,
            bias,
            m,
            n,
            k,
            g,
            x.stride(0),
            xg,
            xh,
            cs,
            head,
            rope_dim,
            bm,
            bn,
            bk,
            split,
            bias is not None,
            num_warps=4 if m == 1 else 8,
            enable_fp_fusion=False,
        )
    else:
        decoded = torch.empty(weight.shape, dtype=torch.bfloat16, device=weight.device)
        _dequant_weight[(triton.cdiv(weight.numel(), 1024),)](
            weight.view(torch.uint8),
            scale,
            decoded,
            weight.numel(),
            1024,
        )
        if rope_dim:
            from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
                _fused_inverse_rope_gptj,
            )

            x = _fused_inverse_rope_gptj(x, positions, cos_sin_cache, rope_dim).view(
                m, g, k
            )
            xg = x.stride(1)
        bm, bn, bk, split = 32, 32, 64, 1
        partial = out
        _grouped_mm[(triton.cdiv(m, bm), triton.cdiv(n, bn), g)](
            x,
            decoded,
            out,
            bias,
            m,
            n,
            k,
            g,
            x.stride(0),
            xg,
            bm,
            bn,
            bk,
            bias is not None,
            num_warps=4,
            num_stages=2,
        )
    if split > 1:
        _reduce_splits[(triton.cdiv(out.numel(), 256),)](
            partial,
            out,
            bias,
            out.numel(),
            g * n,
            split,
            bias is not None,
            256,
            num_warps=4,
        )
    return out


class Sm80Mxfp8BmmLinearKernel(Mxfp8LinearKernel):
    """Opt-in W8A16 grouped projection on CUDA SM8x."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not envs.VLLM_SM80_MXFP8_BMM:
            return False, "VLLM_SM80_MXFP8_BMM is disabled."
        if not current_platform.is_cuda():
            return False, "SM80 MXFP8 BMM requires CUDA."
        device_cap = current_platform.get_device_capability()
        cap = compute_capability or (device_cap.to_int() if device_cap else 0)
        return (
            (True, None) if 80 <= cap < 90 else (False, "SM80 MXFP8 BMM requires SM8x.")
        )

    @classmethod
    def can_implement(cls, config: Mxfp8LinearLayerConfig):
        ok = config.bmm_batch_size is not None and config.bmm_batch_size > 0
        return ok, None if ok else "A positive group count is required."

    def process_weights_after_loading(self, layer):
        w, s = layer.weight.data, layer.weight_scale.data
        g = self.config.bmm_batch_size
        if w.ndim == 3:
            layer._sm80_mxfp8_bmm = self
            return
        assert g is not None and w.ndim == 2 and w.shape[0] % g == 0
        n, k = w.shape[0] // g, w.shape[1]
        assert w.dtype == torch.float8_e4m3fn and k % 32 == 0
        assert s.dtype == torch.uint8 and s.ndim == 2
        layer.weight = Parameter(w.contiguous().view(g, n, k), requires_grad=False)
        layer.weight_scale = Parameter(
            s[: g * n, : k // 32].contiguous().view(g, n, k // 32),
            requires_grad=False,
        )
        layer._sm80_mxfp8_bmm = self

    def apply_weights(self, layer, x, bias=None):
        if x.dtype != torch.bfloat16:
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
                dequant_mxfp8_to_bf16,
            )

            weight = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale).to(x.dtype)
            out = torch.einsum("tgk,gnk->tgn", x, weight)
            return out if bias is None else out + bias.view(out.shape[1:])
        return sm80_mxfp8_bmm(x, layer.weight, layer.weight_scale, bias)

    def apply_with_inverse_rope(self, layer, x, positions, cache, rope_dim):
        return sm80_mxfp8_bmm(
            x,
            layer.weight,
            layer.weight_scale,
            positions=positions,
            cos_sin_cache=cache,
            rope_dim=rope_dim,
        )
