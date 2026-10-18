# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 grouped projection: compact weights, ragged tiles, RoPE and replay.

Compare public kernel outputs with FP32 contraction of the same BF16 operands.
The tolerance covers BF16 output rounding and FP32 split-K accumulation only.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm import envs
from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
from vllm.model_executor.kernels.linear.mxfp8.sm80 import (
    Sm80Mxfp8BmmLinearKernel,
    sm80_mxfp8_bmm,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(80),
    reason="SM80 MXFP8 grouped projection requires CUDA SM8x",
)


def _layer(g, n, k):
    raw = torch.randint(0, 256, (g * n, k), dtype=torch.uint8, device="cuda")
    raw[(raw & 127) == 127] = 0
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        raw.view(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.randint(117, 125, (g * n, k // 32), dtype=torch.uint8, device="cuda"),
        requires_grad=False,
    )
    with patch.object(envs, "VLLM_SM80_MXFP8_BMM", True):
        kernel = init_mxfp8_linear_kernel(bmm_batch_size=g)
    assert isinstance(kernel, Sm80Mxfp8BmmLinearKernel)
    kernel.process_weights_after_loading(layer)
    kernel.process_weights_after_loading(layer)
    assert layer.weight.element_size() == 1
    assert layer.weight.numel() + layer.weight_scale.numel() == g * n * k * 33 // 32
    return layer, kernel


def _inverse_rope(o, positions, cache, rope):
    out = o.clone()
    cs = cache[positions].float().unsqueeze(1)
    a, b = o[..., -rope::2].float(), o[..., -rope + 1 :: 2].float()
    out[..., -rope::2] = (a * cs[..., : rope // 2] + b * cs[..., rope // 2 :]).to(
        o.dtype
    )
    out[..., -rope + 1 :: 2] = (b * cs[..., : rope // 2] - a * cs[..., rope // 2 :]).to(
        o.dtype
    )
    return out


def _check(actual, expected):
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual.float(), expected.bfloat16().float(), rtol=0.008, atol=0.02
    )


@pytest.mark.parametrize("m", [0, 1, 7, 33, 129])
@pytest.mark.parametrize("g,n,k", [(2, 65, 96), (1, 1024, 4096)])
@torch.inference_mode()
def test_grouped_mxfp8_keeps_weights_compact_and_matches_bf16(m, g, n, k):
    """Masked tiles, split-K and prefill agree without a persistent BF16 copy."""
    torch.manual_seed(m + g)
    layer, kernel = _layer(g, n, k)
    x = torch.randn(m, g * 2, k, device="cuda", dtype=torch.bfloat16)[:, ::2]
    bias = torch.randn(g, n, device="cuda", dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x, bias)
    unbiased = kernel.apply_weights(layer, x)
    w = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale)
    ref = torch.einsum("tgk,gnk->tgn", x.float(), w.float())
    _check(unbiased, ref)
    torch.testing.assert_close(out, unbiased + bias, rtol=0, atol=0)


@pytest.mark.parametrize("m,g", [(1, 1), (7, 2), (33, 1), (129, 1)])
@torch.inference_mode()
def test_inverse_rope_grouped_projection_replays_with_new_positions(m, g):
    """Padded head strides and changing positions must survive CUDA capture."""
    torch.manual_seed(m)
    head, rope, heads_per_group, n = 512, 64, 8, 65
    layer, kernel = _layer(g, n, heads_per_group * head)
    o = torch.randn(
        m, g * heads_per_group + 3, head, device="cuda", dtype=torch.bfloat16
    )[:, : g * heads_per_group]
    positions = torch.arange(m, device="cuda", dtype=torch.int64)
    angle = torch.randn(257, rope // 2, device="cuda")
    cache = torch.cat((angle.cos(), angle.sin()), -1)
    w = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale).float()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernel.apply_with_inverse_rope(layer, o, positions, cache, rope)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = kernel.apply_with_inverse_rope(layer, o, positions, cache, rope)
    for offset in [0, 17, 91]:
        positions.copy_((torch.arange(m, device="cuda") + offset) % 257)
        graph.replay()
        rotated = _inverse_rope(o, positions, cache, rope).reshape(m, g, -1)
        ref = torch.einsum("tgk,gnk->tgn", rotated.float(), w)
        _check(out, ref)
    from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
        DeepseekV41AmpereMLAAttention,
    )

    attention = SimpleNamespace(
        wo_a=layer,
        rotary_emb=SimpleNamespace(cos_sin_cache=cache),
        rope_head_dim=rope,
        wo_b=torch.nn.Identity(),
    )
    routed = DeepseekV41AmpereMLAAttention._o_proj(attention, o, positions)
    _check(routed.view(m, g, n), ref)
    assert not hasattr(layer, "_dsv4_wo_a_bf16")
    previous = out.clone()
    graph.replay()
    torch.testing.assert_close(out, previous, rtol=0, atol=0)


@torch.inference_mode()
def test_grouped_mxfp8_exact_exponents_and_zero_rows():
    """Tiny/large valid scales, signed zero and FP8 subnormals decode correctly."""
    raw = torch.arange(256, device="cuda", dtype=torch.uint8).repeat(4, 4)
    raw[(raw & 127) == 127] = 0
    w = raw.view(torch.float8_e4m3fn).view(1, 4, 1024)
    scale = torch.tensor(
        [0, 1, 16, 64, 117, 127, 200, 225], device="cuda", dtype=torch.uint8
    ).repeat(1, 4, 4)
    x = torch.ones(3, 1, 1024, dtype=torch.bfloat16, device="cuda")
    x[1] = 0
    x[2, :, ::2] = -1
    out = sm80_mxfp8_bmm(x, w, scale)
    wb = dequant_mxfp8_to_bf16(w, scale)
    ref = torch.einsum("tgk,gnk->tgn", x.float(), wb.float())
    _check(out, ref)
    assert not out[1].count_nonzero()


@torch.inference_mode()
def test_grouped_mxfp8_exhaustive_value_decode():
    """Every finite E4M3 byte at extreme E8M0 scales matches Torch exactly."""
    codes = torch.arange(256, dtype=torch.uint8, device="cuda")
    codes = codes[(codes & 127) != 127]
    exponents = torch.tensor(
        [0, 1, 8, 64, 117, 127, 200, 240], dtype=torch.uint8, device="cuda"
    )
    raw = codes.repeat(exponents.numel()).view(1, -1, 1).expand(-1, -1, 32)
    weight = raw.contiguous().view(torch.float8_e4m3fn)
    scales = exponents.repeat_interleave(codes.numel()).view(1, -1, 1)
    x = torch.zeros(1, 1, 32, dtype=torch.bfloat16, device="cuda")
    x[..., 0] = 1
    out = sm80_mxfp8_bmm(x, weight, scales)
    ref = dequant_mxfp8_to_bf16(weight, scales)[..., 0].unsqueeze(0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_grouped_projection_requires_explicit_opt_in():
    with patch.object(envs, "VLLM_SM80_MXFP8_BMM", False):
        kernel = init_mxfp8_linear_kernel(bmm_batch_size=1)
    assert not isinstance(kernel, Sm80Mxfp8BmmLinearKernel)
