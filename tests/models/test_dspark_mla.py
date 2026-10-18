# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    mapper = K3DSparkForCausalLM.hf_to_vllm_mapper
    for name in ("confidence_head.weight", "embed_tokens.weight", "lm_head.weight"):
        assert mapper._map_name(name) is None


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []
    context_kv_proj_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    def make_context_kv_proj(*args, **kwargs):
        context_kv_proj_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", make_context_kv_proj)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        kv_lora_rank=3,
        qk_rope_head_dim=1,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        # K3DSparkModel.__init__ sizes its context-KV buffer from the scheduler
        # budget (dspark_mla.py:150-152), so the stand-in config has to carry
        # it. The value is arbitrary here -- nothing in this test reaches the
        # buffer -- but it must exist.
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1
    assert context_kv_proj_calls == [
        (
            (8, [4]),
            {
                "bias": False,
                "return_bias": False,
                "quant_config": None,
                "prefix": "model.layers.0.self_attn.fused_qkv_a_proj",
                "disable_tp": True,
            },
        )
    ]


def test_context_kv_weights_are_loaded_as_merged_linear_shards():
    weights = [
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight_packed",
            torch.arange(4),
        ),
        (
            "layers.1.self_attn.kv_a_proj_with_mqa.weight_scale",
            torch.tensor(0.5),
        ),
    ]

    duplicated = dspark_mla._duplicate_context_kv_weights(weights, 2)
    mapped = list(K3DSparkForCausalLM.hf_to_vllm_mapper.apply(duplicated))

    assert [name for name, _ in mapped] == [
        "model.layers.0.self_attn.fused_qkv_a_proj.weight_packed",
        "model.context_kv_proj.weight_packed",
        "model.layers.1.self_attn.fused_qkv_a_proj.weight_scale",
        "model.context_kv_proj.weight_scale",
    ]
    assert [weight.shard_id for _, weight in mapped] == [1, 0, 1, 1]
    assert mapped[0][1].data_ptr() == mapped[1][1].data_ptr()
    assert mapped[2][1].data_ptr() == mapped[3][1].data_ptr()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "scale_dtype", [torch.uint8, torch.float8_e8m0fnu, torch.float32]
)
@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_module", "shard_id"),
    [
        ("mtp.0.attn.wq_a.scale", "model.layers.0.attn.fused_wqa_wkv", 0),
        ("mtp.0.attn.wkv.scale", "model.layers.0.attn.fused_wqa_wkv", 1),
        ("mtp.0.main_proj.scale", "model.main_proj", None),
        (
            "mtp.0.ffn.shared_experts.w1.scale",
            "model.layers.0.ffn.shared_experts.gate_up_proj",
            0,
        ),
        (
            "mtp.0.ffn.shared_experts.w2.scale",
            "model.layers.0.ffn.shared_experts.down_proj",
            None,
        ),
    ],
)
def test_v41_dspark_loads_linear_scales(
    monkeypatch, scale_dtype, checkpoint_name, runtime_module, shard_id
):
    """Checkpoint ``.scale`` maps to the quant method's scale parameter and
    loads untouched. MXFP8 block-scale expansion lives in the KMxfp8Static
    loader (see tests/quantization/test_modelopt.py), not in load_weights."""
    from vllm.models.deepseek_v4_1.nvidia import dspark

    mxfp8 = scale_dtype != torch.float32
    scale_name = "weight_scale" if mxfp8 else "weight_scale_inv"
    runtime_name = f"{runtime_module}.{scale_name}"
    raw = torch.tensor([[120, 127], [128, 130]], dtype=torch.uint8)
    checkpoint_scale = raw.view(scale_dtype) if mxfp8 else raw.float()
    param = nn.Parameter(torch.empty_like(checkpoint_scale), requires_grad=False)
    shards = []

    def load_scale(param, weight, *args):
        shards.append(args)
        assert weight.dtype == checkpoint_scale.dtype
        param.copy_(weight)

    param.weight_loader = load_scale
    draft = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=4, n_routed_experts=1),
        quant_config=SimpleNamespace(
            weight_block_size=[32, 32] if mxfp8 else [128, 128]
        ),
        linear_scale_name=scale_name,
        pad_shared_expert=False,
        model=SimpleNamespace(
            layers=[SimpleNamespace(ffn=SimpleNamespace(use_mega_moe=False))],
            confidence_head=None,
        ),
        named_parameters=lambda: [(runtime_name, param)],
        process_weights_after_loading=lambda: None,
    )
    draft._remap_dspark_name = lambda name: (
        dspark.DSparkDeepseekV4ForCausalLM._remap_dspark_name(draft, name)
    )
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        dspark, "fused_moe_make_expert_params_mapping", lambda *a, **kw: []
    )

    loaded = dspark.DSparkDeepseekV4ForCausalLM.load_weights(
        draft, [(checkpoint_name, checkpoint_scale)]
    )

    assert loaded == {runtime_name}
    assert shards == [() if shard_id is None else (shard_id,)]
    torch.testing.assert_close(param, checkpoint_scale)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "fallback",
    [
        None,
        "device",
        "block",
        "scale",
        "kernel",
        "new_kernel",
        "geometry",
        "layers",
        "sp",
        "lora",
        "disabled",
    ],
)
def test_v41_context_projection_requires_screened_sm80_configuration(
    monkeypatch, fallback
):
    from vllm.models.deepseek_v4_1.nvidia import dspark

    monkeypatch.setattr(
        dspark.envs, "VLLM_DSV41_DSPARK_CONTEXT_KV", fallback != "disabled"
    )
    monkeypatch.setattr(dspark.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        dspark.current_platform,
        "is_device_capability",
        lambda _cc: fallback != "device",
    )
    config = SimpleNamespace(
        quant_config=SimpleNamespace(
            weight_block_size=[32, 32], is_scale_e8m0=fallback != "scale"
        )
    )
    kernel = object.__new__(dspark.MarlinMxfp8LinearKernel)
    projections = [
        SimpleNamespace(
            input_size=5120,
            output_sizes=[1280, 512],
            disable_tp=True,
            prefix=f"model.layers.{40 + i}.attn.fused_wqa_wkv",
            quant_method=SimpleNamespace(kernel=kernel),
        )
        for i in range(3)
    ]
    if fallback == "kernel":
        projections[1].quant_method.kernel = object()
    if fallback == "geometry":
        projections[1].output_sizes = [1280, 1024]
    if fallback == "block":
        config.quant_config.weight_block_size = [128, 128]
    if fallback == "layers":
        projections.pop()
    if fallback == "lora":
        config.lora_config = SimpleNamespace()
    layers = [
        SimpleNamespace(attn=SimpleNamespace(fused_wqa_wkv=p)) for p in projections
    ]
    constructor = Mock(
        return_value=SimpleNamespace(
            quant_method=SimpleNamespace(
                kernel=object() if fallback == "new_kernel" else kernel
            )
        )
    )
    monkeypatch.setattr(dspark, "MergedColumnParallelLinear", constructor)

    result = dspark._make_sm80_context_kv_proj(config, layers, fallback == "sp")

    if fallback == "new_kernel":
        assert result is None
        constructor.assert_called_once()
    elif fallback is not None:
        assert result is None
        constructor.assert_not_called()
    else:
        assert result is constructor.return_value
        constructor.assert_called_once_with(
            5120,
            [512, 512, 512],
            bias=False,
            return_bias=False,
            quant_config=config.quant_config,
            prefix=projections[0].prefix,
            disable_tp=True,
        )


def _make_v41_context_loader(monkeypatch, rank, scale_dtype):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4_1.nvidia import dspark
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config

    for module in (linear, parameter, dspark):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 8)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
    kernel = Mock()
    kernel.input_quant_key.return_value = None
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.modelopt.init_mxfp8_linear_kernel",
        lambda **kwargs: kernel,
    )
    monkeypatch.setattr(
        dspark, "fused_moe_make_expert_params_mapping", lambda *a, **kw: []
    )
    quant = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[32, 32]
    )
    draft = dspark.DSparkDeepseekV4ForCausalLM.__new__(
        dspark.DSparkDeepseekV4ForCausalLM
    )
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(num_attention_heads=8, n_routed_experts=1)
    draft.quant_config = quant
    draft.linear_scale_name = "weight_scale"
    draft.pad_shared_expert = False
    draft.model = nn.Module()
    draft.model.confidence_head = None
    draft.model.layers = nn.ModuleList()
    for i in range(3):
        layer = nn.Module()
        layer.ffn = SimpleNamespace(use_mega_moe=False)
        layer.attn = nn.Module()
        layer.attn.fused_wqa_wkv = linear.MergedColumnParallelLinear(
            64, [32, 32], bias=False, quant_config=quant, disable_tp=True
        )
        draft.model.layers.append(layer)
    draft.model.context_kv_proj = linear.MergedColumnParallelLinear(
        64, [32] * 3, bias=False, return_bias=False, quant_config=quant, disable_tp=True
    )
    monkeypatch.setattr(draft, "process_weights_after_loading", lambda: None)
    weights = []
    for i in range(3):
        for part, value in (("wq_a", -1), ("wkv", i + 1)):
            weights.append(
                (
                    f"mtp.{i}.attn.{part}.weight",
                    torch.full((32, 64), value, dtype=torch.float32).to(
                        torch.float8_e4m3fn
                    ),
                )
            )
            base = 123 if part == "wq_a" else 127
            scales = torch.tensor([[base + i, base + i + 1]], dtype=torch.uint8)
            weights.append((f"mtp.{i}.attn.{part}.scale", scales.view(scale_dtype)))
    return draft, weights


@pytest.mark.cpu_test
@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
def test_v41_context_projection_loads_native_mxfp8_shards(
    monkeypatch, default_vllm_config, rank, scale_dtype
):
    draft, weights = _make_v41_context_loader(monkeypatch, rank, scale_dtype)
    source = {name: weight.clone() for name, weight in weights}
    loaded = draft.load_weights(weights)

    assert {
        "model.context_kv_proj.weight",
        "model.context_kv_proj.weight_scale",
    } <= loaded
    assert len(loaded) == 8  # Two params per original layer plus the two merged params.
    context = draft.model.context_kv_proj
    assert context.tp_rank == 0 and context.tp_size == 1
    for i, layer in enumerate(draft.model.layers):
        kv = source[f"mtp.{i}.attn.wkv.weight"]
        scales = (
            source[f"mtp.{i}.attn.wkv.scale"].view(torch.uint8).repeat_interleave(32, 0)
        )
        assert torch.equal(
            context.weight[i * 32 : (i + 1) * 32].view(torch.uint8),
            kv.view(torch.uint8),
        )
        assert torch.equal(context.weight_scale[i * 32 : (i + 1) * 32], scales)
        original = layer.attn.fused_wqa_wkv
        assert torch.equal(original.weight[32:].view(torch.uint8), kv.view(torch.uint8))
        assert torch.equal(original.weight_scale[32:], scales)
        q = source[f"mtp.{i}.attn.wq_a.weight"]
        q_scales = (
            source[f"mtp.{i}.attn.wq_a.scale"]
            .view(torch.uint8)
            .repeat_interleave(32, 0)
        )
        assert torch.equal(original.weight[:32].view(torch.uint8), q.view(torch.uint8))
        assert torch.equal(original.weight_scale[:32], q_scales)
    for name, weight in weights:
        assert torch.equal(weight.view(torch.uint8), source[name].view(torch.uint8))


@pytest.mark.cpu_test
@pytest.mark.parametrize("missing", ["weight", "scale"])
def test_v41_context_projection_rejects_missing_layer_shards(
    monkeypatch, default_vllm_config, missing
):
    draft, weights = _make_v41_context_loader(monkeypatch, 7, torch.uint8)
    weights = [(name, w) for name, w in weights if name != f"mtp.2.attn.wkv.{missing}"]
    with pytest.raises(
        ValueError, match="Missing checkpoint shards for context_kv_proj"
    ):
        draft.load_weights(weights)


@pytest.mark.cpu_test
def test_v41_context_projection_rejects_unconfigured_checkpoint_layer(
    monkeypatch, default_vllm_config
):
    draft, weights = _make_v41_context_loader(monkeypatch, 7, torch.uint8)
    weights.append(("mtp.3.attn.wkv.weight", weights[2][1]))
    with pytest.raises(ValueError, match="DSpark layer 3 is out of range"):
        draft.load_weights(weights)


@pytest.mark.cpu_test
@pytest.mark.parametrize("mapping_mode", ["per_layer", "partial_none", "profiling"])
def test_v41_context_projection_reuses_per_layer_norm_rope_cache(
    monkeypatch, mapping_mode
):
    from vllm.models.deepseek_v4_1.nvidia import dspark

    states = torch.arange(40, dtype=torch.float32).reshape(5, 8) / 40
    positions = torch.tensor([0, 63, 64, 127, 128])
    slots = [torch.tensor([i, -1, i + 2, -1, i + 4]) for i in (10, 20, 30)]
    records = []
    monkeypatch.setattr(
        dspark,
        "_insert_context_kv",
        lambda attn, kv, pos, slot: records.append((attn, kv.clone(), pos, slot)),
    )
    weights = [
        torch.arange(64, dtype=torch.float32).reshape(8, 8) / (10 + i) for i in range(3)
    ]
    query_projections = [
        Mock(side_effect=lambda x, w=w: (x @ w.T, None)) for w in weights
    ]
    layers = [
        SimpleNamespace(
            attn=SimpleNamespace(
                fused_wqa_wkv=projection,
                q_lora_rank=4,
                head_dim=4,
                kv_norm=nn.RMSNorm(4, eps=1e-6),
            )
        )
        for projection in query_projections
    ]
    model = SimpleNamespace(layers=layers, context_kv_proj=None)
    mappings = None if mapping_mode == "profiling" else slots
    if mapping_mode == "partial_none":
        assert mappings is not None
        mappings[1] = None
    dspark.DSparkDeepseekV4Model.precompute_and_store_context_kv(
        model, states, positions, mappings
    )
    expected = list(records)
    records.clear()
    grouped_weight = torch.cat([w[4:] for w in weights])
    grouped = Mock(side_effect=lambda x: x @ grouped_weight.T)
    model.context_kv_proj = grouped
    for projection in query_projections:
        projection.reset_mock()
    dspark.DSparkDeepseekV4Model.precompute_and_store_context_kv(
        model, states, positions, mappings
    )
    grouped.assert_called_once_with(states)
    for projection in query_projections:
        projection.assert_not_called()
    expected_count = {"per_layer": 3, "partial_none": 2, "profiling": 0}[mapping_mode]
    assert len(records) == len(expected) == expected_count
    for actual, reference in zip(records, expected):
        assert actual[0] is reference[0]
        torch.testing.assert_close(actual[1], reference[1])
        assert actual[2] is reference[2] is positions
        assert actual[3] is reference[3]
