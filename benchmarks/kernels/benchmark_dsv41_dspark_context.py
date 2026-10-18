# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen DSpark context projection: three fused Q/KV vs grouped KV-only MXFP8.

Both paths use native SM80 Marlin, H=5120, Q=1280, KV=512 and the same three
RMSNorm modules. TP-replicated projections have no collectives. RoPE, quantization
and cache insertion are excluded from timing and remain unchanged. These are
projection-boundary microseconds, not model tokens/s.

--weights accepts a CPU safetensors extract with exactly these 15 tensors:
context_weights.{0,1,2}.{q,kv,q_scale,kv_scale,norm}. q/kv are unpacked E4M3FN
[1280,5120]/[512,5120]; block32 scales are E8M0 bytes [40,160]/[16,160]. Scales
are expanded by 32 rows before native Marlin preparation. Raw CPU weights are
concatenated before each independent pack; no packed weight is ever sliced.
Without --weights, inputs and weights are explicitly synthetic.

Run with external fresh >=4 GiB free VRAM admission and timeout <=240s. This
also checks free VRAM inside and caps Torch at 512 MiB. No profiling, cache flush,
autotuning sweep or clock changes. Default timing is four alternating samples
of 20 graph replays each. All shapes finish and retain numerical differences;
exit2 means strict changed-input/restore/repeat bitwise checks did not all pass.
"""

import argparse
import gc
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
from benchmark_dsv41_native_consumers import Mxfp8
from safetensors import safe_open
from torch import nn

import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8

HIDDEN, Q_RANK, KV_DIM, LAYERS, BLOCK = 5120, 1280, 512, 3, 32


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def load_weights(path, seed):
    metadata = {}
    if path is not None:
        with safe_open(path, framework="pt", device="cpu") as source:
            expected = {
                f"context_weights.{layer}.{name}"
                for layer in range(LAYERS)
                for name in ("q", "kv", "q_scale", "kv_scale", "norm")
            }
            assert set(source.keys()) == expected, "Expected canonical 15 tensor keys"
            weights = {name: source.get_tensor(name) for name in expected}
            metadata = source.metadata() or {}
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weights = {}
        for layer in range(LAYERS):
            for name, rows in (("q", Q_RANK), ("kv", KV_DIM)):
                weights[f"context_weights.{layer}.{name}"] = torch.randn(
                    rows, HIDDEN, generator=generator, dtype=torch.float32
                ).to(torch.float8_e4m3fn)
                weights[f"context_weights.{layer}.{name}_scale"] = torch.randint(
                    119,
                    122,
                    (rows // BLOCK, HIDDEN // BLOCK),
                    generator=generator,
                    dtype=torch.uint8,
                )
            weights[f"context_weights.{layer}.norm"] = torch.ones(
                KV_DIM, dtype=torch.bfloat16
            )
    records = []
    for layer in range(LAYERS):
        tensors = {
            name: weights[f"context_weights.{layer}.{name}"]
            for name in ("q", "kv", "q_scale", "kv_scale", "norm")
        }
        for name, rows in (("q", Q_RANK), ("kv", KV_DIM)):
            value, scales = tensors[name], tensors[f"{name}_scale"]
            assert value.shape == (rows, HIDDEN)
            assert value.dtype == torch.float8_e4m3fn
            assert scales.shape == (rows // BLOCK, HIDDEN // BLOCK)
            assert scales.dtype in (torch.uint8, torch.float8_e8m0fnu)
            tensors[name] = value.contiguous()
            tensors[f"{name}_scale"] = scales.contiguous().view(torch.uint8)
            assert bool((tensors[f"{name}_scale"] != 255).all()), "NaN E8M0 scale"
        assert tensors["norm"].shape == (KV_DIM,)
        assert tensors["norm"].dtype in (torch.bfloat16, torch.float32, torch.float16)
        assert bool(torch.isfinite(tensors["norm"]).all())
        records.append(tensors)
    return records, dict(
        mode="supplied_safetensors" if path else "synthetic",
        sha256=digest(path) if path else None,
        metadata=metadata,
        tensor_specs={
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in weights.items()
        },
    )


def concatenate_fp8(weights):
    # Byte concatenation also works on CPU builds without float8 cat kernels.
    return torch.cat([weight.view(torch.uint8) for weight in weights], dim=0).view(
        torch.float8_e4m3fn
    )


def make_projection(raw_weight, block_scales):
    layer = Mxfp8.__new__(Mxfp8)
    nn.Module.__init__(layer)
    layer.input_size_per_partition = HIDDEN
    layer.output_size_per_partition = raw_weight.shape[0]
    layer.weight = nn.Parameter(raw_weight.to("cuda"), requires_grad=False)
    scales = block_scales.repeat_interleave(BLOCK, dim=0).contiguous()
    layer.weight_scale = nn.Parameter(scales.to("cuda"), requires_grad=False)
    marlin_utils_fp8.prepare_mxfp8_layer_for_marlin(layer)
    return layer


def difference(actual, expected):
    delta = actual.float() - expected.float()
    assert bool(torch.isfinite(delta).all()), "Nonfinite context output"
    return dict(
        bitwise_equal=torch.equal(actual, expected),
        mismatched=int((actual != expected).sum()),
        numel=actual.numel(),
        max_absolute=float(delta.abs().max()),
        relative_l2=float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
    )


def compare(actual, expected):
    return [difference(a, b) for a, b in zip(actual, expected, strict=True)]


def cpu_reference(x, outputs, weights, epsilon):
    """Small independent dequant/matmul check outside the timed boundary."""
    count = min(x.shape[0], 2)
    xc = x[:count].float().cpu()
    records = []
    for i, tensors in enumerate(weights):
        scales = torch.exp2(tensors["kv_scale"].float() - 127)
        scales = scales.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        weight = tensors["kv"].float() * scales
        kv = torch.nn.functional.linear(xc, weight).to(torch.bfloat16)
        normalized = kv.float() * torch.rsqrt(
            kv.float().square().mean(-1, True) + epsilon
        )
        normalized = (normalized.to(tensors["norm"].dtype) * tensors["norm"]).to(
            torch.bfloat16
        )
        raw_error = difference(outputs[LAYERS + i][:count].cpu(), kv)
        norm_error = difference(outputs[i][:count].cpu(), normalized)
        assert raw_error["relative_l2"] < 0.02, raw_error
        assert norm_error["relative_l2"] < 0.02, norm_error
        records.append(dict(layer=i, raw=raw_error, normalized=norm_error))
    return records


def run_shape(args, functions, tokens, weights):
    x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    original = x.clone()
    graphs = {}
    try:
        for name, function in functions.items():
            for _ in range(2):
                function(x)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = function(x)
            graphs[name] = graph, output
        initial = {}
        checks = []
        for step, factor in enumerate((1.0, -0.75, 1.0)):
            x.copy_(original * factor)
            repeats = {}
            for name, (graph, output) in graphs.items():
                graph.replay()
                torch.accelerator.synchronize()
                previous = [value.clone() for value in output]
                graph.replay()
                torch.accelerator.synchronize()
                repeats[name] = compare(output, previous)
                if step == 0:
                    initial[name] = [value.clone() for value in output]
            check = dict(
                factor=factor,
                grouped_vs_three_fused=compare(
                    graphs["grouped_kv"][1], graphs["three_fused_qkv"][1]
                ),
                same_graph_repeat=repeats,
                changed_from_initial={
                    name: compare(output, initial[name])
                    for name, (_, output) in graphs.items()
                }
                if step == 1
                else None,
                restored_initial={
                    name: compare(output, initial[name])
                    for name, (_, output) in graphs.items()
                }
                if step == 2
                else None,
            )
            if step == 1:
                assert all(
                    any(not diff["bitwise_equal"] for diff in values[:LAYERS])
                    for values in check["changed_from_initial"].values()
                )
            checks.append(check)
        reference = (
            cpu_reference(x, graphs["three_fused_qkv"][1], weights, args.norm_eps)
            if tokens == args.tokens[0]
            else None
        )
        samples = {name: [] for name in graphs}
        orders = []
        first = torch.cuda.Event(enable_timing=True)
        last = torch.cuda.Event(enable_timing=True)
        for trial in range(args.samples):
            order = list(graphs) if trial % 2 == 0 else list(reversed(graphs))
            orders.append(order)
            for name in order:
                torch.accelerator.synchronize()
                first.record()
                for _ in range(args.replays):
                    graphs[name][0].replay()
                last.record()
                last.synchronize()
                elapsed_ms = first.elapsed_time(last)
                samples[name].append(
                    dict(
                        event_elapsed_ms=elapsed_ms,
                        replays=args.replays,
                        per_replay_us=elapsed_ms * 1000 / args.replays,
                    )
                )
        medians = {
            name: statistics.median(row["per_replay_us"] for row in values)
            for name, values in samples.items()
        }
        differences = (
            [diff for check in checks for diff in check["grouped_vs_three_fused"]]
            + [
                diff
                for check in checks
                for values in check["same_graph_repeat"].values()
                for diff in values
            ]
            + [
                diff
                for values in checks[-1]["restored_initial"].values()
                for diff in values
            ]
        )
        return dict(
            tokens=tokens,
            median_us=medians,
            samples=samples,
            sample_orders=orders,
            changed_input_restore=checks,
            grouped_over_baseline=medians["grouped_kv"] / medians["three_fused_qkv"],
            all_bitwise_equal=all(diff["bitwise_equal"] for diff in differences),
            cpu_reference=reference,
            output_names=[f"normalized_kv_{i}" for i in range(LAYERS)]
            + [f"raw_kv_{i}" for i in range(LAYERS)],
        )
    finally:
        torch.accelerator.synchronize()
        for graph, _ in graphs.values():
            graph.reset()
        graphs.clear()


def run(args):
    assert args.shared_screening
    from vllm.platforms import current_platform

    assert current_platform.is_device_capability(80)
    assert torch.accelerator.device_count() == 1
    free, total = torch.accelerator.get_memory_info()
    assert free >= 4 * 1024**3, "Fresh >=4 GiB free VRAM required"
    cap = 512 * 1024**2
    torch.cuda.set_per_process_memory_fraction(cap / total)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    weights, provenance = load_weights(args.weights, args.seed)
    manifest = Path("/opt/sm80-series.json")
    revision = args.source_revision
    if revision is None and manifest.is_file():
        revision = json.loads(manifest.read_text()).get("source_revision")
    result = dict(
        status="initializing",
        gpu=torch.cuda.get_device_name(),
        cap_bytes=cap,
        admission_free_bytes=free,
        source_revision=revision,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        vllm_version=vllm.__version__,
        weights=provenance,
        input_mode="synthetic BF16 main_x, shared by both paths",
        rms_norm_priority=["vllm_c", "native"],
        measurement_excludes_cpu_dispatch=True,
        context_is_graphed_in_serving=False,
        official_weight_extract_loaded=bool(
            args.weights
            and provenance["metadata"].get("source_model")
            == "deepseek-ai/DeepSeek-V4.1-Flash"
        ),
        full_model_loaded=False,
        seed=args.seed,
        norm_eps=args.norm_eps,
        rows=[],
        dimensions=dict(hidden=HIDDEN, q_rank=Q_RANK, kv=KV_DIM, layers=LAYERS),
        source_hashes={
            str(Path(__file__).name): digest(__file__),
            "benchmark_dsv41_native_consumers.py": digest(
                Path(__file__).with_name("benchmark_dsv41_native_consumers.py")
            ),
            "marlin_utils_fp8.py": digest(marlin_utils_fp8.__file__),
            "layernorm.py": digest(inspect.getfile(RMSNorm)),
        },
        calls_per_replay=dict(
            three_fused_qkv=dict(marlin=3, rms_norm=3),
            grouped_kv=dict(marlin=1, rms_norm=3),
        ),
        boundary="projection + same per-layer RMSNorm; TP replicated, no collectives",
        excluded=["RoPE", "quantization/cache insertion", "whole-model serving"],
        serving_tokens_per_second_measured=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2))

    try:
        config = VllmConfig()
        config.kernel_config.set_platform_defaults(config)
        config.kernel_config.ir_op_priority.rms_norm = ["vllm_c", "native"]
        with (
            set_current_vllm_config(config),
            torch.inference_mode(),
            config.kernel_config.ir_op_priority.set_priority(),
        ):
            fused = nn.ModuleList(
                [
                    make_projection(
                        concatenate_fp8([row["q"], row["kv"]]),
                        torch.cat([row["q_scale"], row["kv_scale"]], dim=0),
                    )
                    for row in weights
                ]
            )
            grouped = make_projection(
                concatenate_fp8([row["kv"] for row in weights]),
                torch.cat([row["kv_scale"] for row in weights]),
            )
            norms = nn.ModuleList(
                [
                    RMSNorm(KV_DIM, args.norm_eps, dtype=row["norm"].dtype).to("cuda")
                    for row in weights
                ]
            )
            for norm, row in zip(norms, weights, strict=True):
                norm.weight.copy_(row["norm"].to("cuda"))

            def three_fused_qkv(x):
                raw, normalized = [], []
                # Match the original projection -> norm order for each layer.
                for projection, norm in zip(fused, norms, strict=True):
                    raw.append(projection(x)[:, Q_RANK:])
                    normalized.append(norm(raw[-1]))
                return *normalized, *raw

            def grouped_kv(x):
                raw = grouped(x).split(KV_DIM, dim=-1)
                normalized = tuple(
                    norm(value) for norm, value in zip(norms, raw, strict=True)
                )
                return *normalized, *raw

            functions = dict(three_fused_qkv=three_fused_qkv, grouped_kv=grouped_kv)
            save()
            for tokens in args.tokens:
                assert time.monotonic() - started < args.budget_seconds
                print(f"SHAPE {tokens}", flush=True)
                row = run_shape(args, functions, tokens, weights)
                result["rows"].append(row)
                result.update(
                    status="in_progress", elapsed_s=time.monotonic() - started
                )
                save()
                print(json.dumps(row), flush=True)
                gc.collect()
            result["all_bitwise_equal"] = all(
                row["all_bitwise_equal"] for row in result["rows"]
            )
            result.update(
                status="complete"
                if result["all_bitwise_equal"]
                else "complete_with_non_bitwise_outputs",
                elapsed_s=time.monotonic() - started,
                peak_allocated=torch.accelerator.max_memory_allocated(),
                peak_reserved=torch.accelerator.max_memory_reserved(),
            )
            functions.clear()
            fused = grouped = norms = None
            gc.collect()
            torch.accelerator.empty_cache()
            save()
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        save()
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-screening", action="store_true")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        choices=[1, 6, 8, 48, 128, 512, 2048],
        default=[1, 6, 8, 48, 128, 512, 2048],
    )
    parser.add_argument("--samples", type=int, choices=[4, 8], default=4)
    parser.add_argument("--replays", type=int, choices=[20], default=20)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--norm-eps", type=float, default=1e-20)
    parser.add_argument(
        "--budget-seconds", type=int, choices=range(30, 211), default=190
    )
    result = run(parser.parse_args())
    raise SystemExit(0 if result["all_bitwise_equal"] else 2)
