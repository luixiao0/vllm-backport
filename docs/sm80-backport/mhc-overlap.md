# SM80 shifted mHC coefficient overlap

This port adapts upstream [#57603](https://github.com/vllm-project/vllm/pull/57603)
and its FULL-graph safety fix
[#57874](https://github.com/vllm-project/vllm/pull/57874) to the fork's TileLang
SM80 implementation. It retains the original FP32 coefficient weights,
FP32 prenorm accumulation, Sinkhorn arithmetic, and BF16 collapse-before-RMSNorm
rounding. It does not require a converted or requantized checkpoint.

`VLLM_DSV41_MHC_OVERLAP=1` opts in; the default is **0**. It creates a distinct
mHC stream and splits the existing normalization epilogue into input and stats
modes. Input collapse runs on the caller stream. The original TileLang prenorm
GEMM plus coefficient epilogue runs on the side stream while attention or FFN
consumes the collapsed input. The model joins before either post mixing or
returning the FFN coefficients. Cross-stream scratch, inputs, weights, and outputs
are recorded on their consuming stream. Split specializations are registered
with JIT warmup before capture.

The execution boundary is SM80, hidden size 5120, hc_mult 4, explicit JIT warmup,
non-sequence-parallel target layers, no ubatching, and **1–8 tokens in a FULL CUDA
graph capture**. Eager, PIECEWISE, zero/larger batches, draft layers, and other
kernel configurations retain the serial path. Engram injection precedes the
side stream's dependency event, so coefficients see the injected residual.

## A100 screening

The bounded test used one resident-production A100's spare VRAM, a 256 MiB Torch
allocator cap, and independent containers. Production was not changed or
restarted. No clocks, profiling, cache flush, or inference load test was used.

Native tests compare every decoder entry (broadcast, pipeline, residual, Engram)
against the existing Torch reference, and compare native/split TileLang outputs
through changed-input/restore graph replays and scratch allocation reuse. BF16
layer input is exact; FP32 coefficients use atol/rtol 1e-6. Graph/context/config
fallbacks are also checked. See the task's `outputs/report.json` for counts,
source hashes, peaks, and execution logs.

The benchmark times the actual decoder scheduling and native mHC kernels through
two sublayers, including post kernels and stream joins. Its attention consumer
has TP8-sized 5120/1280/4096/1024 projections and a 512-position BF16 attention
proxy. The FFN consumer is either one 2304-wide BF16 expert or the native
`fused_marlin_moe` MXFP4/group32/E8M0 kernel with one expert and top-k 1. Weights
are synthetic; there is no global router, multi-expert routing, or EP collective.
Actual official-weight serving uses different FP8 attention and distributed MoE
resource behavior. These numbers are **not model tokens/s**.

Initial full-boundary screens, four interleaved graph-event samples per case:

| Tokens | BF16 FFN serial → overlap (μs) | Native MXFP4 FFN serial → overlap (μs) |
| --- | --- | --- |
| 1 | 175.10 → 160.26 | 160.77 → 146.43 |
| 6 | 181.25 → 175.62 | 188.42 → 176.64 |
| 12 | 211.97 → 211.97 | 189.44 → 192.00 |
| 24 (native fallback) | 238.59 → 236.54 | 212.99 → 214.02 |
| 48 (native fallback) | 242.18 → 242.69 | 285.18 → 284.67 |

Every output matched bitwise. The MXFP4 consumer's 12-token regression (+1.35%)
is why this SM80 port narrows upstream's 16-token threshold to 8. The final
screen repeats the entry/fallback tests and native-consumer boundary with that
threshold. Keep the option off by default: the probe establishes a useful
scheduling candidate, without establishing a full-model deployment speedup.

The final 8-token-threshold screen passed **44 checks** (30 native GPU/model
checks and 14 policy checks). The native-MXFP4-consumer boundary measured
160.26 → 144.38 μs at 1 token (-9.90%), and 220.16 → 210.43 μs at 6 tokens
(-4.42%). Native fallbacks at 12/24/48 tokens measured 223.74 → 225.28,
253.95 → 252.93, and 285.70 → 285.70 μs. All outputs were bitwise equal.
These are short shared-GPU screens; absolute latency varies between runs.

## Reproduction

On a CUDA installation with sufficient admitted spare VRAM, use the repository
venv. The benchmark enforces the 256 MiB Torch cap; the container must also have
an external runtime limit and a fresh free-VRAM check, as in the saved task
harness.

```bash
VLLM_DSV41_MHC_OVERLAP=1 VLLM_DSV41_SPARSE_MQA=0 \
  uv run --no-project .venv/bin/python \
  benchmarks/kernels/benchmark_dsv41_mhc_overlap.py \
  --shared-screening --ffn-consumer marlin_mxfp4 --output /tmp/mhc-screen.json
```

Use `--ffn-consumer bf16` for the secondary proxy. The sparse-MQA experiment is
independent and remains disabled in this comparison.
