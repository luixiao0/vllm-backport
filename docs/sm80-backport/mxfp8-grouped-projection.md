# MXFP8 grouped output projection: Blackwell to SM80

## Source and hardware barrier

The source operation is upstream vLLM's
[`DeepGemmMxfp8BmmLinearKernel`](https://github.com/vllm-project/vllm/blob/9b959b86577c082c0b2bf9e2c22263255a36ad83/vllm/model_executor/kernels/linear/mxfp8/deep_gemm.py),
introduced by [#56228](https://github.com/vllm-project/vllm/pull/56228), commit
`9b959b86577c082c0b2bf9e2c22263255a36ad83`. The backport upstream imported it in
`e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`. Its MXFP8 grouped BMM explicitly
requires Blackwell. The DSV4.1 NVIDIA output projection applies inverse RoPE,
quantizes activations and calls the grouped einsum with recipe `(1, 1, 32)`.

A100 has neither native E4M3 conversion nor FP8/microscaled matrix instructions.
Removing the capability check cannot make that kernel work. The inherited
Ampere path instead expands WO_A to BF16 at load time, materializes inverse
RoPE output, and runs a BF16 einsum.

## SM80 implementation

`Sm80Mxfp8BmmLinearKernel` ports the quantized grouped projection to SM8x:

- Keep contiguous E4M3 weights `[groups, N, K]` and E8M0 scales
  `[groups, N, K/32]` resident. The scale block size and group boundaries match
  the source operation; no persistent BF16 copy or Marlin repack is used.
- For up to eight tokens, decode weight tiles with integer E4M3 expansion,
  reconstruct E8M0 powers of two, and reduce products in FP32. A tile shares
  decoded weights across tokens. Decode includes signed zero, subnormals and
  NaN values.
- Retain BF16 activations, as in the existing Ampere path. This is a W8A16 port,
  not bitwise emulation of Blackwell's activation-quantized W8A8 arithmetic.
- Fuse inverse GPT-J RoPE into decode activation loads, with padded head strides
  and int64 position offsets. Larger batches use transient BF16 weight decoding,
  the existing inverse-RoPE kernel and SM80 BF16 `mma.sync` with FP32
  accumulation. These temporary weights do not persist between calls.
- Use private FP32 split-K partials at small token counts and a fixed-order
  reduction. There are no global atomic sums, cooperative spin barriers,
  clusters, TMA, WGMMA or SM100 instructions.

Both ordinary grouped BMM and the fused DSV4.1 WO_A path call the same kernel.
The existing WO_B projection follows it. Unsupported platforms and the default
configuration retain their existing kernel selection.

## Enable and rollback

Set `VLLM_SM80_MXFP8_BMM=1` **before loading weights** in a separate deployment.
It selects the new backend only for grouped MXFP8 layers on CUDA SM8x. The
default is `0`. Turning it off requires reloading the model, because the weight
representation is selected at load time; changing the environment after load
does not convert already-loaded tensors.

The optional `docker/Dockerfile.sm80-mxfp8` builds a development Python overlay
on `vllm-sm80-radix:git-20b2ff96f8e4`. The installer checks all preimage hashes
before copying any source, compiles the Python files and writes a source
manifest to `/opt/sm80-mxfp8.json`. It reuses the base's native extensions; it is
not a full native rebuild or a production deployment.

## Verification

Correctness tests live in
`tests/kernels/quantization/test_sm80_mxfp8_bmm.py`. They cover masked tiles,
empty input, split-K, prefill, compact weight residency, bias rounding,
noncontiguous group/head strides, changing positions under CUDA Graph replay,
deterministic replay, and exhaustive finite FP8 value decoding at extreme
E8M0 scales. Matrix products are checked against FP32 contraction of the same
BF16 operands, rather than accepting the default cuBLAS reduction as exact.

`benchmarks/kernels/benchmark_sm80_mxfp8_bmm.py` compares complete inverse-RoPE
plus WO_A operations, excluding the common WO_B. Default timing uses CUPTI,
graphs and cold L2 for an isolated GPU. `--shared-screening` explicitly uses
four interleaved event-timed graph replays, a 256 MiB allocator cap and no
profiler, cache flush or stress loop. Resident weight sizes are counted
separately from the additional BF16 reference stored by the benchmark.

Bounded shared-A100 screening is the owner's accepted pass criterion. It does
not constitute a measurement of full-model throughput or quality.

## Observed tradeoff

For each group with `N=1024, K=4096`, resident weight plus scale storage falls
from 8.125 MiB (BF16 plus retained scales) to 4.125 MiB: a 49.23% reduction.
Transient prefill decoding still requires a BF16 working buffer. This does not
halve total model or KV-cache memory.

The tested warm-cache shared-A100 timings do **not** establish a latency win.
Most token/group shapes are slower than cached BF16 einsum. Earlier fully
fused tensor-core, vector-only and native Marlin variants were also screened;
their results are retained, rather than treating successful compilation as a
speedup. This port is an opt-in resident-memory tradeoff, not the default
throughput optimization.

Final screening: 17 pytest cases passed, including the actual Ampere `_o_proj`
entry point with a lightweight layer, and 27 SM80 cross-compiles passed without
CUDA initialization. Peak PyTorch allocation was 112.46 MiB; reserved memory was
214.00 MiB under the 256 MiB cap. The production container start time remained
unchanged.

| Groups | Tokens | Cached BF16 (us) | SM80 compact (us) | New / baseline |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 16.896 | 19.456 | 1.15 |
| 1 | 7 | 19.456 | 54.784 | 2.82 |
| 1 | 32 | 19.456 | 59.392 | 3.05 |
| 1 | 129 | 27.648 | 68.096 | 2.46 |
| 2 | 1 | 30.208 | 28.672 | 0.95 |
| 2 | 7 | 29.184 | 96.768 | 3.32 |
| 2 | 32 | 33.280 | 84.480 | 2.54 |
| 2 | 129 | 41.984 | 103.424 | 2.46 |

These are shared-GPU, warm-cache event timings with four interleaved graph
replays per path. The one small apparent improvement is insufficient to claim
a general latency win. Full-model serving was not benchmarked.
