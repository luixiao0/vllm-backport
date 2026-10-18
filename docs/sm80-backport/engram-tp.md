# DeepSeek V4.1-Flash Engram tensor-parallel backport

This ports [vLLM #58678](https://github.com/vllm-project/vllm/pull/58678),
merge `4bb804cc5bf0de81ce8344238b5b4522630874fd`, onto the backport fork's
`deepseek_v4_1` model implementation. It targets the actual TP8 Flash deployment.
The upstream H20 results are not A100 measurements.

## Change

Without sequence parallelism, every rank receives the same Engram embedding
vector. Previously each rank stored the full WKV weight and repeated its full
matrix multiplication. `ColumnParallelLinear(gather_output=True)` partitions
the output columns and gathers the complete key/value result. Checkpoint loading
uses the existing linear and MXFP8 loaders; no offline checkpoint conversion is
needed. Sequence parallelism uses `disable_tp=True`, retaining the full weight
because each rank then owns different tokens. TP1 also retains the full weight.

The inspected Flash configuration has two Engram layers (1 and 14), input width
6144 and output width 25600. TP8 partitions each output to width 3200. The A100
backend is the native `MarlinMxfp8LinearKernel`, using FP8 weights and E8M0 scales
with checkpoint blocks `[32,32]`. The projected persistent-weight reduction is
approximately 270.7 MiB per GPU for both layers, including expanded scales.

The fork's staged host embedding lookup, CUDA graph preparation, and fused gate
are preserved. The slower experimental WO_A MXFP8 path is excluded from this
branch and its image.

## Reproduction

Run the existing Engram tests with the normal CUDA test environment:

```bash
uv run --no-project .venv/bin/python -m pytest tests/kernels/test_engram.py -q
```

The new loader regression covers TP1, TP8, and TP8 with sequence parallelism,
both unquantized and MXFP8. A local output width of 40 exercises a partition
boundary crossing a 32-row checkpoint scale block. Tests compare exact loaded
weight bytes and expanded scales before checking the reconstructed projection.

The distributed benchmark uses full-size synthetic WKV weights rather than
loading the model or its large Engram tables. It includes native projection,
all-gather, and output layout conversion. Both variants use the same weights,
inputs, precision, TP size, and graph mode. Synchronization precedes each
measurement; four alternating event samples are reduced with MAX across ranks
before computing their median. Shared mode uses warm-cache CUDA events without
profiling or cache flushing, to keep the probe bounded on occupied GPUs.

```bash
uv run --no-project .venv/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 \
  benchmarks/kernels/benchmark_dsv41_engram_wkv.py \
  --shared-screening --output outputs/engram-tp8
```

Shared mode requires external admission of at least 2048 MiB free on all eight
GPUs, an external timeout, and a separate container. Its PyTorch allocator cap is
512 MiB per rank; CUDA/NCCL context memory is additional. Custom all-reduce is
disabled only in this projection probe; the new all-gather uses the normal vLLM
PyNCCL communicator. The probe performs no expert dispatch or MoE computation.

## Candidate image

`docker/Dockerfile.sm80-serving` layers this Python backport over the previously
tested radix image. The installer verifies the Engram preimage and records its
installed hash, upstream source, and source revision in `/opt/sm80-serving.json`.
The native extensions and dependencies are inherited from the production-based
image; this is a Python overlay rather than a full source rebuild.

Build from a clean, committed checkout with a minimal context containing the
Dockerfile, Engram source, and installer:

```bash
docker build --network=none --cpu-period=100000 --cpu-quota=200000 --memory=4g \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -f docker/Dockerfile.sm80-serving \
  -t vllm-sm80-serving:git-<commit> <minimal-context>
```

The candidate preserves TP8, EP, DSpark five drafts, LMCache, graph sizes,
context length, and batching limits. To include the separately screened radix
selector, set `VLLM_DSV4_CANDIDATE_SELECTION_BACKEND=triton_radix` in the candidate
process. Engram TP sharding is automatic when TP > 1 and SP is disabled.
Rollback selects the previous image. No existing service is modified by the
build or probe commands.

## Recorded validation

On 2026-10-18, the six new loader cases passed on CPU with the MXFP8 emulation
execution backend. On A100, 70 Engram tests passed, including the six loader
cases using native Marlin. One multi-rank test was deselected from that
single-GPU run. The whole run took 45.24 seconds and peaked at 23.9 MiB tensor
allocation and 50 MiB reservation within a 256 MiB allocator cap. The distributed
benchmark independently checks real TP8 output and graph collective replay.

The successful eight-rank shared-A100 run used NCCL 2.30.7 and completed all
five shapes with clean teardown. Rank 0's measured section took 12.83 seconds;
its peak tensor allocation was 454.7 MiB and reservation was 472 MiB, within
the 512 MiB cap. Persistent WKV parameters measured 162201600 bytes before and
20275200 bytes after sharding per layer/rank, saving 270.7 MiB for both layers.
Each entry below is the median of four
interleaved per-sample MAX latencies over all eight ranks. Latencies include
the sharded path's all-gather and layout conversion.

| Tokens | Replicated (us) | TP-sharded (us) | Latency reduction |
| ---: | ---: | ---: | ---: |
| 1 | 103.94 | 57.86 | 44.3% |
| 6 | 106.50 | 63.49 | 40.4% |
| 12 | 109.57 | 69.63 | 36.4% |
| 48 | 123.39 | 101.38 | 17.8% |
| 128 | 300.03 | 178.69 | 40.4% |

Maximum relative L2 error against the replicated Marlin path was 0.000109.
Shared-GPU samples have outliers; these medians describe this bounded run.
The first completed measurement exposed a benchmark teardown lifetime issue:
captured graphs remained alive when their NCCL communicator was destroyed.
The benchmark now resets and releases all graphs before communicator cleanup;
its successful rerun is the table above.

These bounded shared-GPU results are the requested screening criterion. Kernel
latency does not establish whole-model output tokens/s; an unchanged production
container provides no candidate serving A/B result.
