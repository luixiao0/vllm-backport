# Adaptive FP8 consumer MQA and native-consumer screening

This follow-up preserves official FP8/block-32 UE8M0 linears, MXFP4 experts,
FP8 indexer cache bytes, FP32 indexer weights/logits and native ordered top-k.
It changes no production process, checkpoint, installed file or serving flag.

## Candidate preparation

The first candidate-only port sorted 2048 candidate blocks plus 64 tie-filler
blocks in a 4096-lane sorting network. That preparation dominated short rows.
The new implementation uses GPU-resident lengths inside the same FULL graph:

- At most 16384 visible keys: identity block mapping and atomic membership
  flags, without sorting. Native top-k writes directly into contiguous output;
  the restoration kernel exits for identity rows. Strided outputs retain a
  separate compact buffer and restoration.
- Longer rows: a fixed 64-block filler prefix and a 2048-lane sort/dedup of
  the remaining candidates. Fillers preserve lowest-context-index negative
  infinity ties, including empty, partial, duplicate and invalid candidates.
- FP8 query LUT decoding is fused into preparation. FP8 conversion, FP32
  scales, score accumulation and native selection mathematics are unchanged.

An earlier guarded 4096-lane implementation passed correctness but regressed
long rows; it was rejected. A stage ablation of the smaller sort reduced long
preparation from 65.0 to 29.7 microseconds, registers from 128 to 64 and spills
from two to zero, before query decoding was fused.

Seventy-six shared-A100 GPU checks passed, covering both output layouts,
noncontiguous inputs, native ordered ties, NaN cache bytes, partial candidates,
all three context-length layouts, 16384/16385 boundaries and mixed-length graph
replays followed by restoration. A separate 36-shape screen covered 1/6/12/24/48
rows, with changed device lengths under a 131072-key capture bound. All selected
indices matched the dense reference exactly.

For six rows, that screen measured 227.33 to 110.59 microseconds at 131072 keys
and 72.70 to 73.73 at 8192 keys. At 48 rows, the respective pairs were 1789.95
to 265.22 and 208.90 to 148.99. Short six-row cases retain a small regression:
2048 keys measured 58.37 to 62.46 microseconds. Consequently the general profile
keeps `VLLM_DSV41_SPARSE_MQA=0`; the long-context profile opts in.

The inspected deployment's compression ratios are 1 and 2 outside SWA-only
layers. Its 278528-token limit therefore produces compressed capture bounds
278528 and 139264. Both were screened directly. At an 8192-key replay, dense
versus adaptive measured 96.26/71.68 and 73.22/73.73 microseconds respectively.
These are compressed-key coordinates, not original prompt-token counts.

Forcing the persistent native top-k dispatcher improved the short case but
made compact long rows slower; that ablation is excluded from the release.

## Real consumers in the same FULL graph

Run the combined probe with `--native-consumers` to replace its default BF16
attention proxy and single expert. The resulting boundary includes:

- Actual source/consumer sparse-indexer bodies and radix candidate selection.
- Ampere `forward_mqa` and its inherited sparse decode, including ragged
  metadata, SM80 FP8 lookup, split-K and output copy. Synthetic `fp8_ds_mla`
  cache uses the native writer and its padded per-block stride.
- Native Marlin MXFP8 qA/qB/woB, native inverse RoPE and the existing BF16 woA.
- Native `GateLinear.forward` with FP32 logits, sqrtsoftplus routing over 384
  IDs, top-6 normalization and scale 1.5, followed by native group-32/E8M0 MXFP4
  Marlin experts with SwiGLU limit 10. Eight experts are materialized and all
  other IDs are masked; this is not a full 384-expert weight load.
- Both mHC stages, stream joins, final residual mixing and output snapshots.

CPU references check gate projection, dequantized expert math and selected-KV
attention with the sink. The query RoPE table is an identity fixture; KV and
indexer projection producers, shared experts, Engram and TP/EP communication
are outside this single-GPU combined boundary. No official checkpoint is loaded.

The FP32-gate screen used six rows and the same 131072-key capture for every
replay. Latency is microseconds:

| Device keys | Serial/dense | General | Adaptive sparse |
| ---: | ---: | ---: | ---: |
| 131072 | 1031.68 | 893.95 | 782.34 |
| 32768 | 936.96 | 826.37 | 747.01 |
| 8192 | 719.36 | 549.38 | 558.59 |

All outputs and ordered source/consumer indices were bitwise equal; candidate
sets matched and indices remained in range. Changed input and length replays
and restoration passed. Capture counters confirmed the intended dispatch.
The long result is 24.17% below serial/dense and 12.48% below general; the short
result remains 1.68% above general. These are component boundaries, not model
ITL or tokens/s. A prior BF16-gate exploratory result is superseded by this
FP32-gate result.

## Larger mHC batches and distributed scope

The separate native-consumer screen compares input-first and statistics-first
scheduling at 1/6/12/24/48 rows. Raw mHC outputs are bitwise equal at every
shape. The larger complete consumer boundary exhibits small BF16 differences;
48-row serial replay itself also varies. Twelve rows regressed and 24/48 showed
only small timing gains. The production candidate therefore retains its
existing at-most-eight-token FULL-capture gate and FP32 coefficient path.

The independent TP8/EP probe uses actual one-sided FlashInfer NVLink dispatch,
native MXFP4 experts and combine, plus attention-output NCCL all-reduce. Only
one active expert per rank is materialized. Results, numerical tolerances,
per-sample maximum-rank timing and process-exit status are recorded separately;
single-GPU numbers above do not include those collectives.

## Images and limits

The combined Dockerfile applies only the adaptive MQA file to its immutable
previous image. Installation verifies the exact preimage, compiles the module
and records source revision and hashes in `/opt/sm80-adaptive.json`. Other model
runtime source and native extensions are inherited unchanged. Python defaults
and image profiles retain their existing opt-in policy.

Every GPU screen used an independent bounded container, fresh free-VRAM
admission, short alternating CUDA-event graph measurements and a 240-second
outer timeout. No profiler, cold-L2 flush, clock changes or production requests
were used. The native combined probe needs a 512 MiB allocator cap; the kernel
correctness suite fits a 256 MiB cap.

Full-model A/B is not feasible in the observed spare memory. The eight GPUs
offer about 22.5 GiB in total; routed FP4 expert data alone requires at least
31.6 GiB per EP8 rank, before other weights or KV. The checkpoint metadata totals
510286023000 bytes. Full-model candidate throughput and model-quality evaluation
remain unexecuted under the no-production-change constraint.
