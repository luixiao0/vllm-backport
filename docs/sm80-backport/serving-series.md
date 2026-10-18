# DeepSeek V4.1-Flash SM80 serving backports

This series preserves the official FP8/block-32 UE8M0 linear weights, FP4
experts, existing FP8 indexer cache, and FP32 mHC coefficient arithmetic. It
extends the existing Engram TP and SM80 radix-selector candidate. Production
services and installed production files were not changed.

## Candidate-only FP8 sparse MQA

The SM80 implementation sorts and deduplicates candidate blocks, scores at most
16384 positions, applies the native top-k, and remaps token indices with the
same ordered tie behavior. It is integrated into `SparseAttnIndexer`; it is not
limited to a standalone benchmark. An independent fix handles speculative
`(B, next_n)` context lengths without changing vector or `(B, 1)` semantics.

`VLLM_DSV41_SPARSE_MQA=1` opts in. **Leave it at the default 0 for mixed-length
serving.** A host capture bound cannot distinguish short device replay lengths
inside a FULL graph. At a 278528 capture bound, 6 rows with 8K device lengths
regressed from 112.6 to 146.9 microseconds (+30.5%). The corresponding 32K replay
improved from 333.8 to 160.8 microseconds. This is a long-context specialization.

The complete sort/dedup/score/top-k/remap pipeline measured 193 to 144
microseconds for 6 rows and 131072 keys, and 456 to 167 microseconds for 6 rows
and 278528 keys. Native ordered indices were exact through changed-input graph
replays. Nineteen GPU correctness cases plus a final actual-entry check passed.

## Shifted mHC coefficient overlap

The port adapts upstream [#57603](https://github.com/vllm-project/vllm/pull/57603)
and [#57874](https://github.com/vllm-project/vllm/pull/57874) to native SM80
TileLang FP32 math. Collapsed-input normalization stays on the caller stream;
coefficient generation runs on a separate stream while the sublayer consumes
the input, with joins before coefficient use.

`VLLM_DSV41_MHC_OVERLAP=1` opts in; the default is **0**. It requires SM80,
hidden size 5120, hc_mult 4, JIT warmup, no sequence parallelism or ubatching,
target layers, and FULL captures with at most 8 tokens. The 12-token screen
regressed by 1.35%, so 12 and larger retain the original path. The production
capture list's 6-token graph is eligible; its 12-token graph is not.

Forty-four checks passed. The final native-MXFP4-single-expert consumer boundary
measured 160.26 to 144.38 microseconds at 1 token (-9.90%), and 220.16 to 210.43
microseconds at 6 tokens (-4.42%). All benchmark outputs were bitwise equal.
Attention is a BF16 proxy; the native Marlin FFN has one synthetic expert and
no global routing or EP communication. These are not full-model throughput or
ITL improvements. See [mHC details](mhc-overlap.md).

## Cached prompt tails

The port adapts upstream [#58400](https://github.com/vllm-project/vllm/pull/58400)
to this fork's V2 runner, DSpark placeholders, adaptive verification, and
variable-length FULL descriptors. A request with existing computed context and
one scheduled non-draft prompt token can replay a decode graph. Actual
`has_prefill`/`is_prefilling` state remains intact. The separate eligibility and
per-request mask propagate through input batches, PCP, microbatch slices,
drafters, and recurrent-state metadata.

Fresh one-token prompts and genuine multi-token chunks remain prefill. PCP
prefills stay conservative. An ineligible batch also withholds the query-length
bound, so it cannot accidentally match a variable-length FULL graph; DP
agreement conservatively propagates that restriction. Ordinary adaptive
variable-length decode remains eligible.

The CPU cohort passed 85 tests, with 24 GPU/environment skips. Fourteen cases
exercise the real `execute_model` → graph-manager dispatch boundary with uniform
and variable-length graphs, including DSpark5 cached tails and fresh/real prompt
chunks. Additional checks cover DP, PCP, microbatch masks, and Mamba rollback
metadata. This measures graph eligibility and correctness, not model throughput.
The A100 follow-up passed 48 related GPU checks plus one PCP partition check,
covering all 24 CPU CUDA skips. Peak reserved memory was 6 MiB. See
[prompt-tail details](prompt-tail-graphs.md).

This change is automatic for the V2 runner and does not introduce a flag. In
production `VLLM_USE_V2_MODEL_RUNNER` was unset; the installed configuration
rejects DSpark under V1. The fact that DSpark is running therefore supports a
V2 inference; this was not established by a retained startup log.

## Candidate configuration

The candidate retains conservative defaults:

```bash
VLLM_DSV4_CANDIDATE_SELECTION_BACKEND=triton_radix
VLLM_DSV41_SPARSE_MQA=0
VLLM_DSV41_MHC_OVERLAP=0
```

The first flag selects the previously screened SM80 candidate block selector.
The two new kernel paths can be enabled individually for their measured
workloads. Engram TP sharding remains automatic. No checkpoint conversion or
additional weight quantization is needed.

All GPU measurements used independent bounded containers and spare VRAM on a
resident-production A100. They used short interleaved CUDA-event graph samples,
without profiling, L2 flushing, clock changes, or production inference load.
Model tokens/s and a full-model A/B were not measured. The tested source hashes,
commands, raw results, and immutable candidate image are recorded in the task
artifacts.
