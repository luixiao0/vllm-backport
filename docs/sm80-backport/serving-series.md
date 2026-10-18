# DeepSeek V4.1-Flash SM80 serving backports

This series preserves the official FP8/block-32 UE8M0 linear weights, FP4
experts, existing FP8 indexer cache, and FP32 mHC coefficient arithmetic. It
extends the existing Engram TP and SM80 radix-selector candidate. The initial
component screening left production services and installed production files
unchanged. The later authorized maintenance A/B is recorded below.

## Candidate-only FP8 sparse MQA

The SM80 implementation sorts and deduplicates candidate blocks, scores at most
16384 positions, applies the native top-k, and remaps token indices with the
same ordered tie behavior. It is integrated into `SparseAttnIndexer`; it is not
limited to a standalone benchmark. An independent fix handles speculative
`(B, next_n)` context lengths without changing vector or `(B, 1)` semantics.

`VLLM_DSV41_SPARSE_MQA=1` opts in. **Leave it at the default 0 for mixed-length
serving.** A host capture bound cannot distinguish short device replay lengths
inside a FULL graph. In the initial release, at a 278528 capture bound,
6 rows with 8K device lengths
regressed from 112.6 to 146.9 microseconds (+30.5%). The corresponding 32K replay
improved from 333.8 to 160.8 microseconds.

The [adaptive consumer follow-up](adaptive-consumer.md) replaces the preparation
with a device-length-dependent identity/compact path, fuses query decoding and
passes 76 GPU cases. It removes most of the short-row cost and further improves
long rows. Small residual short-row regressions keep sparse MQA opt-in.

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

## Combined deployment profiles

The combined image opts into the screened general profile:

```bash
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_DSV4_CANDIDATE_SELECTION_BACKEND=triton_radix
VLLM_DSV41_MHC_OVERLAP=1
VLLM_DSV41_SPARSE_MQA=0
```

[dsv41-general.env](../../docker/sm80/dsv41-general.env) combines radix candidate
selection, small-FULL-graph mHC overlap, automatic Engram TP sharding and the V2
cached-prompt graph path. The existing TP8/EP/DSpark5, official weights and serving
arguments remain the deployment inputs. Python-level defaults for the two new
kernel flags stay off; the combined image explicitly opts into this profile.

[dsv41-long-context.env](../../docker/sm80/dsv41-long-context.env) additionally
sets `VLLM_DSV41_SPARSE_MQA=1`. Use it for a dedicated long-context workload;
short device replays under a long capture bound retain the documented regression.
The mHC gate still limits overlap to at most 8 tokens in FULL capture and retains
serial fallback elsewhere. No checkpoint conversion or additional quantization
is needed.

`docker/Dockerfile.sm80-combined` wraps the immutable previous combined image,
verifies and installs the single adaptive MQA file, and includes these profiles.
Other model runtime files and native extensions are inherited unchanged.
The probe in `benchmarks/kernels/benchmark_dsv41_sm80_combined.py`
places the actual source/consumer indexer bodies, radix selection, native mHC,
and one native MXFP4 expert inside the same FULL graph. It checks actual dispatch
counts, bitwise outputs, candidate sets, changed device lengths/input, and
restore replay. BF16 attention remains a proxy; Engram TP collectives and global
MoE routing are outside its default joint boundary. `--native-consumers` selects
real FP8 MLA, native Marlin projections, FP32 GateLinear and 384-route/top-6
MXFP4 with eight materialized experts. See the follow-up for current results.

The original proxy-based joint A100 screen uses six rows and a 131072-key capture
bound. The serial
Torch-selector/dense-indexer path takes 872.96 microseconds; the general profile
takes 746.50 (-14.49%), and the long-context profile takes 694.78 (-20.41%). With
8K device lengths under that same capture, general takes 401.41 microseconds
while long-context takes 485.38 (+20.92% versus general). This confirms keeping
sparse MQA out of the general profile. The benchmark includes actual source and
consumer indexers, both native mHC stages and stream joins, the BF16 attention
proxy, the native single-expert Marlin FFN, final post mixing, and small output
snapshots. It is not whole-model ITL or tokens/s.

All three profiles have bitwise-equal outputs and identical candidate sets at
131K, 32K, and 8K device lengths, including changed inputs and a restore replay.
Capture counters establish one radix source selection in the general profile,
one additional compact consumer in the long-context profile, and two overlapped
mHC stages in both. The external harness enforces fresh 2 GiB free-VRAM admission
and a 240-second timeout; the Torch allocator cap is 256 MiB.

The component measurements above used independent bounded containers and spare
VRAM on a resident-production A100. They used short interleaved CUDA-event graph
samples, without profiling, L2 flushing, clock changes, or production inference
load. Those screens did not measure model tokens/s or a full-model A/B. The
tested source hashes, commands, raw results, and immutable candidate image are
recorded in the task artifacts.

## Full-model maintenance validation, 2026-10-19

The user subsequently authorized a service interruption for an exclusive
maintenance A/B on the real 8x A100 deployment, using the official checkpoint.
External business traffic was isolated for this comparison. These measurements
are distinct from the earlier shared-GPU component screens, including the
reported 24.2% component-boundary improvement; that number is not an end-to-end
model result.

For the old combined image's general profile, throughput changes relative to the
original image were:

| Prompt context | Concurrency | Cold throughput change | Warm throughput change |
| --- | ---: | ---: | ---: |
| 2K | 1 | +3.83% | +8.46% |
| 2K | 8 | +0.37% | +2.49% |
| 32K | 8 | +2.01% | -1.71% |
| 128K | 1 | +2.14% | -2.11% |

The results vary with workload and cache state; the general profile did not
establish a uniform improvement. For the old combined long-context profile at
128K context and concurrency 8, warm throughput increased from 506.626 to
562.943 tokens/s (+11.11%). However, cold request duration regressed from
157.069 to 373.307 seconds.

The cold regression exposed runtime compilation that the fixed-bound component
screens had missed. Mixed prefill/decode batches grow their host `max_seq_len`
across steps, while all three sparse-MQA kernels treated `MAX_LEN` as
`tl.constexpr`; rank 0 compiled 58 `_prepare_candidates` variants. The fix makes
`MAX_LEN` a runtime scalar with `do_not_specialize=["MAX_LEN"]` in all three
kernels, retaining the exact clamp rather than rounding the bound upward.
The new GPU regression fails on the old image for recompilation and passes with
the fix, alongside all 146 tests in the MQA logits suite. Growing bounds reuse
the kernels and preserve native ordered top-k with the exact host clamp.

### Post-fix long-context profile

The corrected image was validated as `vllm-sm80-combined:git-156c4c51b4bb`,
source commit `156c4c51b4bbe791be4dcd23064d81cc2aeea54d`, image ID
`sha256:455900c5bebab4fbdb42ae41cc32cefbd1eff4d4d0472deed0e5ceb1d151e2c4`.
These are long-profile validation results, not confirmation of a completed
general-profile rollout. The old image's results above remain separate.

| Prompt context | Concurrency | Cold throughput change | Warm throughput change |
| --- | ---: | ---: | ---: |
| 2K | 1 | -3.27% | +4.93% |
| 2K | 8 | -0.48% | +1.91% |
| 32K | 8 | +1.30% | +0.82% |
| 128K | 1 | +1.64% | -4.89% |
| 128K | 8 | +2.43% | +15.21% |

Throughput is end-to-end output tokens/s, including prefill, summarized by the
median of recorded repetitions. At 128K/C8, the cold pass completed in
153.347264 seconds versus the original image's 157.068524 seconds (-2.37%
duration), eliminating the old bug's 373.307432-second result. Cold throughput
was 26.078 to 26.711 tokens/s (+2.43%); this cold case has one recorded pass.
Warm throughput was 506.626052 to 583.689063 tokens/s (+15.21%), using the
median of three repetitions; the corrected image ranged from 548.857 to 583.849
tokens/s. Warm median TPOT improved from 11.099 to 9.667 ms (-12.90%).
Multi-request TPOT can include interference from other requests' prefills.

After the full-model workload, rank 0 had only 1 `_prepare_candidates`,
1 `_candidate_paged_mqa`, and 2 `_restore_indices` compiled variants. The
comparison files `comparison-fixed-long-{cold,warm}.json` and
`comparison-fixed-128k8-{cold,warm}.json`, and `fixed-jit-variant-counts.txt`, are
retained under the rollout task's `outputs/` directory. The mixed results at
other sizes still do not establish a uniform throughput improvement.

### Post-fix general profile

The same corrected image was separately tested with sparse MQA disabled,
retaining radix candidate selection, small-graph mHC overlap, Engram TP, and
the cached-prompt graph path. Three repetitions per cold and warm case gave:

| Prompt context | Concurrency | Cold throughput change | Warm throughput change |
| --- | ---: | ---: | ---: |
| 2K | 1 | +1.92% | +8.50% |
| 2K | 8 | -0.64% | +3.47% |
| 32K | 8 | +2.48% | -0.94% |
| 128K | 1 | +1.75% | -6.74% |

The general profile remains the deployment default for mixed traffic. These
results show modest gains and a long-single-request warm regression, rather
than a universal improvement. The +15.21% long-profile result at 128K/C8 is a
separate opt-in configuration and must not be attributed to that default.
The exact input-token hashes match the baseline for all compared requests.
Raw results and comparisons are `fixed-general-{warm-,}chat.json` and
`comparison-fixed-general-{cold,warm}.json` in the rollout task's `outputs/`.

At 2026-10-18 18:22:17 UTC (2026-10-19 02:22:17 UTC+8), the corrected
general-profile container was serving under `vllm_deepseek_v41_flash` on port
18005. The published host port returned HTTP 200, and all 15 chat checks passed
through that host port after business ingress was restored. All temporary
maintenance firewall rules were removed. The original container and image
were retained; the rollback script and sanitized evidence are preserved at
`/mnt/disk2/benchmarks/dsv41-sm80-rollout-20261019/` on `a100`.

Fifteen actual chat checks passed, but outputs were not bitwise identical. The
synthetic performance benchmark fixes output length at 512 tokens; it is not a
model-quality evaluation. Neither those chat checks nor the component-level
bitwise checks establish full-model accuracy equivalence.
