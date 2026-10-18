# SM80 device-length-aware candidate selection

This is an opt-in replacement for the complete candidate selector. The default
remains `VLLM_DSV4_CANDIDATE_SELECTION_BACKEND=torch`. Set it to `triton_radix`
in a development process to test the new path. The earlier
`VLLM_DSV4_LENGTH_AWARE_CANDIDATE_SCORES` switch controls only the Torch backend's
score kernel; it does not select this implementation.

## Algorithm and SM80 adaptation

[Upstream #61039](https://github.com/vllm-project/vllm/pull/61039) uses a
length-aware DeepSelect path on supported hardware. This implementation uses
Triton and SM80-compatible warp-match instructions, without DeepSelect,
TMA, thread-block clusters, or a cooperative spin barrier. The backend is gated
to NVIDIA CUDA SM80+; other platforms retain the Torch implementation.

1. A fixed grid scores only each row's live prefix and publishes its length on
   the GPU. The newest-block rule is retained even for bounds beyond the logits
   allocation. Long rows build 512-bin coarse histograms in 4096-score tiles.
   FP16 rounding is used only to form monotone coarse buckets. Warp matching
   combines equal-key updates before global atomics; each CTA owns its histogram,
   with a CTA barrier between initialization and updates.
2. A row kernel finds the coarse boundary bin and resets its compaction state.
3. Independent CTAs compact only that boundary bin. Atomic reservations may
   change scratch order, which has no effect on the exact threshold.
4. A row kernel refines the threshold using full FP32 keys. It skips refinement
   for a constant boundary bucket. Long rows use parallel counts, a prefix scan,
   and parallel scatter to emit IDs in input order; equal boundary keys choose
   lower input IDs. Short rows use a direct radix path; rows with at most K
   blocks require no selection. Coarse rounding never determines the final set.

No length is read back to the host, and graph shapes stay fixed. Only the live
score prefix is initialized or read. Scratch memory and the coarse histogram
are allocated statically from shape, so this does not reduce score-buffer
allocation to the live length. The isolated benchmark captures the complete
wrapper's GPU operations and graph-managed scratch. Host allocation and
compilation costs are outside replay timing and require a serving comparison.

## Numerical contract

Candidate IDs feed `apply_candidate_mask`, which discards their order. Validate
the exact candidate multiset and the resulting logits mask against the Torch
backend; direct candidate-array order equality is not the consumer contract.
This is distinct from token top-k output order, which can affect sparse-attention
rounding and is unchanged by this implementation.

NaNs rank first during selection but become `-1` on output, preserving the
existing store behavior. Positive infinity and newest-block pins remain valid;
negative infinity becomes `-1`. Packed starts, repeated decode bounds, strided
inputs/outputs, and padding are supported for FP32, FP16, and BF16. Unsupported
dtypes retain the Torch path. The existing full-width backend is the explicit
fallback, and it remains the default until release criteria are met.

## Validation and measurement

Run the existing regression suite in a CUDA development environment:

```bash
.venv/bin/python -m pytest \
  tests/model_executor/layers/test_mla_short_prefill_indexer.py \
  -q -k 'candidate'
.venv/bin/python benchmarks/kernels/benchmark_dsv41_candidate_scores.py \
  --backends torch score_only triton_radix --rows 1 6 48 --width 278528 \
  --output outputs/candidate-selector.json
```

The benchmark checks candidate multisets, alternates backend order, and reports
the complete selector plus its stages. `coarse_and_compact` includes the coarse
kernel because the reservation counter must be reset on every invocation; stage
times are not additive. Use an isolated allocation for CUPTI, cold-L2, and any
performance acceptance claim. A bounded shared-GPU probe is only a screening
tool; it cannot establish serving throughput or ITL improvements.

The first linear-scan radix experiment preserved masks but showed a large
long-row regression in shared-GPU screening. That failed experiment is retained
in the task evidence. The hierarchical design addresses its repeated full-row
histogram scans rather than hiding the long-context cases.

Stage screening also rejected rebuilding a large histogram for every small
score tile and collecting a long row serially. Coarse histograms now process
4096 scores per tile; long-row output uses parallel counts and scatter. A CPU
FP16 counterexample caught an early-exit error after bucket compaction: a
partial FP32 prefix can admit values from lower coarse bins. Nonconstant
compacted buckets therefore refine all 32 key bits before global collection.
The reference histogram branch remains available to kernel probes with
`WARP_AGG=False`; CPU interpreter results use this branch because inline PTX is
not supported there. Actual CUDA probes must check the warp-aggregated counts
and the complete selector against their references.

The development Docker overlay verifies every base preimage and records the
three installed source hashes. Its installer is an explicit Python file, so both
legacy Docker and BuildKit execute the same installation command. Always verify
the manifest and module hashes before running a GPU probe. An overlay reuses the
base's native extensions and is not a source-built release image.

On the recorded shared A100 development image, actual pytest execution of
`-k candidate` passed all 31 selected tests (nine unrelated tests deselected),
including the 18 new radix configurations. Tests cover FP32/FP16/BF16, packed
prefill, repeated decode bounds, noncontiguous tensors, NaNs/infinities, FP32
values outside FP16's range, exact consumer masks, and identical graph replays.
The mounted test module ran without the repository-wide conftest environment.
Six separate GPU histogram comparisons matched every reference bin. The CPU
reference-histogram interpreter passed 16 cases, and 34 explicit-target SM80
kernel compilations passed without CUDA initialization.

A 53.45-second bounded probe also screened width 278528, rows 1/6/48, and
contexts 8192/32768/131072/278528 using four alternating CUDA-event samples per
path. Complete-selector median new/old ratios ranged from 0.183 to 0.784 in
that run. These are descriptive shared-GPU measurements, not an isolated
speedup claim or a serving result. Peak tensor allocation was 66.0 MiB and
reservation 120 MiB, within a 256 MiB allocator limit; CUDA context overhead
is additional. The existing production container was not changed or restarted.

Full-source image validation, isolated kernel timing, serving A/B, DSpark
acceptance, and model quality remain release requirements. Existing services
must not be patched, reconfigured, stopped, or restarted for this work.
