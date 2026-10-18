# First port: length-aware DSV4.1 candidate selection

Status: experimental score-kernel prototype, disabled by default. The complete
selector still uses full-width `torch.topk`; this is not a completed port of
DeepSelect or a validated serving optimization.

## Running the prototype

Set `VLLM_DSV4_LENGTH_AWARE_CANDIDATE_SCORES=1` before starting the test process.
The score kernel skips loads and reductions for tiles entirely outside the
row's valid context, writes their padding scores, and preserves the original
newest-block rule. Score layout, top-k, candidate storage, call sites, and graph
shapes remain unchanged. Reductions use FP32, with scores stored in the original
dtype. Unset the flag or set it to `0` to select the baseline score path.

On an isolated CUDA development allocation, run both paths' existing tests:

```bash
.venv/bin/python -m pytest \
  tests/model_executor/layers/test_mla_short_prefill_indexer.py \
  -q -k 'candidate or sparse_topk_remap'
.venv/bin/python benchmarks/kernels/benchmark_dsv41_candidate_scores.py \
  --rows 1 6 48 --width 278528 --output outputs/candidate-scores.json
```

The benchmark checks candidate equality before timing, alternates baseline and
prototype order, and reports score, top-k, store, and complete-selector timings
separately. It requests CUDA graph replay and cold L2 from FlashInfer's CUPTI
helper; preserve its logs because that helper can fall back when CUPTI is absent.
The score bandwidth metric counts valid logits read once plus all score writes;
it excludes allocation, compilation, and other selector stages. These are kernel
replay measurements, not serving latency or quality measurements.

For a development overlay on the known v0.13.1 base:

```bash
docker build --file docker/Dockerfile.sm80-candidate \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --tag vllm-sm80-candidate:dev .
```

Set `BASE_IMAGE` to the tested base image's immutable reference. The overlay
checks the original source hashes before replacing either file, records package
versions and new source hashes in `/opt/sm80-candidate.json`, and leaves the flag
off. It reuses the base's native extensions; it does not validate a full source
build, publish an image, or start a model server.

## Validation evidence

- 60 CPU Triton-interpreter cases passed: FP32/FP16, packed and decode bounds,
  strides, repeated row metadata, empty and shortened contexts, ends beyond
  allocated width, tied values, NaN and infinities. Both flag states matched
  the original kernel's scores and top-k IDs and an independent PyTorch score
  reference. The check used import/env stubs for the exact kernel source.
- 12 explicit-target SM80 compilations passed: FP32/FP16/BF16, packed/decode,
  and both flag states. Each produced SM80 PTX and cubin without initializing
  CUDA. These configurations compile; this does not prove GPU execution.
- BF16 interpreter execution is unavailable in Triton and is not counted as a
  passed correctness case.
- Eight additional CPU-interpreter output-storage cases passed, covering strided
  outputs, untouched guard columns, NaN filtering, and `-1` padding.
- The development overlay built successfully; a CPU-only smoke check verified
  both installed source hashes and the flag's default/off/on behavior. Its
  preimage guard also rejected an already modified base.
- A later bounded probe on spare VRAM ran 13 extracted regression cases on
  a shared A100 using the real installed module and env registry. Both flag
  states, packed/decode bounds, empty rows, padding, and FP32/FP16/BF16 CUDA graph
  replays with changing bounds passed. Peak PyTorch allocation was 9.5 MiB and
  reservation 28 MiB, plus CUDA context overhead. The probe used a 256 MiB
  allocator cap, exposed one GPU only, and exited without changing the existing
  service. This was not the full pytest module or a serving test.
- A second bounded shared-A100 probe checked allocated width 278528, block size
  8, top-k 2048, rows 1/6/48, and contexts 8192/32768/131072/278528. All 12 FP32
  configurations matched baseline scores and candidate IDs exactly during graph
  replay; multi-row cases included an empty row and reused graphs across changing
  bounds. The probe completed in 12.94 seconds, with 67.9 MiB peak tensor
  allocation and 120 MiB reservation, plus CUDA context overhead. Four alternating
  A/B event samples per path showed no compelling complete-selector benefit.
  The first baseline score sample in each configuration was an outlier around
  2.3 ms. Keep those raw samples; these timings cannot establish a speedup or
  regression on a shared GPU. No profiler or cache-flushing loop was used.
- Full GPU-suite execution, isolated kernel timings, serving A/B, and model
  quality evaluation remain pending. Shared-GPU probes establish no speedup and
  provide no basis for changing the default.

## Problem and source

DSV4.1's candidate source reduces logits into block scores, pins the newest block,
and chooses candidate block IDs for later sparse indexers. CUDA graphs require
static allocation sizes, although each row's valid context may be much shorter.

The current `select_candidate_blocks` scores the full allocated width and calls
`torch.topk` over all block scores. Upstream
[vLLM #61039](https://github.com/vllm-project/vllm/pull/61039), head
`5e952d607749297e28d96050e2023fc3a6295b5f`, avoids this waste with a valid-length
aware DeepSelect path. Its `has_deep_select()` requires CUDA family 100 and a
compiled extension. A100 cannot use that path; the upstream fallback still
processes the full width.

Port the algorithmic idea while retaining SM80-compatible operations. Removing
the architecture check would not implement a port.

## Implementation boundaries

The main implementation is
`vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`. Preserve its
call interface and the existing sparse-indexer call sites. The input bounds
are device tensors; the replacement must not copy lengths to the CPU or make
the graph shape depend on host reads of live lengths.

The initial implementation sequence is:

1. Measure score reduction, top-k, and output storage separately and together
   for the actual A100 workload. Record allocation overhead separately from
   replay timing, and include it in the serving comparison where appropriate.
2. Prototype valid-length score reduction using ordinary Triton operations
   available on SM80. Keep unprocessed score storage initialized or masked;
   graph replay and short rows must never expose stale candidates.
3. Measure that prototype against the original before selecting a top-k design.
   Merely reducing score work while retaining full-width `torch.topk` is a
   partial optimization. Do not call it a completed length-aware selector.
4. Implement and compare a device-length-aware selection strategy only if top-k
   is material in the profile. Avoid new SM90/SM100 instructions, DeepSelect
   binary dependencies, per-request JIT, and graph-time host synchronization.
5. Keep the new path opt-in until kernel and serving validation is complete.

## Correctness contract

Reuse `tests/model_executor/layers/test_mla_short_prefill_indexer.py`, including
the existing candidate empty-row and packed-bounds tests.

- Preserve request-local IDs, packed starts/ends, `row_repeat`, strides, and
  output padding with `-1`.
- Preserve block maximum scores and newest-block selection for finite inputs.
- Initially preserve the baseline's tie choices. A different set of equally
  scored blocks can change downstream token candidates; it is not automatically
  an equivalent model computation.
- Cover zero width, empty rows, fewer valid blocks than K, K above the allocated
  block count, partial final blocks, ends beyond the allocated width, and
  non-contiguous output.
- Exercise NaN and infinity inputs, warmup buffers, and consecutive graph
  replays with changing row lengths. Check bounds and stale output explicitly;
  document any intentional difference from the baseline.
- Keep model outputs and DSpark acceptance in the evaluation. Kernel-level
  membership tests alone do not establish serving quality.

Run the relevant existing suite in the CUDA development environment:

```bash
.venv/bin/python -m pytest \
  tests/model_executor/layers/test_mla_short_prefill_indexer.py \
  -q -k 'candidate or sparse_topk_remap'
```

The full GPU suite above remains pending. The bounded shared-GPU probe used
extracted regression bodies; it does not replace the complete test environment.

## Measurement and acceptance

Start with allocated width 278528 and derive the compressed widths from actual
indexer call sites. Sweep request concurrency 1, 2, 4, and 8, with one-token
decode and six-token DSpark verification where applicable. Use contexts such
as 8K, 32K, 128K, and the configured maximum; include mixed lengths and padded
rows. Do not assume that global batch size equals each rank's local row count.

Use the repository's FlashInfer CUPTI timing pattern with CUDA graph replay and
cold-L2 treatment. Warm up compilation outside the measured region and alternate
baseline/candidate runs. Report score-kernel time, selection time, complete
selector time, scratch memory, and end-to-end ITL/throughput. Keep TTFT and long
prefill regressions visible.

The port is accepted only after correctness and model evaluation pass and a
repeatable benefit is observed in the target serving workload. A smaller
microbenchmark number without a serving benefit is an experiment result, not
a reason to change the default. If top-k is not a meaningful bottleneck at
concurrency at most eight, record that result and reprioritize the next port.

Performance measurements require an isolated allocation. With the user's
authorization, short correctness probes may use spare VRAM in a separate
container with explicit allocation and runtime limits. Do not modify, stop, or
restart the existing service, and do not treat shared-GPU timing as performance
evidence. The serving container is not a test or benchmark harness.
