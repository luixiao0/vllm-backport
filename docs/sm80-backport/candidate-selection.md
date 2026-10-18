# First port: length-aware DSV4.1 candidate selection

Status: design. No replacement kernel is enabled or implemented by this change.

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

This command has not been run as part of the fork bootstrap.

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

GPU measurements must use an isolated allocation. The currently serving A100
container is not a benchmark harness; this bootstrap schedules no GPU work.
