# Cached prompt tails in FULL decode graphs

This ports upstream [vLLM #58400](https://github.com/vllm-project/vllm/pull/58400)
and its hybrid-model prerequisite
[#58434](https://github.com/vllm-project/vllm/pull/58434) to this fork's V2 runner.
It applies to the official DeepSeek V4.1-Flash deployment with DSpark, without
changing weights, KV-cache precision or the DSpark draft block.

A request with existing context can schedule one new prompt token, optionally
padded with placeholder drafts. Previously, any prefilling request disqualified
the whole batch from a uniform FULL decode graph. The runner now classifies
these rows from request state before dispatch. For example, a 129-token prompt
with 128 tokens computed and one prompt token plus five placeholder drafts can
reuse the six-token FULL verification graph. Fresh one-token prompts and real
multi-token prompt chunks keep their prefill path. A one-token mid-prompt chunk
over existing context also qualifies. Graph selection still requires a captured
shape; eligibility alone does not create a new graph.

`has_prefill` and `is_prefilling_np` retain their request-state meaning, so prompt
input preparation still runs. Separate `decode_graph_eligible` and
`prefill_runs_as_decode_np` fields control graph selection and hybrid attention
metadata. Draft counts are computed once before adaptive verification compacts
the batch. Autoregressive and multi-module MTP drafters use the same eligibility;
DSpark inherits DFlash's independently uniform query-block graph and retains its
existing behavior. PCP conservatively keeps the old prefill classification;
microbatches slice the per-request mask.

This fork also has variable-length FULL decode graphs. Unlike uniform graphs,
they match on a query-length bound and have no uniform-token constraint.
Ineligible prefills therefore withhold both the uniform count and the dispatch
query bound. The existing DP handshake propagates the missing bound to every
rank. Eligible adaptive variable-length verification retains its FULL path.
The attention metadata's actual query length is not erased.

## Validation

The isolated test overlay uses the existing native extensions and checks the
SHA-256 preimage of each of eight Python runtime files before installation.
Installed hashes are checked again inside the test process. Production files,
service configuration and weights are unchanged.

The CPU cohort passes **85 tests**, with **24 CUDA-only cases skipped**:

```text
tests/v1/worker/test_gpu_batch_ordering.py
tests/v1/spec_decode/test_dynamic_sd_cug.py
tests/v1/worker/test_gpu_ubatch_slicing.py
tests/v1/worker/test_gpu_batch_shard.py
tests/v1/worker/test_gpu_pcp_manager.py
tests/v1/worker/test_mamba_hybrid_model_state.py
tests/v1/attention/test_gdn_metadata_builder.py
tests/v1/attention/test_mamba_update_block_table.py
```

Fourteen cases execute the real runner gather and `execute_model` dispatch
through `CudaGraphManager`, with uniform and variable-length FULL descriptors.
They cover cached padded tails, one-token mid-prompt chunks, fresh one-token
prompts, genuine two/three/six-token chunks and normal verification. An additional
adaptive replay shape stays eligible. Tests also cover mixed-batch masks,
state-slot reordering, DP propagation of an ineligible rank, PCP refusal,
microbatch mask slicing, and hybrid rollback draft counts after compaction.

The single-A100 GPU cohort passes **48 tests**, plus **one PCP partition test**
that checks the local eligibility and cleared mask. This covers all 24 CPU-skip
cases above. Each probe has a 256 MiB Torch allocator cap, a fresh 2 GiB free-VRAM
admission check, one exposed GPU, no network and a 240-second external timeout.
The 48-case probe takes 39.0 seconds; its peak allocation is 248,832 bytes and its
peak reservation is 6 MiB. The PCP probe takes 14.69 seconds and reserves 2 MiB.
The GPU returns to 2,877 MiB free after each probe.

These tests establish graph selection and metadata correctness. They do not
measure full-model throughput or end-to-end generation latency. The production
model is never loaded in a test container. The sparse-MQA and mHC overlap flags
remain disabled during this validation.
