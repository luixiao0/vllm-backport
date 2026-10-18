# SM80 backport development

This fork targets A100 serving, beginning with DeepSeek V4.1 and DSpark.
The first kernel project is a
[length-aware candidate selector](candidate-selection.md) inspired by upstream
vLLM #61039. Its SM100 DeepSelect implementation cannot run directly on SM80.
The score-kernel prototype and complete selector have passed bounded
shared-A100 screening. The owner accepts this screening as the pass criterion;
full-model throughput and model-quality measurements have not been performed.

A complete experimental backend replaces full-width `torch.topk` with an opt-in
SM80-compatible [hierarchical radix selector](candidate-radix.md). It computes
device-side lengths, parallel coarse histograms, and an exact boundary-bucket
selection. Its 31 candidate tests passed, and 12 shared timing cases were faster
than the comparison path. The [MXFP8 grouped projection port](mxfp8-grouped-projection.md)
additionally supports compact quantized WO_A weights on SM80. It reduces
resident weight memory but currently regresses latency in most screened cases,
so it remains an explicit tradeoff behind a disabled-by-default flag.

## Branches and provenance

- `master` follows `wtdcode/vllm-backport`; keep fork changes on `sm80` and
  topic branches.
- `sm80` is the development branch. Its starting commit is
  `29e66dad461b11caae6039f510eaa7913f9dfb19` from the backport upstream.
- The production reference for comparison is upstream tag `v0.13.1`, commit
  `0cc2bedf83dc8cae3b6860f3ed82a4c22c7941c3`. Starting from newer upstream
  source does not establish production compatibility.
- The initial LMCache build pin is
  `2acb7b24bea5ac55c5dfc6a41366099236ec525c`, matching the inherited Dockerfile.
- Local remotes: `origin` is this fork, `upstream` is wtdcode's backport,
  and `reference-buliaoyin` is a research reference. Changes from that reference
  are not part of the development baseline.

For each port, record the source URL/commit, unavailable hardware features,
replacement implementation, correctness evidence, performance evidence, and
rollback. Preserve the original implementation as the comparison path until
the replacement has passed its release criteria.

## Build

From a clean checkout, inspect the command before running a build:

```bash
tools/sm80/build-image.sh --dry-run
tools/sm80/build-image.sh
```

The script builds `linux/amd64` for CUDA architecture `8.0`, includes the pinned
LMCache fork, and loads a local image named
`vllm-backport-sm80:git-<commit>`. It does not publish an image. A Linux x86-64
builder with sufficient storage and CUDA build support is the intended host;
the local macOS lint environment does not validate a CUDA build.

Build parameters can be set explicitly:

```bash
SM80_MAX_JOBS=8 SM80_NVCC_THREADS=1 tools/sm80/build-image.sh
```

`SM80_IMAGE` changes the local image name. `SM80_LMCACHE_REF` accepts a full
commit SHA for deliberate LMCache updates. The source revision, architecture,
and LMCache revision are explicit, but upstream dependency and base-image
resolution means these are not bit-for-bit reproducible builds. Record the
resulting image digest and installed package versions when testing a release.

The inherited Docker Publish workflow is restricted to its original repository.
Our publishing destination and build runner will be configured separately.

## Validation and release

Existing production deployments must not be changed as part of this work.
On serving hosts, use read-only inspection or separate probes and tests. Short
correctness probes may use spare VRAM with bounded allocation and runtime, but
must not patch installed production files, change service configuration, or stop
or restart any existing service. Shared-GPU timing must be labelled with its
measurement scope; it is the owner's accepted screening evidence and does not
measure full-model serving performance.

Use the repository's `AGENTS.md`, kernel-writing guidance, and microbenchmark
guidance. Run existing correctness tests before timing and extend the nearest
suite for missing boundary cases. Put kernel benchmarks in `benchmarks/kernels/`.

The target measurement envelope is A100 SM80, TP8, DSpark with five drafts,
`fp8_ds_mla`, CUDA graphs, and LMCache. Test concurrency 1 through 8 and both
short and long contexts. Keep the device, software versions, shapes, precision,
graph mode, and cache treatment in each result artifact.

A port records design, opt-in implementation, correctness, kernel A/B and a
pinned candidate image. Serving and model-quality evaluation are separate
measurements and require resources allowed by the owner. The default path
changes only when the evidence supports it. Performance claims from other
GPU generations are hypotheses for this fork, not A100 measurements.

General serving fixes from the preceding review remain a separate patch series:
KV-load admission (#57104), grammar polling (#55931), finish reasons (#46303),
choice escaping (#48115), and reasoning boundaries (#56200 then #56635).
The bootstrap baseline already includes the backport fork's DSML wrapper fix
(#89). Do not reapply it.
