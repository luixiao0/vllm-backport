#!/usr/bin/env bash
set -euo pipefail

dry_run=false
case "${1:-}" in
  --dry-run) dry_run=true ;;
  "") ;;
  *) printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf 'Usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi

repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
revision=$(git -C "$repo_root" rev-parse HEAD)
image=${SM80_IMAGE:-vllm-backport-sm80:git-${revision:0:12}}
max_jobs=${SM80_MAX_JOBS:-4}
nvcc_threads=${SM80_NVCC_THREADS:-1}
lmcache_ref=${SM80_LMCACHE_REF:-2acb7b24bea5ac55c5dfc6a41366099236ec525c}

if [[ ! "$max_jobs" =~ ^[1-9][0-9]*$ || ! "$nvcc_threads" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Build parallelism must be a positive integer.\n' >&2
  exit 2
fi
if [[ ! "$lmcache_ref" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'SM80_LMCACHE_REF must be a full commit SHA.\n' >&2
  exit 2
fi

command_args=(
  docker buildx build
  --platform linux/amd64
  --file "$repo_root/docker/Dockerfile"
  --target vllm-openai
  --build-arg torch_cuda_arch_list=8.0
  --build-arg "max_jobs=$max_jobs"
  --build-arg "nvcc_threads=$nvcc_threads"
  --build-arg INSTALL_LMCACHE=true
  --build-arg "LMCACHE_REF=$lmcache_ref"
  --label org.opencontainers.image.source=https://github.com/luixiao0/vllm-backport
  --label "org.opencontainers.image.revision=$revision"
  --tag "$image"
  --load
  "$repo_root"
)

if "$dry_run"; then
  printf '%q ' "${command_args[@]}"
  printf '\n'
  exit 0
fi
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=normal)" ]]; then
  printf 'Commit or isolate working-tree changes before building a revision-tagged image.\n' >&2
  exit 1
fi

exec "${command_args[@]}"
