#!/usr/bin/env bash
set -euo pipefail

: "${RECIPE_IMAGE:?Set the tested image tag or digest}"
: "${RECIPE_MODEL_DIR:?Set the model directory containing tokenizer.json}"
: "${RECIPE_UPSTREAM_MODEL:?Set the actual model name returned by vLLM /v1/models}"

recipe_name="${RECIPE_CONTAINER_NAME:-deepseek-recipe-candidate}"
if docker container inspect "${recipe_name}" >/dev/null 2>&1; then
  echo "Container ${recipe_name} exists; inspect it before replacing it." >&2
  exit 1
fi
test -r "${RECIPE_MODEL_DIR}/tokenizer.json"

recipe_args=(
  --detach --name "${recipe_name}" --runtime runc --network host
  --read-only --tmpfs "/tmp:rw,nosuid,size=64m"
  --cpus 1 --memory 1g --memory-swap 1g --pids-limit 128
  --cap-drop ALL --security-opt no-new-privileges
  --restart unless-stopped --log-opt max-size=10m --log-opt max-file=3
  --label app=deepseek-recipe --label role=candidate
  --env "RECIPE_UPSTREAM_URL=${RECIPE_UPSTREAM_URL:-http://127.0.0.1:18005}"
  --env "RECIPE_UPSTREAM_MODEL=${RECIPE_UPSTREAM_MODEL}"
  --env "RECIPE_SERVED_MODEL=${RECIPE_SERVED_MODEL:-${RECIPE_UPSTREAM_MODEL}}"
  --env RECIPE_TOKENIZER_PATH=/model/tokenizer.json
  --env "RECIPE_MAX_MODEL_LEN=${RECIPE_MAX_MODEL_LEN:-278528}"
  --env "RECIPE_MAX_TOKENS=${RECIPE_MAX_TOKENS:-262144}"
  --mount "type=bind,source=${RECIPE_MODEL_DIR}/tokenizer.json,target=/model/tokenizer.json,readonly"
)
if [[ -n "${RECIPE_API_KEY_FILE:-}" ]]; then
  recipe_args+=(
    --mount "type=bind,source=${RECIPE_API_KEY_FILE},target=/run/secrets/recipe-key,readonly"
    --env RECIPE_API_KEY_FILE=/run/secrets/recipe-key
  )
fi
if [[ -n "${RECIPE_UPSTREAM_KEY_FILE:-}" ]]; then
  recipe_args+=(
    --mount "type=bind,source=${RECIPE_UPSTREAM_KEY_FILE},target=/run/secrets/upstream-key,readonly"
    --env RECIPE_UPSTREAM_KEY_FILE=/run/secrets/upstream-key
  )
fi
docker run "${recipe_args[@]}" "${RECIPE_IMAGE}"
