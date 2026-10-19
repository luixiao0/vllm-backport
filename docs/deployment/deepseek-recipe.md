# Official DeepSeek recipe gateway

This CPU gateway uses `deepseek-recipe==0.1.1` for request conversion, V4.1
prompt encoding, thinking/tool parsing, and streaming response generation. It
connects to an existing vLLM `/v1/completions` endpoint using token IDs. The
existing vLLM tokenizer and tool parsers do not process these requests again.
The gateway does not load weights or use a GPU.

Supported endpoints are `POST /v1/chat/completions`, `POST /v1/responses`, and
`POST /v1/messages`, each with JSON and SSE responses. Function tools execute
on the client. `GET /v1/models`, `/healthz`, and `/readyz` expose the configured
model and readiness. Request/response conversion follows the official library;
this is a stateless gateway, including for Responses requests.

## Official behavior

The official recipe defaults to thinking enabled with high effort (budget 75),
while the existing fork maps `high` to 50. This gateway adopts the official
semantics. The recipe API accepts its supported effort names, not arbitrary
numeric reasoning budgets. To disable thinking, Chat Completions and Messages
use `thinking.type=disabled`; Responses uses `reasoning.effort=none`.
Sampling defaults are `temperature=1` and
`top_p=1`, matching the 1017 model card. Caller-specified sampling values and
thinking token budgets are forwarded to vLLM.

An omitted output limit defaults to 262144 tokens, bounded by the remaining
configured model context. Explicit limits exceeding that remaining context
return an error. Set `RECIPE_MAX_MODEL_LEN` to the actual vLLM model limit;
this setting does not enlarge vLLM's context or KV cache.

This gateway currently supports text and client function tools. Image requests
are rejected before any URL is fetched; keep using the original multimodal vLLM
endpoint for images. Enforcing serial-only tool use is not available through
raw completions and returns an explicit error. The official library's other
limitations, including unsupported document/audio/video input and server-side
tool execution, continue to apply. It does not enforce JSON Schema output or
tool `strict` constraints; do not use those as an application validation boundary.

## Build and run

Build from the repository root. Dependencies and wheel hashes are locked;
record the built image digest and base digest for a deployment.

```bash
docker build -f docker/Dockerfile.deepseek-recipe \
  -t deepseek-recipe-vllm:git-<revision> .
```

On the vLLM host, use the actual backend model name and matching tokenizer.
Installing this gateway does not switch the backend to 1017 weights. Its
served name defaults to the backend's name so an old checkpoint is not
advertised as 1017.

```bash
export RECIPE_IMAGE=deepseek-recipe-vllm:git-<revision>
export RECIPE_MODEL_DIR=/mnt/disk2/models/deepseek-ai/DeepSeek-V4.1-Flash
export RECIPE_UPSTREAM_MODEL=deepseek-v4.1-flash
export RECIPE_MAX_MODEL_LEN=278528
bash tools/deepseek-recipe/run.sh
curl --fail http://127.0.0.1:18006/readyz
```

The container binds to loopback on port 18006 using the host network, runs
without the NVIDIA runtime, and mounts only the tokenizer file. CPU, memory,
process count, and logs are bounded. It uses a read-only filesystem and runs
as an unprivileged user. `run.sh` refuses to replace an existing container.

Optional `RECIPE_API_KEY_FILE` enables Bearer or `x-api-key` authentication;
`RECIPE_UPSTREAM_KEY_FILE` supplies a separate backend Bearer token. Files must
be readable by container UID 65532. Tokens are read from mounted files and are
not placed in command arguments. Keep this candidate on loopback until an
authenticated ingress is deliberately configured. A local tunnel can inspect
the candidate without changing the existing public endpoint:

```bash
ssh -L 18006:127.0.0.1:18006 a100
```

Rollback stops only the gateway; retain its container for inspection:

```bash
docker stop deepseek-recipe-candidate
```

## Verification

Run the CPU suite in an environment containing the locked dependencies, pytest,
and tokenizers. No model weights or inference service are required:

```bash
uv run --no-project .venv/bin/python -m pytest --noconftest \
  tests/entrypoints/openai/test_deepseek_recipe_gateway.py -q
```

The suite covers all three protocols, official effort defaults, split Unicode
tokens and DSML tool calls, JSON/SSE output, output limits, upstream failures,
client disconnection cleanup, and cached-token accounting. vLLM reports cache
usage in its final SSE event. The gateway attaches that authoritative usage to
the terminal recipe response; Messages streaming clients should apply final
`message_delta.usage` updates to the initial `message_start` usage.

References:

- [Official recipe](https://github.com/deepseek-ai/deepseek-recipe)
- [Official Python HTTP example](https://github.com/deepseek-ai/deepseek-recipe/tree/8cadfede7063c896b944e7bae05daa3549ae97ea/server-py)
- [1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
