# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official DeepSeek recipe protocols backed by vLLM raw completions.

The protocol flow follows deepseek-ai/deepseek-recipe's server-py example;
inference always comes from the configured vLLM server, never a mock backend.
"""

import hmac
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import httpx
from anyio import CancelScope
from deepseek_recipe import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ConversionError,
    ConversionOptions,
    DeepseekV41Encoding,
    InferenceChunk,
    InferenceFinishReason,
    MessagesRequest,
    MessagesResponse,
    PromptUsage,
    ResponsesRequest,
    ResponsesResponse,
    StreamProcessor,
    Tokenizer,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

PROTOCOLS = {
    "chat_completions": (ChatCompletionRequest, ChatCompletionResponse),
    "responses": (ResponsesRequest, ResponsesResponse),
    "messages": (MessagesRequest, MessagesResponse),
}
MAX_REQUEST_BYTES = 16 * 1024 * 1024


class GatewayError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class Settings:
    upstream_url: str
    upstream_model: str
    served_model: str
    tokenizer_path: str
    max_model_len: int = 278528
    default_max_tokens: int = 262144
    upstream_key: str = ""
    api_key: str = ""

    @classmethod
    def from_env(cls):
        def key_file(name):
            path = os.environ.get(name)
            if not path:
                return ""
            value = Path(path).read_text().strip()
            if not value:
                raise ValueError(f"Configured key file for {name} is empty")
            return value

        return cls(
            upstream_url=os.environ["RECIPE_UPSTREAM_URL"].rstrip("/"),
            upstream_model=os.environ["RECIPE_UPSTREAM_MODEL"],
            served_model=os.environ["RECIPE_SERVED_MODEL"],
            tokenizer_path=os.environ["RECIPE_TOKENIZER_PATH"],
            max_model_len=int(os.environ.get("RECIPE_MAX_MODEL_LEN", "278528")),
            default_max_tokens=int(os.environ.get("RECIPE_MAX_TOKENS", "262144")),
            upstream_key=key_file("RECIPE_UPSTREAM_KEY_FILE"),
            api_key=key_file("RECIPE_API_KEY_FILE"),
        )


def error_response(message, status_code):
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": "upstream_error"
                if status_code >= 500
                else "invalid_request_error",
                "message": message,
            },
        },
    )


def sse_frame(event, data):
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


def output_json(output, response_type, backend_usage):
    """Attach authoritative vLLM usage, which arrives after streamed tokens."""
    body = json.loads(output.to_json())
    if not backend_usage:
        return json.dumps(body, ensure_ascii=False)
    prompt = backend_usage["prompt_tokens"]
    completion = backend_usage["completion_tokens"]
    cached = (backend_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if response_type is ChatCompletionResponse and body.get("usage") is not None:
        body["usage"].update(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            prompt_tokens_details={"cached_tokens": cached},
            prompt_cache_hit_tokens=cached,
            prompt_cache_miss_tokens=prompt - cached,
        )
    elif response_type is ResponsesResponse:
        target = body.get("response", body)
        if target.get("usage") is not None:
            target["usage"].update(
                input_tokens=prompt,
                output_tokens=completion,
                total_tokens=prompt + completion,
                input_tokens_details={"cached_tokens": cached},
            )
    elif response_type is MessagesResponse and body.get("usage") is not None:
        body["usage"].update(
            input_tokens=prompt - cached,
            cache_read_input_tokens=cached,
            output_tokens=completion,
        )
    return json.dumps(body, ensure_ascii=False)


async def sse_json(response) -> AsyncIterator[dict]:
    data = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            payload = "\n".join(data)
            data.clear()
            if payload == "[DONE]":
                return
            yield json.loads(payload)
    if data:
        payload = "\n".join(data)
        if payload != "[DONE]":
            yield json.loads(payload)


class ClosingStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                await self.body_iterator.aclose()


def create_app(settings: Settings | None = None, *, transport=None) -> FastAPI:
    settings = settings or Settings.from_env()
    tokenizer = Tokenizer.from_file(settings.tokenizer_path)
    encoding = DeepseekV41Encoding().with_tokenizer(tokenizer)
    eos_ids = tokenizer.encode("<｜end▁of▁sentence｜>")
    if len(eos_ids) != 1:
        raise ValueError("The model tokenizer must encode the V4.1 EOS as one token")
    eos_id = eos_ids[0]
    if settings.max_model_len <= 1 or settings.default_max_tokens < 1:
        raise ValueError("Model and output token limits must be positive")
    url = httpx.URL(settings.upstream_url)
    if url.scheme not in {"http", "https"} or url.username or url.password:
        raise ValueError("Use an HTTP upstream URL and a separate key file")

    @asynccontextmanager
    async def lifespan(app):
        headers = {}
        if settings.upstream_key:
            headers["Authorization"] = f"Bearer {settings.upstream_key}"
        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(3600, connect=10),
            transport=transport,
            trust_env=False,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="DeepSeek recipe / vLLM", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request, call_next):
        if settings.api_key and request.url.path not in {"/healthz", "/readyz"}:
            key = request.headers.get("x-api-key", "")
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                key = auth[7:]
            if not hmac.compare_digest(key.encode(), settings.api_key.encode()):
                return error_response("Invalid API key", 401)
        return await call_next(request)

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(request, exc):
        return error_response(exc.message, exc.status_code)

    @app.exception_handler(ConversionError)
    async def handle_conversion_error(request, exc):
        return Response(exc.body, exc.status_code, media_type="application/json")

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "recipe_version": "0.1.1"}

    @app.get("/readyz")
    async def ready():
        try:
            response = await app.state.client.get(
                f"{settings.upstream_url}/health", timeout=5
            )
        except httpx.HTTPError:
            return error_response("vLLM is unavailable", 503)
        if response.status_code != 200:
            return error_response("vLLM is not ready", 503)
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": settings.served_model, "object": "model", "owned_by": "local"}
            ],
        }

    def prepare(protocol, body):
        request_type, response_type = PROTOCOLS[protocol]
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("Request body must be an object")
            if data.get("model") != settings.served_model:
                raise GatewayError("Unknown model", 404)
            request = request_type(body)
            include_usage = (
                request.include_usage() if protocol == "chat_completions" else False
            )
            custom_tools = (
                request.custom_tool_names() if protocol == "responses" else set()
            )
            converted = request.convert(ConversionOptions())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise GatewayError(str(exc)) from exc
        rendered = encoding.render_conversation(converted.conversation)
        if rendered.image_sources:
            raise GatewayError(
                "This raw-completions gateway supports text and client tools. "
                "Use the original vLLM multimodal endpoint for images."
            )
        if converted.inference_options.disable_parallel_tool_use:
            raise GatewayError(
                "The raw-completions backend cannot enforce serial tools"
            )
        prompt_ids = encoding.encode(converted.conversation)
        available = settings.max_model_len - len(prompt_ids)
        if available < 1:
            raise GatewayError("Prompt exceeds the configured model context")
        options = converted.inference_options
        max_tokens = options.max_tokens
        if max_tokens is None:
            max_tokens = min(settings.default_max_tokens, available)
        if max_tokens > available:
            raise GatewayError("Requested output exceeds the remaining model context")
        payload = {
            "model": settings.upstream_model,
            "prompt": prompt_ids,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "max_tokens": max_tokens,
            "temperature": options.temperature
            if options.temperature is not None
            else 1.0,
            "top_p": options.top_p if options.top_p is not None else 1.0,
        }
        if options.thinking_budget_tokens is not None:
            payload["thinking_token_budget"] = options.thinking_budget_tokens
        if converted.parsing_options.stop_sequences:
            payload["stop"] = converted.parsing_options.stop_sequences
            payload["include_stop_str_in_output"] = True
        response_id = str(uuid4())
        generator = request_type.chunk_generator(
            converted, response_id, settings.served_model
        )
        if protocol == "chat_completions":
            generator = generator.with_include_usage(include_usage)
        elif protocol == "responses":
            generator = generator.with_custom_tool_names(custom_tools)
        processor = StreamProcessor(generator, converted.parsing_options, tokenizer)
        accumulator = response_type(
            response_id, settings.served_model, int(time.time()), len(prompt_ids), 0
        )
        return converted.stream, payload, processor, accumulator, response_type

    async def response_parts(
        upstream, payload, processor, accumulator, response_type, stream
    ):
        finish_reason = None
        saw_eos = False
        backend_usage = None
        terminal_outputs = []

        def outputs_for(chunk):
            outputs = processor.push(chunk)
            if processor.finished:
                terminal_outputs.extend(outputs)
                return []
            return outputs

        try:
            initial = InferenceChunk.ready(
                prompt_usage=PromptUsage(prompt_tokens=len(payload["prompt"]))
            )
            for output in outputs_for(initial):
                if stream:
                    yield sse_frame(
                        response_type.chunk_event_type(output), output.to_json()
                    )
                else:
                    accumulator.append(output)
            async for message in sse_json(upstream):
                if "error" in message:
                    raise GatewayError("vLLM failed during generation", 502)
                usage = message.get("usage")
                chunks = []
                if usage:
                    backend_usage = usage
                    details = usage.get("prompt_tokens_details") or {}
                    chunks.append(
                        InferenceChunk.ready(
                            prompt_usage=PromptUsage(
                                prompt_tokens=usage["prompt_tokens"],
                                prompt_cache_hit_tokens=details.get("cached_tokens", 0),
                            )
                        )
                    )
                for choice in message.get("choices", []):
                    if choice.get("index") != 0:
                        raise GatewayError("Unexpected upstream choice", 502)
                    token_ids = choice.get("token_ids")
                    if token_ids is None:
                        raise GatewayError("vLLM must support return_token_ids", 502)
                    for token in token_ids:
                        if saw_eos:
                            raise GatewayError("vLLM returned tokens after EOS", 502)
                        if token == eos_id:
                            saw_eos = True
                        else:
                            chunks.append(InferenceChunk.token(token))
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                for chunk in chunks:
                    if processor.finished:
                        break
                    for output in outputs_for(chunk):
                        if stream:
                            yield sse_frame(
                                response_type.chunk_event_type(output), output.to_json()
                            )
                        else:
                            accumulator.append(output)
            reasons = {
                "stop": InferenceFinishReason.Stop,
                "length": InferenceFinishReason.Length,
                "content_filter": InferenceFinishReason.ContentFilter,
            }
            if finish_reason not in reasons:
                raise GatewayError("vLLM stream ended without a finish reason", 502)
            if not processor.finished:
                outputs_for(InferenceChunk.finish(reasons[finish_reason]))
            for output in terminal_outputs:
                if stream:
                    yield sse_frame(
                        response_type.chunk_event_type(output),
                        output_json(output, response_type, backend_usage),
                    )
                else:
                    accumulator.append(output)
            if stream:
                done = response_type.done_message()
                if done is not None:
                    yield sse_frame(None, done)
            else:
                yield output_json(accumulator, response_type, backend_usage)
        except (
            httpx.HTTPError,
            ValueError,
            KeyError,
            RuntimeError,
            GatewayError,
        ) as exc:
            if stream:
                yield sse_frame(
                    "error",
                    json.dumps(
                        {
                            "error": {
                                "type": "upstream_error",
                                "message": "Generation failed",
                            }
                        }
                    ),
                )
            else:
                raise GatewayError("vLLM generation failed", 502) from exc
        finally:
            processor.close()
            with CancelScope(shield=True):
                await upstream.aclose()

    def handler(protocol):
        async def generate(request: Request):
            body = bytearray()
            async for part in request.stream():
                body.extend(part)
                if len(body) > MAX_REQUEST_BYTES:
                    raise GatewayError("Request is too large", 413)
            (
                stream,
                payload,
                processor,
                accumulator,
                response_type,
            ) = await run_in_threadpool(prepare, protocol, bytes(body))
            upstream = None
            try:
                outgoing = app.state.client.build_request(
                    "POST", f"{settings.upstream_url}/v1/completions", json=payload
                )
                upstream = await app.state.client.send(outgoing, stream=True)
                if upstream.status_code != 200:
                    raise GatewayError(
                        "vLLM rejected the request", upstream.status_code
                    )
            except BaseException as exc:
                processor.close()
                if upstream is not None:
                    with CancelScope(shield=True):
                        await upstream.aclose()
                if isinstance(exc, GatewayError):
                    raise
                if isinstance(exc, httpx.HTTPError):
                    raise GatewayError("vLLM is unavailable", 502) from exc
                raise
            output = response_parts(
                upstream, payload, processor, accumulator, response_type, stream
            )
            if stream:
                return ClosingStreamingResponse(
                    output,
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return Response(
                "".join([part async for part in output]), media_type="application/json"
            )

        return generate

    for path, protocol in (
        ("/v1/chat/completions", "chat_completions"),
        ("/v1/responses", "responses"),
        ("/v1/messages", "messages"),
    ):
        app.add_api_route(path, handler(protocol), methods=["POST"])
    return app
