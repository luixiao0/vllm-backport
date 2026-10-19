# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the official recipe bindings against a streamed vLLM contract."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "deepseek_recipe_gateway", ROOT / "tools/deepseek-recipe/gateway.py"
)
assert SPEC is not None and SPEC.loader is not None
GATEWAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATEWAY
SPEC.loader.exec_module(GATEWAY)

CASES: list[tuple[str, dict[str, Any]]] = [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/responses", {"input": "hi"}),
    ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}]}),
]


@pytest.fixture
def tokenizer(tmp_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["Hello world! Singapore lookup reason 你好，世界！"],
        trainers.BpeTrainer(
            vocab_size=400,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[
                "<｜begin▁of▁sentence｜>",
                "<｜end▁of▁sentence｜>",
                "<｜User｜>",
                "<｜Assistant｜>",
                "<｜System｜>",
                "<think>",
                "</think>",
                "｜DSML｜",
                "<｜deepseek_image｜>",
            ],
        ),
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return tokenizer, path


class Stream(httpx.AsyncByteStream):
    def __init__(self, messages):
        self.messages = messages
        self.closed = False

    async def __aiter__(self):
        for message in self.messages:
            yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


def setup_client(
    tokenizer, answer="reason</think>你好，world!", *, mode="ok", **settings
):
    codec, path = tokenizer
    calls, streams = [], []

    def upstream(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        calls.append(body)
        if mode == "http_error":
            return httpx.Response(503)
        tokens = codec.encode(answer, add_special_tokens=False).ids
        # vLLM includes EOS in token_ids even though its decoded text excludes it.
        tokens.append(codec.token_to_id("<｜end▁of▁sentence｜>"))
        messages: list[dict[str, Any]] = [
            {"choices": [{"index": 0, "token_ids": [token], "finish_reason": None}]}
            for token in tokens
        ]
        if mode != "truncated":
            messages.append(
                {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": [],
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        if mode == "no_token_ids":
            messages = [
                {"choices": [{"index": 0, "text": answer, "finish_reason": "stop"}]}
            ]
        messages.append(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": len(body["prompt"]),
                    "completion_tokens": len(tokens),
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            }
        )
        stream = Stream(messages)
        streams.append(stream)
        return httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        )

    config = GATEWAY.Settings(
        upstream_url="http://engine",
        upstream_model="backend-model",
        served_model="flash",
        tokenizer_path=str(path),
        max_model_len=4096,
        default_max_tokens=128,
        **settings,
    )
    return (
        TestClient(GATEWAY.create_app(config, transport=httpx.MockTransport(upstream))),
        calls,
        streams,
    )


def response_text(path, body):
    if path == "/v1/chat/completions":
        return body["choices"][0]["message"]["content"]
    if path == "/v1/responses":
        return "".join(
            part["text"]
            for item in body["output"]
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )
    return "".join(part["text"] for part in body["content"] if part["type"] == "text")


@pytest.mark.parametrize("path,body", CASES)
@pytest.mark.parametrize("stream", [False, True])
def test_protocols_use_real_completion_contract_and_official_defaults(
    tokenizer, path, body, stream
):
    client, calls, streams = setup_client(tokenizer)
    with client:
        response = client.post(path, json={"model": "flash", "stream": stream, **body})
    assert response.status_code == 200, response.text
    sent = calls[0]
    assert sent["model"] == "backend-model"
    prompt = tokenizer[0].decode(sent["prompt"], skip_special_tokens=False)
    assert "Reasoning Effort: 75 " in prompt
    assert prompt.count("<｜begin▁of▁sentence｜>") == 1
    assert sent["add_special_tokens"] is False
    assert sent["return_token_ids"] is True
    assert sent["temperature"] == sent["top_p"] == 1.0
    assert streams[0].closed
    if stream:
        assert "text/event-stream" in response.headers["content-type"]
        assert "reason" in response.text
        assert "event: error" not in response.text
    else:
        assert response_text(path, response.json()) == "你好，world!"


def test_usage_and_sampling_are_forwarded_without_double_bos(tokenizer):
    client, calls, _ = setup_client(tokenizer)
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "flash",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.4,
                "top_p": 0.8,
                "max_tokens": 32,
            },
        )
    assert response.status_code == 200, response.text
    assert calls[0]["max_tokens"] == 32
    assert calls[0]["temperature"] == pytest.approx(0.4)
    assert calls[0]["top_p"] == pytest.approx(0.8)
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == len(calls[0]["prompt"])
    assert usage["prompt_cache_hit_tokens"] == 3


@pytest.mark.parametrize(
    "mode,status", [("http_error", 503), ("truncated", 502), ("no_token_ids", 502)]
)
def test_upstream_failure_is_not_reported_as_success(tokenizer, mode, status):
    client, _, streams = setup_client(tokenizer, mode=mode)
    with client:
        response = client.post(
            "/v1/chat/completions", json={"model": "flash", **CASES[0][1]}
        )
    assert response.status_code == status, response.text
    assert all(stream.closed for stream in streams)


def test_stream_failure_has_error_and_no_success_marker(tokenizer):
    client, _, streams = setup_client(tokenizer, mode="truncated")
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "flash",
                "stream": True,
                **CASES[0][1],
            },
        )
    assert "event: error" in response.text
    assert "[DONE]" not in response.text
    assert streams[0].closed


@pytest.mark.parametrize(
    "body,status",
    [
        ({"model": "other", **CASES[0][1]}, 404),
        (
            {
                "model": "flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "http://127.0.0.1/private"},
                            }
                        ],
                    }
                ],
            },
            400,
        ),
        ({"model": "flash", "reasoning_effort": 50, **CASES[0][1]}, 400),
        ({"model": "flash", "max_tokens": 4096, **CASES[0][1]}, 400),
    ],
)
def test_invalid_or_unsupported_input_never_reaches_engine(tokenizer, body, status):
    client, calls, _ = setup_client(tokenizer)
    with client:
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == status, response.text
    assert calls == []


def test_authentication_and_readiness(tokenizer):
    client, calls, _ = setup_client(tokenizer, api_key="test-only")
    with client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get("/v1/models", headers={"x-api-key": "test-only"}).status_code
            == 200
        )
        assert (
            client.get(
                "/v1/models", headers={"Authorization": "Bearer test-only"}
            ).status_code
            == 200
        )
    assert calls == []


@pytest.mark.parametrize("path,body", CASES)
def test_final_usage_uses_upstream_cache_counts(tokenizer, path, body):
    client, calls, _ = setup_client(tokenizer)
    with client:
        response = client.post(path, json={"model": "flash", **body})
    usage = response.json()["usage"]
    prompt = len(calls[0]["prompt"])
    if path == "/v1/chat/completions":
        assert usage["prompt_tokens"] == prompt
        assert usage["prompt_cache_hit_tokens"] == 3
    elif path == "/v1/responses":
        assert usage["input_tokens"] == prompt
        assert usage["input_tokens_details"]["cached_tokens"] == 3
    else:
        assert usage["input_tokens"] + usage["cache_read_input_tokens"] == prompt
        assert usage["cache_read_input_tokens"] == 3


@pytest.mark.parametrize("path,body", CASES)
def test_official_tool_parser_handles_split_dsml(tokenizer, path, body):
    answer = (
        '</think><｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
        '<｜DSML｜ parameter name="q" string="true">Singapore</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    if path == "/v1/messages":
        tools = [{"name": "lookup", "input_schema": schema}]
    elif path == "/v1/responses":
        tools = [{"type": "function", "name": "lookup", "parameters": schema}]
    else:
        tools = [
            {"type": "function", "function": {"name": "lookup", "parameters": schema}}
        ]
    client, _, streams = setup_client(tokenizer, answer)
    with client:
        response = client.post(path, json={"model": "flash", "tools": tools, **body})
    assert response.status_code == 200, response.text
    result = response.json()
    if path == "/v1/chat/completions":
        tool = result["choices"][0]["message"]["tool_calls"][0]["function"]
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert json.loads(tool["arguments"]) == {"q": "Singapore"}
    elif path == "/v1/responses":
        tool = next(
            item for item in result["output"] if item["type"] == "function_call"
        )
        assert json.loads(tool["arguments"]) == {"q": "Singapore"}
    else:
        tool = next(item for item in result["content"] if item["type"] == "tool_use")
        assert result["stop_reason"] == "tool_use"
        assert tool["input"] == {"q": "Singapore"}
    assert tool["name"] == "lookup"
    assert streams[0].closed


def test_thinking_disabled_and_stop_sequence_use_official_parsing(tokenizer):
    client, calls, streams = setup_client(tokenizer, "answer STOP hidden")
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "flash",
                **CASES[0][1],
                "thinking": {"type": "disabled"},
                "stop": ["STOP"],
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "answer "
    assert calls[0]["stop"] == ["STOP"]
    assert calls[0]["include_stop_str_in_output"] is True
    prompt = tokenizer[0].decode(calls[0]["prompt"], skip_special_tokens=False)
    assert prompt.endswith("<｜Assistant｜></think>")
    assert "Reasoning Effort" not in prompt
    assert streams[0].closed


def test_disconnect_closes_inference_iterator():
    closed = []

    async def run():
        disconnected = asyncio.Event()

        async def source():
            try:
                yield "data: first\n\n"
                await asyncio.Event().wait()
            finally:
                closed.append(True)

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                disconnected.set()

        response = GATEWAY.ClosingStreamingResponse(source())
        await asyncio.wait_for(
            response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send),
            timeout=2,
        )

    asyncio.run(run())
    assert closed == [True]


@pytest.mark.parametrize("path,body", CASES)
def test_streamed_final_usage_includes_late_cache_counts(tokenizer, path, body):
    client, calls, _ = setup_client(tokenizer)
    payload = {"model": "flash", "stream": True, **body}
    if path == "/v1/chat/completions":
        payload["stream_options"] = {"include_usage": True}
    with client:
        response = client.post(path, json=payload)
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    if path == "/v1/chat/completions":
        usage = next(event["usage"] for event in reversed(events) if event.get("usage"))
        assert usage["prompt_cache_hit_tokens"] == 3
    elif path == "/v1/responses":
        usage = next(
            event["response"]["usage"]
            for event in events
            if event.get("type") == "response.completed"
        )
        assert usage["input_tokens_details"]["cached_tokens"] == 3
    else:
        usage = next(
            event["usage"] for event in events if event.get("type") == "message_delta"
        )
        assert usage["cache_read_input_tokens"] == 3
        assert usage["input_tokens"] + 3 == len(calls[0]["prompt"])
