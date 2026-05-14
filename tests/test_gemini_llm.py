"""Tests for GeminiLLMClient — issue #241.

Verifies the streaming delta shape, tool-call accumulation across chunks,
429 backoff behaviour, and null/empty content chunk handling.

All Gemini SDK calls are mocked so these tests have zero network/API-key
dependencies.
"""

from __future__ import annotations
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot_comic.gemini_llm import GeminiLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> GeminiLLMClient:
    """Return a GeminiLLMClient with a dummy API key."""
    return GeminiLLMClient(api_key="DUMMY", model="gemini-2.5-flash")


def _make_text_chunk(text: str) -> Any:
    """Build a fake streaming chunk that carries a text part."""
    part = MagicMock()
    part.text = text
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = None

    chunk = MagicMock()
    chunk.candidates = [candidate]
    return chunk


def _make_tool_chunk(name: str, args: dict[str, Any], index: int = 0) -> Any:
    """Build a fake streaming chunk that carries a function-call part."""
    fc = MagicMock()
    fc.name = name
    fc.args = args

    part = MagicMock()
    part.text = None
    part.function_call = fc

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = None

    chunk = MagicMock()
    chunk.candidates = [candidate]
    return chunk


def _make_finish_chunk(finish_reason: str = "stop") -> Any:
    """Build a fake streaming chunk that signals end-of-stream."""
    fr = MagicMock()
    fr.name = finish_reason.upper()

    content = MagicMock()
    content.parts = []

    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = fr

    chunk = MagicMock()
    chunk.candidates = [candidate]
    return chunk


def _make_null_chunk() -> Any:
    """Build a chunk with a None/empty-text part — should be skipped."""
    part = MagicMock()
    part.text = None
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = None

    chunk = MagicMock()
    chunk.candidates = [candidate]
    return chunk


async def _collect_deltas(client: GeminiLLMClient, chunks: list[Any]) -> list[dict[str, Any]]:
    """Run stream_completion against *chunks* and collect all deltas."""

    async def _fake_stream(*_args: Any, **_kwargs: Any):  # type: ignore[no-untyped-def]
        for c in chunks:
            yield c

    with patch.object(client._client.aio.models, "generate_content_stream", return_value=_fake_stream()):
        deltas: list[dict[str, Any]] = []
        async for d in client.stream_completion(messages=[], tools=[], system_instruction="test"):
            deltas.append(d)
        return deltas


# ---------------------------------------------------------------------------
# Tests — delta shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_delta_shape() -> None:
    """Text parts produce text_delta dicts with the correct keys."""
    client = _make_client()
    chunks = [_make_text_chunk("Hello, "), _make_text_chunk("world!"), _make_finish_chunk()]
    deltas = await _collect_deltas(client, chunks)

    text_deltas = [d for d in deltas if d["type"] == "text_delta"]
    assert len(text_deltas) == 2
    assert text_deltas[0] == {"type": "text_delta", "content": "Hello, "}
    assert text_deltas[1] == {"type": "text_delta", "content": "world!"}


@pytest.mark.asyncio
async def test_finish_reason_delta() -> None:
    """A finish chunk yields a finish_reason delta."""
    client = _make_client()
    chunks = [_make_text_chunk("Hi"), _make_finish_chunk("stop")]
    deltas = await _collect_deltas(client, chunks)

    finish_deltas = [d for d in deltas if d["type"] == "finish_reason"]
    assert len(finish_deltas) == 1
    assert finish_deltas[0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_tool_call_delta_shape() -> None:
    """Tool-call parts produce tool_call_delta dicts with the expected keys."""
    client = _make_client()
    chunks = [
        _make_tool_chunk("greet", {"intensity": "high"}, index=0),
        _make_finish_chunk("tool_calls"),
    ]
    deltas = await _collect_deltas(client, chunks)

    tc_deltas = [d for d in deltas if d["type"] == "tool_call_delta"]
    assert len(tc_deltas) == 1
    tc = tc_deltas[0]
    assert tc["name"] == "greet"
    assert tc["index"] == 0
    assert "id" in tc
    # arguments should be valid JSON encoding the args dict
    assert json.loads(tc["arguments"]) == {"intensity": "high"}


@pytest.mark.asyncio
async def test_multiple_tool_calls_get_separate_indices() -> None:
    """Multiple tool calls in the same stream each get a distinct index.

    Regression guard: tool-call accumulation must not merge separate calls
    into the same index slot (mirrors the PR #180 regression guard pattern).
    """
    client = _make_client()
    chunks = [
        _make_tool_chunk("greet", {"name": "Bob"}, index=0),
        _make_tool_chunk("dance", {"style": "jive"}, index=0),  # Gemini gives each its own chunk
        _make_finish_chunk("tool_calls"),
    ]
    deltas = await _collect_deltas(client, chunks)

    tc_deltas = [d for d in deltas if d["type"] == "tool_call_delta"]
    assert len(tc_deltas) == 2
    # Indices must be distinct because each chunk is a separate tool call
    indices = [d["index"] for d in tc_deltas]
    assert indices[0] != indices[1], "Each tool call chunk must receive a distinct index"


@pytest.mark.asyncio
async def test_null_content_chunks_are_skipped() -> None:
    """Chunks whose text is None must not produce text_delta deltas."""
    client = _make_client()
    chunks = [
        _make_null_chunk(),
        _make_text_chunk("real content"),
        _make_null_chunk(),
        _make_finish_chunk(),
    ]
    deltas = await _collect_deltas(client, chunks)

    text_deltas = [d for d in deltas if d["type"] == "text_delta"]
    assert len(text_deltas) == 1
    assert text_deltas[0]["content"] == "real content"


@pytest.mark.asyncio
async def test_empty_string_text_not_yielded() -> None:
    """Empty-string text parts must not produce a text_delta."""
    client = _make_client()

    empty_part = MagicMock()
    empty_part.text = ""
    empty_part.function_call = None
    content = MagicMock()
    content.parts = [empty_part]
    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = None
    chunk = MagicMock()
    chunk.candidates = [candidate]

    chunks = [chunk, _make_finish_chunk()]
    deltas = await _collect_deltas(client, chunks)

    text_deltas = [d for d in deltas if d["type"] == "text_delta"]
    assert text_deltas == []


# ---------------------------------------------------------------------------
# Tests — call_completion accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_completion_accumulates_text() -> None:
    """call_completion joins text delta fragments into a single string."""
    client = _make_client()
    chunks = [
        _make_text_chunk("You "),
        _make_text_chunk("ugly "),
        _make_text_chunk("mug."),
        _make_finish_chunk(),
    ]

    async def _fake_stream(*_args: Any, **_kwargs: Any):  # type: ignore[no-untyped-def]
        for c in chunks:
            yield c

    with patch.object(client._client.aio.models, "generate_content_stream", return_value=_fake_stream()):
        text, tool_calls, raw_msg = await client.call_completion([], [], "sys")

    assert text == "You ugly mug."
    assert tool_calls == []
    assert raw_msg["role"] == "assistant"
    assert raw_msg["content"] == "You ugly mug."


@pytest.mark.asyncio
async def test_call_completion_with_tool_call() -> None:
    """call_completion returns parsed tool_calls list."""
    client = _make_client()
    chunks = [
        _make_tool_chunk("play_emotion", {"name": "annoyed"}, index=0),
        _make_finish_chunk("tool_calls"),
    ]

    async def _fake_stream(*_args: Any, **_kwargs: Any):  # type: ignore[no-untyped-def]
        for c in chunks:
            yield c

    with patch.object(client._client.aio.models, "generate_content_stream", return_value=_fake_stream()):
        text, tool_calls, raw_msg = await client.call_completion([], [], "sys")

    assert text == ""
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["function"]["name"] == "play_emotion"
    assert isinstance(tc["function"]["arguments"], dict)
    assert tc["function"]["arguments"] == {"name": "annoyed"}
    assert "tool_calls" in raw_msg


# ---------------------------------------------------------------------------
# Tests — 429 retry / backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_429_triggers_backoff_and_retry() -> None:
    """A 429 error on the first attempt causes a sleep and a retry."""
    from google.genai.errors import ClientError

    response_json = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "1s",
                }
            ],
        }
    }
    rate_limit_exc = ClientError(429, response_json, None)

    client = _make_client()
    call_count = 0

    async def _fake_stream_raises(*_args: Any, **_kwargs: Any):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise rate_limit_exc
        # Second attempt succeeds
        yield _make_text_chunk("retry worked")
        yield _make_finish_chunk()

    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    with (
        patch.object(client._client.aio.models, "generate_content_stream", side_effect=_fake_stream_raises),
        patch("robot_comic.gemini_llm.asyncio.sleep", side_effect=_fake_sleep),
    ):
        deltas: list[dict[str, Any]] = []
        async for d in client.stream_completion([], [], "sys"):
            deltas.append(d)

    assert call_count == 2, "Should have retried once after 429"
    assert len(sleep_calls) == 1, "Should have slept exactly once"
    assert sleep_calls[0] >= 1.0, "Sleep should honour retry-after >= 1s"
    text_deltas = [d for d in deltas if d["type"] == "text_delta"]
    assert any("retry worked" in d["content"] for d in text_deltas)


@pytest.mark.asyncio
async def test_exhausted_retries_raise() -> None:
    """After all retries are exhausted the exception propagates to the caller."""
    from google.genai.errors import ClientError

    response_json = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota", "details": []}}
    exc = ClientError(429, response_json, None)
    client = _make_client()

    async def _always_raises(*_args: Any, **_kwargs: Any):  # type: ignore[no-untyped-def]
        raise exc
        yield  # make it an async generator

    with (
        patch.object(client._client.aio.models, "generate_content_stream", side_effect=_always_raises),
        patch("robot_comic.gemini_llm.asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(ClientError):
            async for _ in client.stream_completion([], [], "sys"):
                pass
