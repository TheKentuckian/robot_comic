"""Unit tests for _stream_llm_deltas and _call_llm streaming edge cases.

Covers the unit-testable subset of issue #115:
  A — Multi-chunk SSE tool_call delta accumulation (name-in-one-chunk, null skipping)
  B — Auto-generated tool_call id when missing; verbatim id preservation
  C — Error-path: mid-stream HTTP drop and malformed/missing-keys JSON chunks

Real bugs uncovered (do not fix here — production code is out of scope):
  BUG-1: _call_llm leaves ``id`` as "" when the SSE stream omits the ``id`` field;
         auto-generation only happens in ``_start_tool_calls``, not in the
         accumulator. Tests marked xfail(strict=True) guard these regressions.
  BUG-2: ``KeyError`` for missing ``choices`` or ``delta`` keys inside a valid-JSON
         SSE chunk is not caught (only ``json.JSONDecodeError`` is caught). The
         unhandled KeyError bubbles out of the SSE loop, trips the retry logic,
         and eventually re-raises after exhausting retries. Tests marked
         xfail(strict=True) document the current (broken) behaviour so that
         fixing the production code will make them pass.
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest
from _helpers import make_stream_response, make_stream_response_with_error


# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------


def _make_handler():
    from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler
    from robot_comic.tools.core_tools import ToolDependencies

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTChatterboxHandler(deps)
    handler._http = MagicMock()
    return handler


# ===========================================================================
# A — Multi-chunk tool_call delta accumulation
# ===========================================================================


@pytest.mark.asyncio
async def test_tool_call_name_appears_in_first_chunk_only() -> None:
    """Function name arrives in the first chunk; argument fragments arrive later.

    Regression guard for PR #153: the name must be captured even when it is
    absent from all subsequent chunks.
    """
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                # First chunk: name + id, no arguments yet
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"dance","arguments":""}}]},"finish_reason":null}]}',
                # Second chunk: arguments fragment, name absent
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"style\\":"}}]},"finish_reason":null}]}',
                # Third chunk: closing fragment, name still absent
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"wave\\"}"}}]},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )
    )

    deltas = []
    async for delta in handler._stream_llm_deltas():
        deltas.append(delta)

    tool_deltas = [d for d in deltas if d["type"] == "tool_call_delta"]
    # Name must be captured in the first tool_call_delta
    name_deltas = [d for d in tool_deltas if d.get("name")]
    assert len(name_deltas) >= 1, "Expected at least one delta carrying the function name"
    assert name_deltas[0]["name"] == "dance"

    # All argument fragments must be present
    all_args = "".join(d["arguments"] for d in tool_deltas if d.get("arguments") is not None)
    assert "wave" in all_args


@pytest.mark.asyncio
async def test_tool_call_full_accumulation_via_call_llm() -> None:
    """_call_llm assembles name + accumulated arguments correctly.

    Name is only in the first SSE chunk (end-to-end through the accumulator loop).
    """
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"play_emotion","arguments":""}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"emotion\\":"}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"happy1\\"}"}}]},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )
    )

    _text, tool_calls, _raw = await handler._call_llm()

    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["function"]["name"] == "play_emotion"
    assert tc["function"]["arguments"] == {"emotion": "happy1"}
    assert tc["id"] == "call_xyz"


@pytest.mark.asyncio
async def test_null_content_delta_is_skipped() -> None:
    """Chunks where delta.content is null must not yield a text_delta.

    Regression guard for PR #151: OpenAI-compatible servers set content=null
    when a chunk carries only tool_calls.
    """
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                # content=null — must be skipped
                'data: {"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"dance","arguments":"{}"}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":"OK, dancing!"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
    )

    deltas = []
    async for delta in handler._stream_llm_deltas():
        deltas.append(delta)

    text_deltas = [d for d in deltas if d["type"] == "text_delta"]
    # Only the non-null content chunk should produce a text_delta
    assert len(text_deltas) == 1
    assert text_deltas[0]["content"] == "OK, dancing!"


@pytest.mark.asyncio
async def test_null_arguments_fragment_is_skipped() -> None:
    """A tool_call delta with arguments=null must not be emitted.

    PR #151 fix: null argument fragments in the SSE stream are skipped so
    downstream string concatenation stays safe.
    """
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                # First chunk: name + null arguments
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","type":"function","function":{"name":"head_track","arguments":null}}]},"finish_reason":null}]}',
                # Second chunk: actual arguments
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )
    )

    deltas = []
    async for delta in handler._stream_llm_deltas():
        deltas.append(delta)

    tool_deltas = [d for d in deltas if d["type"] == "tool_call_delta"]
    # The null-arguments delta should carry arguments=None (not the string "null")
    null_arg_deltas = [d for d in tool_deltas if d.get("arguments") == "null"]
    assert len(null_arg_deltas) == 0, "null string must not appear in arguments fragments"


# ===========================================================================
# B — Auto-generated tool_call id when missing / id preserved when present
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG-1: _call_llm leaves id='' when SSE stream omits the id field. "
        "Auto-generation only happens in _start_tool_calls, not in the accumulator. "
        "File a bug and fix in production code; remove xfail then."
    ),
)
async def test_tool_call_id_auto_generated_when_absent() -> None:
    """When the SSE stream omits the ``id`` field, _call_llm must auto-generate one.

    Currently xfail: see BUG-1 in module docstring.
    """
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                # No "id" field at all
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"dance","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )
    )

    _text, tool_calls, _raw = await handler._call_llm()

    assert len(tool_calls) == 1
    tc_id = tool_calls[0].get("id")
    assert tc_id, "id must be non-empty when auto-generated"
    assert len(tc_id) == 8
    assert all(c in "0123456789abcdef" for c in tc_id)


@pytest.mark.asyncio
async def test_tool_call_id_preserved_verbatim_when_present() -> None:
    """When the SSE stream includes an ``id``, it must be passed through unchanged."""
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_MY_EXACT_ID","type":"function","function":{"name":"dance","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )
    )

    _text, tool_calls, _raw = await handler._call_llm()

    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_MY_EXACT_ID"


@pytest.mark.asyncio
async def test_tool_call_id_from_delta_stream_level() -> None:
    """_stream_llm_deltas carries the id field in the yielded dict when present."""
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_VERBATIM","type":"function","function":{"name":"camera","arguments":""}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )
    )

    tool_deltas = []
    async for delta in handler._stream_llm_deltas():
        if delta["type"] == "tool_call_delta":
            tool_deltas.append(delta)

    id_bearing = [d for d in tool_deltas if d.get("id") is not None]
    assert len(id_bearing) >= 1
    assert id_bearing[0]["id"] == "call_VERBATIM"


@pytest.mark.asyncio
async def test_tool_call_id_absent_yields_none_not_missing() -> None:
    """When id is absent from the chunk the delta has id=None (not KeyError)."""
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                # No id field
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"dance","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )
    )

    tool_deltas = []
    async for delta in handler._stream_llm_deltas():
        if delta["type"] == "tool_call_delta":
            tool_deltas.append(delta)

    assert len(tool_deltas) >= 1
    # id key must be present in the dict, value must be None (not KeyError)
    assert "id" in tool_deltas[0]
    assert tool_deltas[0]["id"] is None


# ===========================================================================
# C — Error-path tests
# ===========================================================================


@pytest.mark.asyncio
async def test_midstream_http_error_raises_after_retries() -> None:
    """Mid-stream HTTP drop: after exhausting retries the exception propagates.

    Each retry attempt gets a fresh stream mock (via side_effect) so that
    the iteration state is not shared across retries.  asyncio.sleep is
    patched to keep the test fast.
    """
    from unittest.mock import AsyncMock as _AsyncMock
    from unittest.mock import patch

    import httpx

    handler = _make_handler()

    def _fresh_error_response():
        return make_stream_response_with_error(
            [
                'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
                "data: NEVER_REACHED",  # raise fires before this line is yielded
            ],
            raise_at_index=1,
            exc=httpx.ReadError("connection dropped"),
        )

    # side_effect is called once per handler._http.stream(...) call
    handler._http.stream = MagicMock(side_effect=lambda *_a, **_kw: _fresh_error_response())

    with patch("robot_comic.llama_base.asyncio.sleep", new_callable=_AsyncMock):
        with pytest.raises(httpx.ReadError):
            async for _delta in handler._stream_llm_deltas():
                pass


@pytest.mark.asyncio
async def test_midstream_http_error_logs_warning() -> None:
    """Mid-stream HTTP drop: intermediate retry attempts are logged at WARNING level."""
    from unittest.mock import AsyncMock as _AsyncMock
    from unittest.mock import patch

    import httpx

    handler = _make_handler()

    def _fresh_error_response():
        return make_stream_response_with_error(
            [
                'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
                "data: NEVER_REACHED",  # raise fires before this line is yielded
            ],
            raise_at_index=1,
            exc=httpx.ReadError("connection dropped"),
        )

    handler._http.stream = MagicMock(side_effect=lambda *_a, **_kw: _fresh_error_response())

    import logging as _logging

    with patch("robot_comic.llama_base.asyncio.sleep", new_callable=_AsyncMock):
        import io

        log_stream = io.StringIO()
        log_handler = _logging.StreamHandler(log_stream)
        log_handler.setLevel(_logging.WARNING)
        target_logger = _logging.getLogger("robot_comic.llama_base")
        target_logger.addHandler(log_handler)
        try:
            with pytest.raises(httpx.ReadError):
                async for _delta in handler._stream_llm_deltas():
                    pass
        finally:
            target_logger.removeHandler(log_handler)

    log_output = log_stream.getvalue()
    assert "retrying" in log_output.lower() or "failed" in log_output.lower(), (
        f"Expected a retry warning in logs; got: {log_output!r}"
    )


@pytest.mark.asyncio
async def test_midstream_partial_text_before_drop() -> None:
    """Deltas received before the mid-stream drop are yielded before the error.

    _LLM_MAX_RETRIES is patched to 1 so we can inspect yielded values vs
    the raised exception without waiting for multiple retry cycles.
    """
    from unittest.mock import AsyncMock as _AsyncMock
    from unittest.mock import patch

    import httpx

    handler = _make_handler()

    def _fresh_error_response():
        return make_stream_response_with_error(
            [
                'data: {"choices":[{"delta":{"content":"Partial"},"finish_reason":null}]}',
                "data: NEVER_REACHED",  # raise fires here (index 1) before this is yielded
            ],
            raise_at_index=1,
            exc=httpx.ReadError("dropped"),
        )

    handler._http.stream = MagicMock(side_effect=lambda *_a, **_kw: _fresh_error_response())

    collected: list[dict] = []

    with patch("robot_comic.llama_base._LLM_MAX_RETRIES", 1):
        with patch("robot_comic.llama_base.asyncio.sleep", new_callable=_AsyncMock):
            with pytest.raises(httpx.ReadError):
                async for delta in handler._stream_llm_deltas():
                    collected.append(delta)

    # "Partial" text chunk should have been yielded before the error
    text_deltas = [d for d in collected if d["type"] == "text_delta"]
    assert len(text_deltas) == 1
    assert text_deltas[0]["content"] == "Partial"


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG-2: KeyError for missing 'choices' key in a valid-JSON SSE chunk is not caught. "
        "Only json.JSONDecodeError is caught; KeyError bubbles up and trips the retry logic. "
        "Fix: add KeyError to the except clause in _stream_llm_deltas."
    ),
)
async def test_chunk_missing_choices_key_is_skipped() -> None:
    """A valid-JSON SSE chunk missing the 'choices' key should be skipped silently.

    Currently xfail: see BUG-2 in module docstring.
    """
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                'data: {"choices":[{"delta":{"content":"Good"},"finish_reason":null}]}',
                # Valid JSON but missing "choices" key entirely
                'data: {"model":"llama","usage":{}}',
                'data: {"choices":[{"delta":{"content":"Bye"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
    )

    deltas = []
    async for delta in handler._stream_llm_deltas():
        deltas.append(delta)

    text_deltas = [d for d in deltas if d["type"] == "text_delta"]
    contents = [d["content"] for d in text_deltas]
    assert "Good" in contents
    assert "Bye" in contents


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG-2: KeyError for missing 'delta' sub-key is not caught inside the SSE try/except. "
        "The unhandled KeyError bubbles out, triggers retry logic, and eventually re-raises. "
        "Fix: broaden the except clause in _stream_llm_deltas to include KeyError."
    ),
)
async def test_chunk_missing_delta_key_is_skipped() -> None:
    """A chunk with 'choices' but no 'delta' sub-key should be skipped gracefully.

    Currently xfail: see BUG-2 in module docstring.
    """
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                'data: {"choices":[{"delta":{"content":"First"},"finish_reason":null}]}',
                # choices[0] lacks the "delta" key — triggers KeyError
                'data: {"choices":[{"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":"Last"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
    )

    deltas = []
    async for delta in handler._stream_llm_deltas():
        deltas.append(delta)

    text_deltas = [d for d in deltas if d["type"] == "text_delta"]
    contents = [d["content"] for d in text_deltas]
    assert "First" in contents
    assert "Last" in contents


@pytest.mark.asyncio
async def test_non_data_sse_lines_are_ignored() -> None:
    """SSE event/comment/empty lines that don't start with 'data: ' are ignored."""
    handler = _make_handler()

    handler._http.stream = MagicMock(
        return_value=make_stream_response(
            [
                "event: start",
                ": this is a comment",
                "",
                'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
    )

    deltas = []
    async for delta in handler._stream_llm_deltas():
        deltas.append(delta)

    text_deltas = [d for d in deltas if d["type"] == "text_delta"]
    assert len(text_deltas) == 1
    assert text_deltas[0]["content"] == "Hi"


# ---------------------------------------------------------------------------
# make_stream_response_with_error helper self-test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_helper_raises_at_correct_index() -> None:
    """make_stream_response_with_error raises after yielding lines before the index."""
    import httpx

    response = make_stream_response_with_error(
        ["line0", "line1", "line2"],
        raise_at_index=2,
        exc=httpx.ReadError("test"),
    )

    collected: list[str] = []
    async with response:
        with pytest.raises(httpx.ReadError):
            async for line in response.aiter_lines():
                collected.append(line)

    assert collected == ["line0", "line1"]


@pytest.mark.asyncio
async def test_helper_raises_at_index_zero() -> None:
    """make_stream_response_with_error raises immediately when raise_at_index=0."""
    import httpx

    response = make_stream_response_with_error(
        ["line0"],
        raise_at_index=0,
        exc=httpx.ReadError("immediate"),
    )

    collected: list[str] = []
    async with response:
        with pytest.raises(httpx.ReadError):
            async for line in response.aiter_lines():
                collected.append(line)  # pragma: no cover

    assert collected == []
