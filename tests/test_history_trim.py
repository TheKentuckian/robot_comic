"""Tests for the bounded conversation history helper."""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from robot_comic.history_trim import (
    DEFAULT_MAX_HISTORY_TURNS,
    get_max_history_turns,
    trim_history_in_place,
)
from robot_comic.tools.core_tools import ToolDependencies


# ---------------------------------------------------------------------------
# trim_history_in_place
# ---------------------------------------------------------------------------


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _tool(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_no_trim_when_under_cap() -> None:
    history = [_user("hi"), _assistant("hello")]
    removed = trim_history_in_place(history, max_turns=20)
    assert removed == 0
    assert history == [_user("hi"), _assistant("hello")]


def test_trim_drops_oldest_user_turn_with_its_assistant_reply() -> None:
    history: list[dict] = []
    for i in range(5):
        history.append(_user(f"u{i}"))
        history.append(_assistant(f"a{i}"))
    # Cap=3: keep only u2/a2, u3/a3, u4/a4
    removed = trim_history_in_place(history, max_turns=3)
    assert removed == 4
    contents = [item["content"] for item in history]
    assert contents == ["u2", "a2", "u3", "a3", "u4", "a4"]


def test_trim_drops_user_assistant_in_pairs() -> None:
    """Trim cuts at user boundaries so assistant turns aren't orphaned."""
    history: list[dict] = []
    for i in range(10):
        history.append(_user(f"u{i}"))
        history.append(_assistant(f"a{i}"))
    trim_history_in_place(history, max_turns=4)
    # Each remaining user turn should have its assistant reply directly after.
    roles = [item["role"] for item in history]
    assert roles == ["user", "assistant"] * 4


def test_trim_keeps_tool_round_trip_attached_to_user_turn() -> None:
    """Tool calls + tool results belong to the user turn that triggered them."""
    history = [
        _user("u0"),
        _assistant("looking..."),
        _tool("call_0", '{"result": "ok"}'),
        _assistant("here it is"),
        _user("u1"),
        _assistant("a1"),
        _user("u2"),
        _assistant("a2"),
    ]
    trim_history_in_place(history, max_turns=2)
    # Should drop the u0 group (4 items) and keep u1/a1, u2/a2.
    contents = [item.get("content") for item in history]
    assert contents == ["u1", "a1", "u2", "a2"]
    assert not any(item.get("role") == "tool" for item in history)


def test_trim_preserves_system_prompt_when_present() -> None:
    """Even if a system prompt leaked into history, it survives trimming."""
    history = [{"role": "system", "content": "SYSTEM PROMPT"}]
    for i in range(5):
        history.append(_user(f"u{i}"))
        history.append(_assistant(f"a{i}"))
    trim_history_in_place(history, max_turns=2)
    assert history[0]["role"] == "system"
    assert history[0]["content"] == "SYSTEM PROMPT"
    # Plus the last two user/assistant pairs.
    tail = [item["content"] for item in history[1:]]
    assert tail == ["u3", "a3", "u4", "a4"]


def test_trim_disabled_when_cap_is_zero() -> None:
    history = [_user("u0"), _assistant("a0"), _user("u1"), _assistant("a1")]
    removed = trim_history_in_place(history, max_turns=0)
    assert removed == 0
    assert len(history) == 4


def test_trim_uses_custom_role_key() -> None:
    """Gemini uses 'role' too but with 'parts'; the role_key arg keeps it generic."""
    history = [
        {"role": "user", "parts": [{"text": "u0"}]},
        {"role": "model", "parts": [{"text": "a0"}]},
        {"role": "user", "parts": [{"text": "u1"}]},
        {"role": "model", "parts": [{"text": "a1"}]},
        {"role": "user", "parts": [{"text": "u2"}]},
        {"role": "model", "parts": [{"text": "a2"}]},
    ]
    trim_history_in_place(history, role_key="role", max_turns=2)
    texts = [item["parts"][0]["text"] for item in history]
    # Only "user" role boundaries count; model turns ride along.
    assert texts == ["u1", "a1", "u2", "a2"]


def test_get_max_history_turns_default(monkeypatch) -> None:
    monkeypatch.delenv("REACHY_MINI_MAX_HISTORY_TURNS", raising=False)
    assert get_max_history_turns() == DEFAULT_MAX_HISTORY_TURNS


def test_get_max_history_turns_invalid_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("REACHY_MINI_MAX_HISTORY_TURNS", "not-a-number")
    assert get_max_history_turns() == DEFAULT_MAX_HISTORY_TURNS


def test_get_max_history_turns_negative_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("REACHY_MINI_MAX_HISTORY_TURNS", "-3")
    assert get_max_history_turns() == DEFAULT_MAX_HISTORY_TURNS


def test_get_max_history_turns_zero_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("REACHY_MINI_MAX_HISTORY_TURNS", "0")
    assert get_max_history_turns() == 0


def test_get_max_history_turns_custom_value(monkeypatch) -> None:
    monkeypatch.setenv("REACHY_MINI_MAX_HISTORY_TURNS", "5")
    assert get_max_history_turns() == 5


# ---------------------------------------------------------------------------
# Integration: trimming runs from inside the handler's dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_tts_handler_trims_history_on_dispatch(monkeypatch) -> None:
    """After enough turns, oldest user/model pairs drop.

    The system prompt is fed separately by the handler and is never stored in
    ``_conversation_history``, so it survives regardless of the cap.
    """
    from robot_comic.gemini_tts import GeminiTTSResponseHandler

    monkeypatch.setenv("REACHY_MINI_MAX_HISTORY_TURNS", "3")

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = GeminiTTSResponseHandler(deps)
    handler._client = MagicMock()

    async def fake_llm() -> str:
        return "response"

    async def fake_tts(text: str, system_instruction: str | None = None) -> bytes:
        return b""

    handler._run_llm_with_tools = fake_llm  # type: ignore[method-assign]
    handler._call_tts_with_retry = fake_tts  # type: ignore[method-assign]

    for i in range(5):
        await handler._dispatch_completed_transcript(f"transcript {i}")

    # 3 user turns × (user + model) = 6 entries
    user_turns = [m for m in handler._conversation_history if m.get("role") == "user"]
    assert len(user_turns) == 3
    texts = [m["parts"][0]["text"] for m in user_turns]
    assert texts == ["transcript 2", "transcript 3", "transcript 4"]
    # The two oldest user/model pairs are gone
    flat = [m["parts"][0]["text"] for m in handler._conversation_history]
    assert "transcript 0" not in flat
    assert "transcript 1" not in flat


@pytest.mark.asyncio
async def test_llama_handler_trims_history_before_request(monkeypatch) -> None:
    """Trim happens before the LLM builds its payload (issue #92 bullet 3).

    The primary LLM pass now goes through _stream_response_and_synthesize, so we
    mock that method directly to observe the history length at call time.
    """
    from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler

    monkeypatch.setenv("REACHY_MINI_MAX_HISTORY_TURNS", "2")

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTChatterboxHandler(deps)

    # Seed older turns so the new user message will push us past the cap.
    for i in range(4):
        handler._conversation_history.append(_user(f"old{i}"))
        handler._conversation_history.append(_assistant(f"oldA{i}"))

    observed_history_lengths: list[int] = []

    # The primary LLM pass uses _stream_response_and_synthesize (not _call_llm)
    async def fake_stream_response(extra_messages=None, tts_span=None):
        observed_history_lengths.append(len(handler._conversation_history))
        return "ok", [], {"role": "assistant", "content": "ok"}

    async def fake_tts(text: str, *, exaggeration=None, cfg_weight=None) -> bytes:
        import numpy as np

        return np.zeros(2400, dtype=np.int16).tobytes()

    handler._stream_response_and_synthesize = fake_stream_response  # type: ignore[method-assign]
    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]

    await handler._dispatch_completed_transcript("new")

    # The LLM saw a trimmed history: cap=2 user turns. At call time, the new
    # user message is in but the assistant reply is not, so we expect the
    # most-recent two user turns + the one assistant reply attached = 3 items.
    # The pre-trim history would have been 9 (4 old user/asst pairs + new user).
    assert observed_history_lengths == [3]

    # And the surviving turns are the most-recent user.
    user_msgs = [m for m in handler._conversation_history if m.get("role") == "user"]
    assert [m["content"] for m in user_msgs] == ["old3", "new"]
