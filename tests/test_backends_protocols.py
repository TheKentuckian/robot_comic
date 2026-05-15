"""Tests for the STT/LLM/TTS Protocols (Phase 1 of pipeline refactor).

These Protocols are the contract Phase 2's ``ConversationHandler`` will
consume via dependency injection. This file validates:

1. The Protocols are well-formed (importable, ``runtime_checkable``).
2. A reference in-memory implementation of each Protocol satisfies an
   ``isinstance(obj, Protocol)`` check — proving the surface matches what
   ``runtime_checkable`` expects.
3. The data classes (LLMResponse, ToolCall, AudioFrame) round-trip cleanly.

Phase 1 deliberately does NOT validate the production handler classes
(``ElevenLabsTTSResponseHandler`` etc.) against the Protocols — those
handlers will get adapter shims in Phase 2 and the Protocol coverage
shifts there.
"""

from __future__ import annotations
from typing import Any, AsyncIterator

import pytest

from robot_comic.backends import (
    ToolCall,
    AudioFrame,
    LLMBackend,
    STTBackend,
    TTSBackend,
    LLMResponse,
)


# ---------------------------------------------------------------------------
# Reference in-memory implementations
# ---------------------------------------------------------------------------


class _MockSTT:
    """Minimum STT implementation: stores callback, echoes the last fed transcript."""

    def __init__(self) -> None:
        self.on_completed = None
        self.audio_frames: list[AudioFrame] = []
        self.stopped = False

    async def start(self, on_completed) -> None:
        self.on_completed = on_completed

    async def feed_audio(self, frame: AudioFrame) -> None:
        self.audio_frames.append(frame)

    async def stop(self) -> None:
        self.stopped = True


class _MockLLM:
    """Minimum LLM implementation: echoes back the last user message."""

    def __init__(self) -> None:
        self.prepared = False
        self.shutdown_called = False
        self.last_messages: list[dict[str, Any]] | None = None

    async def prepare(self) -> None:
        self.prepared = True

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.last_messages = messages
        user_text = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return LLMResponse(text=f"echo: {user_text}")

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _MockTTS:
    """Minimum TTS implementation: yields one silent frame per call."""

    def __init__(self) -> None:
        self.prepared = False
        self.shutdown_called = False
        self.last_text: str | None = None
        self.last_tags: tuple[str, ...] | None = None

    async def prepare(self) -> None:
        self.prepared = True

    async def synthesize(self, text: str, tags: tuple[str, ...] = ()) -> AsyncIterator[AudioFrame]:
        self.last_text = text
        self.last_tags = tags
        yield AudioFrame(samples=[0] * 480, sample_rate=24000)

    async def shutdown(self) -> None:
        self.shutdown_called = True


# ---------------------------------------------------------------------------
# isinstance() against runtime_checkable Protocols
# ---------------------------------------------------------------------------


def test_mock_stt_satisfies_protocol() -> None:
    assert isinstance(_MockSTT(), STTBackend)


def test_mock_llm_satisfies_protocol() -> None:
    assert isinstance(_MockLLM(), LLMBackend)


def test_mock_tts_satisfies_protocol() -> None:
    assert isinstance(_MockTTS(), TTSBackend)


def test_unrelated_class_does_not_satisfy_stt_protocol() -> None:
    class _NotSTT:
        async def start(self, on_completed) -> None: ...

        # Missing feed_audio + stop

    # runtime_checkable Protocols only check method NAMES, so a class missing
    # a required method should fail the isinstance check.
    assert not isinstance(_NotSTT(), STTBackend)


def test_unrelated_class_does_not_satisfy_llm_protocol() -> None:
    class _NotLLM:
        async def prepare(self) -> None: ...

    assert not isinstance(_NotLLM(), LLMBackend)


def test_unrelated_class_does_not_satisfy_tts_protocol() -> None:
    class _NotTTS:
        async def prepare(self) -> None: ...

    assert not isinstance(_NotTTS(), TTSBackend)


# ---------------------------------------------------------------------------
# prepare() idempotency — docstring claims "Safe to call multiple times"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_prepare_is_idempotent() -> None:
    llm = _MockLLM()
    await llm.prepare()
    await llm.prepare()
    assert llm.prepared is True


@pytest.mark.asyncio
async def test_tts_prepare_is_idempotent() -> None:
    tts = _MockTTS()
    await tts.prepare()
    await tts.prepare()
    assert tts.prepared is True


# ---------------------------------------------------------------------------
# Behavioural smoke — the Protocols compose into a working pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_pipeline_round_trips_a_turn() -> None:
    """The three reference mocks compose into a turn: STT → LLM → TTS."""
    stt = _MockSTT()
    llm = _MockLLM()
    tts = _MockTTS()

    await llm.prepare()
    await tts.prepare()
    captured: list[str] = []

    async def _on_transcript(transcript: str) -> None:
        captured.append(transcript)

    await stt.start(_on_transcript)
    assert stt.on_completed is _on_transcript

    # Simulate STT emitting a transcript.
    await _on_transcript("hello robot")
    assert captured == ["hello robot"]

    # Orchestrator round-trips through the LLM.
    response = await llm.chat([{"role": "user", "content": "hello robot"}])
    assert response.text == "echo: hello robot"
    assert response.tool_calls == ()

    # Then the TTS produces frames.
    frames = [frame async for frame in tts.synthesize(response.text, tags=("slow",))]
    assert len(frames) == 1
    assert frames[0].sample_rate == 24000
    assert tts.last_tags == ("slow",)

    await stt.stop()
    await llm.shutdown()
    await tts.shutdown()
    assert stt.stopped and llm.shutdown_called and tts.shutdown_called


# ---------------------------------------------------------------------------
# Data class shapes
# ---------------------------------------------------------------------------


def test_tool_call_is_frozen_dataclass() -> None:
    tc = ToolCall(id="t-1", name="dance", args={"name": "happy"})
    assert tc.id == "t-1"
    assert tc.name == "dance"
    assert tc.args == {"name": "happy"}
    with pytest.raises(AttributeError):
        tc.id = "t-2"  # type: ignore[misc]


def test_llm_response_defaults_to_empty() -> None:
    r = LLMResponse()
    assert r.text == ""
    assert r.tool_calls == ()
    assert r.metadata == {}


def test_llm_response_with_tool_calls() -> None:
    calls = (ToolCall(id="t-1", name="x", args={}),)
    r = LLMResponse(tool_calls=calls)
    assert r.tool_calls == calls
    assert r.text == ""


def test_audio_frame_carries_samples_and_rate() -> None:
    af = AudioFrame(samples=[1, 2, 3], sample_rate=16000)
    assert af.samples == [1, 2, 3]
    assert af.sample_rate == 16000
