"""Tests for ComposablePipeline (Phase 2 of pipeline refactor)."""

from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator

import pytest

from robot_comic.backends import ToolCall, AudioFrame, LLMResponse
from robot_comic.composable_pipeline import ComposablePipeline


# ---------------------------------------------------------------------------
# Programmable mock backends
# ---------------------------------------------------------------------------


class _ProgrammableSTT:
    """STT mock that holds the bound callback so tests can fire transcripts."""

    def __init__(self) -> None:
        self.on_completed = None
        self.audio_frames: list[AudioFrame] = []
        self.stopped = False
        self.started = False

    async def start(self, on_completed) -> None:
        self.started = True
        self.on_completed = on_completed

    async def feed_audio(self, frame: AudioFrame) -> None:
        self.audio_frames.append(frame)

    async def stop(self) -> None:
        self.stopped = True


class _ScriptedLLM:
    """LLM mock that returns a sequence of pre-built LLMResponses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.prepared = False
        self.shutdown_called = False

    async def prepare(self) -> None:
        self.prepared = True

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        # Snapshot the messages as the orchestrator saw them this round.
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise RuntimeError("scripted LLM exhausted")
        return self._responses.pop(0)

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _RecordingTTS:
    """TTS mock that yields one frame and records the text/tags it saw."""

    def __init__(self) -> None:
        self.prepared = False
        self.shutdown_called = False
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def prepare(self) -> None:
        self.prepared = True

    async def synthesize(self, text: str, tags: tuple[str, ...] = ()) -> AsyncIterator[AudioFrame]:
        self.calls.append((text, tags))
        yield AudioFrame(samples=[0, 0, 0], sample_rate=24000)

    async def shutdown(self) -> None:
        self.shutdown_called = True


async def _wait_for_callback(stt: _ProgrammableSTT) -> None:
    """Yield the loop until the pipeline binds the STT callback."""
    for _ in range(20):
        await asyncio.sleep(0)
        if stt.on_completed is not None:
            return
    raise AssertionError("STT.on_completed was never bound")


# ---------------------------------------------------------------------------
# Single-turn happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_text_turn_produces_one_frame() -> None:
    """One transcript → one LLM call → one TTS synth → one frame on the output queue."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hello there!")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)
    assert stt.started and llm.prepared and tts.prepared

    await stt.on_completed("hi robot")

    frame = await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    assert frame.sample_rate == 24000
    assert tts.calls == [("Hello there!", ())]
    assert pipeline.conversation_history == [
        {"role": "user", "content": "hi robot"},
        {"role": "assistant", "content": "Hello there!"},
    ]

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Tool-call loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_calls_dispatched_then_followup_text_spoken() -> None:
    """LLM requests a tool, orchestrator dispatches it, follow-up text is spoken."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(tool_calls=(ToolCall(id="t-1", name="dance", args={"name": "happy"}),)),
            LLMResponse(text="Done dancing!"),
        ]
    )
    tts = _RecordingTTS()

    dispatched: list[ToolCall] = []

    async def _dispatch(call: ToolCall) -> str:
        dispatched.append(call)
        return f"ok:{call.name}"

    pipeline = ComposablePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        tool_dispatcher=_dispatch,
        tools_spec=[{"name": "dance"}],
    )

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("dance for me")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(dispatched) == 1
    assert dispatched[0].name == "dance"
    assert dispatched[0].args == {"name": "happy"}
    assert len(llm.calls) == 2

    assert pipeline.conversation_history[0] == {"role": "user", "content": "dance for me"}
    assert pipeline.conversation_history[1]["role"] == "tool"
    assert pipeline.conversation_history[1]["tool_call_id"] == "t-1"
    assert pipeline.conversation_history[1]["content"] == "ok:dance"
    assert pipeline.conversation_history[2] == {"role": "assistant", "content": "Done dancing!"}

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_tool_call_without_dispatcher_short_circuits() -> None:
    """When no tool_dispatcher is configured, tool_calls are ignored gracefully."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(tool_calls=(ToolCall(id="t-1", name="x", args={}),))])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("do a thing")
    for _ in range(5):
        await asyncio.sleep(0)

    assert tts.calls == []
    assert pipeline.output_queue.empty()

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_tool_loop_respects_max_rounds() -> None:
    """An LLM that endlessly requests tools is bounded by max_tool_rounds."""
    stt = _ProgrammableSTT()
    looper = LLMResponse(tool_calls=(ToolCall(id="t-1", name="loop", args={}),))
    llm = _ScriptedLLM([looper] * 50)
    tts = _RecordingTTS()

    async def _dispatch(call: ToolCall) -> str:
        return "ok"

    pipeline = ComposablePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        tool_dispatcher=_dispatch,
        max_tool_rounds=3,
    )

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("loop please")
    for _ in range(10):
        await asyncio.sleep(0)

    assert len(llm.calls) == 3
    assert tts.calls == []

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_releases_start_up_and_closes_backends() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await pipeline.shutdown()
    await asyncio.wait_for(task, timeout=1.0)
    assert stt.stopped and llm.shutdown_called and tts.shutdown_called


@pytest.mark.asyncio
async def test_shutdown_before_start_up_is_safe() -> None:
    """Calling shutdown() before start_up() must not deadlock or crash."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    await pipeline.shutdown()
    assert not stt.stopped
    # Subsequent start_up returns immediately because stop_event is set,
    # AND it must not have prepared any backends — those would never get
    # torn down (shutdown's "not started" early-return already ran).
    await asyncio.wait_for(pipeline.start_up(), timeout=1.0)
    assert not stt.started
    assert not llm.prepared
    assert not tts.prepared


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_seeded_into_history() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="hi")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="You are Bill Hicks.")

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hello")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert pipeline.conversation_history[0] == {"role": "system", "content": "You are Bill Hicks."}
    assert llm.calls[0][0] == {"role": "system", "content": "You are Bill Hicks."}
    assert llm.calls[0][1] == {"role": "user", "content": "hello"}

    await pipeline.shutdown()
    await task


def test_reset_history_keeps_system_prompt_by_default() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="persona")
    pipeline.conversation_history.append({"role": "user", "content": "x"})
    pipeline.conversation_history.append({"role": "assistant", "content": "y"})
    pipeline.reset_history()
    assert pipeline.conversation_history == [{"role": "system", "content": "persona"}]


def test_reset_history_can_drop_system_prompt() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="persona")
    pipeline.conversation_history.append({"role": "user", "content": "x"})
    pipeline.reset_history(keep_system=False)
    assert pipeline.conversation_history == []


# ---------------------------------------------------------------------------
# Audio input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_audio_delegates_to_stt() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    frame = AudioFrame(samples=[1, 2], sample_rate=16000)
    await pipeline.feed_audio(frame)
    assert stt.audio_frames == [frame]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_transcript_is_ignored() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])  # would raise if called
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("")
    for _ in range(5):
        await asyncio.sleep(0)

    assert llm.calls == []
    assert pipeline.conversation_history == []

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_empty_assistant_text_does_not_speak() -> None:
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="   ")])  # whitespace-only
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi")
    for _ in range(5):
        await asyncio.sleep(0)

    assert tts.calls == []
    assert pipeline.output_queue.empty()

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_tool_dispatch_exception_records_error_in_history() -> None:
    """A tool that raises gets a stringified-error result back to the LLM."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(tool_calls=(ToolCall(id="t-1", name="bad", args={}),)),
            LLMResponse(text="I see the tool failed; carrying on."),
        ]
    )
    tts = _RecordingTTS()

    async def _dispatch(call: ToolCall) -> str:
        raise ValueError("oops")

    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, tool_dispatcher=_dispatch)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("try the bad tool")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    tool_msg = pipeline.conversation_history[1]
    assert tool_msg["role"] == "tool"
    assert "tool error" in tool_msg["content"]
    assert "ValueError" in tool_msg["content"]

    await pipeline.shutdown()
    await task
