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
    """STT mock that holds the bound callbacks so tests can fire events."""

    def __init__(self) -> None:
        self.on_completed = None
        self.on_partial = None
        self.on_speech_started = None
        self.audio_frames: list[AudioFrame] = []
        self.stopped = False
        self.started = False

    async def start(
        self,
        on_completed,
        on_partial=None,
        on_speech_started=None,
    ) -> None:
        self.started = True
        self.on_completed = on_completed
        self.on_partial = on_partial
        self.on_speech_started = on_speech_started

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
        self.marker_refs: list[list[float] | None] = []

    async def prepare(self) -> None:
        self.prepared = True

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        self.calls.append((text, tags))
        self.marker_refs.append(first_audio_marker)
        # Populate the marker like a real adapter would so the orchestrator
        # has a stamp to read out, in case future test cases want it.
        if first_audio_marker is not None:
            import time as _time

            first_audio_marker.append(_time.monotonic())
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


# ---------------------------------------------------------------------------
# Lifecycle Hook #4 — record_joke_history (#337)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_joke_history_called_for_non_empty_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must invoke ``record_joke_history`` after a final speak round.

    Legacy parity: ``llama_base.py:578-594`` and ``gemini_tts.py:380-394``
    capture the punchline + topic after the LLM produces the final
    assistant text. The composable path bypasses both legacy sites; the
    orchestrator must call ``record_joke_history`` itself.
    """
    from robot_comic import composable_pipeline as mod

    captured: list[str] = []

    async def _recorder(text: str) -> None:
        captured.append(text)

    monkeypatch.setattr(mod, "record_joke_history", _recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="That's the joke!")])
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("tell me one")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert captured == ["That's the joke!"]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_record_joke_history_not_called_for_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only assistant text must not trigger joke-history capture.

    Mirrors the legacy ``if response_text and JOKE_HISTORY_ENABLED:`` guard
    in ``llama_base.py:579``.
    """
    from robot_comic import composable_pipeline as mod

    captured: list[str] = []

    async def _recorder(text: str) -> None:
        captured.append(text)

    monkeypatch.setattr(mod, "record_joke_history", _recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="   ")])  # whitespace-only
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi")
    for _ in range(5):
        await asyncio.sleep(0)

    assert captured == []

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_record_joke_history_not_called_on_tool_only_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-call-only LLM rounds must not record joke history.

    Legacy ``_run_turn`` captures after the assistant text is final, NOT on
    intermediate tool rounds. The orchestrator hook must respect that —
    fire exactly once per turn, on the final speak round.
    """
    from robot_comic import composable_pipeline as mod

    captured: list[str] = []

    async def _recorder(text: str) -> None:
        captured.append(text)

    monkeypatch.setattr(mod, "record_joke_history", _recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(tool_calls=(ToolCall(id="t-1", name="dance", args={"name": "happy"}),)),
            LLMResponse(text="Done dancing!"),
        ]
    )
    tts = _RecordingTTS()

    async def _dispatch(call: ToolCall) -> str:
        return f"ok:{call.name}"

    pipeline = mod.ComposablePipeline(
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

    # Exactly one capture, with the final assistant text — not the tool round.
    assert captured == ["Done dancing!"]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_record_joke_history_exception_does_not_crash_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If record_joke_history somehow raises, the TTS still runs.

    The helper itself swallows exceptions, but if a future change ever
    leaks one, the orchestrator must not let it prevent speech.
    """
    from robot_comic import composable_pipeline as mod

    async def _bad_recorder(text: str) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(mod, "record_joke_history", _bad_recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hello!")])
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi")
    # Yield several loop iterations so the turn either completes or fails.
    for _ in range(20):
        await asyncio.sleep(0)
        if not tts.calls == []:
            break

    # TTS must have run even though the hook raised.
    assert tts.calls == [("Hello!", ())]

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


# ---------------------------------------------------------------------------
# Lifecycle Hook #5 — trim_history_in_place (#337)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trim_history_called_once_per_user_turn_before_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must invoke ``trim_history_in_place`` once per user turn.

    Legacy parity: all three legacy ``_dispatch_completed_transcript``
    sites (``llama_base.py:506``, ``gemini_tts.py:365``,
    ``elevenlabs_tts.py:565``) trim once per user turn before the LLM
    loop. The composable path bypasses every one of those — the
    orchestrator must trim itself.
    """
    from robot_comic import composable_pipeline as mod

    call_log: list[str] = []

    def _trim_recorder(history: list[dict[str, Any]], **kwargs: Any) -> int:
        call_log.append("trim")
        return 0

    monkeypatch.setattr(mod, "trim_history_in_place", _trim_recorder)

    class _LoggingLLM(_ScriptedLLM):
        def __init__(self, responses: list[LLMResponse]) -> None:
            super().__init__(responses)

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
        ) -> LLMResponse:
            call_log.append("chat")
            return await super().chat(messages, tools)

    stt = _ProgrammableSTT()
    llm = _LoggingLLM([LLMResponse(text="Hello!")])
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("hi robot")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    # Exactly one trim, fired before the LLM chat() call.
    assert call_log == ["trim", "chat"]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_trim_history_uses_orchestrator_history_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trim must operate on the orchestrator's own history list, by identity.

    A passed-in copy would mean the cap doesn't shrink subsequent
    requests. Identity check guards against an accidental ``list(...)``
    or slice at the call site.
    """
    from robot_comic import composable_pipeline as mod

    seen: list[list[dict[str, Any]]] = []

    def _trim_recorder(history: list[dict[str, Any]], **kwargs: Any) -> int:
        seen.append(history)
        return 0

    monkeypatch.setattr(mod, "trim_history_in_place", _trim_recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="ok")])
    tts = _RecordingTTS()
    pipeline = mod.ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("test")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(seen) == 1
    # Identity, not equality — the trim must mutate the orchestrator's list
    # in place. After the LLM call the orchestrator appends the assistant
    # turn, so equality wouldn't hold anyway, but ``is`` proves the contract.
    assert seen[0] is pipeline._conversation_history

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_trim_history_called_once_per_turn_not_per_tool_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-call rounds inside a single user turn must NOT re-trim.

    Legacy ``_dispatch_completed_transcript`` trims once before the LLM
    loop and never re-trims inside it. The orchestrator must respect
    that cadence — fire exactly once per user turn even when several
    LLM rounds run.
    """
    from robot_comic import composable_pipeline as mod

    trim_calls: list[int] = []

    def _trim_recorder(history: list[dict[str, Any]], **kwargs: Any) -> int:
        trim_calls.append(len(history))
        return 0

    monkeypatch.setattr(mod, "trim_history_in_place", _trim_recorder)

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(tool_calls=(ToolCall(id="t-1", name="dance", args={"name": "happy"}),)),
            LLMResponse(text="Done dancing!"),
        ]
    )
    tts = _RecordingTTS()

    async def _dispatch(call: ToolCall) -> str:
        return f"ok:{call.name}"

    pipeline = mod.ComposablePipeline(
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

    # Two LLM rounds, but exactly ONE trim call.
    assert len(trim_calls) == 1
    assert len(llm.calls) == 2

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_trim_history_cap_respected_across_user_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: with ``REACHY_MINI_MAX_HISTORY_TURNS=2`` history shrinks to 2 user turns.

    Exercises the real ``trim_history_in_place`` helper (no monkeypatch
    on the trim itself) to prove the orchestrator → helper wiring
    delivers the bound legacy already provides.
    """
    monkeypatch.setenv("REACHY_MINI_MAX_HISTORY_TURNS", "2")

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM(
        [
            LLMResponse(text="reply 1"),
            LLMResponse(text="reply 2"),
            LLMResponse(text="reply 3"),
        ]
    )
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    for transcript in ("turn one", "turn two", "turn three"):
        await stt.on_completed(transcript)
        await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    user_msgs = [m for m in pipeline.conversation_history if m["role"] == "user"]
    assert len(user_msgs) == 2
    assert user_msgs[0]["content"] == "turn two"
    assert user_msgs[1]["content"] == "turn three"

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Phase 5a.2 — delivery-tag + first-audio-marker plumbing through orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speak_assistant_text_threads_delivery_tags_to_tts() -> None:
    """Phase 5a.2: ``LLMResponse.delivery_tags`` flows into
    ``TTSBackend.synthesize(tags=...)``. The orchestrator passes the tuple
    through unchanged; adapters decide whether to consume from the param or
    fall back to text parsing."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hi!", delivery_tags=("fast",))])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    await stt.on_completed("ping")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert tts.calls == [("Hi!", ("fast",))]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_speak_assistant_text_passes_empty_tags_when_response_unpopulated() -> None:
    """When the LLM response has the default empty ``delivery_tags``, the
    orchestrator still passes an empty tuple — adapters fall back to their
    text-parsing path. Pins the fallback contract."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Plain reply.")])  # delivery_tags=() default
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)
    await stt.on_completed("ping")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert tts.calls == [("Plain reply.", ())]

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_speak_assistant_text_allocates_first_audio_marker() -> None:
    """Phase 5a.2: orchestrator allocates a fresh ``list[float] = []`` per
    turn and passes it to ``TTSBackend.synthesize(first_audio_marker=...)``.
    Population is the adapter's responsibility — the orchestrator only owns
    the allocation."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="Hi.")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)
    await stt.on_completed("ping")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(tts.marker_refs) == 1
    marker = tts.marker_refs[0]
    assert marker is not None
    # The _RecordingTTS mock populates the marker (mimicking real-adapter
    # behaviour) so the orchestrator could read a stamp out after iteration.
    assert len(marker) == 1
    assert isinstance(marker[0], float)

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_speak_assistant_text_allocates_fresh_marker_per_turn() -> None:
    """Each turn gets its own marker list — no cross-turn aliasing."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([LLMResponse(text="One."), LLMResponse(text="Two.")])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)
    await stt.on_completed("first")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)
    await stt.on_completed("second")
    await asyncio.wait_for(pipeline.output_queue.get(), timeout=1.0)

    assert len(tts.marker_refs) == 2
    assert tts.marker_refs[0] is not tts.marker_refs[1]

    await pipeline.shutdown()
    await task


# ---------------------------------------------------------------------------
# Phase 5c.2 — apply_personality moved onto ComposablePipeline
# ---------------------------------------------------------------------------


class _ResettableTTS(_RecordingTTS):
    """Recording TTS that also tracks ``reset_per_session_state`` awaits."""

    def __init__(self) -> None:
        super().__init__()
        self.reset_calls: int = 0

    async def reset_per_session_state(self) -> None:
        self.reset_calls += 1


@pytest.mark.asyncio
async def test_apply_personality_resets_history_and_reseeds_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pipeline.apply_personality`` wipes history (system + user) and
    reseeds with ``get_session_instructions()``.

    Phase 5c.2 moves the wrapper's logic onto the pipeline; this test pins
    the new owner's behaviour. The wrapper-level equivalent in
    ``test_composable_conversation_handler.py`` is the post-5c.2
    "forwards to pipeline" test.
    """
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "set_custom_profile", lambda profile: None)
    monkeypatch.setattr(mod, "get_session_instructions", lambda: "fresh instructions")

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _ResettableTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="old persona")
    # Pre-seed some history that should be wiped.
    pipeline._conversation_history.append({"role": "user", "content": "hi"})

    result = await pipeline.apply_personality("rodney")

    assert "Applied personality 'rodney'" in result
    assert pipeline.conversation_history == [
        {"role": "system", "content": "fresh instructions"},
    ]


@pytest.mark.asyncio
async def test_apply_personality_awaits_tts_reset_per_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pipeline.apply_personality`` awaits ``tts.reset_per_session_state``
    exactly once on the success path. Persona switch is a hard cut on
    listening state; the TTS adapter clears its wrapped handler's
    echo-guard accumulators in this call.
    """
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "set_custom_profile", lambda profile: None)
    monkeypatch.setattr(mod, "get_session_instructions", lambda: "fresh")

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _ResettableTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    await pipeline.apply_personality("rodney")

    assert tts.reset_calls == 1, (
        f"apply_personality must await tts.reset_per_session_state exactly once; got {tts.reset_calls}"
    )


@pytest.mark.asyncio
async def test_apply_personality_returns_failure_message_on_set_custom_profile_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``set_custom_profile`` raises, the pipeline returns a failure
    string and does NOT touch history or per-session state.

    Pins the partial-failure contract: a profile-name typo does not
    half-reset the session.
    """
    from robot_comic import composable_pipeline as mod

    def _boom(profile: str | None) -> None:
        raise RuntimeError("bad profile")

    monkeypatch.setattr(mod, "set_custom_profile", _boom)
    monkeypatch.setattr(mod, "get_session_instructions", lambda: "fresh")

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _ResettableTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="old")
    pipeline._conversation_history.append({"role": "user", "content": "untouched"})
    original_history = list(pipeline._conversation_history)

    result = await pipeline.apply_personality("broken")

    assert "Failed to apply personality" in result
    assert "bad profile" in result
    # TTS reset must NOT have fired on the failure path.
    assert tts.reset_calls == 0, "tts.reset_per_session_state must NOT run when set_custom_profile fails"
    # History must be untouched on the failure path.
    assert pipeline._conversation_history == original_history, (
        "history must be untouched when set_custom_profile fails"
    )


@pytest.mark.asyncio
async def test_apply_personality_orders_history_reset_before_tts_reset_and_reseed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering: ``reset_history`` → ``tts.reset_per_session_state`` →
    history-append. Pins the ordering so a future refactor that, say,
    appends the system prompt before resetting history (which would leak
    the new prompt into the old session's tail) gets caught.
    """
    from robot_comic import composable_pipeline as mod

    monkeypatch.setattr(mod, "set_custom_profile", lambda profile: None)
    monkeypatch.setattr(mod, "get_session_instructions", lambda: "FRESH-MARKER")

    call_log: list[str] = []

    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])

    class _SequencingTTS(_RecordingTTS):
        async def reset_per_session_state(self) -> None:
            call_log.append("tts_reset")

    tts = _SequencingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts, system_prompt="old persona")
    pipeline._conversation_history.append({"role": "user", "content": "u"})

    original_reset_history = pipeline.reset_history

    def _spy_reset_history(*, keep_system: bool = True) -> None:
        call_log.append(f"reset_history(keep_system={keep_system})")
        original_reset_history(keep_system=keep_system)

    pipeline.reset_history = _spy_reset_history  # type: ignore[method-assign]

    await pipeline.apply_personality("rodney")

    # reset_history fires first, then tts_reset. The history-append is the
    # last step (visible in the final history shape).
    assert call_log == [
        "reset_history(keep_system=False)",
        "tts_reset",
    ], f"unexpected call ordering: {call_log!r}"
    # Final history reflects the post-append shape.
    assert pipeline.conversation_history == [
        {"role": "system", "content": "FRESH-MARKER"},
    ]


# ---------------------------------------------------------------------------
# Phase 5e.2 — STT host concerns landed on ComposablePipeline
# ---------------------------------------------------------------------------


def _make_mock_deps(
    *,
    head_wobbler: Any = None,
    pause_controller: Any = None,
    recent_user_transcripts: list[str] | None = None,
) -> Any:
    """Return a SimpleNamespace standing in for ``ToolDependencies``.

    The pipeline only reads ``movement_manager.set_listening``,
    ``head_wobbler.reset``, ``pause_controller.handle_transcript``, and
    ``recent_user_transcripts``; a SimpleNamespace with the relevant
    attributes is enough.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    return SimpleNamespace(
        movement_manager=MagicMock(),
        head_wobbler=head_wobbler,
        pause_controller=pause_controller,
        recent_user_transcripts=recent_user_transcripts if recent_user_transcripts is not None else [],
    )


@pytest.mark.asyncio
async def test_start_up_subscribes_partial_and_speech_started_callbacks() -> None:
    """The pipeline wires all three STT callbacks through ``stt.start``."""
    stt = _ProgrammableSTT()
    llm = _ScriptedLLM([])
    tts = _RecordingTTS()
    pipeline = ComposablePipeline(stt=stt, llm=llm, tts=tts)

    task = asyncio.create_task(pipeline.start_up())
    await _wait_for_callback(stt)

    assert stt.on_completed is not None
    assert stt.on_partial is not None
    assert stt.on_speech_started is not None

    await pipeline.shutdown()
    await task


@pytest.mark.asyncio
async def test_speech_started_callback_opens_turn_span() -> None:
    """``_on_speech_started`` opens a root ``turn`` span + child ``stt.infer`` span."""
    stt = _ProgrammableSTT()
    pipeline = ComposablePipeline(stt=stt, llm=_ScriptedLLM([]), tts=_RecordingTTS())

    assert pipeline._turn_span is None
    assert pipeline._stt_infer_span is None

    await pipeline._on_speech_started()

    assert pipeline._turn_span is not None
    assert pipeline._stt_infer_span is not None


@pytest.mark.asyncio
async def test_speech_started_callback_calls_set_listening_true_when_deps_provided() -> None:
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS(), deps=deps)

    await pipeline._on_speech_started()

    deps.movement_manager.set_listening.assert_called_once_with(True)


@pytest.mark.asyncio
async def test_speech_started_callback_calls_head_wobbler_reset_when_provided() -> None:
    from unittest.mock import MagicMock

    head_wobbler = MagicMock()
    deps = _make_mock_deps(head_wobbler=head_wobbler)
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS(), deps=deps)

    await pipeline._on_speech_started()

    head_wobbler.reset.assert_called_once()


@pytest.mark.asyncio
async def test_speech_started_callback_calls_clear_queue_when_set() -> None:
    from unittest.mock import MagicMock

    clear_cb = MagicMock()
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())
    pipeline._clear_queue = clear_cb

    await pipeline._on_speech_started()

    clear_cb.assert_called_once()


@pytest.mark.asyncio
async def test_speech_started_callback_no_deps_does_not_raise() -> None:
    """When ``deps=None`` the callback opens the span but skips movement/wobbler."""
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())

    await pipeline._on_speech_started()  # must not raise

    assert pipeline._turn_span is not None  # span opened
    # No deps means no movement/wobbler calls to assert; the test passes
    # if no exception was raised.


@pytest.mark.asyncio
async def test_partial_callback_publishes_user_partial_to_output_queue() -> None:
    from fastrtc import AdditionalOutputs

    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())

    await pipeline._on_partial_transcript("hello rob")

    item = pipeline.output_queue.get_nowait()
    assert isinstance(item, AdditionalOutputs)
    assert item.args[0] == {"role": "user_partial", "content": "hello rob"}


@pytest.mark.asyncio
async def test_partial_callback_does_not_publish_empty_string() -> None:
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())

    await pipeline._on_partial_transcript("")

    assert pipeline.output_queue.empty()


@pytest.mark.asyncio
async def test_completed_callback_publishes_user_to_output_queue_when_deps_provided() -> None:
    from fastrtc import AdditionalOutputs

    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="reply")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    await pipeline._on_transcript_completed("hi there")

    # First item on the queue is the user-role transcript marker; the TTS
    # frame follows.
    item = pipeline.output_queue.get_nowait()
    assert isinstance(item, AdditionalOutputs)
    assert item.args[0] == {"role": "user", "content": "hi there"}


@pytest.mark.asyncio
async def test_completed_callback_records_user_transcript_when_deps_provided() -> None:
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="reply")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    await pipeline._on_transcript_completed("hello robot")

    assert "hello robot" in deps.recent_user_transcripts


@pytest.mark.asyncio
async def test_completed_callback_calls_set_listening_false_when_deps_provided() -> None:
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="reply")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    await pipeline._on_transcript_completed("hi")

    deps.movement_manager.set_listening.assert_called_with(False)


@pytest.mark.asyncio
async def test_completed_callback_suppresses_duplicate_within_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror of mixin's 0.75s duplicate-suppression window.

    Pins the perf-counter window to a deterministic sequence so the
    test is independent of test-runner wallclock delays (e.g.
    pytest-xdist worker overhead, joke-history file I/O on the first
    LLM round-trip).
    """
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="r1"), LLMResponse(text="r2")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    # First call sets the dedup baseline at t=100.0; second call at
    # t=100.1 is well inside the 0.75s window and must be dropped.
    # We also intercept the post-LLM perf_counter reads (in
    # telemetry.record_stt timing) by stamping a stable sequence.
    times = iter([100.0, 100.0, 100.05, 100.1, 100.1, 100.15])

    def _fake_perf_counter() -> float:
        try:
            return next(times)
        except StopIteration:
            return 100.2

    monkeypatch.setattr("robot_comic.composable_pipeline.time.perf_counter", _fake_perf_counter)

    await pipeline._on_transcript_completed("repeat me")
    # Immediate duplicate within the 0.75s window must be ignored.
    await pipeline._on_transcript_completed("repeat me")

    user_items = []
    while not pipeline.output_queue.empty():
        item = pipeline.output_queue.get_nowait()
        from fastrtc import AdditionalOutputs as AO

        if isinstance(item, AO) and item.args[0].get("role") == "user":
            user_items.append(item)
    assert len(user_items) == 1, "duplicate completion within 0.75s window must be dropped"


@pytest.mark.asyncio
async def test_completed_callback_pause_controller_handled_drops_transcript() -> None:
    from unittest.mock import MagicMock

    from robot_comic.pause import TranscriptDisposition

    pause = MagicMock()
    pause.handle_transcript.return_value = TranscriptDisposition.HANDLED
    deps = _make_mock_deps(pause_controller=pause)

    llm = _ScriptedLLM([])  # exhausted — would error if dispatch happens
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=llm, tts=_RecordingTTS(), deps=deps)

    await pipeline._on_transcript_completed("pause please")

    pause.handle_transcript.assert_called_once_with("pause please")
    # LLM must not have been invoked since pause handled the transcript.
    assert llm.calls == []


@pytest.mark.asyncio
async def test_completed_callback_pause_controller_dispatch_proceeds() -> None:
    from unittest.mock import MagicMock

    from robot_comic.pause import TranscriptDisposition

    pause = MagicMock()
    pause.handle_transcript.return_value = TranscriptDisposition.DISPATCH
    deps = _make_mock_deps(pause_controller=pause)

    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="ok")]),
        tts=_RecordingTTS(),
        deps=deps,
    )

    await pipeline._on_transcript_completed("normal speech")

    pause.handle_transcript.assert_called_once_with("normal speech")
    # Pipeline proceeded to LLM dispatch since pause returned DISPATCH.
    assert pipeline.conversation_history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_completed_callback_welcome_gate_waiting_drops_on_no_match() -> None:
    from robot_comic.welcome_gate import WelcomeGate

    gate = WelcomeGate(["rickles"])
    deps = _make_mock_deps()
    llm = _ScriptedLLM([])  # exhausted — drop must prevent dispatch
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=llm,
        tts=_RecordingTTS(),
        deps=deps,
        welcome_gate=gate,
    )

    await pipeline._on_transcript_completed("hello world")

    # Gate still WAITING since "rickles" not heard.
    from robot_comic.welcome_gate import GateState

    assert gate.state is GateState.WAITING
    assert llm.calls == []


@pytest.mark.asyncio
async def test_completed_callback_welcome_gate_waiting_opens_on_match_and_dispatches() -> None:
    from robot_comic.welcome_gate import GateState, WelcomeGate

    gate = WelcomeGate(["rickles"])
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="hey kid")]),
        tts=_RecordingTTS(),
        deps=deps,
        welcome_gate=gate,
    )

    await pipeline._on_transcript_completed("hey rickles")

    assert gate.state is GateState.GATED
    # LLM did run (gate just opened on the matching transcript).
    assert pipeline.conversation_history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_completed_callback_welcome_gate_gated_dispatches_immediately() -> None:
    from robot_comic.welcome_gate import GateState, WelcomeGate

    gate = WelcomeGate(["rickles"])
    gate.state = GateState.GATED  # pre-opened
    deps = _make_mock_deps()
    pipeline = ComposablePipeline(
        stt=_ProgrammableSTT(),
        llm=_ScriptedLLM([LLMResponse(text="reply")]),
        tts=_RecordingTTS(),
        deps=deps,
        welcome_gate=gate,
    )

    await pipeline._on_transcript_completed("anything")

    assert pipeline.conversation_history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_speech_started_no_op_for_legacy_pipeline_without_callbacks() -> None:
    """A pipeline constructed without ``deps`` still wires the callbacks safely.

    Pre-5e.2 callers (host-coupled triples) don't pass ``deps`` to the
    pipeline; the mixin handles speech-start/partial on the host. The
    pipeline must not crash when its own callbacks fire in that mode.
    """
    pipeline = ComposablePipeline(stt=_ProgrammableSTT(), llm=_ScriptedLLM([]), tts=_RecordingTTS())
    # All three callbacks must be safe no-ops with no deps / no gate.
    await pipeline._on_speech_started()
    await pipeline._on_partial_transcript("partial")
    # Completed without deps/gate falls through to the LLM loop; provide one
    # response so the loop doesn't exhaust.
    pipeline.llm = _ScriptedLLM([LLMResponse(text="ok")])
    await pipeline._on_transcript_completed("hi")
