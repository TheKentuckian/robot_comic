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
    llm = _ScriptedLLM(
        [LLMResponse(text="One."), LLMResponse(text="Two.")]
    )
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
