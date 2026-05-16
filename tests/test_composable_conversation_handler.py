"""Tests for the Phase 4a `ComposableConversationHandler` wrapper."""

from __future__ import annotations
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from robot_comic.conversation_handler import AudioFrame, ConversationHandler


def _make_wrapper() -> Any:
    """Build a wrapper with all collaborators mocked. Returns the wrapper instance."""
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    pipeline = MagicMock()
    pipeline.output_queue = asyncio.Queue()
    pipeline._conversation_history = []
    pipeline.start_up = AsyncMock()
    pipeline.shutdown = AsyncMock()
    pipeline.feed_audio = AsyncMock()
    pipeline.reset_history = MagicMock()

    tts_handler = MagicMock()

    def _build() -> ComposableConversationHandler:
        return _make_wrapper()

    return ComposableConversationHandler(
        pipeline=pipeline,
        tts_handler=tts_handler,
        deps=MagicMock(),
        build=_build,
    )


def test_wrapper_implements_conversation_handler_abc() -> None:
    wrapper = _make_wrapper()
    assert isinstance(wrapper, ConversationHandler)


@pytest.mark.asyncio
async def test_start_up_delegates_to_pipeline() -> None:
    wrapper = _make_wrapper()
    await wrapper.start_up()
    wrapper.pipeline.start_up.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_up_emits_handler_start_up_complete_after_delegating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper must emit ``handler.start_up.complete`` *after* the
    pipeline finishes preparing its adapters.

    Boot memo PR #383 §"weird things to know" #2 and instrumentation audit
    PR #385 §6 gap #3 flagged the previous emit-on-entry timing as a
    misnomer: the row labelled "complete" was firing on wrapper entry
    rather than on pipeline readiness. This test pins the corrected timing
    — the emit lives in a ``try/finally`` so it still fires if the pipeline
    raises, but the normal-path ordering is pipeline-first, then emit.
    """
    from robot_comic import telemetry

    wrapper = _make_wrapper()

    emit_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    pipeline_seen_emits: list[int] = []

    async def _record_start_up(*_a: Any, **_kw: Any) -> None:
        pipeline_seen_emits.append(len(emit_calls))

    wrapper.pipeline.start_up = AsyncMock(side_effect=_record_start_up)

    def _record_emit(*args: Any, **kwargs: Any) -> None:
        emit_calls.append((args, kwargs))

    monkeypatch.setattr(telemetry, "emit_supporting_event", _record_emit)

    await wrapper.start_up()

    # The emit fired exactly once with a numeric dur_ms kwarg.
    complete_calls = [(a, kw) for (a, kw) in emit_calls if a and a[0] == "handler.start_up.complete"]
    assert len(complete_calls) == 1, f"expected one handler.start_up.complete emit, got {emit_calls!r}"
    _args, kwargs = complete_calls[0]
    assert "dur_ms" in kwargs
    assert isinstance(kwargs["dur_ms"], float) and kwargs["dur_ms"] >= 0

    # Pipeline.start_up was awaited, and no emit had fired yet when the
    # pipeline entered. (The emit is the LAST thing start_up does.)
    wrapper.pipeline.start_up.assert_awaited_once()
    assert pipeline_seen_emits == [0], (
        "emit_supporting_event must fire AFTER pipeline.start_up returns; "
        f"observed emit_count at pipeline entry = {pipeline_seen_emits!r}"
    )


@pytest.mark.asyncio
async def test_start_up_emit_fires_even_when_pipeline_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``try/finally`` wrap ensures ``handler.start_up.complete`` still
    fires if the pipeline raises during prepare. Downstream monitor consumers
    get an early-exit signal instead of a hung "no event ever arrived" state.
    """
    from robot_comic import telemetry

    wrapper = _make_wrapper()

    async def _boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("prepare blew up")

    wrapper.pipeline.start_up = AsyncMock(side_effect=_boom)

    emit_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _record_emit(*args: Any, **kwargs: Any) -> None:
        emit_calls.append((args, kwargs))

    monkeypatch.setattr(telemetry, "emit_supporting_event", _record_emit)

    with pytest.raises(RuntimeError, match="prepare blew up"):
        await wrapper.start_up()

    complete_calls = [(a, kw) for (a, kw) in emit_calls if a and a[0] == "handler.start_up.complete"]
    assert len(complete_calls) == 1, (
        f"handler.start_up.complete must fire from the finally branch even on prepare failure; got {emit_calls!r}"
    )


@pytest.mark.asyncio
async def test_start_up_emit_failure_does_not_break_pipeline_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raised ``telemetry.emit_supporting_event`` must not prevent the
    wrapper from delegating to ``pipeline.start_up``. Telemetry never blocks
    boot — matches the ``try/except`` shape the legacy ElevenLabs handler
    uses at the emit site."""
    from robot_comic import telemetry

    wrapper = _make_wrapper()

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("export wiring broken")

    monkeypatch.setattr(telemetry, "emit_supporting_event", _raise)

    await wrapper.start_up()

    wrapper.pipeline.start_up.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_delegates_to_pipeline() -> None:
    wrapper = _make_wrapper()
    await wrapper.shutdown()
    wrapper.pipeline.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_converts_and_forwards_to_feed_audio() -> None:
    from robot_comic.backends import AudioFrame as BackendAudioFrame

    wrapper = _make_wrapper()
    samples = np.zeros(160, dtype=np.int16)
    frame: AudioFrame = (16000, samples)
    await wrapper.receive(frame)
    wrapper.pipeline.feed_audio.assert_awaited_once()
    (forwarded,), _ = wrapper.pipeline.feed_audio.call_args
    assert isinstance(forwarded, BackendAudioFrame)
    assert forwarded.sample_rate == 16000
    assert forwarded.samples is samples


@pytest.mark.asyncio
async def test_emit_pulls_from_output_queue() -> None:
    wrapper = _make_wrapper()
    sentinel = (24000, np.ones(48, dtype=np.int16))
    await wrapper.pipeline.output_queue.put(sentinel)
    result = await wrapper.emit()
    assert result is sentinel


def test_get_current_voice_delegates_to_pipeline_tts() -> None:
    """Phase 5c.1: wrapper forwards voice queries through ``pipeline.tts``.

    Pre-5c.1 the wrapper forwarded to ``self._tts_handler`` directly;
    the Protocol extension lets the adapter own the surface.
    ``_tts_handler`` is still held for ``_reset_tts_per_session_state``
    (Phase 5c.2 / 5d will revisit).
    """
    wrapper = _make_wrapper()
    wrapper.pipeline.tts = MagicMock()
    wrapper.pipeline.tts.get_current_voice = MagicMock(return_value="Brian")
    assert wrapper.get_current_voice() == "Brian"
    wrapper.pipeline.tts.get_current_voice.assert_called_once()


@pytest.mark.asyncio
async def test_get_available_voices_delegates_to_pipeline_tts() -> None:
    """Phase 5c.1: wrapper forwards voice queries through ``pipeline.tts``."""
    wrapper = _make_wrapper()
    wrapper.pipeline.tts = MagicMock()
    wrapper.pipeline.tts.get_available_voices = AsyncMock(return_value=["A", "B"])
    assert await wrapper.get_available_voices() == ["A", "B"]
    wrapper.pipeline.tts.get_available_voices.assert_awaited_once()


@pytest.mark.asyncio
async def test_change_voice_delegates_to_pipeline_tts() -> None:
    """Phase 5c.1: wrapper forwards voice queries through ``pipeline.tts``."""
    wrapper = _make_wrapper()
    wrapper.pipeline.tts = MagicMock()
    wrapper.pipeline.tts.change_voice = AsyncMock(return_value="Voice changed to X.")
    assert await wrapper.change_voice("X") == "Voice changed to X."
    wrapper.pipeline.tts.change_voice.assert_awaited_once_with("X")


@pytest.mark.asyncio
async def test_apply_personality_forwards_to_pipeline() -> None:
    """Phase 5c.2: wrapper.apply_personality is a thin pass-through.

    The persona-switch state surgery (history reset, per-session TTS
    state reset via :meth:`TTSBackend.reset_per_session_state`,
    system-prompt re-seed) lives on the pipeline as of Phase 5c.2; the
    wrapper just satisfies the :class:`ConversationHandler` ABC contract
    by forwarding through. Behavioural assertions for the pipeline-side
    work live in ``tests/test_composable_pipeline.py`` and end-to-end
    coverage with a real adapter lives in
    ``tests/test_composable_persona_reset.py``.
    """
    wrapper = _make_wrapper()
    wrapper.pipeline.apply_personality = AsyncMock(return_value="sentinel-result")

    result = await wrapper.apply_personality("rodney")

    assert result == "sentinel-result"
    wrapper.pipeline.apply_personality.assert_awaited_once_with("rodney")


def _make_fake_pipeline() -> Any:
    p = MagicMock()
    p.output_queue = asyncio.Queue()
    p._conversation_history = []
    return p


def test_copy_returns_new_instance_from_build_closure() -> None:
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    build_count = {"n": 0}

    def _build() -> ComposableConversationHandler:
        build_count["n"] += 1
        return ComposableConversationHandler(
            pipeline=_make_fake_pipeline(),
            tts_handler=MagicMock(),
            deps=MagicMock(),
            build=_build,
        )

    original = _build()
    build_count["n"] = 0  # reset after constructing the original
    copy = original.copy()
    assert copy is not original
    assert build_count["n"] == 1


def test_copy_does_not_share_pipeline_state() -> None:
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        return ComposableConversationHandler(
            pipeline=_make_fake_pipeline(),
            tts_handler=MagicMock(),
            deps=MagicMock(),
            build=_build,
        )

    original = _build()
    copy = original.copy()

    original.pipeline._conversation_history.append({"role": "user", "content": "hi"})
    assert copy.pipeline._conversation_history == []
    assert copy.pipeline is not original.pipeline


@pytest.mark.asyncio
async def test_integration_transcript_to_audio_frame() -> None:
    """End-to-end through a real ``ComposablePipeline`` with stubbed backends."""
    from robot_comic.backends import AudioFrame as BackendsAudioFrame
    from robot_comic.backends import LLMResponse, TranscriptCallback
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    callback_holder: dict[str, TranscriptCallback] = {}

    class StubSTT:
        async def start(self, on_completed: TranscriptCallback) -> None:
            callback_holder["fn"] = on_completed

        async def feed_audio(self, frame: Any) -> None:  # noqa: ARG002
            pass

        async def stop(self) -> None:
            pass

    class StubLLM:
        async def prepare(self) -> None:
            pass

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
        ) -> LLMResponse:
            return LLMResponse(text="hello world", tool_calls=())

        async def shutdown(self) -> None:
            pass

    class StubTTS:
        async def prepare(self) -> None:
            pass

        async def synthesize(
            self,
            text: str,  # noqa: ARG002
            tags: tuple[str, ...] = (),  # noqa: ARG002
            first_audio_marker: list[float] | None = None,  # noqa: ARG002
        ):
            yield BackendsAudioFrame(
                samples=np.ones(48, dtype=np.int16),
                sample_rate=24000,
            )

        async def shutdown(self) -> None:
            pass

    pipeline = ComposablePipeline(StubSTT(), StubLLM(), StubTTS())

    def _build_unused() -> ComposableConversationHandler:
        raise AssertionError("copy() not exercised in this test")

    wrapper = ComposableConversationHandler(
        pipeline=pipeline,
        tts_handler=MagicMock(),
        deps=MagicMock(),
        build=_build_unused,
    )

    # start_up blocks until shutdown — run it in the background.
    start_task = asyncio.create_task(wrapper.start_up())
    # Let prepare()/start() run and register the STT callback.
    for _ in range(5):
        if "fn" in callback_holder:
            break
        await asyncio.sleep(0)
    assert "fn" in callback_holder, "pipeline did not register STT callback"

    # Drive a "completed transcript" through the registered callback.
    await callback_holder["fn"]("hello")

    # The TTS frame should now be on the wrapper's output queue.
    frame = await asyncio.wait_for(wrapper.emit(), timeout=1.0)
    assert isinstance(frame, BackendsAudioFrame)
    assert frame.sample_rate == 24000
    assert frame.samples.shape == (48,)

    await wrapper.shutdown()
    await asyncio.wait_for(start_task, timeout=1.0)


def test_clear_queue_assignment_propagates_to_tts_handler() -> None:
    """LocalStream sets handler._clear_queue on the wrapper; the LocalSTTInputMixin
    listener reads it off the legacy handler. Forward the assignment so barge-in
    on the composable path still reaches console.clear_audio_queue."""
    wrapper = _make_wrapper()

    def cb() -> None:
        return None

    wrapper._clear_queue = cb
    assert wrapper._clear_queue is cb
    assert wrapper._tts_handler._clear_queue is cb


def test_clear_queue_assignment_handles_none() -> None:
    wrapper = _make_wrapper()
    wrapper._clear_queue = lambda: None
    wrapper._clear_queue = None
    assert wrapper._clear_queue is None
    assert wrapper._tts_handler._clear_queue is None


def test_output_queue_getter_returns_pipeline_queue() -> None:
    """The wrapper reads output_queue through to the pipeline, no caching."""
    wrapper = _make_wrapper()
    fresh: asyncio.Queue[Any] = asyncio.Queue()
    wrapper.pipeline.output_queue = fresh
    assert wrapper.output_queue is fresh


def test_output_queue_setter_replaces_pipeline_queue() -> None:
    """Rebinding wrapper.output_queue must replace the pipeline's queue
    (what emit() actually reads). Otherwise console.clear_audio_queue is a
    no-op on the composable path: it would swap the wrapper's queue while
    leaving stale TTS frames on the pipeline's queue for emit() to drain."""
    wrapper = _make_wrapper()
    fresh: asyncio.Queue[Any] = asyncio.Queue()
    wrapper.output_queue = fresh
    assert wrapper.pipeline.output_queue is fresh


@pytest.mark.asyncio
async def test_emit_reads_from_replaced_queue_after_clear() -> None:
    """End-to-end: after clear_audio_queue rebinds output_queue, emit() reads
    from the new queue and ignores stale frames on the old queue."""
    wrapper = _make_wrapper()
    # Stale frame on the original queue — must NOT be returned.
    await wrapper.pipeline.output_queue.put("stale")
    # Simulate console.clear_audio_queue's rebind.
    wrapper.output_queue = asyncio.Queue()
    # Fresh frame on the new queue.
    await wrapper.output_queue.put("fresh")
    result = await asyncio.wait_for(wrapper.emit(), timeout=1.0)
    assert result == "fresh"
