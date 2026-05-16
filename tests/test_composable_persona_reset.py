"""Regression tests for per-persona echo-guard reset on the composable path.

Phase 5a.1 wired this through the wrapper's ``apply_personality`` reaching
into ``self._tts_handler`` directly. Phase 5c.2 moves the body onto
``ComposablePipeline.apply_personality`` and the per-session-state reset
onto the :class:`TTSBackend` Protocol as ``reset_per_session_state``,
so the wrapper is now a thin pass-through and the per-handler reset
happens inside the TTS adapter.

These tests preserve the original end-to-end coverage: a non-zero
``_speaking_until`` on the wrapped handler at the moment of persona switch
must be cleared so ``LocalSTTInputMixin._handle_local_stt_event``
(``local_stt_realtime.py:619``) does not keep dropping the operator's
next few transcripts to the new persona for the remaining
``ECHO_COOLDOWN_MS`` window. The failure surface now points at the
adapter's ``reset_per_session_state`` rather than the wrapper's
helper, which is the right level of regression coverage after the
Phase 5c.2 move.

Design rationale and audit findings live in
``docs/superpowers/specs/2026-05-16-phase-5a1-echo-guard-persona-reset.md``
(audit) and
``docs/superpowers/specs/2026-05-16-phase-5c2-apply-personality-to-pipeline.md``
(the move).
"""

from __future__ import annotations
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from robot_comic.backends import AudioFrame, LLMResponse, TranscriptCallback
from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.composable_pipeline import ComposablePipeline


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


class _NoopSTT:
    """Minimal STT stub — never fires the transcript callback."""

    async def start(self, on_completed: TranscriptCallback) -> None:
        return None

    async def feed_audio(self, frame: AudioFrame) -> None:
        return None

    async def stop(self) -> None:
        return None


class _NoopLLM:
    """Minimal LLM stub — never called in these tests."""

    async def prepare(self) -> None:
        return None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:  # pragma: no cover — not exercised
        return LLMResponse(text="")

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_apply_personality_clears_tts_handler_echo_guard_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``pipeline.apply_personality`` must reset the wrapped TTS
    handler's per-session echo-guard accumulators via the adapter.

    Phase 5a.1 wired this through the wrapper; Phase 5c.2 moved the body
    onto the pipeline and the per-handler reset onto
    ``TTSBackend.reset_per_session_state``. The end-state contract is
    unchanged: a non-zero ``_speaking_until`` from a still-playing TTS
    turn at the moment of persona switch must be cleared so
    ``LocalSTTInputMixin`` does not suppress the operator's first few
    transcripts to the new persona.

    Uses a real ``ElevenLabsTTSResponseHandler`` and a real
    ``ElevenLabsTTSAdapter`` so the regression fires if either the
    pipeline's await of ``tts.reset_per_session_state`` or the adapter's
    forward into the wrapped handler is dropped.
    """
    from robot_comic import elevenlabs_tts as mod
    from robot_comic import composable_pipeline as pipeline_mod
    from robot_comic.elevenlabs_tts import ElevenLabsTTSResponseHandler
    from robot_comic.adapters.elevenlabs_tts_adapter import ElevenLabsTTSAdapter

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    monkeypatch.setattr(pipeline_mod, "set_custom_profile", lambda profile: None)
    monkeypatch.setattr(pipeline_mod, "get_session_instructions", lambda: "fresh instructions")

    tts_handler = ElevenLabsTTSResponseHandler(_make_deps())
    # Simulate the wrapped handler being mid-playback at the moment of switch.
    tts_handler._speaking_until = 999.0
    tts_handler._response_start_ts = 500.0
    tts_handler._response_audio_bytes = 9600

    adapter = ElevenLabsTTSAdapter(tts_handler)  # type: ignore[arg-type]
    pipeline = ComposablePipeline(stt=_NoopSTT(), llm=_NoopLLM(), tts=adapter)

    result = await pipeline.apply_personality("rodney")

    assert "Applied personality 'rodney'" in result, f"apply_personality should succeed; got {result!r}"
    # Phase 5c.2 contract: per-session echo-guard accumulators on the
    # wrapped TTS handler are cleared so persona switch is a hard cut on
    # listening state. The reset happens in the adapter's
    # ``reset_per_session_state``, which is awaited by the pipeline.
    assert tts_handler._speaking_until == 0.0, (
        f"_speaking_until must be reset on persona switch; got {tts_handler._speaking_until!r}"
    )
    assert tts_handler._response_start_ts == 0.0, (
        f"_response_start_ts must be reset; got {tts_handler._response_start_ts!r}"
    )
    assert tts_handler._response_audio_bytes == 0, (
        f"_response_audio_bytes must be reset; got {tts_handler._response_audio_bytes!r}"
    )


class _GeminiShapedTTS:
    """TTSBackend stub whose wrapped handler has no echo-guard fields.

    Mirrors ``GeminiTTSResponseHandler`` for the purposes of this test:
    no ``_speaking_until`` / ``_response_start_ts`` / ``_response_audio_bytes``
    on the wrapped object. The adapter's ``reset_per_session_state`` must
    be a clean no-op against this shape.
    """

    def __init__(self) -> None:
        # Bare handler — no echo-guard fields. The adapter uses
        # hasattr-guarded setattr so a plain object suffices.
        self._handler: Any = type("_BareHandler", (), {})()

    async def prepare(self) -> None:
        return None

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:  # pragma: no cover — not exercised
        if False:
            yield AudioFrame(samples=[], sample_rate=0)

    async def shutdown(self) -> None:
        return None

    async def get_available_voices(self) -> list[str]:  # pragma: no cover
        return []

    def get_current_voice(self) -> str:  # pragma: no cover
        return ""

    async def change_voice(self, voice: str) -> str:  # pragma: no cover
        return ""

    async def reset_per_session_state(self) -> None:
        """Same hasattr-guarded shape as the real Gemini adapter."""
        for field, value in (
            ("_speaking_until", 0.0),
            ("_response_start_ts", 0.0),
            ("_response_audio_bytes", 0),
        ):
            if hasattr(self._handler, field):
                setattr(self._handler, field, value)


@pytest.mark.asyncio
async def test_apply_personality_no_op_on_handler_without_echo_guard_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reset path must be a clean no-op on handlers without echo-guard
    state — no ``AttributeError`` from blind ``setattr``, no spurious
    field creation. ``GeminiTTSResponseHandler`` has no such fields in
    production today (see Phase 5a.1 audit).

    Pins the no-op contract so future handlers without echo-guard (or a
    future refactor that removes the fields) don't crash persona
    switching. Failure surface now points at the adapter's
    ``reset_per_session_state`` rather than the wrapper's helper.
    """
    from robot_comic import composable_pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "set_custom_profile", lambda profile: None)
    monkeypatch.setattr(pipeline_mod, "get_session_instructions", lambda: "fresh instructions")

    tts = _GeminiShapedTTS()
    pipeline = ComposablePipeline(stt=_NoopSTT(), llm=_NoopLLM(), tts=tts)

    # Must not raise. Must succeed.
    result = await pipeline.apply_personality("rodney")
    assert "Applied personality 'rodney'" in result, (
        f"apply_personality should still succeed on handlers without echo-guard state; got {result!r}"
    )
    # The wrapped handler must not have grown the fields (defensive —
    # verifies the guard is hasattr-based, not blind setattr).
    assert not hasattr(tts._handler, "_speaking_until"), (
        "Reset must not create _speaking_until on a handler that did not have it"
    )
    assert not hasattr(tts._handler, "_response_start_ts"), (
        "Reset must not create _response_start_ts on a handler that did not have it"
    )
    assert not hasattr(tts._handler, "_response_audio_bytes"), (
        "Reset must not create _response_audio_bytes on a handler that did not have it"
    )
