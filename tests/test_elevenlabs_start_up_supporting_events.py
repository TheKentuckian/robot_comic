"""Tests for supporting-event emission inside ElevenLabsTTSResponseHandler.start_up (#337).

Before this fix, ``handler.start_up.complete`` was emitted by the
``console.py`` wrapper *after* ``await _handler.start_up()`` returned — but
``start_up()`` blocks on ``self._stop_event.wait()`` for the entire lifetime
of the handler, so the event only fired on app shutdown. The fix moves the
emit inside the handler itself, right before it blocks on the stop event.

These tests also pin ``first_greeting.tts_first_audio`` emission from
``_stream_tts_to_queue``: the original ``emit_first_greeting_audio_once``
hook had call sites in every backend module *except* ``elevenlabs_tts.py``,
so the local-STT + ElevenLabs config (which we ship as default) never
closed the boot timeline.
"""

from __future__ import annotations
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from robot_comic.tools.core_tools import ToolDependencies


def _make_handler():
    """Construct an ElevenLabsTTSResponseHandler with http + LLM clients mocked out."""
    from robot_comic.elevenlabs_tts import ElevenLabsTTSResponseHandler

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = ElevenLabsTTSResponseHandler(deps)
    handler._http = MagicMock()
    handler._client = MagicMock()
    return handler


# ---------------------------------------------------------------------------
# handler.start_up.complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_up_emits_complete_event_before_blocking_on_stop_event() -> None:
    """``handler.start_up.complete`` must fire once the handler is ready to
    receive audio (credentials prepared, startup-trigger task scheduled) —
    *not* when start_up returns on shutdown."""
    handler = _make_handler()
    handler._prepare_startup_credentials = AsyncMock()
    handler._send_startup_trigger = AsyncMock()

    with patch("robot_comic.telemetry.emit_supporting_event") as emit:
        task = asyncio.create_task(handler.start_up())
        # Give the event loop a turn so start_up reaches the
        # ``await self._stop_event.wait()`` line.
        for _ in range(5):
            await asyncio.sleep(0)
            if any(c.args and c.args[0] == "handler.start_up.complete" for c in emit.call_args_list):
                break
        # Snapshot emits *before* signalling shutdown.
        complete_calls = [c for c in emit.call_args_list if c.args and c.args[0] == "handler.start_up.complete"]
        # Now release start_up and clean up.
        handler._stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)

    assert len(complete_calls) == 1, (
        f"expected exactly one handler.start_up.complete emit before shutdown, got {emit.call_args_list}"
    )
    _args, kwargs = complete_calls[0]
    assert "dur_ms" in kwargs
    assert isinstance(kwargs["dur_ms"], float) and kwargs["dur_ms"] >= 0


@pytest.mark.asyncio
async def test_start_up_does_not_double_emit_on_shutdown() -> None:
    """The handler must emit ``.complete`` exactly once per process, not twice
    (once at ready, once at shutdown)."""
    handler = _make_handler()
    handler._prepare_startup_credentials = AsyncMock()
    handler._send_startup_trigger = AsyncMock()

    with patch("robot_comic.telemetry.emit_supporting_event") as emit:
        task = asyncio.create_task(handler.start_up())
        for _ in range(5):
            await asyncio.sleep(0)
        handler._stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)

    complete_calls = [c for c in emit.call_args_list if c.args and c.args[0] == "handler.start_up.complete"]
    assert len(complete_calls) == 1, f"expected one emit, got {len(complete_calls)}"


# ---------------------------------------------------------------------------
# first_greeting.tts_first_audio
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamCM:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc: Any) -> None:
        return None


def _silent_pcm_bytes(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype=np.int16).tobytes()


@pytest.mark.asyncio
async def test_first_audio_chunk_triggers_first_greeting_supporting_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first TTS PCM chunk through the elevenlabs handler must invoke
    ``telemetry.emit_first_greeting_audio_once`` so the boot timeline closes
    with the audio-out event (#337). Before the fix, only gemini_live /
    chatterbox / llama paths called this hook — the elevenlabs path didn't,
    so the default local-STT+ElevenLabs config never closed the timeline."""
    from robot_comic import telemetry
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    handler = _make_handler()

    handler._http.stream = lambda *a, **kw: _FakeStreamCM(_FakeStreamResponse([_silent_pcm_bytes(2400)]))

    # Reset the once-guard so the function is reachable in this test.
    monkeypatch.setattr(telemetry, "_FIRST_GREETING_EMITTED", False)
    with patch.object(telemetry, "emit_first_greeting_audio_once") as once_hook:
        await handler._stream_tts_to_queue("hi", first_audio_marker=[__import__("time").perf_counter()])

    once_hook.assert_called_once()


@pytest.mark.asyncio
async def test_first_greeting_hook_skipped_when_marker_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hook only fires when first_audio_marker is non-empty (i.e. the very
    first audio of the turn). Empty marker = mid-turn streaming, no event."""
    from robot_comic import telemetry
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    handler = _make_handler()

    handler._http.stream = lambda *a, **kw: _FakeStreamCM(_FakeStreamResponse([_silent_pcm_bytes(2400)]))

    with patch.object(telemetry, "emit_first_greeting_audio_once") as once_hook:
        await handler._stream_tts_to_queue("hi", first_audio_marker=[])

    once_hook.assert_not_called()
