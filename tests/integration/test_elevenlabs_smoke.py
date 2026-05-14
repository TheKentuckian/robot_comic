"""End-to-end integration smoke test — LocalSTTElevenLabsHandler lifecycle.

Boots the handler in sim mode, injects a synthetic transcript through the
public ``_dispatch_completed_transcript`` entry point, drains the output queue,
and asserts that at least one PCM audio frame was produced.

Network boundaries mocked:
  * ``_client.aio.models.generate_content`` → canned Gemini LLM response.
  * ``_http.stream``                         → canned ElevenLabs PCM bytes.

Run with::

    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import robot_comic.elevenlabs_tts as elevenlabs_mod
from .conftest import drain_queue, make_tool_deps
from robot_comic.elevenlabs_tts import LocalSTTElevenLabsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int = 4800) -> bytes:
    """Return *n_samples* of silence as int16 PCM bytes (200 ms at 24 kHz)."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_fake_llm_response(text: str) -> Any:
    """Build a minimal Gemini response object containing one text part."""
    part = MagicMock()
    part.text = text
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_fake_http_stream(pcm_data: bytes) -> Any:
    """Return an async-context-manager mock for ``_http.stream(...)`` calls.

    Delivers the supplied *pcm_data* as a single chunk then closes.
    """

    class _FakeStreamResponse:
        async def __aenter__(self) -> "_FakeStreamResponse":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        def raise_for_status(self) -> None:
            pass

        async def aiter_bytes(self) -> AsyncIterator[bytes]:
            yield pcm_data

    ctx = _FakeStreamResponse()
    stream_cm = MagicMock()
    stream_cm.return_value = ctx
    return stream_cm


def _make_handler() -> LocalSTTElevenLabsHandler:
    deps = make_tool_deps()
    handler = LocalSTTElevenLabsHandler(deps, sim_mode=True)
    # Pre-initialise the Gemini client mock so no real API calls are made.
    handler._client = MagicMock()
    handler._http = MagicMock()
    return handler


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_elevenlabs_dispatch_produces_pcm_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler lifecycle: inject transcript → drain queue → assert PCM frame.

    This test exercises:
      1. ``_dispatch_completed_transcript`` calling the Gemini LLM path.
      2. LLM response feeding into the ElevenLabs TTS streaming path.
      3. PCM bytes flowing from _http.stream → output_queue as (rate, array) tuples.
    """
    monkeypatch.setattr(elevenlabs_mod, "get_session_instructions", lambda: "Be funny.")
    monkeypatch.setattr(elevenlabs_mod, "get_active_tool_specs", lambda _: [])
    # Provide a stable voice ID so the handler does not hit the voice catalog.
    monkeypatch.setattr(
        "robot_comic.elevenlabs_tts.ElevenLabsTTSResponseHandler._resolve_voice_id",
        lambda self: "test_voice_id",
    )
    # Set a dummy API key so the handler does not bail out early.
    monkeypatch.setattr(
        "robot_comic.elevenlabs_tts.config",
        MagicMock(
            ELEVENLABS_API_KEY="test_key",
            ECHO_COOLDOWN_MS=300,
            REACHY_MINI_MAX_HISTORY_TURNS=20,
            REACHY_MINI_CUSTOM_PROFILE=None,
            JOKE_HISTORY_ENABLED=False,
        ),
    )

    handler = _make_handler()

    # --- Mock Gemini LLM (no tool calls, returns plain text) -----------------
    handler._client.aio = MagicMock()
    handler._client.aio.models = MagicMock()
    handler._client.aio.models.generate_content = AsyncMock(
        return_value=_make_fake_llm_response("Hello there, friend!")
    )

    # --- Mock ElevenLabs HTTP stream (return canned PCM) ---------------------
    # 4800 int16 samples = 200 ms at 24 kHz
    pcm_data = _pcm_bytes(4800)
    handler._http.stream = _make_fake_http_stream(pcm_data)

    # --- Run the dispatch ----------------------------------------------------
    await handler._dispatch_completed_transcript("hello")

    # --- Assertions ----------------------------------------------------------
    all_items = drain_queue(handler.output_queue)

    audio_frames = [item for item in all_items if isinstance(item, tuple)]
    assert len(audio_frames) >= 1, (
        f"Expected at least one PCM audio frame in output_queue, got {len(audio_frames)}. "
        f"All item types: {[type(i).__name__ for i in all_items]}"
    )

    sample_rate, pcm_array = audio_frames[0]
    assert sample_rate == 24000, f"Expected sample rate 24000, got {sample_rate}"
    assert isinstance(pcm_array, np.ndarray), "PCM data must be a numpy array"
    assert pcm_array.dtype == np.int16, f"PCM dtype should be int16, got {pcm_array.dtype}"
