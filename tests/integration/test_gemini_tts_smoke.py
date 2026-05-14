"""End-to-end integration smoke test — GeminiTTSResponseHandler lifecycle.

Boots the handler in sim mode, injects a synthetic transcript through the
public ``_dispatch_completed_transcript`` entry point, drains the output queue,
and asserts that at least one PCM audio frame was produced.

Network boundaries mocked:
  * ``_client.aio.models.generate_content`` (LLM call)  → canned text response.
  * ``_client.aio.models.generate_content`` (TTS call)  → canned PCM bytes.

The same ``generate_content`` coroutine is used for both the LLM and TTS
model calls; the mock returns different values based on call order.

Run with::

    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations
import base64
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import robot_comic.gemini_tts as gemini_tts_mod
from .conftest import drain_queue, make_tool_deps
from robot_comic.gemini_tts import GeminiTTSResponseHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int = 4800) -> bytes:
    """Return *n_samples* of silence as int16 PCM bytes (200 ms at 24 kHz)."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_fake_llm_response(text: str) -> Any:
    """Build a minimal Gemini LLM response object containing one text part."""
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


def _make_fake_tts_response(pcm_bytes: bytes) -> Any:
    """Build a minimal Gemini TTS response object with base64-encoded PCM."""
    encoded = base64.b64encode(pcm_bytes).decode()

    inline_data = MagicMock()
    inline_data.data = encoded

    part = MagicMock()
    part.inline_data = inline_data

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_handler() -> GeminiTTSResponseHandler:
    deps = make_tool_deps()
    handler = GeminiTTSResponseHandler(deps, sim_mode=True)
    handler._client = MagicMock()
    return handler


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_tts_dispatch_produces_pcm_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler lifecycle: inject transcript → drain queue → assert PCM frame.

    This test exercises:
      1. ``_dispatch_completed_transcript`` calling the Gemini LLM path.
      2. LLM response text feeding into the Gemini TTS path.
      3. TTS audio bytes flowing into output_queue as (rate, ndarray) tuples.
    """
    monkeypatch.setattr(gemini_tts_mod, "get_session_instructions", lambda: "Be funny.")
    monkeypatch.setattr(gemini_tts_mod, "get_active_tool_specs", lambda _: [])
    monkeypatch.setattr(gemini_tts_mod, "load_profile_tts_instruction", lambda: "Deliver briskly.")
    monkeypatch.setattr(
        "robot_comic.gemini_tts.config",
        MagicMock(
            GEMINI_API_KEY="test_key",
            REACHY_MINI_MAX_HISTORY_TURNS=20,
            REACHY_MINI_CUSTOM_PROFILE=None,
            JOKE_HISTORY_ENABLED=False,
        ),
    )

    handler = _make_handler()

    pcm_data = _pcm_bytes(4800)

    llm_response = _make_fake_llm_response("Hello there, friend!")
    tts_response = _make_fake_tts_response(pcm_data)

    # First call → LLM (text generation), subsequent calls → TTS (audio generation).
    call_count: list[int] = [0]

    async def _fake_generate_content(**kwargs: Any) -> Any:
        call_count[0] += 1
        if call_count[0] == 1:
            return llm_response
        return tts_response

    handler._client.aio = MagicMock()
    handler._client.aio.models = MagicMock()
    handler._client.aio.models.generate_content = _fake_generate_content

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
