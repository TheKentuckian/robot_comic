"""End-to-end integration smoke test — Moonshine + Llama + ElevenLabs lifecycle.

Boots the handler in sim mode, injects a synthetic transcript through the
public ``_dispatch_completed_transcript`` entry point, drains the output queue,
and asserts that at least one PCM audio frame was produced.

Network boundaries mocked:
  * ``_http.stream`` (llama-server SSE) → canned one-sentence LLM response.
  * ``_stream_tts_to_queue`` (ElevenLabs TTS) → pushes canned PCM frames directly.

Run with::

    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import robot_comic.llama_base as llama_base_mod
from .conftest import drain_queue, make_tool_deps
from robot_comic.local_stt_realtime import LocalSTTInputMixin
from robot_comic.llama_elevenlabs_tts import LlamaElevenLabsTTSResponseHandler


# Phase 4e (#337) retired LocalSTTLlamaElevenLabsHandler. The composable
# factory composes LocalSTTInputMixin over LlamaElevenLabsTTSResponseHandler
# via a factory-private host class; we mirror that shape here so the smoke
# test still exercises both halves end-to-end.
class _LocalSTTLlamaElevenLabsHost(LocalSTTInputMixin, LlamaElevenLabsTTSResponseHandler):
    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        # Route past LocalSTTInputMixin's OpenAI-realtime default —
        # mirrors the factory-private host shape.
        await LlamaElevenLabsTTSResponseHandler._dispatch_completed_transcript(self, transcript)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int = 4800) -> bytes:
    """Return *n_samples* of silence as int16 PCM bytes (200 ms at 24 kHz)."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_sse_stream(lines: list[str]) -> MagicMock:
    """Return an async-context-manager mock for ``_http.stream(...)`` calls.

    Mirrors the helper used in ``tests/integration/test_chatterbox_smoke.py``.
    """
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    response.raise_for_status = MagicMock()

    async def aiter_lines() -> Any:
        for line in lines:
            yield line

    response.aiter_lines = aiter_lines
    return response


def _make_handler() -> _LocalSTTLlamaElevenLabsHost:
    deps = make_tool_deps()
    handler = _LocalSTTLlamaElevenLabsHost(deps, sim_mode=True)
    handler._http = AsyncMock()
    return handler


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_OUTPUT_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400  # 100 ms at 24 kHz


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llama_elevenlabs_dispatch_produces_pcm_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler lifecycle: inject transcript → drain queue → assert PCM frame.

    This test exercises:
      1. ``_dispatch_completed_transcript`` calling the llama LLM SSE path.
      2. Token accumulation and sentence splitting.
      3. ``_stream_tts_to_queue`` being called for each sentence.
      4. PCM frames landing in ``output_queue`` as (sample_rate, ndarray) tuples.
    """
    monkeypatch.setattr(llama_base_mod, "get_session_instructions", lambda: "Be funny.")
    monkeypatch.setattr(llama_base_mod, "get_active_tool_specs", lambda _: [])
    monkeypatch.setattr(
        "robot_comic.llama_elevenlabs_tts.config",
        MagicMock(
            ELEVENLABS_API_KEY="test_key",
            REACHY_MINI_MAX_HISTORY_TURNS=20,
            REACHY_MINI_CUSTOM_PROFILE=None,
            LLAMA_CPP_URL="http://localhost:8080",
            JOKE_HISTORY_ENABLED=False,
            ECHO_COOLDOWN_MS=300,
        ),
    )
    # Provide a stable voice ID so the handler skips the voice catalog.
    monkeypatch.setattr(
        "robot_comic.llama_elevenlabs_tts.LlamaElevenLabsTTSResponseHandler._resolve_voice_id",
        lambda self: "test_voice_id",
    )

    handler = _make_handler()

    # --- Mock llama-server LLM SSE stream ------------------------------------
    handler._http.stream = MagicMock(
        return_value=_make_sse_stream(
            [
                'data: {"choices":[{"delta":{"content":"Hello there, friend!"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
    )

    # --- Mock ElevenLabs TTS: inject canned PCM frames directly --------------
    tts_calls: list[str] = []

    async def _fake_stream_tts_to_queue(
        text: str,
        first_audio_marker: Any = None,
        tags: Any = None,
        target_queue: Any = None,
    ) -> bool:
        tts_calls.append(text)
        out = target_queue if target_queue is not None else handler.output_queue
        pcm = np.zeros(_CHUNK_SAMPLES * 2, dtype=np.int16)
        await out.put((_OUTPUT_SAMPLE_RATE, pcm))
        return True

    handler._stream_tts_to_queue = _fake_stream_tts_to_queue

    # --- Run the dispatch ----------------------------------------------------
    await handler._dispatch_completed_transcript("hello")

    # --- Assertions ----------------------------------------------------------
    all_items = drain_queue(handler.output_queue)

    audio_frames = [item for item in all_items if isinstance(item, tuple)]
    assert len(audio_frames) >= 1, (
        f"Expected at least one PCM audio frame in output_queue, got {len(audio_frames)}. "
        f"All item types: {[type(i).__name__ for i in all_items]}"
    )

    assert len(tts_calls) >= 1, "Expected ElevenLabs TTS to be called at least once"

    sample_rate, pcm_array = audio_frames[0]
    assert sample_rate == _OUTPUT_SAMPLE_RATE, f"Expected sample rate {_OUTPUT_SAMPLE_RATE}, got {sample_rate}"
    assert isinstance(pcm_array, np.ndarray), "PCM data must be a numpy array"
    assert pcm_array.dtype == np.int16, f"PCM dtype should be int16, got {pcm_array.dtype}"
