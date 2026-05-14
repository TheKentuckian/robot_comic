"""End-to-end integration smoke test — LocalSTTChatterboxHandler lifecycle.

Boots the handler in sim mode, injects a synthetic transcript through the
public ``_dispatch_completed_transcript`` entry point, drains the output queue,
and asserts that at least one PCM audio frame was produced.

Network boundaries mocked:
  * ``_http.stream`` (llama-server SSE) → canned one-sentence response.
  * ``_call_chatterbox_tts``             → returns canned PCM bytes.

Run with::

    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from .conftest import drain_queue, make_tool_deps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_bytes(n_samples: int = 2400) -> bytes:
    """Return *n_samples* of silence as int16 PCM bytes."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_sse_stream(lines: list[str]) -> MagicMock:
    """Return an async-context-manager mock for ``_http.stream(...)`` calls.

    Mirrors the helper in ``tests/_helpers.py`` without importing from a
    sibling directory (the integration conftest adds the right paths).
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


def _make_handler() -> Any:
    from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler

    deps = make_tool_deps()
    handler = LocalSTTChatterboxHandler(deps, sim_mode=True)
    # Replace the HTTP client so no real network calls are made.
    handler._http = AsyncMock()
    return handler


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chatterbox_dispatch_produces_pcm_audio() -> None:
    """Handler lifecycle: inject transcript → drain queue → assert PCM frame.

    This test exercises the full wiring of:
      1. ``_dispatch_completed_transcript`` calling the LLM streaming path.
      2. Token accumulation and sentence splitting.
      3. ``_call_chatterbox_tts`` being called for each sentence.
      4. PCM bytes being chunked and pushed into ``output_queue``.
    """
    handler = _make_handler()

    # --- Mock LLM response (one sentence) -----------------------------------
    handler._http.stream = MagicMock(
        return_value=_make_sse_stream(
            [
                'data: {"choices":[{"delta":{"content":"Hello there, friend!"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
    )

    # --- Mock Chatterbox TTS (return canned PCM) ----------------------------
    tts_calls: list[str] = []

    async def fake_tts(text: str, *, exaggeration: float | None = None, cfg_weight: float | None = None) -> bytes:
        tts_calls.append(text)
        return _pcm_bytes(4800)  # 200 ms of silence at 24 kHz

    handler._call_chatterbox_tts = fake_tts  # type: ignore[method-assign]

    # --- Run the dispatch (the unit under test) ------------------------------
    await handler._dispatch_completed_transcript("hello")

    # --- Assertions ---------------------------------------------------------
    all_items = drain_queue(handler.output_queue)

    # At least one PCM audio frame must have been queued as a (sample_rate, array) tuple.
    audio_frames = [item for item in all_items if isinstance(item, tuple)]
    assert len(audio_frames) >= 1, (
        f"Expected at least one PCM audio frame in output_queue, got {len(audio_frames)}. "
        f"All items: {[type(i).__name__ for i in all_items]}"
    )

    # TTS must have been called at least once.
    assert len(tts_calls) >= 1, "Expected TTS to be called at least once"

    # Each audio frame should be a (int, ndarray) pair with int16 dtype.
    sample_rate, pcm_array = audio_frames[0]
    assert isinstance(sample_rate, int), "First element of audio frame must be sample rate (int)"
    assert hasattr(pcm_array, "dtype"), "Second element of audio frame must be a numpy array"
    assert pcm_array.dtype == np.int16, f"PCM array dtype should be int16, got {pcm_array.dtype}"
