"""Regression tests for the composable-path echo guard.

The legacy TTS handlers (``ElevenLabsTTSResponseHandler``,
``LlamaElevenLabsTTSResponseHandler``) update ``_speaking_until`` from
``emit()`` — the legacy fastrtc consumer call site. The composable factory
path replaces that consumer with ``ComposableConversationHandler``, which
drains ``ComposablePipeline.output_queue`` instead. The TTS adapter's
``synthesize()`` becomes the sole producer of audio frames, but the
legacy ``emit()`` is bypassed entirely — so ``_speaking_until`` stays at
``0.0`` for the wrapped handler's lifetime and the echo guard never fires.

These regression tests exercise both wrapped handlers via
``ElevenLabsTTSAdapter`` and assert ``_speaking_until`` is updated. They
fail before the Option A fix (move ``_speaking_until`` derivation into
``_enqueue_audio_frame``) and pass afterwards.
"""

from __future__ import annotations
from typing import Any
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.adapters.elevenlabs_tts_adapter import ElevenLabsTTSAdapter


# ---------------------------------------------------------------------------
# Shared httpx streaming helpers (same shape as test_llama_elevenlabs_tts.py)
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes], status_code: int = 200) -> None:
        self._chunks = chunks
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            req = httpx.Request("POST", "https://example/")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=req, response=resp)

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


def _stream_factory(*responses: _FakeStreamResponse):
    it = iter(responses)

    def factory(method: str, url: str, **kwargs: Any) -> _FakeStreamCM:
        return _FakeStreamCM(next(it))

    return factory


def _silent_pcm_bytes(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


# ---------------------------------------------------------------------------
# Regression: ElevenLabsTTSResponseHandler via ElevenLabsTTSAdapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_elevenlabs_adapter_updates_speaking_until_on_streamed_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composable path: ElevenLabsTTSAdapter.synthesize() must update _speaking_until.

    Before the Option A fix, ``_speaking_until`` is only written from the
    legacy ``ElevenLabsTTSResponseHandler.emit()`` consumer — which the
    composable pipeline never calls. The echo guard would stay at 0.0 and
    the robot would hear itself speak. This test wraps the real handler in
    the adapter, drains a streamed response, and asserts the guard fires.
    """
    from robot_comic import elevenlabs_tts as mod
    from robot_comic.elevenlabs_tts import ElevenLabsTTSResponseHandler

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})

    handler = ElevenLabsTTSResponseHandler(_make_deps())
    handler._http = MagicMock()
    handler._client = MagicMock()

    # 4800 samples = 2 full _CHUNK_SAMPLES frames at 24 kHz.
    chunks = [_silent_pcm_bytes(2400), _silent_pcm_bytes(2400)]
    handler._http.stream = _stream_factory(_FakeStreamResponse(chunks))

    adapter = ElevenLabsTTSAdapter(handler)
    frames = [frame async for frame in adapter.synthesize("hello")]

    # Sanity: we actually streamed audio.
    assert len(frames) >= 1
    # The fix: _speaking_until is updated by _enqueue_audio_frame, which the
    # adapter exercises indirectly via _stream_tts_to_queue. Before the fix
    # this stays at 0.0.
    assert handler._speaking_until > 0.0, (
        "Echo guard not updated on composable path; "
        "_enqueue_audio_frame must write _speaking_until."
    )
    # Byte accumulators must also be tracked.
    assert handler._response_audio_bytes > 0
    assert handler._response_start_ts > 0.0


# ---------------------------------------------------------------------------
# Regression: LlamaElevenLabsTTSResponseHandler via ElevenLabsTTSAdapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llama_elevenlabs_adapter_updates_speaking_until_on_streamed_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composable path: same regression for the llama-LLM variant.

    The llama variant has its own ``_stream_tts_to_queue`` (independent
    HTTP path) and its own legacy ``emit()`` on ``BaseLlamaResponseHandler``.
    The composable factory routes around both. This test ensures the fix
    applies to this handler family too — otherwise the
    ``(moonshine, llama, elevenlabs)`` triple stays broken at 4d default flip.
    """
    from robot_comic import llama_elevenlabs_tts as mod
    from robot_comic.llama_elevenlabs_tts import LlamaElevenLabsTTSResponseHandler

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})

    handler = LlamaElevenLabsTTSResponseHandler(_make_deps())
    handler._http = MagicMock()

    chunks = [_silent_pcm_bytes(2400), _silent_pcm_bytes(2400)]
    handler._http.stream = _stream_factory(_FakeStreamResponse(chunks))

    adapter = ElevenLabsTTSAdapter(handler)
    frames = [frame async for frame in adapter.synthesize("hello")]

    assert len(frames) >= 1
    assert handler._speaking_until > 0.0, (
        "Echo guard not updated on composable path for llama variant; "
        "BaseLlamaResponseHandler._enqueue_audio_frame must write _speaking_until."
    )
    assert handler._response_audio_bytes > 0
    assert handler._response_start_ts > 0.0
