"""Tests for LlamaElevenLabsTTSResponseHandler — the TTS-side override.

The LLM/tool dispatch path is inherited from BaseLlamaResponseHandler and is
covered by test_llama_base.py. These tests focus on:
- _stream_tts_to_queue: streaming, chunking, telemetry, retries.
- _synthesize_and_enqueue: tag stripping, [short pause] silence, error queue.
- Voice resolution including PVC voice_id precedence.
"""

from __future__ import annotations
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from robot_comic.tools.core_tools import ToolDependencies


def _make_deps() -> ToolDependencies:
    """Build minimal ToolDependencies with mocked robot + movement manager."""
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _make_handler():
    """Construct a LlamaElevenLabsTTSResponseHandler with http client mocked out."""
    from robot_comic.llama_elevenlabs_tts import LlamaElevenLabsTTSResponseHandler

    handler = LlamaElevenLabsTTSResponseHandler(_make_deps())
    handler._http = MagicMock()
    return handler


# ---------------------------------------------------------------------------
# Fake httpx streaming helpers
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
    """Build a stand-in for ``handler._http.stream`` that yields the given responses across calls."""
    it = iter(responses)

    def factory(method: str, url: str, **kwargs: Any) -> _FakeStreamCM:
        return _FakeStreamCM(next(it))

    return factory


def _silent_pcm_bytes(n_samples: int) -> bytes:
    """Return n_samples of silent 16-bit PCM as raw bytes."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def test_resolve_voice_id_prefers_profile_custom_voice_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile-level voice_id (PVC clone) overrides the named-voice mapping."""
    from robot_comic import llama_elevenlabs_tts as mod

    monkeypatch.setattr(
        mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "custom_pvc_xyz", "voice": "Rachel"}
    )
    handler = _make_handler()
    assert handler._resolve_voice_id() == "custom_pvc_xyz"


def test_resolve_voice_id_falls_back_to_named_voice_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without voice_id in profile, the named-voice override maps to the ElevenLabs catalog ID."""
    from robot_comic import llama_elevenlabs_tts as mod

    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {})
    handler = _make_handler()
    handler._voice_override = "Rachel"
    assert handler._resolve_voice_id() == "21m00Tcm4ijWNoXd58YU"


# ---------------------------------------------------------------------------
# _stream_tts_to_queue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_aligned_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed-size streamed chunks are re-chunked into fixed 2400-sample frames."""
    from robot_comic import llama_elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    handler = _make_handler()
    handler._http.stream = _stream_factory(_FakeStreamResponse([_silent_pcm_bytes(1000), _silent_pcm_bytes(3800)]))

    ok = await handler._stream_tts_to_queue("hi", first_audio_marker=[])
    assert ok is True

    items = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    # 4800 samples → 2 full 2400-sample frames, no partial.
    assert [len(f) for _, f in items] == [2400, 2400]


@pytest.mark.asyncio
async def test_stream_flushes_partial_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trailing remainder smaller than a full frame is flushed rather than dropped."""
    from robot_comic import llama_elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    handler = _make_handler()
    handler._http.stream = _stream_factory(_FakeStreamResponse([_silent_pcm_bytes(2500)]))

    await handler._stream_tts_to_queue("hi", first_audio_marker=[])

    sizes: list[int] = []
    while not handler.output_queue.empty():
        _, f = handler.output_queue.get_nowait()
        sizes.append(len(f))
    assert sizes == [2400, 100]


@pytest.mark.asyncio
async def test_stream_fires_first_audio_telemetry_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """First chunk fires record_tts_first_audio; the marker clears so subsequent sentences don't refire."""
    from robot_comic import telemetry
    from robot_comic import llama_elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    handler = _make_handler()

    recorded: list[tuple[float, dict[str, Any]]] = []
    monkeypatch.setattr(
        telemetry,
        "record_tts_first_audio",
        lambda elapsed, attrs: recorded.append((elapsed, attrs)),
    )

    handler._http.stream = _stream_factory(_FakeStreamResponse([_silent_pcm_bytes(2400), _silent_pcm_bytes(2400)]))
    marker = [time.perf_counter() - 0.05]  # ~50ms ago
    await handler._stream_tts_to_queue("hi", first_audio_marker=marker)

    assert len(recorded) == 1
    assert recorded[0][1] == {"gen_ai.system": "elevenlabs"}
    assert recorded[0][0] > 0
    assert marker == []


@pytest.mark.asyncio
async def test_stream_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ELEVENLABS_API_KEY, _stream_tts_to_queue bails before issuing any HTTP request."""
    from robot_comic import llama_elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "", raising=False)
    handler = _make_handler()
    handler._http.stream = MagicMock(side_effect=AssertionError("should not be called"))

    assert await handler._stream_tts_to_queue("hi") is False


@pytest.mark.asyncio
async def test_stream_401_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 (bad API key) bails on the first attempt rather than burning retries."""
    from robot_comic import llama_elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    monkeypatch.setattr(mod.asyncio, "sleep", AsyncMock())
    handler = _make_handler()

    calls = [0]

    def factory(method: str, url: str, **kw: Any) -> _FakeStreamCM:
        calls[0] += 1
        return _FakeStreamCM(_FakeStreamResponse([], status_code=401))

    handler._http.stream = factory
    ok = await handler._stream_tts_to_queue("hi")
    assert ok is False
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_stream_429_exhausts_and_marks_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    """All retries returning 429 leaves _last_tts_rate_limited set so the caller can surface it."""
    from robot_comic import llama_elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    monkeypatch.setattr(mod.asyncio, "sleep", AsyncMock())
    handler = _make_handler()

    handler._http.stream = _stream_factory(
        _FakeStreamResponse([], status_code=429),
        _FakeStreamResponse([], status_code=429),
        _FakeStreamResponse([], status_code=429),
    )
    ok = await handler._stream_tts_to_queue("hi")
    assert ok is False
    assert handler._last_tts_rate_limited is True


# ---------------------------------------------------------------------------
# _synthesize_and_enqueue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_strips_tags_from_spoken_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delivery tags like [fast] are removed before the spoken text hits the TTS stream call."""
    from robot_comic import llama_elevenlabs_tts as mod

    handler = _make_handler()
    captured: list[str] = []

    async def fake_stream(self, text: str, first_audio_marker=None):  # type: ignore[no-untyped-def]
        captured.append(text)
        return True

    monkeypatch.setattr(mod.LlamaElevenLabsTTSResponseHandler, "_stream_tts_to_queue", fake_stream)

    await handler._synthesize_and_enqueue("[fast] You hockey puck!")
    assert captured == ["You hockey puck!"]


@pytest.mark.asyncio
async def test_synthesize_empty_response_emits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty response_text returns early without touching TTS or the output queue."""
    from robot_comic import llama_elevenlabs_tts as mod

    handler = _make_handler()
    fake = AsyncMock(return_value=True)
    monkeypatch.setattr(mod.LlamaElevenLabsTTSResponseHandler, "_stream_tts_to_queue", fake)

    await handler._synthesize_and_enqueue("")
    fake.assert_not_called()
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_synthesize_short_pause_queues_silence_before_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    """[short pause] enqueues silence frames before the per-sentence TTS stream call."""
    from robot_comic import llama_elevenlabs_tts as mod

    handler = _make_handler()
    qdepths: list[int] = []

    async def fake_stream(self, text: str, first_audio_marker=None):  # type: ignore[no-untyped-def]
        qdepths.append(self.output_queue.qsize())
        return True

    monkeypatch.setattr(mod.LlamaElevenLabsTTSResponseHandler, "_stream_tts_to_queue", fake_stream)
    await handler._synthesize_and_enqueue("[short pause] Hi.")
    # Silence frames must already be queued by the time TTS is called.
    assert qdepths and qdepths[0] > 0


@pytest.mark.asyncio
async def test_synthesize_all_failures_pushes_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every sentence's stream call fails, an error AdditionalOutputs is queued for the UI."""
    from robot_comic import llama_elevenlabs_tts as mod

    handler = _make_handler()

    async def failing(self, text: str, first_audio_marker=None):  # type: ignore[no-untyped-def]
        return False

    monkeypatch.setattr(mod.LlamaElevenLabsTTSResponseHandler, "_stream_tts_to_queue", failing)
    await handler._synthesize_and_enqueue("One. Two.")

    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    assert any(isinstance(i, AdditionalOutputs) and "error" in str(i.args).lower() for i in items)


@pytest.mark.asyncio
async def test_synthesize_rate_limited_message_when_all_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the rate-limit flag is set and no audio plays, the surfaced message names the cause."""
    from robot_comic import llama_elevenlabs_tts as mod

    handler = _make_handler()

    async def rl(self, text: str, first_audio_marker=None):  # type: ignore[no-untyped-def]
        self._last_tts_rate_limited = True
        return False

    monkeypatch.setattr(mod.LlamaElevenLabsTTSResponseHandler, "_stream_tts_to_queue", rl)
    await handler._synthesize_and_enqueue("Hi.")

    msgs = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, AdditionalOutputs):
            msgs.append(str(item.args))
    assert any("rate-limited" in m for m in msgs)


# ---------------------------------------------------------------------------
# Delivery tag → voice_settings mapping
# ---------------------------------------------------------------------------


def test_apply_voice_settings_deltas_no_tags() -> None:
    """Without tags, voice_settings are returned unchanged."""
    from robot_comic.llama_elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.75, [])
    assert result == {"stability": 0.5, "similarity_boost": 0.75}


def test_apply_voice_settings_deltas_fast_tag() -> None:
    """[fast] tag increases similarity_boost and decreases stability."""
    from robot_comic.llama_elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.75, ["fast"])
    assert result == {"stability": 0.4, "similarity_boost": 0.95}


def test_apply_voice_settings_deltas_multiple_tags() -> None:
    """Multiple tags combine their effects."""
    from robot_comic.llama_elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.5, ["fast", "aggression"])
    # fast: +0.2 similarity, -0.1 stability
    # aggression: +0.4 similarity, -0.2 stability
    # = 0.5 + 0.6 = 1.1 → 1.0, 0.5 - 0.3 = 0.2
    assert result == {"stability": 0.2, "similarity_boost": 1.0}


@pytest.mark.asyncio
async def test_stream_tts_applies_voice_settings_from_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delivery tags are mapped to voice_settings deltas in the TTS payload."""
    from robot_comic import llama_elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    handler = _make_handler()

    captured_payloads: list[dict[str, Any]] = []

    def capture_stream(method: str, url: str, json: Any = None, **kwargs: Any) -> _FakeStreamCM:
        if json:
            captured_payloads.append(json)
        return _FakeStreamCM(_FakeStreamResponse([_silent_pcm_bytes(2400)]))

    handler._http.stream = capture_stream

    # Call with [fast] tag
    await handler._stream_tts_to_queue("Hello", tags=["fast"])

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    vs = payload["voice_settings"]
    # [fast]: stability -0.1, similarity_boost +0.2 from base (0.5, 0.75)
    assert vs["stability"] == 0.4
    assert vs["similarity_boost"] == 0.95


@pytest.mark.asyncio
async def test_synthesize_applies_tags_before_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthesize extracts tags and passes them to _stream_tts_to_queue."""
    from robot_comic import llama_elevenlabs_tts as mod

    handler = _make_handler()

    captured_calls: list[tuple[str, list[str]]] = []

    async def fake_stream(self, text: str, first_audio_marker=None, tags=None):  # type: ignore[no-untyped-def]
        captured_calls.append((text, tags or []))
        return True

    monkeypatch.setattr(mod.LlamaElevenLabsTTSResponseHandler, "_stream_tts_to_queue", fake_stream)
    await handler._synthesize_and_enqueue("[fast] Zoom in!")

    assert captured_calls
    text, tags = captured_calls[0]
    assert text == "Zoom in!"
    assert "fast" in tags
