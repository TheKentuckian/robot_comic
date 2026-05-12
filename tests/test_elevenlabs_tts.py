"""Tests for ElevenLabsTTSResponseHandler (the direct Gemini-LLM + ElevenLabs-TTS handler).

Focus is on the parts unique to this handler:
- _stream_tts_to_queue: HTTP streaming, frame chunking, first-audio telemetry,
  retries, 429 / 401 handling, voice resolution.
- _dispatch_completed_transcript's TTS-side loop: tag stripping, [short pause]
  silence injection, error AdditionalOutputs when nothing speaks.

The Gemini-LLM side is exercised indirectly via _dispatch tests; we stub the
LLM round-trip with _run_llm_with_tools mock to keep tests focused on TTS.
"""

from __future__ import annotations
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
    """Construct an ElevenLabsTTSResponseHandler with http + LLM clients mocked out."""
    from robot_comic.elevenlabs_tts import ElevenLabsTTSResponseHandler

    handler = ElevenLabsTTSResponseHandler(_make_deps())
    handler._http = MagicMock()
    handler._client = MagicMock()
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
    """A `voice_id=` line in the profile elevenlabs.txt overrides the named-voice mapping."""
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(
        mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "custom_pvc_xyz", "voice": "Rachel"}
    )
    handler = _make_handler()
    assert handler._resolve_voice_id() == "custom_pvc_xyz"


def test_resolve_voice_id_falls_back_to_named_voice_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without voice_id in profile, the named-voice override maps to the ElevenLabs catalog ID."""
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {})
    handler = _make_handler()
    handler._voice_override = "Rachel"
    assert handler._resolve_voice_id() == "21m00Tcm4ijWNoXd58YU"


def test_resolve_voice_id_returns_none_for_unknown_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognised voice name resolves to None (caller surfaces an error)."""
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {})
    handler = _make_handler()
    handler._voice_override = "NoSuchVoice"
    assert handler._resolve_voice_id() is None


# ---------------------------------------------------------------------------
# _stream_tts_to_queue — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_aligned_frames_into_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arbitrary-sized PCM chunks from the API are re-chunked into _CHUNK_SAMPLES frames."""
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    handler = _make_handler()

    # Three "weird-sized" chunks that together produce exactly 2 full frames (4800 samples).
    chunks = [
        _silent_pcm_bytes(1000),
        _silent_pcm_bytes(2000),
        _silent_pcm_bytes(1800),
    ]
    handler._http.stream = _stream_factory(_FakeStreamResponse(chunks))

    ok = await handler._stream_tts_to_queue("hi", first_audio_marker=[])
    assert ok is True

    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    # 4800 samples / 2400 per frame = 2 aligned frames; no remainder, so exactly 2 items.
    assert len(items) == 2
    for sr, frame in items:
        assert sr == 24000
        assert len(frame) == 2400


@pytest.mark.asyncio
async def test_stream_emits_trailing_partial_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trailing remainder smaller than a full frame is still flushed (not dropped)."""
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    handler = _make_handler()

    # 2400 + 100 samples → one full frame + one partial frame of 100 samples.
    handler._http.stream = _stream_factory(_FakeStreamResponse([_silent_pcm_bytes(2500)]))

    ok = await handler._stream_tts_to_queue("hi", first_audio_marker=[])
    assert ok is True

    frames = []
    while not handler.output_queue.empty():
        frames.append(handler.output_queue.get_nowait())
    assert [len(f) for _, f in frames] == [2400, 100]


@pytest.mark.asyncio
async def test_stream_fires_first_audio_telemetry_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """On the first chunk only, telemetry.record_tts_first_audio fires and the marker clears."""
    from robot_comic import telemetry
    from robot_comic import elevenlabs_tts as mod

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

    marker = [0.0]  # use 0 as "long ago" so elapsed is a real positive number
    await handler._stream_tts_to_queue("hi", first_audio_marker=marker)

    assert len(recorded) == 1
    assert recorded[0][1] == {"gen_ai.system": "elevenlabs"}
    assert recorded[0][0] > 0
    assert marker == []  # cleared so a second call in the same turn doesn't refire


@pytest.mark.asyncio
async def test_stream_does_not_fire_telemetry_without_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty first_audio_marker (already-fired or opted-out) suppresses telemetry."""
    from robot_comic import telemetry
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    handler = _make_handler()

    fired: list[Any] = []
    monkeypatch.setattr(telemetry, "record_tts_first_audio", lambda *a, **k: fired.append(a))

    handler._http.stream = _stream_factory(_FakeStreamResponse([_silent_pcm_bytes(2400)]))
    await handler._stream_tts_to_queue("hi", first_audio_marker=[])
    assert fired == []


# ---------------------------------------------------------------------------
# _stream_tts_to_queue — failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_missing_api_key_returns_false_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ELEVENLABS_API_KEY, _stream_tts_to_queue bails before issuing any HTTP request."""
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "", raising=False)
    handler = _make_handler()
    handler._http.stream = MagicMock(side_effect=AssertionError("should not be called"))

    ok = await handler._stream_tts_to_queue("hi", first_audio_marker=[0.0])
    assert ok is False


@pytest.mark.asyncio
async def test_stream_missing_voice_id_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the voice can't be resolved to an ElevenLabs ID, the stream call returns False."""
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {})
    handler = _make_handler()
    handler._voice_override = "NoSuchVoice"

    ok = await handler._stream_tts_to_queue("hi")
    assert ok is False


@pytest.mark.asyncio
async def test_stream_401_fails_fast_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 means a bad API key — bail immediately rather than burn retries."""
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    monkeypatch.setattr(mod.asyncio, "sleep", AsyncMock())  # belt-and-braces
    handler = _make_handler()

    calls = [0]

    def factory(method: str, url: str, **kw: Any) -> _FakeStreamCM:
        calls[0] += 1
        return _FakeStreamCM(_FakeStreamResponse([], status_code=401))

    handler._http.stream = factory
    ok = await handler._stream_tts_to_queue("hi")
    assert ok is False
    assert calls[0] == 1  # no retry


@pytest.mark.asyncio
async def test_stream_429_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient 429 retries with backoff and the next attempt's 200 streams audio."""
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "v"})
    monkeypatch.setattr(mod.asyncio, "sleep", AsyncMock())  # don't actually sleep in tests
    handler = _make_handler()

    handler._http.stream = _stream_factory(
        _FakeStreamResponse([], status_code=429),
        _FakeStreamResponse([_silent_pcm_bytes(2400)], status_code=200),
    )
    ok = await handler._stream_tts_to_queue("hi")
    assert ok is True
    # _last_tts_rate_limited is set on the 429 but the call ultimately succeeded.
    assert handler._last_tts_rate_limited is True


@pytest.mark.asyncio
async def test_stream_429_exhausted_marks_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    """All retries returning 429 leaves _last_tts_rate_limited set so the caller can surface it."""
    from robot_comic import elevenlabs_tts as mod

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
# _dispatch_completed_transcript — TTS-side behaviour (tag strip, silence, error)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_strips_tags_before_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """[fast]/[annoyance] tags are stripped before the spoken text hits TTS."""
    from robot_comic import elevenlabs_tts as mod

    handler = _make_handler()

    async def fake_llm(self):  # type: ignore[no-untyped-def]
        return "[fast] You hockey puck!"

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_run_llm_with_tools", fake_llm)

    captured: list[str] = []

    async def fake_stream(self, text: str, first_audio_marker=None):  # type: ignore[no-untyped-def]
        captured.append(text)
        return True

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_stream_tts_to_queue", fake_stream)
    await handler._dispatch_completed_transcript("hi")

    assert captured == ["You hockey puck!"]


@pytest.mark.asyncio
async def test_dispatch_short_pause_inserts_silence_before_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """[short pause] queues silence frames *before* invoking the TTS stream."""
    from robot_comic import elevenlabs_tts as mod

    handler = _make_handler()

    async def fake_llm(self):  # type: ignore[no-untyped-def]
        return "[short pause] Hello."

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_run_llm_with_tools", fake_llm)

    stream_call_qsize: list[int] = []

    async def fake_stream(self, text: str, first_audio_marker=None):  # type: ignore[no-untyped-def]
        # Capture queue depth at the moment TTS is invoked — silence should already be queued.
        stream_call_qsize.append(self.output_queue.qsize())
        return True

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_stream_tts_to_queue", fake_stream)
    await handler._dispatch_completed_transcript("hi")

    # First item is the assistant AdditionalOutputs, then silence frames, then the TTS call.
    assert stream_call_qsize and stream_call_qsize[0] > 1


@pytest.mark.asyncio
async def test_dispatch_all_tts_failures_pushes_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every sentence's stream call returns False, an error AdditionalOutputs is queued."""
    from robot_comic import elevenlabs_tts as mod

    handler = _make_handler()

    async def fake_llm(self):  # type: ignore[no-untyped-def]
        return "One. Two."

    async def failing_stream(self, text: str, first_audio_marker=None):  # type: ignore[no-untyped-def]
        return False

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_run_llm_with_tools", fake_llm)
    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_stream_tts_to_queue", failing_stream)

    await handler._dispatch_completed_transcript("hi")

    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    error_items = [i for i in items if isinstance(i, AdditionalOutputs) and "error" in str(i.args).lower()]
    assert error_items, f"Expected an error AdditionalOutputs, got {items!r}"


@pytest.mark.asyncio
async def test_dispatch_rate_limited_message_when_all_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """When all TTS attempts 429 out, the user-visible message names the rate-limit cause."""
    from robot_comic import elevenlabs_tts as mod

    handler = _make_handler()

    async def fake_llm(self):  # type: ignore[no-untyped-def]
        return "One."

    async def rl_stream(self, text: str, first_audio_marker=None):  # type: ignore[no-untyped-def]
        self._last_tts_rate_limited = True
        return False

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_run_llm_with_tools", fake_llm)
    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_stream_tts_to_queue", rl_stream)

    await handler._dispatch_completed_transcript("hi")

    msgs = []
    while not handler.output_queue.empty():
        item = handler.output_queue.get_nowait()
        if isinstance(item, AdditionalOutputs):
            msgs.append(str(item.args))
    assert any("rate-limited" in m for m in msgs)
