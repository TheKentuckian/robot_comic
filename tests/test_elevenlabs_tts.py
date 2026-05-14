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


def test_resolve_voice_id_unknown_voice_falls_through_to_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognised voice name now falls through to the Adam fallback rather than None.

    The resolver in elevenlabs_voices guarantees a voice ID as long as the
    catalog is non-empty — exact match → prefix match → case-insensitive →
    fallback ("Adam") → first premade. The legacy contract of returning None
    for unknown names led to the production "[TTS error]" outage (issue #263).
    """
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {})
    handler = _make_handler()
    handler._voice_override = "NoSuchVoice"
    # Falls through to the "Adam" fallback in the hardcoded catalog.
    assert handler._resolve_voice_id() == "pNInz6obpgDQGcFmaJgB"


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
    """When the voice can't be resolved to an ElevenLabs ID, the stream call returns False.

    Post-#263 the resolver always returns *something* as long as the catalog is
    non-empty, so we have to force the empty-catalog edge here (e.g. API
    unreachable AND the hardcoded fallback also wiped) to exercise this
    defensive path in ``_stream_tts_to_queue``.
    """
    from robot_comic import elevenlabs_tts as mod

    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {})
    monkeypatch.setattr(mod, "resolve_voice_id_by_name", lambda *a, **kw: None)
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

    async def fake_stream(self, text: str, first_audio_marker=None, tags=None, target_queue=None):  # type: ignore[no-untyped-def]
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

    async def fake_stream(self, text: str, first_audio_marker=None, tags=None, target_queue=None):  # type: ignore[no-untyped-def]
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

    async def failing_stream(self, text: str, first_audio_marker=None, tags=None, target_queue=None):  # type: ignore[no-untyped-def]
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

    async def rl_stream(self, text: str, first_audio_marker=None, tags=None, target_queue=None):  # type: ignore[no-untyped-def]
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


# ---------------------------------------------------------------------------
# [Skipped TTS:] status markers must not leak into _conversation_history.
# See issue #306 — these strings are monitor-visibility cues, not LLM turns.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_tool_call_limit_marker_not_in_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """[Skipped TTS: tool-call limit reached] goes to output queue, not history."""
    from robot_comic import elevenlabs_tts as mod

    handler = _make_handler()

    async def raising_llm(self):  # type: ignore[no-untyped-def]
        raise mod._LLMToolCallLimitExceeded("tool-call limit reached")

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_run_llm_with_tools", raising_llm)

    await handler._dispatch_completed_transcript("hi")

    # Marker reaches the monitor.
    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    marker_items = [i for i in items if isinstance(i, AdditionalOutputs) and "[Skipped TTS:" in str(i.args)]
    assert marker_items, f"Expected [Skipped TTS:] AdditionalOutputs, got {items!r}"

    # … but never reaches LLM history.
    history_texts = [
        (turn.get("parts") or [{}])[0].get("text", "")
        for turn in handler._conversation_history
        if turn.get("role") == "model"
    ]
    assert all("[Skipped TTS:" not in t for t in history_texts), history_texts


@pytest.mark.asyncio
async def test_dispatch_synthetic_response_text_skipped_from_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the LLM (hypothetically) returns a marker string, it is filtered out of history."""
    from robot_comic import elevenlabs_tts as mod

    handler = _make_handler()

    async def fake_llm(self):  # type: ignore[no-untyped-def]
        # Simulate any future code path that ends up putting a status marker
        # into response_text — the guard at the history-append site stops it
        # from being persisted to _conversation_history.
        return "[Skipped TTS: empty LLM response]"

    async def fake_stream(self, text: str, first_audio_marker=None, tags=None, target_queue=None):  # type: ignore[no-untyped-def]
        return True

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_run_llm_with_tools", fake_llm)
    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_stream_tts_to_queue", fake_stream)

    await handler._dispatch_completed_transcript("hi")

    # Monitor still saw the marker.
    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    assistant_outputs = [i for i in items if isinstance(i, AdditionalOutputs) and "[Skipped TTS:" in str(i.args)]
    assert assistant_outputs

    # History only carries the user turn; no model marker.
    model_turns = [t for t in handler._conversation_history if t.get("role") == "model"]
    assert model_turns == []


@pytest.mark.asyncio
async def test_dispatch_real_llm_text_is_persisted_to_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: normal LLM-emitted content STILL lands in history."""
    from robot_comic import elevenlabs_tts as mod

    handler = _make_handler()

    async def fake_llm(self):  # type: ignore[no-untyped-def]
        return "Hello, friend."

    async def fake_stream(self, text: str, first_audio_marker=None, tags=None, target_queue=None):  # type: ignore[no-untyped-def]
        return True

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_run_llm_with_tools", fake_llm)
    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_stream_tts_to_queue", fake_stream)

    await handler._dispatch_completed_transcript("hi")

    model_turns = [
        (turn.get("parts") or [{}])[0].get("text", "")
        for turn in handler._conversation_history
        if turn.get("role") == "model"
    ]
    assert model_turns == ["Hello, friend."]


# ---------------------------------------------------------------------------
# Delivery tag → voice_settings mapping
# ---------------------------------------------------------------------------


def test_apply_voice_settings_deltas_no_tags() -> None:
    """Without tags, voice_settings are returned unchanged."""
    from robot_comic.elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.75, [])
    assert result == {"stability": 0.5, "similarity_boost": 0.75}


def test_apply_voice_settings_deltas_fast_tag() -> None:
    """[fast] tag increases similarity_boost and decreases stability."""
    from robot_comic.elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.75, ["fast"])
    assert result == {"stability": 0.4, "similarity_boost": 0.95}


def test_apply_voice_settings_deltas_annoyance_tag() -> None:
    """[annoyance] tag increases similarity_boost more and decreases stability more."""
    from robot_comic.elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.75, ["annoyance"])
    assert result == {"stability": 0.35, "similarity_boost": 1.0}  # clamped to 1.0


def test_apply_voice_settings_deltas_aggression_tag() -> None:
    """[aggression] tag further increases similarity_boost and decreases stability."""
    from robot_comic.elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.75, ["aggression"])
    assert result == {"stability": 0.3, "similarity_boost": 1.0}  # clamped to 1.0


def test_apply_voice_settings_deltas_slow_tag() -> None:
    """[slow] tag increases stability and decreases similarity_boost."""
    from robot_comic.elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.75, ["slow"])
    assert result == {"stability": 0.6, "similarity_boost": 0.65}


def test_apply_voice_settings_deltas_amusement_tag() -> None:
    """[amusement] tag moderately increases similarity_boost."""
    from robot_comic.elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.75, ["amusement"])
    assert result == {"stability": 0.5, "similarity_boost": 0.9}


def test_apply_voice_settings_deltas_enthusiasm_tag() -> None:
    """[enthusiasm] tag increases similarity_boost and slightly decreases stability."""
    from robot_comic.elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.75, ["enthusiasm"])
    assert result == {"stability": 0.45, "similarity_boost": 0.95}


def test_apply_voice_settings_deltas_multiple_tags() -> None:
    """Multiple tags combine their effects."""
    from robot_comic.elevenlabs_tts import apply_voice_settings_deltas

    result = apply_voice_settings_deltas(0.5, 0.5, ["fast", "aggression"])
    # fast: +0.2 similarity, -0.1 stability
    # aggression: +0.4 similarity, -0.2 stability
    # = 0.5 + 0.6 = 1.1 → 1.0, 0.5 - 0.3 = 0.2
    assert result == {"stability": 0.2, "similarity_boost": 1.0}


def test_apply_voice_settings_deltas_clamps_to_range() -> None:
    """Values are clamped to [0.0, 1.0] range."""
    from robot_comic.elevenlabs_tts import apply_voice_settings_deltas

    # Start with high values and apply decreasing tags
    result = apply_voice_settings_deltas(0.9, 0.9, ["fast", "aggression"])
    assert result["stability"] >= 0.0
    assert result["similarity_boost"] <= 1.0
    assert 0.0 <= result["stability"] <= 1.0
    assert 0.0 <= result["similarity_boost"] <= 1.0


@pytest.mark.asyncio
async def test_stream_tts_applies_voice_settings_from_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delivery tags are mapped to voice_settings deltas in the TTS payload."""
    from robot_comic import elevenlabs_tts as mod

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
async def test_dispatch_applies_tags_before_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch extracts tags and passes them to _stream_tts_to_queue."""
    from robot_comic import elevenlabs_tts as mod

    handler = _make_handler()

    async def fake_llm(self):  # type: ignore[no-untyped-def]
        return "[fast] Zoom in!"

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_run_llm_with_tools", fake_llm)

    captured_calls: list[tuple[str, list[str]]] = []

    async def fake_stream(self, text: str, first_audio_marker=None, tags=None, target_queue=None):  # type: ignore[no-untyped-def]
        captured_calls.append((text, tags or []))
        return True

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_stream_tts_to_queue", fake_stream)
    await handler._dispatch_completed_transcript("hi")

    assert captured_calls
    text, tags = captured_calls[0]
    assert text == "Zoom in!"
    assert "fast" in tags


# ---------------------------------------------------------------------------
# OTel instrumentation (issue #268)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_otel_spans_emitted_for_full_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end turn through ElevenLabsTTSResponseHandler emits the expected
    span hierarchy: a parent ``turn`` span with ``llm.request`` and
    ``tts.synthesize`` children carrying the documented attributes.

    Regression guard for #268, where the handler inherited no OTel
    instrumentation and the monitor saw only the ``stt.infer`` mixin span.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from robot_comic import telemetry
    from robot_comic import elevenlabs_tts as mod

    # Swap in a private TracerProvider so the test only sees its own spans
    # (the global provider is process-wide and shared with other tests).
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "get_tracer", lambda: provider.get_tracer("test"))

    # Open a synthetic outer turn span (simulating LocalSTTInputMixin's role
    # in the live path) so the handler attaches its llm.request / tts.synthesize
    # spans as children.
    outer = provider.get_tracer("test").start_span(
        "turn",
        attributes={
            "turn.id": "test-turn",
            "session.id": "test-session",
            "robot.mode": "local_stt",  # will be overwritten with the handler label
        },
    )
    handler = _make_handler()
    handler._turn_span = outer

    # Stub the LLM round-trip with a minimal "no tool calls, just text"
    # Gemini response so we exercise the real _run_llm_with_tools wrapper
    # (which is what opens the llm.request span).
    class _FakePart:
        def __init__(self, text: str) -> None:
            self.text = text
            self.function_call = None

    class _FakeContent:
        def __init__(self, text: str) -> None:
            self.parts = [_FakePart(text)]

    class _FakeCandidate:
        def __init__(self, text: str) -> None:
            self.content = _FakeContent(text)
            self.finish_reason = "STOP"
            self.safety_ratings = None

    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self.candidates = [_FakeCandidate(text)]
            self.prompt_feedback = None

    async def fake_backoff(self, contents: Any, cfg: Any) -> Any:
        return _FakeResponse("Hello there.")

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_llm_generate_with_backoff", fake_backoff)

    # Stub the TTS HTTP stream so _stream_tts_to_queue exercises its span wrap.
    monkeypatch.setattr(mod.config, "ELEVENLABS_API_KEY", "k", raising=False)
    monkeypatch.setattr(mod, "load_profile_elevenlabs_config", lambda: {"voice_id": "voice-abc"})
    handler._http.stream = _stream_factory(_FakeStreamResponse([_silent_pcm_bytes(2400)]))

    await handler._dispatch_completed_transcript("hi")

    # _dispatch_completed_transcript_impl closes the outer span as part of its
    # finally block — no need to end() it manually here.
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert names.count("turn") == 1, f"Expected exactly one turn span, got {names}"
    assert "llm.request" in names, f"Missing llm.request span; got {names}"
    assert "tts.synthesize" in names, f"Missing tts.synthesize span; got {names}"

    turn_span = next(s for s in spans if s.name == "turn")
    llm_span = next(s for s in spans if s.name == "llm.request")
    tts_span = next(s for s in spans if s.name == "tts.synthesize")

    # Attribute parity with BaseLlamaResponseHandler.
    assert turn_span.attributes is not None
    assert turn_span.attributes.get("robot.mode") == "gemini_text_elevenlabs"
    assert turn_span.attributes.get("turn.outcome") == "success"
    assert turn_span.attributes.get("gen_ai.system") == "gemini"

    assert llm_span.attributes is not None
    assert llm_span.attributes.get("gen_ai.system") == "gemini"
    assert llm_span.attributes.get("gen_ai.operation.name") == "chat"
    assert llm_span.attributes.get("finish_reason") == "STOP"

    assert tts_span.attributes is not None
    assert tts_span.attributes.get("gen_ai.system") == "elevenlabs"
    assert tts_span.attributes.get("tts.voice_id") == "voice-abc"

    # Children should belong to the outer turn's trace.
    assert llm_span.context.trace_id == turn_span.context.trace_id
    assert tts_span.context.trace_id == turn_span.context.trace_id


@pytest.mark.asyncio
async def test_otel_turn_outcome_on_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the LLM call raises, the outer turn span still closes with
    ``turn.outcome=llm_error`` and is not left open."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from robot_comic import telemetry
    from robot_comic import elevenlabs_tts as mod

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "get_tracer", lambda: provider.get_tracer("test"))

    outer = provider.get_tracer("test").start_span("turn")
    handler = _make_handler()
    handler._turn_span = outer

    async def boom(self, contents: Any, cfg: Any) -> Any:
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(mod.ElevenLabsTTSResponseHandler, "_llm_generate_with_backoff", boom)

    await handler._dispatch_completed_transcript("hi")

    spans = exporter.get_finished_spans()
    turn_spans = [s for s in spans if s.name == "turn"]
    assert len(turn_spans) == 1
    assert turn_spans[0].attributes is not None
    assert turn_spans[0].attributes.get("turn.outcome") == "llm_error"
    # Handler state should be reset so the next turn doesn't see a stale span.
    assert handler._turn_span is None
