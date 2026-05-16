"""Parametric ``TTSBackend`` contract suite — test-infra #339.

Holds assertions that *every* TTS adapter must satisfy. Per-adapter test
files keep adapter-specific behaviour (Gemini's tag parsing, Chatterbox's
``AdditionalOutputs`` drop, ElevenLabs's duck-typed handler shape, etc.).

Each parametrised case constructs a fresh ``(adapter, stub_handler)`` pair
via :func:`_build_adapter`. The stubs are the same shape the per-adapter
test files use — duplicated locally here so the contract suite is self-
contained and survives per-adapter file refactors. Keep these stubs
minimal: they exist only to satisfy the contract assertions below.

If you add a new TTS adapter, register it in :data:`ADAPTERS`. If you
add a new Protocol-level requirement to ``TTSBackend``, add a new
parametrised test here — not three copies in the per-adapter files.
"""

from __future__ import annotations
import asyncio
from typing import Any

import numpy as np
import pytest

from robot_comic.backends import AudioFrame, TTSBackend
from robot_comic.adapters.gemini_tts_adapter import GeminiTTSAdapter
from robot_comic.adapters.chatterbox_tts_adapter import ChatterboxTTSAdapter
from robot_comic.adapters.elevenlabs_tts_adapter import ElevenLabsTTSAdapter


# ---------------------------------------------------------------------------
# Per-adapter stub handlers — minimal shape for contract assertions only.
# ---------------------------------------------------------------------------


class _ElevenLabsStub:
    """Mimics ElevenLabsTTSResponseHandler's queue-push streaming."""

    def __init__(self, frames: list[tuple[int, Any]] | None = None, raise_exc: Exception | None = None) -> None:
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._http: Any = None
        self._frames = list(frames or [])
        self._raise = raise_exc
        self.prepare_called = False

    async def _prepare_startup_credentials(self) -> None:
        self.prepare_called = True

    async def _stream_tts_to_queue(
        self,
        text: str,
        first_audio_marker: list[float] | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        if self._raise is not None:
            raise self._raise
        for sr, frame in self._frames:
            await self.output_queue.put((sr, frame))
        return bool(self._frames)

    # Voice-method surface (Phase 5c.1): match the lenient legacy contract —
    # ``change_voice`` stores any string and returns a confirmation, no
    # validation. Matches ``elevenlabs_tts.ElevenLabsTTSResponseHandler``.
    async def get_available_voices(self) -> list[str]:
        return ["Brian", "Adam"]

    def get_current_voice(self) -> str:
        return "Brian"

    async def change_voice(self, voice: str) -> str:
        return f"Voice changed to {voice}."


class _ChatterboxStub:
    """Mimics ChatterboxTTSResponseHandler's queue-push streaming."""

    def __init__(self, frames: list[Any] | None = None, raise_exc: Exception | None = None) -> None:
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._http: Any = None
        self._frames = list(frames or [])
        self._raise = raise_exc
        self.prepare_called = False

    async def _prepare_startup_credentials(self) -> None:
        self.prepare_called = True

    async def _synthesize_and_enqueue(
        self,
        response_text: str,
        tts_start: float | None = None,
        target_queue: "asyncio.Queue[Any] | None" = None,
    ) -> None:
        if self._raise is not None:
            raise self._raise
        for item in self._frames:
            await self.output_queue.put(item)

    # Voice-method surface (Phase 5c.1): match ChatterboxTTSResponseHandler's
    # contract — ``get_current_voice`` returns the stored voice ref,
    # ``change_voice`` sets the override and returns a confirmation.
    async def get_available_voices(self) -> list[str]:
        return ["voice_a.wav", "voice_b.wav"]

    def get_current_voice(self) -> str:
        return "voice_a.wav"

    async def change_voice(self, voice: str) -> str:
        return f"Voice changed to {voice}."


def _pcm_bytes(n_samples: int, fill: int = 0) -> bytes:
    return np.full(n_samples, fill, dtype=np.int16).tobytes()


class _GeminiStub:
    """Mimics GeminiTTSResponseHandler's bytes-returning surface."""

    def __init__(self, tts_results: list[bytes | None] | None = None, raise_exc: Exception | None = None) -> None:
        self._client: Any = None
        self._conversation_history: list[dict[str, Any]] = []
        self._tts_results = list(tts_results or [])
        self._raise = raise_exc
        self.prepare_called = False

    async def _prepare_startup_credentials(self) -> None:
        self.prepare_called = True

    async def _call_tts_with_retry(self, text: str, system_instruction: str | None = None) -> bytes | None:
        if self._raise is not None:
            raise self._raise
        if self._tts_results:
            return self._tts_results.pop(0)
        return _pcm_bytes(1200)

    async def _run_llm_with_tools(self) -> str:  # pragma: no cover — Protocol requirement
        return ""

    # Voice-method surface (Phase 5c.1): match GeminiTTSResponseHandler's
    # contract — fixed list of supported voices, ``change_voice`` stores the
    # override without validation.
    async def get_available_voices(self) -> list[str]:
        return ["Achird", "Achernar"]

    def get_current_voice(self) -> str:
        return "Achird"

    async def change_voice(self, voice: str) -> str:
        return f"Voice changed to {voice}."


# ---------------------------------------------------------------------------
# Adapter factory registry. Each entry returns (adapter, handler_stub).
# ---------------------------------------------------------------------------


def _build_elevenlabs(
    *, frames: list[tuple[int, Any]] | None = None, raise_exc: Exception | None = None
) -> tuple[Any, Any]:
    stub = _ElevenLabsStub(frames=frames, raise_exc=raise_exc)
    return ElevenLabsTTSAdapter(stub), stub  # type: ignore[arg-type]


def _build_chatterbox(
    *, frames: list[tuple[int, Any]] | None = None, raise_exc: Exception | None = None
) -> tuple[Any, Any]:
    # Chatterbox accepts the same (sr, ndarray) tuples in its output_queue.
    stub_frames: list[Any] = list(frames or [])
    stub = _ChatterboxStub(frames=stub_frames, raise_exc=raise_exc)
    return ChatterboxTTSAdapter(stub), stub  # type: ignore[arg-type]


def _build_gemini(
    *, frames: list[tuple[int, Any]] | None = None, raise_exc: Exception | None = None
) -> tuple[Any, Any]:
    # Gemini's surface is bytes; translate frame count into N PCM blobs so the
    # contract "yields N AudioFrames" test still exercises N inputs → ≥1 frames.
    if frames is None:
        tts_results: list[bytes | None] = []
    else:
        # One sentence per frame, one chunk per blob (2400 samples = one
        # _CHUNK_SAMPLES at 24 kHz).
        tts_results = [_pcm_bytes(2400, fill=1) for _ in frames]
    stub = _GeminiStub(tts_results=tts_results, raise_exc=raise_exc)
    return GeminiTTSAdapter(stub), stub  # type: ignore[arg-type]


ADAPTERS = [
    pytest.param(_build_elevenlabs, id="elevenlabs"),
    pytest.param(_build_chatterbox, id="chatterbox"),
    pytest.param(_build_gemini, id="gemini"),
]


# ---------------------------------------------------------------------------
# Contract tests — every TTSBackend must pass these.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", ADAPTERS)
def test_adapter_satisfies_tts_backend_protocol(build: Any) -> None:
    """Every adapter passes ``isinstance(adapter, TTSBackend)`` structurally."""
    adapter, _ = build()
    assert isinstance(adapter, TTSBackend)


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_prepare_invokes_handler_prepare(build: Any) -> None:
    """``await adapter.prepare()`` triggers the handler's credentials hook."""
    adapter, handler = build()
    await adapter.prepare()
    assert handler.prepare_called is True


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_synthesize_yields_audio_frame_instances(build: Any) -> None:
    """The async generator yields :class:`AudioFrame` objects at a non-zero
    sample rate. The exact frame count is adapter-specific (Gemini chunks
    internally, queue-push adapters mirror their stub's input); the contract
    is just *that* AudioFrames come out and *that* they declare a sample rate."""
    # Two input frames → at least one output frame for every adapter shape.
    frames = [(24000, [1, 2, 3]), (24000, [4, 5, 6])]
    adapter, _ = build(frames=frames)
    # Gemini ignores the protocol `tags` arg; the queue-push adapters accept
    # it and either forward (elevenlabs) or drop (chatterbox). Pass an empty
    # tuple so the contract test stays uniform.
    out = [frame async for frame in adapter.synthesize("Hello there.", tags=())]
    assert len(out) >= 1
    for frame in out:
        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate > 0


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_synthesize_with_no_input_yields_nothing(build: Any) -> None:
    """Empty input → empty output. Queue-push adapters get no stub frames;
    the Gemini adapter gets an empty string (which short-circuits before
    ``_call_tts_with_retry``)."""
    adapter, _ = build(frames=None)
    out = [frame async for frame in adapter.synthesize("")]
    assert out == []


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_synthesize_accepts_tags_kwarg_without_error(build: Any) -> None:
    """The Protocol's ``tags`` kwarg is part of the contract — every adapter
    accepts it, even if the underlying backend drops it (Chatterbox) or
    parses tags from text instead (Gemini)."""
    frames = [(24000, [1, 2, 3])]
    adapter, _ = build(frames=frames)
    # ``tags=("fast", "annoyance")`` is a realistic delivery-cue pair.
    out = [frame async for frame in adapter.synthesize("Hello.", tags=("fast", "annoyance"))]
    # At least one frame for adapters that produce audio; we only pin
    # "no exception raised on tags kwarg" here.
    assert isinstance(out, list)


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_shutdown_is_safe_with_no_open_resource(build: Any) -> None:
    """``await adapter.shutdown()`` on a fresh adapter (no I/O performed)
    does not raise. Adapter-specific shutdown semantics — closing
    ``handler._http``, no-op for Gemini — are pinned in the per-adapter
    files."""
    adapter, _ = build()
    await adapter.shutdown()  # must not raise


# ---------------------------------------------------------------------------
# Voice-method contract (Phase 5c.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_get_available_voices_returns_list_of_strings(build: Any) -> None:
    """``await adapter.get_available_voices()`` returns a ``list[str]``.

    The exact catalog is adapter-specific (Gemini has a fixed list,
    Chatterbox HTTP-fetches with fallback, ElevenLabs returns its
    constant set). The contract is just the shape."""
    adapter, _ = build()
    voices = await adapter.get_available_voices()
    assert isinstance(voices, list)
    for v in voices:
        assert isinstance(v, str)


@pytest.mark.parametrize("build", ADAPTERS)
def test_get_current_voice_returns_string(build: Any) -> None:
    """``adapter.get_current_voice()`` is sync and returns a string.

    Matches the legacy handler signatures — admin-UI call sites expect a
    synchronous read."""
    adapter, _ = build()
    voice = adapter.get_current_voice()
    assert isinstance(voice, str)


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_change_voice_returns_resolved_id(build: Any) -> None:
    """``await adapter.change_voice(voice)`` returns a confirmation string.

    All three legacy TTS handlers return ``f"Voice changed to {voice}."``;
    the contract just pins "string return", not the exact message
    format."""
    adapter, _ = build()
    result = await adapter.change_voice("SomeVoice")
    assert isinstance(result, str)


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_change_voice_to_unknown_voice_does_not_raise(build: Any) -> None:
    """The legacy contract is lenient: unknown voices are stored as-is and
    resolved at synthesis time (or the synthesis path falls back). Pin this
    so future hardening (validate-at-change) is an explicit contract change."""
    adapter, _ = build()
    result = await adapter.change_voice("definitely-not-a-real-voice-id-12345")
    assert isinstance(result, str)
