"""Tests for ``GeminiTTSAdapter`` — Phase 4c.5 adapter wiring.

A stub TTS handler simulates the legacy ``_call_tts_with_retry`` bytes-returning
behaviour without touching real Gemini. The adapter replicates the legacy
``_dispatch_completed_transcript`` per-sentence loop (``split_sentences`` →
``strip_gemini_tags`` → ``extract_delivery_tags`` → optional silence frame for
``[short pause]`` → ``_call_tts_with_retry`` → ``_pcm_to_frames``) and yields
:class:`AudioFrame` instances. The tests assert that behaviour against the
stub.

Differences from :mod:`tests.adapters.test_elevenlabs_tts_adapter` /
:mod:`tests.adapters.test_chatterbox_tts_adapter`:

- The Gemini TTS handler returns raw PCM ``bytes`` from
  ``_call_tts_with_retry`` rather than pushing tuples into ``output_queue``;
  the adapter chunks inline via ``_pcm_to_frames`` so there is no temp-queue
  swap.
- ``shutdown()`` is a no-op (``genai.Client`` has no explicit close path).
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pytest

from robot_comic.backends import AudioFrame
from robot_comic.gemini_tts import (
    SHORT_PAUSE_MS,
    SHORT_PAUSE_TAG,
    GEMINI_TTS_OUTPUT_SAMPLE_RATE,
    GeminiTTSResponseHandler,
)
from robot_comic.adapters.gemini_tts_adapter import GeminiTTSAdapter


def _pcm_bytes(n_samples: int, fill: int = 0) -> bytes:
    """Build *n_samples* of int16 PCM as raw bytes."""
    return np.full(n_samples, fill, dtype=np.int16).tobytes()


class _StubGeminiTTSHandler:
    """Mimics GeminiTTSResponseHandler's TTS-relevant surface.

    Captures every call to ``_call_tts_with_retry`` and returns a canned
    payload from ``tts_results`` (FIFO). If ``tts_results`` is empty, returns
    a 1200-sample (50 ms) silence blob to keep the loop progressing.
    """

    def __init__(
        self,
        tts_results: list[bytes | None] | None = None,
    ) -> None:
        self._client: Any = None
        self._conversation_history: list[dict[str, Any]] = []
        # Adapter does not touch output_queue but the Protocol declares it.
        self._output_queue_marker: object = object()
        self._tts_results: list[bytes | None] = list(tts_results or [])
        self.prepare_called = False
        self.tts_calls: list[tuple[str, str | None]] = []

    async def _prepare_startup_credentials(self) -> None:
        self.prepare_called = True

    async def _call_tts_with_retry(
        self, text: str, system_instruction: str | None = None
    ) -> bytes | None:
        self.tts_calls.append((text, system_instruction))
        if self._tts_results:
            return self._tts_results.pop(0)
        return _pcm_bytes(1200)  # 50 ms at 24 kHz

    async def _run_llm_with_tools(self) -> str:  # pragma: no cover — required by Protocol
        return ""


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_calls_handler_prepare() -> None:
    handler = _StubGeminiTTSHandler()
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]
    await adapter.prepare()
    assert handler.prepare_called is True


# ---------------------------------------------------------------------------
# synthesize() — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_yields_audio_frames_for_one_sentence() -> None:
    """One sentence → one TTS call → N chunks → N AudioFrames at 24 kHz."""
    # 4800 samples = 200 ms; with _CHUNK_SAMPLES=2400 → 2 chunks.
    pcm = _pcm_bytes(4800, fill=1)
    handler = _StubGeminiTTSHandler(tts_results=[pcm])
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]

    out = [frame async for frame in adapter.synthesize("Hello there.")]

    assert len(out) == 2
    for frame in out:
        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == GEMINI_TTS_OUTPUT_SAMPLE_RATE
    assert len(handler.tts_calls) == 1
    assert handler.tts_calls[0][0] == "Hello there."


@pytest.mark.asyncio
async def test_synthesize_yields_audio_frames_for_multiple_sentences() -> None:
    """Multi-sentence text → multiple TTS calls; total frames concatenate."""
    handler = _StubGeminiTTSHandler(
        tts_results=[_pcm_bytes(2400), _pcm_bytes(2400), _pcm_bytes(2400)]
    )
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]

    out = [
        frame
        async for frame in adapter.synthesize(
            "First sentence. Second sentence. Third sentence."
        )
    ]

    assert len(out) == 3
    assert len(handler.tts_calls) == 3
    spoken = [c[0] for c in handler.tts_calls]
    assert "First sentence." in spoken[0]
    assert "Second sentence." in spoken[1]
    assert "Third sentence." in spoken[2]


@pytest.mark.asyncio
async def test_synthesize_strips_gemini_tags_from_spoken_text() -> None:
    """Delivery tags (``[fast]``, ``[annoyance]``) are stripped from the
    *spoken* text passed to ``_call_tts_with_retry``."""
    handler = _StubGeminiTTSHandler(tts_results=[_pcm_bytes(2400)])
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]

    [_ async for _ in adapter.synthesize("[fast] Hello [annoyance] world.")]

    assert len(handler.tts_calls) == 1
    spoken, _ = handler.tts_calls[0]
    assert "[fast]" not in spoken
    assert "[annoyance]" not in spoken
    assert "Hello" in spoken
    assert "world" in spoken


@pytest.mark.asyncio
async def test_synthesize_inserts_silence_for_short_pause_tag() -> None:
    """``[short pause]`` tag emits silence frames *before* the spoken-text frames."""
    pcm = _pcm_bytes(2400, fill=7)  # one 100 ms chunk of non-zero
    handler = _StubGeminiTTSHandler(tts_results=[pcm])
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]

    out = [
        frame
        async for frame in adapter.synthesize("[short pause] Now I speak.")
    ]

    # silence_pcm at SHORT_PAUSE_MS (400 ms) = 9600 samples → 4 chunks of
    # _CHUNK_SAMPLES=2400. Plus 1 chunk of spoken audio = 5 frames total.
    expected_silence_samples = int(
        GEMINI_TTS_OUTPUT_SAMPLE_RATE * SHORT_PAUSE_MS / 1000
    )
    expected_silence_chunks = (expected_silence_samples + 2400 - 1) // 2400
    assert len(out) == expected_silence_chunks + 1

    # Silence frames must come first; the final frame is the spoken audio.
    for frame in out[:expected_silence_chunks]:
        samples = np.asarray(frame.samples)
        assert np.all(samples == 0)
    spoken_frame = np.asarray(out[-1].samples)
    assert np.any(spoken_frame != 0)
    # SHORT_PAUSE_TAG is documented in the source; sanity that the test
    # exercises the expected tag.
    assert SHORT_PAUSE_TAG == "short pause"


@pytest.mark.asyncio
async def test_synthesize_forwards_delivery_tags_to_system_instruction() -> None:
    """A ``[fast]`` tag injects the corresponding delivery cue into the
    ``system_instruction`` arg of ``_call_tts_with_retry``.

    ``build_tts_system_instruction`` appends a ``Delivery cues for this line:``
    suffix when at least one (non-short-pause) tag is present. The test pins
    the presence of that suffix to avoid coupling to the base persona
    instruction text (which already contains words like ``rapid-fire``).
    """
    handler = _StubGeminiTTSHandler(tts_results=[_pcm_bytes(2400)])
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]

    [_ async for _ in adapter.synthesize("[fast] Punch line.")]

    assert len(handler.tts_calls) == 1
    _, instruction = handler.tts_calls[0]
    assert instruction is not None
    assert "Delivery cues for this line:" in instruction


# ---------------------------------------------------------------------------
# synthesize() — error / empty paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_skips_sentence_when_tts_returns_none() -> None:
    """``_call_tts_with_retry`` returning ``None`` → no frame emitted; loop continues."""
    handler = _StubGeminiTTSHandler(tts_results=[None, _pcm_bytes(2400)])
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]

    out = [
        frame
        async for frame in adapter.synthesize("First. Second.")
    ]

    # First sentence → None → 0 frames. Second sentence → 1 frame.
    assert len(out) == 1
    assert len(handler.tts_calls) == 2


@pytest.mark.asyncio
async def test_synthesize_with_empty_text_yields_nothing() -> None:
    handler = _StubGeminiTTSHandler()
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]
    out = [frame async for frame in adapter.synthesize("")]
    assert out == []
    # Empty text → no TTS calls.
    assert handler.tts_calls == []


@pytest.mark.asyncio
async def test_synthesize_ignores_protocol_tags_arg() -> None:
    """The Protocol's ``tags`` kwarg is accepted and silently ignored — the
    adapter parses tags from the LLM text itself (where the persona prompt
    embeds them)."""
    handler = _StubGeminiTTSHandler(tts_results=[_pcm_bytes(2400)])
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]

    # ``tags=("fast",)`` would inject a delivery cue if the adapter honoured
    # it — but the text has no tag markers, so the resulting instruction
    # should NOT mention rapid-fire.
    [_ async for _ in adapter.synthesize("Plain sentence.", tags=("fast",))]

    assert len(handler.tts_calls) == 1
    _, instruction = handler.tts_calls[0]
    # No tag in the text → adapter computed an empty tag list → the cue
    # suffix from ``build_tts_system_instruction`` is absent. Pin this via
    # the suffix marker rather than vocabulary that may overlap with the
    # base persona instruction.
    assert instruction is not None
    assert "Delivery cues for this line:" not in instruction


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_is_noop() -> None:
    """``shutdown()`` does not raise and does not touch the handler."""
    handler = _StubGeminiTTSHandler()
    adapter = GeminiTTSAdapter(handler)  # type: ignore[arg-type]
    await adapter.shutdown()
    # No flag set; no exception.
    assert handler.prepare_called is False


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_satisfies_tts_backend_protocol() -> None:
    """``GeminiTTSAdapter`` passes ``isinstance(TTSBackend)``."""
    from robot_comic.backends import TTSBackend

    adapter = GeminiTTSAdapter(_StubGeminiTTSHandler())  # type: ignore[arg-type]
    assert isinstance(adapter, TTSBackend)


def test_real_pcm_to_frames_static_method_chunking() -> None:
    """Sanity that the real ``_pcm_to_frames`` static (used by the adapter)
    chunks at 2400 samples (100 ms at 24 kHz)."""
    # 6000 samples → ceil(6000/2400) = 3 chunks of sizes [2400, 2400, 1200].
    chunks = GeminiTTSResponseHandler._pcm_to_frames(_pcm_bytes(6000))
    assert len(chunks) == 3
    assert len(chunks[0]) == 2400
    assert len(chunks[1]) == 2400
    assert len(chunks[2]) == 1200
