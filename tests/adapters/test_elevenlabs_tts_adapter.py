"""Tests for ``ElevenLabsTTSAdapter`` — Phase 3b adapter wiring.

A stub TTS handler simulates the legacy ``_stream_tts_to_queue`` push-to-queue
behaviour without touching real HTTP. The adapter swaps the handler's
``output_queue`` for a temp queue per call, so the stub just pushes
``(sample_rate, frame)`` tuples into ``self.output_queue`` like the real
handler does — and the tests assert that the adapter yields equivalent
:class:`AudioFrame` instances.
"""

from __future__ import annotations
import asyncio
from typing import Any

import pytest

from robot_comic.backends import AudioFrame
from robot_comic.adapters.elevenlabs_tts_adapter import ElevenLabsTTSAdapter


class _StubElevenLabsHandler:
    """Mimics ElevenLabsTTSResponseHandler's queue-push streaming."""

    def __init__(
        self,
        frames_to_push: list[tuple[int, Any]] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._http = None
        self._frames = list(frames_to_push or [])
        self._raise = raise_exc
        self.prepare_called = False
        self.last_text: str | None = None
        self.last_tags: list[str] | None = None

    async def _prepare_startup_credentials(self) -> None:
        self.prepare_called = True

    async def _stream_tts_to_queue(
        self,
        text: str,
        first_audio_marker: list[float] | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        self.last_text = text
        self.last_tags = tags
        if self._raise is not None:
            raise self._raise
        for sample_rate, frame in self._frames:
            await self.output_queue.put((sample_rate, frame))
        return bool(self._frames)


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_calls_handler_prepare() -> None:
    handler = _StubElevenLabsHandler()
    adapter = ElevenLabsTTSAdapter(handler)
    await adapter.prepare()
    assert handler.prepare_called is True


# ---------------------------------------------------------------------------
# synthesize() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_yields_one_audio_frame_per_pushed_item() -> None:
    """Each (sample_rate, ndarray) tuple → one AudioFrame."""
    frames = [
        (24000, [1, 2, 3]),
        (24000, [4, 5, 6]),
        (24000, [7, 8, 9]),
    ]
    handler = _StubElevenLabsHandler(frames_to_push=frames)
    adapter = ElevenLabsTTSAdapter(handler)
    out = [frame async for frame in adapter.synthesize("hi")]

    assert len(out) == 3
    for i, frame in enumerate(out):
        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == 24000
        assert frame.samples == frames[i][1]


@pytest.mark.asyncio
async def test_synthesize_forwards_text_and_tags_to_handler() -> None:
    handler = _StubElevenLabsHandler(frames_to_push=[(24000, [0])])
    adapter = ElevenLabsTTSAdapter(handler)
    async for _ in adapter.synthesize("Hello!", tags=("fast", "annoyance")):
        pass
    assert handler.last_text == "Hello!"
    assert handler.last_tags == ["fast", "annoyance"]


@pytest.mark.asyncio
async def test_synthesize_no_tags_passes_none_to_legacy_handler() -> None:
    """Empty tags tuple → ``tags=None`` for the legacy method (its default)."""
    handler = _StubElevenLabsHandler(frames_to_push=[(24000, [0])])
    adapter = ElevenLabsTTSAdapter(handler)
    async for _ in adapter.synthesize("x"):
        pass
    assert handler.last_tags is None


# ---------------------------------------------------------------------------
# synthesize() — queue isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_does_not_leak_into_handlers_original_output_queue() -> None:
    """The handler's original ``output_queue`` must be untouched by the adapter."""
    handler = _StubElevenLabsHandler(frames_to_push=[(24000, [1]), (24000, [2])])
    original_queue: asyncio.Queue[Any] = handler.output_queue
    adapter = ElevenLabsTTSAdapter(handler)

    async for _ in adapter.synthesize("hi"):
        pass

    # Original queue is restored AND empty (frames went to the temp queue).
    assert handler.output_queue is original_queue
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_synthesize_restores_original_queue_after_exception() -> None:
    """Even if the streaming task raises, the original queue is restored."""
    handler = _StubElevenLabsHandler(raise_exc=RuntimeError("tts boom"))
    original_queue = handler.output_queue
    adapter = ElevenLabsTTSAdapter(handler)

    with pytest.raises(RuntimeError, match="tts boom"):
        async for _ in adapter.synthesize("hi"):
            pass

    assert handler.output_queue is original_queue


# ---------------------------------------------------------------------------
# synthesize() — empty / error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_with_no_frames_yields_nothing() -> None:
    handler = _StubElevenLabsHandler(frames_to_push=[])
    adapter = ElevenLabsTTSAdapter(handler)
    out = [frame async for frame in adapter.synthesize("hi")]
    assert out == []


@pytest.mark.asyncio
async def test_synthesize_propagates_handler_exception() -> None:
    handler = _StubElevenLabsHandler(raise_exc=ValueError("nope"))
    adapter = ElevenLabsTTSAdapter(handler)
    with pytest.raises(ValueError, match="nope"):
        async for _ in adapter.synthesize("hi"):
            pass


# ---------------------------------------------------------------------------
# Phase 5a.2 — first_audio_marker channel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_appends_first_audio_marker_on_first_frame() -> None:
    """Phase 5a.2: the adapter appends a ``time.monotonic()`` timestamp to the
    caller-supplied ``first_audio_marker`` list on the first yielded frame.

    The orchestrator allocates ``marker: list[float] = []`` per call and
    can read ``marker[0]`` once iteration begins to record per-turn
    first-audio latency. Single-shot per call.
    """
    frames = [(24000, [1, 2, 3]), (24000, [4, 5, 6]), (24000, [7, 8, 9])]
    handler = _StubElevenLabsHandler(frames_to_push=frames)
    adapter = ElevenLabsTTSAdapter(handler)

    marker: list[float] = []
    async for _ in adapter.synthesize("hi", first_audio_marker=marker):
        pass

    assert len(marker) == 1, "marker must be appended exactly once"
    assert isinstance(marker[0], float)


@pytest.mark.asyncio
async def test_synthesize_marker_is_only_appended_once_across_frames() -> None:
    """Multiple frames in one call → still exactly one marker entry."""
    frames = [(24000, [i]) for i in range(5)]
    handler = _StubElevenLabsHandler(frames_to_push=frames)
    adapter = ElevenLabsTTSAdapter(handler)

    marker: list[float] = []
    out = [frame async for frame in adapter.synthesize("hi", first_audio_marker=marker)]
    assert len(out) == 5
    assert len(marker) == 1


@pytest.mark.asyncio
async def test_synthesize_does_not_touch_marker_when_none() -> None:
    """``first_audio_marker=None`` (default) → adapter skips the append path."""
    handler = _StubElevenLabsHandler(frames_to_push=[(24000, [0])])
    adapter = ElevenLabsTTSAdapter(handler)
    # No assertion needed beyond "no exception raised" — the test passes
    # iff the default path doesn't blow up on ``None``.
    async for _ in adapter.synthesize("hi"):
        pass


@pytest.mark.asyncio
async def test_synthesize_does_not_append_marker_when_no_frames() -> None:
    """No frames yielded → marker stays empty."""
    handler = _StubElevenLabsHandler(frames_to_push=[])
    adapter = ElevenLabsTTSAdapter(handler)
    marker: list[float] = []
    async for _ in adapter.synthesize("hi", first_audio_marker=marker):
        pass
    assert marker == []


# ---------------------------------------------------------------------------
# synthesize() — consumer abandonment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_cleans_up_when_consumer_breaks_early() -> None:
    """If the consumer abandons the generator, the streaming task is cancelled."""

    class _SlowHandler:
        def __init__(self) -> None:
            self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
            self._http = None
            self.stream_cancelled = False

        async def _prepare_startup_credentials(self) -> None: ...

        async def _stream_tts_to_queue(
            self,
            text: str,
            first_audio_marker: list[float] | None = None,
            tags: list[str] | None = None,
        ) -> bool:
            try:
                await self.output_queue.put((24000, [1]))
                # Block forever so the consumer can abandon mid-stream.
                await asyncio.sleep(60)
                return True
            except asyncio.CancelledError:
                self.stream_cancelled = True
                raise

    handler = _SlowHandler()
    adapter = ElevenLabsTTSAdapter(handler)

    gen = adapter.synthesize("hi")
    first = await gen.__anext__()
    assert first.samples == [1]
    # Abandon the generator: close it to trigger the finally block.
    await gen.aclose()

    # Give the cancellation a moment to propagate.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert handler.stream_cancelled is True


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_closes_handler_http() -> None:
    class _Closeable:
        def __init__(self) -> None:
            self.aclose_called = False

        async def aclose(self) -> None:
            self.aclose_called = True

    closeable = _Closeable()
    handler = _StubElevenLabsHandler()
    handler._http = closeable
    adapter = ElevenLabsTTSAdapter(handler)
    await adapter.shutdown()
    assert closeable.aclose_called is True
    assert handler._http is None


@pytest.mark.asyncio
async def test_shutdown_with_no_open_http_is_safe() -> None:
    handler = _StubElevenLabsHandler()
    adapter = ElevenLabsTTSAdapter(handler)
    await adapter.shutdown()
    assert handler._http is None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_satisfies_tts_backend_protocol() -> None:
    """``ElevenLabsTTSAdapter`` passes ``isinstance(TTSBackend)``."""
    from robot_comic.backends import TTSBackend

    adapter = ElevenLabsTTSAdapter(_StubElevenLabsHandler())
    assert isinstance(adapter, TTSBackend)


# ---------------------------------------------------------------------------
# Phase 4c.3 — duck-typed acceptance for GeminiTextElevenLabsResponseHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_accepts_duck_typed_gemini_elevenlabs_handler_shape() -> None:
    """Phase 4c.3: the adapter accepts any handler matching the Protocol surface.

    Mimics ``GeminiTextElevenLabsResponseHandler``'s diamond-MRO shape:
    exposes ``_prepare_startup_credentials``, ``output_queue``,
    ``_stream_tts_to_queue``, and ``_http`` without inheriting from
    ``ElevenLabsTTSResponseHandler``. No ``# type: ignore`` is needed
    because the broadened Protocol annotation accepts the structural match.
    """

    class _DuckGeminiElevenLabs:
        def __init__(self) -> None:
            self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
            self._http: Any = None
            self.prepare_called = False
            self.streamed_text: str | None = None

        async def _prepare_startup_credentials(self) -> None:
            self.prepare_called = True

        async def _stream_tts_to_queue(
            self,
            text: str,
            first_audio_marker: list[float] | None = None,
            tags: list[str] | None = None,
        ) -> bool:
            self.streamed_text = text
            await self.output_queue.put((24000, [42]))
            return True

    handler = _DuckGeminiElevenLabs()
    adapter = ElevenLabsTTSAdapter(handler)
    await adapter.prepare()
    assert handler.prepare_called is True

    out = [frame async for frame in adapter.synthesize("hello, world")]
    assert len(out) == 1
    assert out[0].sample_rate == 24000
    assert handler.streamed_text == "hello, world"

    await adapter.shutdown()
