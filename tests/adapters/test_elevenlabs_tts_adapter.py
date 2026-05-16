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
# synthesize() — adapter-specific: text + tags forwarding
# ---------------------------------------------------------------------------
#
# Shared "prepare invokes handler", "synthesize yields AudioFrames", and
# Protocol-conformance assertions now live in
# ``tests/adapters/test_tts_backend_contract.py``. The tests below pin
# behaviour that is *specific* to the ElevenLabs adapter (text/tags
# forwarding shape, queue isolation, exception propagation, duck-typed
# handler acceptance, _http shutdown).


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
# synthesize() — error paths (empty-frames case lives in the contract suite)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_propagates_handler_exception() -> None:
    handler = _StubElevenLabsHandler(raise_exc=ValueError("nope"))
    adapter = ElevenLabsTTSAdapter(handler)
    with pytest.raises(ValueError, match="nope"):
        async for _ in adapter.synthesize("hi"):
            pass


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
