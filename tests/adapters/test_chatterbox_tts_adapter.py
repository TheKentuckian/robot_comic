"""Tests for ``ChatterboxTTSAdapter`` — Phase 4c.1 adapter wiring.

A stub TTS handler simulates the legacy ``_synthesize_and_enqueue`` push-to-
queue behaviour without touching real HTTP. The adapter swaps the handler's
``output_queue`` for a temp queue per call, so the stub just pushes
``(sample_rate, frame)`` tuples into ``self.output_queue`` like the real
handler does — and the tests assert that the adapter yields equivalent
:class:`AudioFrame` instances.

Differences from :mod:`tests.adapters.test_elevenlabs_tts_adapter`:

- The chatterbox handler's surface is ``_synthesize_and_enqueue(response_text,
  tts_start, target_queue)`` (no ``tags`` parameter, no ``first_audio_marker``).
- Non-tuple queue items (the legacy handler pushes ``AdditionalOutputs``
  error sentinels) are dropped by the adapter; one test pins that.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any

import pytest

from robot_comic.backends import AudioFrame
from robot_comic.adapters.chatterbox_tts_adapter import ChatterboxTTSAdapter


class _StubChatterboxHandler:
    """Mimics ChatterboxTTSResponseHandler's queue-push streaming."""

    def __init__(
        self,
        frames_to_push: list[Any] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._http: Any = None
        self._frames = list(frames_to_push or [])
        self._raise = raise_exc
        self.prepare_called = False
        self.last_text: str | None = None

    async def _prepare_startup_credentials(self) -> None:
        self.prepare_called = True

    async def _synthesize_and_enqueue(
        self,
        response_text: str,
        tts_start: float | None = None,
        target_queue: "asyncio.Queue[Any] | None" = None,
    ) -> None:
        self.last_text = response_text
        if self._raise is not None:
            raise self._raise
        for item in self._frames:
            await self.output_queue.put(item)


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_calls_handler_prepare() -> None:
    handler = _StubChatterboxHandler()
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    await adapter.prepare()
    assert handler.prepare_called is True


# ---------------------------------------------------------------------------
# synthesize() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_yields_one_audio_frame_per_pushed_item() -> None:
    """Each (sample_rate, ndarray) tuple → one AudioFrame."""
    frames: list[Any] = [
        (24000, [1, 2, 3]),
        (24000, [4, 5, 6]),
        (24000, [7, 8, 9]),
    ]
    handler = _StubChatterboxHandler(frames_to_push=frames)
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    out = [frame async for frame in adapter.synthesize("hi")]

    assert len(out) == 3
    for i, frame in enumerate(out):
        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == 24000
        assert frame.samples == frames[i][1]


@pytest.mark.asyncio
async def test_synthesize_forwards_text_to_handler() -> None:
    handler = _StubChatterboxHandler(frames_to_push=[(24000, [0])])
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    async for _ in adapter.synthesize("Hello!", tags=("fast", "annoyance")):
        pass
    # Tags are accepted for Protocol compliance and silently dropped — the
    # chatterbox handler has no channel for them today.
    assert handler.last_text == "Hello!"


# ---------------------------------------------------------------------------
# synthesize() — queue isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_does_not_leak_into_handlers_original_output_queue() -> None:
    """The handler's original ``output_queue`` must be untouched by the adapter."""
    handler = _StubChatterboxHandler(frames_to_push=[(24000, [1]), (24000, [2])])
    original_queue: asyncio.Queue[Any] = handler.output_queue
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]

    async for _ in adapter.synthesize("hi"):
        pass

    assert handler.output_queue is original_queue
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_synthesize_restores_original_queue_after_exception() -> None:
    """Even if the streaming task raises, the original queue is restored."""
    handler = _StubChatterboxHandler(raise_exc=RuntimeError("tts boom"))
    original_queue = handler.output_queue
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="tts boom"):
        async for _ in adapter.synthesize("hi"):
            pass

    assert handler.output_queue is original_queue


# ---------------------------------------------------------------------------
# synthesize() — empty / error / non-tuple paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_with_no_frames_yields_nothing() -> None:
    handler = _StubChatterboxHandler(frames_to_push=[])
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    out = [frame async for frame in adapter.synthesize("hi")]
    assert out == []


@pytest.mark.asyncio
async def test_synthesize_propagates_handler_exception() -> None:
    handler = _StubChatterboxHandler(raise_exc=ValueError("nope"))
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nope"):
        async for _ in adapter.synthesize("hi"):
            pass


@pytest.mark.asyncio
async def test_synthesize_drops_non_tuple_items() -> None:
    """``AdditionalOutputs``-shaped items don't yield audio frames.

    The legacy chatterbox handler emits an ``AdditionalOutputs`` sentinel
    when all retries fail and no audio was produced. The adapter has no
    Protocol channel for them today; dropping is the documented behaviour
    (matches the parallel TODO in :class:`ElevenLabsTTSAdapter`).
    """

    class _FakeAdditionalOutputs:
        # Stand-in for fastrtc.AdditionalOutputs to avoid importing fastrtc
        # in this unit test.
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

    handler = _StubChatterboxHandler(
        frames_to_push=[
            (24000, [1]),
            _FakeAdditionalOutputs({"role": "assistant", "content": "[TTS error]"}),
            (24000, [2]),
        ]
    )
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    out = [frame async for frame in adapter.synthesize("hi")]

    # Only the two 2-tuple items become AudioFrame instances; the
    # AdditionalOutputs-shaped item is dropped silently.
    assert len(out) == 2
    assert all(isinstance(f, AudioFrame) for f in out)
    assert [f.samples for f in out] == [[1], [2]]


# ---------------------------------------------------------------------------
# synthesize() — consumer abandonment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_cleans_up_when_consumer_breaks_early() -> None:
    """If the consumer abandons the generator, the streaming task is cancelled."""

    class _SlowHandler:
        def __init__(self) -> None:
            self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
            self._http: Any = None
            self.stream_cancelled = False

        async def _prepare_startup_credentials(self) -> None: ...

        async def _synthesize_and_enqueue(
            self,
            response_text: str,
            tts_start: float | None = None,
            target_queue: "asyncio.Queue[Any] | None" = None,
        ) -> None:
            try:
                await self.output_queue.put((24000, [1]))
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.stream_cancelled = True
                raise

    handler = _SlowHandler()
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]

    gen = adapter.synthesize("hi")
    first = await gen.__anext__()
    assert first.samples == [1]
    await gen.aclose()

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
    handler = _StubChatterboxHandler()
    handler._http = closeable
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    await adapter.shutdown()
    assert closeable.aclose_called is True
    assert handler._http is None


@pytest.mark.asyncio
async def test_shutdown_with_no_open_http_is_safe() -> None:
    handler = _StubChatterboxHandler()
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    await adapter.shutdown()
    assert handler._http is None


# ---------------------------------------------------------------------------
# Phase 5a.2 — first_audio_marker channel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_appends_first_audio_marker_on_first_frame() -> None:
    """Phase 5a.2: marker append on first yielded frame; single-shot per call."""
    frames: list[Any] = [(24000, [1]), (24000, [2]), (24000, [3])]
    handler = _StubChatterboxHandler(frames_to_push=frames)
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]

    marker: list[float] = []
    async for _ in adapter.synthesize("hi", first_audio_marker=marker):
        pass

    assert len(marker) == 1
    assert isinstance(marker[0], float)


@pytest.mark.asyncio
async def test_synthesize_does_not_touch_marker_when_none() -> None:
    """``first_audio_marker=None`` (default) → no exception."""
    handler = _StubChatterboxHandler(frames_to_push=[(24000, [0])])
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    async for _ in adapter.synthesize("hi"):
        pass


@pytest.mark.asyncio
async def test_synthesize_does_not_append_marker_when_no_frames() -> None:
    """No frames → marker stays empty (no spurious append)."""
    handler = _StubChatterboxHandler(frames_to_push=[])
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    marker: list[float] = []
    async for _ in adapter.synthesize("hi", first_audio_marker=marker):
        pass
    assert marker == []


@pytest.mark.asyncio
async def test_synthesize_drops_non_tuple_items_does_not_count_as_first_frame() -> None:
    """A leading ``AdditionalOutputs`` sentinel must NOT trigger the marker
    append — only real audio frames count as "first audio out"."""

    class _FakeAdditionalOutputs:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

    handler = _StubChatterboxHandler(
        frames_to_push=[
            _FakeAdditionalOutputs({"role": "assistant", "content": "[TTS error]"}),
            (24000, [1]),
        ]
    )
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]
    marker: list[float] = []
    out = [frame async for frame in adapter.synthesize("hi", first_audio_marker=marker)]
    assert len(out) == 1
    # Marker appended on the real audio frame, not the dropped sentinel.
    assert len(marker) == 1


# ---------------------------------------------------------------------------
# Phase 5a.2 — non-empty tags log at DEBUG
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_logs_non_empty_tags_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Phase 5a.2: when the orchestrator passes non-empty delivery tags, the
    chatterbox adapter logs at DEBUG that it dropped them — the chatterbox
    handler reads delivery from the active persona, not from caller tags.
    A future refactor can wire structured tag forwarding into the legacy
    handler; for now this log is the audit trail."""
    handler = _StubChatterboxHandler(frames_to_push=[(24000, [0])])
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]

    with caplog.at_level(logging.DEBUG, logger="robot_comic.adapters.chatterbox_tts_adapter"):
        async for _ in adapter.synthesize("hi", tags=("fast",)):
            pass

    # The log message mentions tags being dropped/forwarded gap.
    matching = [r for r in caplog.records if "tag" in r.message.lower()]
    assert matching, f"expected a DEBUG log mentioning tags; got {[r.message for r in caplog.records]}"


@pytest.mark.asyncio
async def test_synthesize_empty_tags_does_not_log_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When ``tags=()`` (the default) the chatterbox adapter stays quiet —
    no log because there's nothing being dropped."""
    handler = _StubChatterboxHandler(frames_to_push=[(24000, [0])])
    adapter = ChatterboxTTSAdapter(handler)  # type: ignore[arg-type]

    with caplog.at_level(logging.DEBUG, logger="robot_comic.adapters.chatterbox_tts_adapter"):
        async for _ in adapter.synthesize("hi"):
            pass

    matching = [r for r in caplog.records if "tag" in r.message.lower()]
    assert matching == [], f"expected no tag-related log on empty tags; got {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_satisfies_tts_backend_protocol() -> None:
    """``ChatterboxTTSAdapter`` passes ``isinstance(TTSBackend)``."""
    from robot_comic.backends import TTSBackend

    adapter = ChatterboxTTSAdapter(_StubChatterboxHandler())  # type: ignore[arg-type]
    assert isinstance(adapter, TTSBackend)
