"""Tests for ``MoonshineSTTAdapter`` — Phase 3c adapter wiring.

A stub handler simulates the LocalSTTInputMixin's surface (``receive``,
``shutdown``, ``_prepare_startup_credentials``, ``_dispatch_completed_transcript``)
without loading a real Moonshine model. The adapter's dispatch-hijacking is
the part under test — the stub fires its ``_dispatch_completed_transcript``
on demand to simulate Moonshine surfacing a completed line.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pytest

from robot_comic.backends import AudioFrame
from robot_comic.adapters.moonshine_stt_adapter import MoonshineSTTAdapter


class _StubMoonshineHandler:
    """Mimics LocalSTTInputMixin's externally-visible surface."""

    def __init__(self) -> None:
        self.received_frames: list[tuple[int, np.ndarray[Any, Any]]] = []
        self.prepare_called = False
        self.shutdown_called = False
        # The mixin's _dispatch_completed_transcript is called by the Moonshine
        # listener thread; tests can fire ``simulate_transcript`` to exercise
        # the adapter's bridge.

    async def _prepare_startup_credentials(self) -> None:
        self.prepare_called = True

    async def receive(self, frame: tuple[int, np.ndarray[Any, Any]]) -> None:
        self.received_frames.append(frame)

    async def _dispatch_completed_transcript(self, transcript: str) -> None:
        """Legacy dispatch — adapter monkey-patches this in start()."""
        # Default behaviour: do nothing. The adapter replaces this with a
        # bridge to the Protocol callback.

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def simulate_transcript(self, text: str) -> None:
        """Test helper: pretend Moonshine just completed a line."""
        await self._dispatch_completed_transcript(text)


# ---------------------------------------------------------------------------
# start() + transcript bridging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_calls_handler_prepare() -> None:
    handler = _StubMoonshineHandler()
    adapter = MoonshineSTTAdapter(handler)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    assert handler.prepare_called is True


@pytest.mark.asyncio
async def test_completed_transcript_routes_to_protocol_callback() -> None:
    """The handler's dispatch is hijacked to fire the registered callback."""
    handler = _StubMoonshineHandler()
    adapter = MoonshineSTTAdapter(handler)
    captured: list[str] = []

    async def _cb(transcript: str) -> None:
        captured.append(transcript)

    await adapter.start(_cb)
    await handler.simulate_transcript("Hello robot")
    assert captured == ["Hello robot"]


@pytest.mark.asyncio
async def test_callback_exception_is_swallowed_not_propagated_to_listener() -> None:
    """A misbehaving callback must not crash the Moonshine listener path."""
    handler = _StubMoonshineHandler()
    adapter = MoonshineSTTAdapter(handler)

    async def _bad(_t: str) -> None:
        raise ValueError("user code bug")

    await adapter.start(_bad)
    # Must NOT raise: the bridge swallows callback exceptions.
    await handler.simulate_transcript("anything")


@pytest.mark.asyncio
async def test_start_rolls_back_dispatch_when_prepare_raises() -> None:
    """If ``_prepare_startup_credentials`` raises, the dispatch swap must be
    rolled back so the handler is no worse off than before start()."""

    class _BadPrepare(_StubMoonshineHandler):
        async def _prepare_startup_credentials(self) -> None:
            raise RuntimeError("prepare boom")

    handler = _BadPrepare()
    adapter = MoonshineSTTAdapter(handler)
    captured: list[str] = []

    async def _cb(t: str) -> None:
        captured.append(t)

    with pytest.raises(RuntimeError, match="prepare boom"):
        await adapter.start(_cb)

    # Dispatch is restored — transcripts fired after a failed start() must
    # NOT route to our callback (the handler should look untouched).
    await handler.simulate_transcript("post-failed-start")
    assert captured == []


@pytest.mark.asyncio
async def test_start_swaps_dispatch_and_stop_restores_legacy_behaviour() -> None:
    """During start..stop the dispatch routes to the callback; after stop it
    falls back to the handler's own behaviour (the no-op stub default here)."""
    handler = _StubMoonshineHandler()
    adapter = MoonshineSTTAdapter(handler)
    captured: list[str] = []

    async def _cb(t: str) -> None:
        captured.append(t)

    # Before start: dispatch is the legacy (stub) no-op.
    await handler.simulate_transcript("before-start")
    assert captured == []

    await adapter.start(_cb)
    # During: dispatch routes to our callback.
    await handler.simulate_transcript("during")
    assert captured == ["during"]

    await adapter.stop()
    # After stop: dispatch reverts to the legacy no-op.
    await handler.simulate_transcript("after-stop")
    assert captured == ["during"]


# ---------------------------------------------------------------------------
# feed_audio()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_audio_forwards_ndarray_frame_as_tuple() -> None:
    handler = _StubMoonshineHandler()
    adapter = MoonshineSTTAdapter(handler)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)

    samples = np.array([1, 2, 3], dtype=np.int16)
    await adapter.feed_audio(AudioFrame(samples=samples, sample_rate=16000))

    assert len(handler.received_frames) == 1
    sr, frame = handler.received_frames[0]
    assert sr == 16000
    assert frame is samples  # passed through without copy


@pytest.mark.asyncio
async def test_feed_audio_coerces_list_samples_to_int16_ndarray() -> None:
    handler = _StubMoonshineHandler()
    adapter = MoonshineSTTAdapter(handler)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)

    await adapter.feed_audio(AudioFrame(samples=[1, 2, 3], sample_rate=16000))

    sr, frame = handler.received_frames[0]
    assert sr == 16000
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.int16
    assert list(frame) == [1, 2, 3]


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_calls_handler_shutdown() -> None:
    handler = _StubMoonshineHandler()
    adapter = MoonshineSTTAdapter(handler)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.stop()
    assert handler.shutdown_called is True


@pytest.mark.asyncio
async def test_stop_restores_original_dispatch() -> None:
    """After stop, the bridge is gone — transcripts no longer reach the callback."""
    handler = _StubMoonshineHandler()
    adapter = MoonshineSTTAdapter(handler)
    captured: list[str] = []

    async def _cb(t: str) -> None:
        captured.append(t)

    await adapter.start(_cb)
    await adapter.stop()

    await handler.simulate_transcript("post-stop")
    assert captured == []


@pytest.mark.asyncio
async def test_stop_is_safe_even_if_handler_shutdown_raises() -> None:
    """A misbehaving shutdown shouldn't prevent dispatch restoration."""

    class _BadShutdown(_StubMoonshineHandler):
        async def shutdown(self) -> None:
            raise RuntimeError("shutdown boom")

    handler = _BadShutdown()
    adapter = MoonshineSTTAdapter(handler)
    captured: list[str] = []

    async def _cb(t: str) -> None:
        captured.append(t)

    await adapter.start(_cb)
    await adapter.stop()  # Must not raise (errors are best-effort-logged).

    # Dispatch is restored even though shutdown raised — transcripts no
    # longer route to our callback.
    await handler.simulate_transcript("post-bad-stop")
    assert captured == []


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_satisfies_stt_backend_protocol() -> None:
    """``MoonshineSTTAdapter`` passes ``isinstance(STTBackend)``."""
    from robot_comic.backends import STTBackend

    adapter = MoonshineSTTAdapter(_StubMoonshineHandler())
    assert isinstance(adapter, STTBackend)
