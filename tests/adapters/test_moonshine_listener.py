"""Tests for the standalone :class:`MoonshineListener` (Phase 5e.1).

The listener is the host-free split of :class:`LocalSTTInputMixin`'s
STT-only pieces. Tests stub the Moonshine transcriber + stream so we
don't need the optional ``moonshine_voice`` dependency at test time.
"""

from __future__ import annotations
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from robot_comic.adapters.moonshine_listener import (
    EVENT_ERROR,
    EVENT_PARTIAL,
    EVENT_STARTED,
    EVENT_COMPLETED,
    MoonshineListener,
    _StandaloneListenerBridge,
)


# ---------------------------------------------------------------------------
# Listener bridge — verifies callback fan-out + #279 rearm flag wiring.
# ---------------------------------------------------------------------------


def _make_listener_with_stub_loop() -> tuple[MoonshineListener, list[tuple[str, str]]]:
    """Build a MoonshineListener whose _schedule_event captures events.

    The bridge tests want to verify what the listener schedules without
    needing a running asyncio loop. We replace ``_schedule_event`` with
    a synchronous capture so the bridge can fire directly from the test
    body.
    """
    listener = MoonshineListener()
    captured: list[tuple[str, str]] = []
    listener._schedule_event = lambda kind, text: captured.append((kind, text))  # type: ignore[assignment]
    return listener, captured


def test_bridge_on_line_started_fires_started_event() -> None:
    listener, captured = _make_listener_with_stub_loop()
    bridge = _StandaloneListenerBridge(listener)
    event = SimpleNamespace(line=SimpleNamespace(text="hello"))

    bridge.on_line_started(event)

    assert captured == [(EVENT_STARTED, "hello")]


def test_bridge_on_line_updated_fires_partial_event() -> None:
    listener, captured = _make_listener_with_stub_loop()
    bridge = _StandaloneListenerBridge(listener)

    bridge.on_line_updated(SimpleNamespace(line=SimpleNamespace(text="hello world")))

    assert captured == [(EVENT_PARTIAL, "hello world")]


def test_bridge_on_line_completed_fires_completed_event_and_sets_pending_rearm() -> None:
    """#279 parity: completed must flag a rearm so the next frame rebuilds the stream."""
    listener, captured = _make_listener_with_stub_loop()
    assert listener._pending_rearm is False
    bridge = _StandaloneListenerBridge(listener)

    bridge.on_line_completed(SimpleNamespace(line=SimpleNamespace(text="bye")))

    assert captured == [(EVENT_COMPLETED, "bye")]
    assert listener._pending_rearm is True


def test_bridge_on_error_fires_error_event_and_sets_pending_rearm() -> None:
    """Stream errors also wedge the C handle — rearm is the only recovery."""
    listener, captured = _make_listener_with_stub_loop()
    bridge = _StandaloneListenerBridge(listener)

    err = RuntimeError("oops")
    bridge.on_error(SimpleNamespace(error=err))

    assert len(captured) == 1
    kind, text = captured[0]
    assert kind == EVENT_ERROR
    assert "oops" in text
    assert listener._pending_rearm is True


# ---------------------------------------------------------------------------
# Event scheduling — confirms async callback fan-out via the asyncio loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_event_invokes_callback_on_running_loop() -> None:
    listener = MoonshineListener()
    captured: list[tuple[str, str]] = []

    async def _cb(kind: str, text: str) -> None:
        captured.append((kind, text))

    listener._on_event = _cb
    listener._loop = asyncio.get_running_loop()

    listener._schedule_event(EVENT_COMPLETED, "scheduled")
    # call_soon_threadsafe schedules; we need one loop tick for the
    # task to run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # nested create_task may need a second tick

    assert captured == [(EVENT_COMPLETED, "scheduled")]


@pytest.mark.asyncio
async def test_schedule_event_swallows_callback_exception() -> None:
    """A misbehaving callback must not propagate to the listener thread."""
    listener = MoonshineListener()
    listener._loop = asyncio.get_running_loop()

    async def _bad(_k: str, _t: str) -> None:
        raise ValueError("user code bug")

    listener._on_event = _bad
    listener._schedule_event(EVENT_PARTIAL, "anything")
    # Two ticks to ensure the task ran and its exception was logged.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # No raise = pass.


def test_schedule_event_noop_when_no_callback_bound() -> None:
    """Before start() (or after stop()), scheduling should silently drop."""
    listener = MoonshineListener()
    # No _on_event, no _loop.
    listener._schedule_event(EVENT_PARTIAL, "ignored")  # must not raise


# ---------------------------------------------------------------------------
# feed_audio — resample + rearm-before-push.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_audio_pushes_resampled_payload_at_target_rate() -> None:
    listener = MoonshineListener()
    stream = MagicMock()
    listener._stream = stream

    # 240 samples @ 24kHz → 160 samples @ 16kHz (Moonshine target).
    samples = np.arange(240, dtype=np.int16)
    await listener.feed_audio(24000, samples)

    stream.add_audio.assert_called_once()
    payload, sr = stream.add_audio.call_args.args
    assert sr == 16000
    assert isinstance(payload, list)
    assert len(payload) == 160


@pytest.mark.asyncio
async def test_feed_audio_rebuilds_stream_when_pending_rearm_flag_set() -> None:
    """#279 parity: the next frame after a completion must rearm before push."""
    listener = MoonshineListener()
    original_stream = MagicMock()
    rebuilt_stream = MagicMock()
    listener._stream = original_stream
    listener._pending_rearm = True

    def _fake_rearm() -> None:
        listener._stream = rebuilt_stream
        listener._pending_rearm = False

    listener._rearm_stream = _fake_rearm  # type: ignore[method-assign]

    samples = np.zeros(160, dtype=np.int16)
    await listener.feed_audio(16000, samples)

    original_stream.add_audio.assert_not_called()
    rebuilt_stream.add_audio.assert_called_once()
    assert listener._pending_rearm is False


@pytest.mark.asyncio
async def test_feed_audio_clears_rearm_flag_and_logs_when_rearm_fails() -> None:
    """A failed rearm must clear the flag to avoid a tight retry loop."""
    listener = MoonshineListener()
    listener._stream = MagicMock()
    listener._pending_rearm = True

    def _broken_rearm() -> None:
        raise RuntimeError("rearm broke")

    listener._rearm_stream = _broken_rearm  # type: ignore[method-assign]

    samples = np.zeros(160, dtype=np.int16)
    await listener.feed_audio(16000, samples)  # must not raise

    assert listener._pending_rearm is False


@pytest.mark.asyncio
async def test_feed_audio_drops_silently_when_stream_is_none() -> None:
    listener = MoonshineListener()
    listener._stream = None
    # No raise.
    await listener.feed_audio(16000, np.zeros(160, dtype=np.int16))


# ---------------------------------------------------------------------------
# stop — cleanup + idempotency.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_closes_stream_and_transcriber() -> None:
    listener = MoonshineListener()
    stream = MagicMock()
    transcriber = MagicMock()
    listener._stream = stream
    listener._transcriber = transcriber
    listener._listener = MagicMock()
    listener._on_event = lambda *_: None  # type: ignore[assignment]

    await listener.stop()

    stream.stop.assert_called_once()
    stream.close.assert_called_once()
    transcriber.close.assert_called_once()
    assert listener._stream is None
    assert listener._transcriber is None
    assert listener._listener is None
    assert listener._on_event is None


@pytest.mark.asyncio
async def test_stop_is_idempotent() -> None:
    listener = MoonshineListener()
    # First stop with nothing wired up.
    await listener.stop()  # must not raise
    # Second stop after a fake start.
    listener._stream = MagicMock()
    await listener.stop()
    # Third stop is a no-op.
    await listener.stop()  # must not raise


@pytest.mark.asyncio
async def test_stop_swallows_close_exceptions() -> None:
    """Close-time exceptions must not propagate; cleanup is best-effort."""
    listener = MoonshineListener()
    stream = MagicMock()
    stream.stop.side_effect = RuntimeError("stop boom")
    stream.close.side_effect = RuntimeError("close boom")
    transcriber = MagicMock()
    transcriber.close.side_effect = RuntimeError("xb boom")
    listener._stream = stream
    listener._transcriber = transcriber

    # Must not raise.
    await listener.stop()


# ---------------------------------------------------------------------------
# _rearm_stream — direct unit test mirroring the mixin's behaviour.
# ---------------------------------------------------------------------------


def test_rearm_stream_creates_new_stream_on_same_transcriber() -> None:
    listener = MoonshineListener()
    old_stream = MagicMock()
    new_stream = MagicMock()
    transcriber = MagicMock()
    transcriber.create_stream.return_value = new_stream
    listener._stream = old_stream
    listener._transcriber = transcriber
    listener._listener = MagicMock()
    listener._effective_update_interval = 0.42

    class _DummyBase:
        def on_line_started(self, event: Any) -> None: ...
        def on_line_updated(self, event: Any) -> None: ...
        def on_line_text_changed(self, event: Any) -> None: ...
        def on_line_completed(self, event: Any) -> None: ...
        def on_error(self, event: Any) -> None: ...

    listener._listener_base_cls = _DummyBase
    listener._pending_rearm = True

    listener._rearm_stream()

    old_stream.stop.assert_called_once()
    old_stream.close.assert_called_once()
    transcriber.create_stream.assert_called_once_with(update_interval=0.42)
    new_stream.add_listener.assert_called_once()
    new_stream.start.assert_called_once()
    assert listener._stream is new_stream
    assert listener._listener is not None
    assert listener._pending_rearm is False


def test_rearm_stream_noop_when_transcriber_already_gone() -> None:
    """A rearm racing with shutdown must not blow up."""
    listener = MoonshineListener()
    old_stream = MagicMock()
    listener._stream = old_stream
    listener._transcriber = None
    listener._pending_rearm = True

    listener._rearm_stream()

    old_stream.stop.assert_called_once()
    old_stream.close.assert_called_once()
    assert listener._stream is None
    assert listener._pending_rearm is False


# ---------------------------------------------------------------------------
# start — defers heavy build to a worker thread; safe to call once.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_calls_build_stream_in_thread_when_transcriber_unset() -> None:
    """start() must run _build_stream off the asyncio loop."""
    listener = MoonshineListener()
    build_calls: list[int] = []

    def _fake_build() -> None:
        build_calls.append(1)
        listener._transcriber = MagicMock()

    listener._build_stream = _fake_build  # type: ignore[method-assign]

    async def _cb(_k: str, _t: str) -> None: ...

    await listener.start(_cb)

    assert build_calls == [1]
    assert listener._on_event is _cb
    assert listener._loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_start_skips_build_when_already_built() -> None:
    listener = MoonshineListener()
    listener._transcriber = MagicMock()  # pretend already built

    build_calls: list[int] = []

    def _fake_build() -> None:
        build_calls.append(1)

    listener._build_stream = _fake_build  # type: ignore[method-assign]

    async def _cb(_k: str, _t: str) -> None: ...

    await listener.start(_cb)

    assert build_calls == []
    assert listener._on_event is _cb
