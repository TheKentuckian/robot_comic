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


# ---------------------------------------------------------------------------
# Standalone mode (Phase 5e.1)
#
# The adapter can be constructed without a host handler. In that mode it
# owns a :class:`MoonshineListener` internally and surfaces only
# ``completed`` events to the ``STTBackend`` callback. We stub the
# listener so these tests don't need the ``moonshine_voice`` optional
# dependency installed.
# ---------------------------------------------------------------------------


class _StubMoonshineListener:
    """Mimics MoonshineListener's externally-visible surface (no model load)."""

    def __init__(self) -> None:
        self.start_called = False
        self.stop_called = False
        self.fed_frames: list[tuple[int, np.ndarray[Any, Any]]] = []
        self._on_event: Any = None

    async def start(self, on_event: Any) -> None:
        self.start_called = True
        self._on_event = on_event

    async def feed_audio(self, sample_rate: int, samples: np.ndarray[Any, Any]) -> None:
        self.fed_frames.append((sample_rate, samples))

    async def stop(self) -> None:
        self.stop_called = True

    async def fire_event(self, kind: str, text: str) -> None:
        """Test helper: pretend Moonshine fired a stream event."""
        assert self._on_event is not None, "fire_event called before start()"
        await self._on_event(kind, text)


def _install_stub_listener(monkeypatch: pytest.MonkeyPatch) -> _StubMoonshineListener:
    """Replace MoonshineListener in the adapter module with a stub.

    Returns the stub instance so tests can poke it (fire events, assert
    state). The factory always returns the SAME stub so the adapter's
    internal handle and the test's reference point at the same object.
    """
    stub = _StubMoonshineListener()

    def _factory(**_kwargs: Any) -> _StubMoonshineListener:
        return stub

    monkeypatch.setattr("robot_comic.adapters.moonshine_stt_adapter.MoonshineListener", _factory)
    return stub


def test_standalone_mode_constructor_accepts_no_handler() -> None:
    """``MoonshineSTTAdapter()`` is valid; no host handler required."""
    adapter = MoonshineSTTAdapter()
    assert adapter._standalone is True
    assert adapter._handler is None


def test_standalone_mode_satisfies_stt_backend_protocol() -> None:
    """Standalone-shape ``MoonshineSTTAdapter`` passes ``isinstance(STTBackend)``."""
    from robot_comic.backends import STTBackend

    adapter = MoonshineSTTAdapter()
    assert isinstance(adapter, STTBackend)


@pytest.mark.asyncio
async def test_standalone_start_invokes_listener_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    assert stub.start_called is True


@pytest.mark.asyncio
async def test_standalone_completed_event_routes_to_protocol_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ``completed`` events surface to the STTBackend callback."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()
    captured: list[str] = []

    async def _cb(transcript: str) -> None:
        captured.append(transcript)

    await adapter.start(_cb)
    await stub.fire_event("completed", "hello robot")

    assert captured == ["hello robot"]


@pytest.mark.asyncio
async def test_standalone_drops_started_partial_and_error_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """started / partial / error are orchestration concerns; not surfaced."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()
    captured: list[str] = []

    async def _cb(transcript: str) -> None:
        captured.append(transcript)

    await adapter.start(_cb)
    await stub.fire_event("started", "hello")
    await stub.fire_event("partial", "hello rob")
    await stub.fire_event("error", "boom")

    assert captured == []


@pytest.mark.asyncio
async def test_standalone_callback_exception_does_not_crash_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misbehaving callback must not propagate to the listener thread."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()

    async def _bad(_t: str) -> None:
        raise ValueError("user code bug")

    await adapter.start(_bad)
    # Must not raise.
    await stub.fire_event("completed", "boom")


@pytest.mark.asyncio
async def test_standalone_feed_audio_forwards_to_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    samples = np.array([1, 2, 3], dtype=np.int16)
    await adapter.feed_audio(AudioFrame(samples=samples, sample_rate=24000))

    assert len(stub.fed_frames) == 1
    sr, frame = stub.fed_frames[0]
    assert sr == 24000
    assert frame is samples


@pytest.mark.asyncio
async def test_standalone_feed_audio_coerces_list_samples_to_int16_ndarray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.feed_audio(AudioFrame(samples=[10, 20, 30], sample_rate=16000))

    sr, frame = stub.fed_frames[0]
    assert sr == 16000
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.int16
    assert list(frame) == [10, 20, 30]


@pytest.mark.asyncio
async def test_standalone_stop_calls_listener_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.stop()

    assert stub.stop_called is True


@pytest.mark.asyncio
async def test_standalone_feed_audio_before_start_drops_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frames arriving before ``start()`` must not crash.

    Regression for the Phase 5e race: ``ComposablePipeline.start_up``
    prepares LLM + TTS before calling ``stt.start()``, while
    ``console.py``'s ``record_loop`` runs concurrently and may begin
    feeding captured frames as soon as the ALSA source opens. Pre-fix
    this raised ``AssertionError`` and propagated up through
    ``record_loop``, crashing the app — observable on hardware as a
    head-slam (#371) when the daemon-driven service cycle happens to
    coincide with a real audio frame arriving.
    """
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()

    samples = np.array([1, 2, 3], dtype=np.int16)
    # No await adapter.start() — frame arrives during the start-up window.
    await adapter.feed_audio(AudioFrame(samples=samples, sample_rate=24000))

    assert stub.fed_frames == []


@pytest.mark.asyncio
async def test_standalone_feed_audio_after_stop_drops_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frames arriving after ``stop()`` must not crash.

    Same race as the pre-start case but on the teardown side. ``stop()``
    tears down the listener; a frame queued in ``record_loop`` between
    teardown and task cancellation must not assert.
    """
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    await adapter.stop()
    stub.fed_frames.clear()

    samples = np.array([4, 5, 6], dtype=np.int16)
    await adapter.feed_audio(AudioFrame(samples=samples, sample_rate=16000))

    assert stub.fed_frames == []
    assert adapter._listener is None


@pytest.mark.asyncio
async def test_standalone_stop_is_safe_when_never_started() -> None:
    """stop() before start() must not raise."""
    adapter = MoonshineSTTAdapter()
    await adapter.stop()  # must not raise
    assert adapter._listener is None


@pytest.mark.asyncio
async def test_standalone_start_failure_clears_callback_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the listener's start raises, the adapter must be re-startable."""

    class _BadListener(_StubMoonshineListener):
        async def start(self, on_event: Any) -> None:
            raise RuntimeError("listener start failed")

    bad = _BadListener()
    monkeypatch.setattr(
        "robot_comic.adapters.moonshine_stt_adapter.MoonshineListener",
        lambda **_k: bad,
    )

    adapter = MoonshineSTTAdapter()

    async def _cb(_t: str) -> None: ...

    with pytest.raises(RuntimeError, match="listener start failed"):
        await adapter.start(_cb)

    assert adapter._on_completed is None
    assert adapter._listener is None


# ---------------------------------------------------------------------------
# Standalone mode partial / speech-started routing (Phase 5e.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standalone_start_invokes_on_partial_for_partial_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``on_partial`` fires for partial events with non-empty text."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()
    partials: list[str] = []

    async def _on_completed(_t: str) -> None: ...

    async def _on_partial(t: str) -> None:
        partials.append(t)

    await adapter.start(_on_completed, on_partial=_on_partial)
    await stub.fire_event("partial", "hello ro")
    await stub.fire_event("partial", "hello robot")

    assert partials == ["hello ro", "hello robot"]


@pytest.mark.asyncio
async def test_standalone_start_invokes_on_speech_started_for_started_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``on_speech_started`` fires once per ``started`` event."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()
    started_calls = 0

    async def _on_completed(_t: str) -> None: ...

    async def _on_speech_started() -> None:
        nonlocal started_calls
        started_calls += 1

    await adapter.start(_on_completed, on_speech_started=_on_speech_started)
    await stub.fire_event("started", "")
    await stub.fire_event("started", "")

    assert started_calls == 2


@pytest.mark.asyncio
async def test_standalone_start_does_not_invoke_partial_for_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed event must NOT fire ``on_partial``."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()
    partials: list[str] = []

    async def _on_completed(_t: str) -> None: ...

    async def _on_partial(t: str) -> None:
        partials.append(t)

    await adapter.start(_on_completed, on_partial=_on_partial)
    await stub.fire_event("completed", "done")
    assert partials == []


@pytest.mark.asyncio
async def test_standalone_start_does_not_invoke_speech_started_for_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial events must NOT fire ``on_speech_started``."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()
    started_calls = 0

    async def _on_completed(_t: str) -> None: ...

    async def _on_speech_started() -> None:
        nonlocal started_calls
        started_calls += 1

    await adapter.start(_on_completed, on_speech_started=_on_speech_started)
    await stub.fire_event("partial", "in progress")
    assert started_calls == 0


@pytest.mark.asyncio
async def test_standalone_start_with_no_partial_callback_does_not_crash_on_partial_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backwards-compat: omitting ``on_partial`` must drop partial events safely."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()

    async def _on_completed(_t: str) -> None: ...

    await adapter.start(_on_completed)
    # Must not raise — adapter silently drops the partial.
    await stub.fire_event("partial", "hello")
    await stub.fire_event("started", "")


@pytest.mark.asyncio
async def test_standalone_partial_callback_drops_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty partial transcripts (Moonshine fires these at first VAD) are dropped."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()
    partials: list[str] = []

    async def _on_completed(_t: str) -> None: ...

    async def _on_partial(t: str) -> None:
        partials.append(t)

    await adapter.start(_on_completed, on_partial=_on_partial)
    await stub.fire_event("partial", "")
    assert partials == []


# ---------------------------------------------------------------------------
# should_drop_frame callback (Phase 5e.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standalone_should_drop_frame_when_callback_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``feed_audio`` does NOT forward when ``should_drop_frame`` is truthy."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter(should_drop_frame=lambda: True)

    async def _on_completed(_t: str) -> None: ...

    await adapter.start(_on_completed)
    samples = np.zeros(160, dtype=np.int16)
    await adapter.feed_audio(AudioFrame(samples=samples, sample_rate=16000))

    assert stub.fed_frames == []


@pytest.mark.asyncio
async def test_standalone_should_drop_frame_when_callback_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``feed_audio`` forwards when ``should_drop_frame`` is falsy."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter(should_drop_frame=lambda: False)

    async def _on_completed(_t: str) -> None: ...

    await adapter.start(_on_completed)
    samples = np.zeros(160, dtype=np.int16)
    await adapter.feed_audio(AudioFrame(samples=samples, sample_rate=16000))

    assert len(stub.fed_frames) == 1


@pytest.mark.asyncio
async def test_standalone_should_drop_frame_default_none_forwards_every_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a ``should_drop_frame`` callback every frame is forwarded."""
    stub = _install_stub_listener(monkeypatch)
    adapter = MoonshineSTTAdapter()  # no should_drop_frame

    async def _on_completed(_t: str) -> None: ...

    await adapter.start(_on_completed)
    samples = np.zeros(160, dtype=np.int16)
    await adapter.feed_audio(AudioFrame(samples=samples, sample_rate=16000))
    await adapter.feed_audio(AudioFrame(samples=samples, sample_rate=16000))

    assert len(stub.fed_frames) == 2


@pytest.mark.asyncio
async def test_host_coupled_mode_ignores_should_drop_frame_callback() -> None:
    """``should_drop_frame`` only applies in standalone mode; host-coupled
    forwarding still goes through the host's ``receive`` (which has its own
    echo-guard in :class:`LocalSTTInputMixin`)."""
    handler = _StubMoonshineHandler()
    # The callback says "drop everything" — host-coupled must ignore it.
    adapter = MoonshineSTTAdapter(handler, should_drop_frame=lambda: True)

    async def _cb(_t: str) -> None: ...

    await adapter.start(_cb)
    samples = np.array([1, 2, 3], dtype=np.int16)
    await adapter.feed_audio(AudioFrame(samples=samples, sample_rate=16000))

    assert len(handler.received_frames) == 1


# ---------------------------------------------------------------------------
# Backwards-compat: the host-coupled shape is unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_host_coupled_mode_is_default_when_handler_provided() -> None:
    """Providing a handler explicitly opts out of standalone mode."""
    handler = _StubMoonshineHandler()
    adapter = MoonshineSTTAdapter(handler)
    assert adapter._standalone is False
    assert adapter._handler is handler
