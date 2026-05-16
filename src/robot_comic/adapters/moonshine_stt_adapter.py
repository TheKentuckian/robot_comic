"""MoonshineSTTAdapter: expose Moonshine streaming STT as :class:`STTBackend`.

Two construction shapes are supported:

1. **Standalone (Phase 5e.1, target shape)**::

       adapter = MoonshineSTTAdapter()

   Owns its own :class:`MoonshineListener` — no host handler required.
   Future STT backends (faster-whisper, Distil-Whisper, Deepgram, …)
   follow this shape. The adapter forwards every audio frame straight
   to the listener and surfaces only ``"completed"`` events to the
   :class:`STTBackend` ``on_completed`` callback (started / partial /
   error are dropped; they're orchestration concerns).

   This shape is not yet used by any factory call site — subsequent
   PRs (5e.2–5e.6) migrate each composable triple onto it.

2. **Host-coupled (pre-5e.1 legacy, surviving until 5e.6)**::

       adapter = MoonshineSTTAdapter(host_handler)

   Wraps a host instance that mixes in :class:`LocalSTTInputMixin`
   (the factory-private ``_LocalSTT*Host`` shells in
   ``handler_factory.py``). The mixin still owns the Moonshine listener
   + audio ingestion + the conversation-handler concerns (turn span,
   output queue, pause controller, welcome gate, echo guard, …); the
   adapter monkey-patches ``handler._dispatch_completed_transcript`` so
   the ``STTBackend`` ``on_completed`` callback fires before the legacy
   LLM/TTS chain runs. The original method is restored in ``stop()`` so
   the handler can be re-used.

   All five composable factory builders use this shape today. They
   migrate to the standalone shape one-at-a-time in the 5e.* PRs.

## Audio frame conversion

``STTBackend.feed_audio`` receives an :class:`AudioFrame` (samples may
be ``list`` or ``numpy.ndarray``). The legacy ``receive`` expects
``(sample_rate, np.ndarray[int16])``. The adapter coerces lists to a
16-bit numpy array on the way through in either mode.
"""

from __future__ import annotations
import logging
from typing import Any, Callable

import numpy as np

from robot_comic.backends import (
    AudioFrame,
    TranscriptCallback,
    SpeechStartedCallback,
)
from robot_comic.adapters.moonshine_listener import (
    EVENT_PARTIAL,
    EVENT_STARTED,
    EVENT_COMPLETED,
    MoonshineListener,
)


logger = logging.getLogger(__name__)


class MoonshineSTTAdapter:
    """Adapter exposing Moonshine streaming STT as :class:`STTBackend`.

    See the module docstring for the two construction shapes
    (standalone vs host-coupled).
    """

    def __init__(
        self,
        handler: Any = None,
        *,
        should_drop_frame: Callable[[], bool] | None = None,
    ) -> None:
        """Construct in standalone (``handler=None``) or host-coupled mode.

        ``handler`` — when provided, must expose the
        :class:`LocalSTTInputMixin` surface
        (``_prepare_startup_credentials``, ``receive``, ``shutdown``,
        ``_dispatch_completed_transcript``). When ``None``, the adapter
        owns its own :class:`MoonshineListener` and runs in standalone
        mode.

        ``should_drop_frame`` (Phase 5e.2, standalone mode only) — when
        provided, the adapter calls this callable before forwarding
        each audio frame to the underlying listener. Returning truthy
        drops the frame. Intended to wire the echo-guard skip the
        legacy :class:`LocalSTTInputMixin.receive` did via
        ``_speaking_until``. Ignored in host-coupled mode (the mixin
        owns its own echo guard there).
        """
        self._handler = handler
        self._should_drop_frame = should_drop_frame
        self._on_completed: TranscriptCallback | None = None
        # Host-coupled mode only: tracks the original dispatch we monkey-patched.
        self._original_dispatch: Any = None
        # Standalone mode only: the owned listener.
        self._listener: MoonshineListener | None = None

    @property
    def _standalone(self) -> bool:
        """Return True iff this adapter is in standalone mode (no host)."""
        return self._handler is None

    async def start(
        self,
        on_completed: TranscriptCallback,
        on_partial: TranscriptCallback | None = None,
        on_speech_started: SpeechStartedCallback | None = None,
    ) -> None:
        """Bind transcript callbacks and initialise Moonshine state.

        See :class:`robot_comic.backends.STTBackend.start` for the
        callback contract. ``on_partial`` and ``on_speech_started`` are
        forwarded to the standalone listener; in host-coupled mode they
        are ignored (the :class:`LocalSTTInputMixin` handles speech-start
        and partial-publishing itself there).
        """
        self._on_completed = on_completed
        if self._standalone:
            await self._start_standalone(on_completed, on_partial, on_speech_started)
        else:
            await self._start_host_coupled(on_completed)

    async def _start_standalone(
        self,
        on_completed: TranscriptCallback,
        on_partial: TranscriptCallback | None,
        on_speech_started: SpeechStartedCallback | None,
    ) -> None:
        """Standalone path: own a :class:`MoonshineListener` and route events.

        - ``completed`` → fires ``on_completed(text)``.
        - ``partial`` with non-empty text → fires ``on_partial(text)`` when
          a callback is provided; dropped otherwise.
        - ``started`` → fires ``on_speech_started()`` when a callback is
          provided; dropped otherwise. The started event's text is usually
          empty and not load-bearing.
        - ``error`` → dropped (telemetry already counts these in
          :class:`MoonshineListener._StandaloneListenerBridge.on_error`).
        """
        listener = MoonshineListener()

        async def _on_event(kind: str, text: str) -> None:
            try:
                if kind == EVENT_COMPLETED:
                    await on_completed(text)
                elif kind == EVENT_PARTIAL:
                    if on_partial is not None and text:
                        await on_partial(text)
                elif kind == EVENT_STARTED:
                    if on_speech_started is not None:
                        await on_speech_started()
                # EVENT_ERROR is dropped; counted in MoonshineListener telemetry.
            except Exception:  # pragma: no cover — best-effort, never crash the listener
                logger.exception("STT %s callback raised", kind)

        try:
            await listener.start(_on_event)
        except Exception:
            # Roll back so a re-attempt isn't blocked by half-built state.
            self._on_completed = None
            raise
        self._listener = listener

    async def _start_host_coupled(self, on_completed: TranscriptCallback) -> None:
        """Host-coupled path (pre-5e.1): monkey-patch the host's dispatch hook."""
        # Replace the handler's dispatch with our callback bridge BEFORE
        # _prepare_startup_credentials runs, so any post-init events fire
        # through the bridge. If prepare raises, restore the original
        # dispatch so the handler isn't left in a half-patched state.
        self._original_dispatch = self._handler._dispatch_completed_transcript

        async def _bridge(transcript: str) -> None:
            try:
                await on_completed(transcript)
            except Exception:  # pragma: no cover — best-effort, never crash the listener
                logger.exception("STT transcript callback raised")

        self._handler._dispatch_completed_transcript = _bridge

        try:
            await self._handler._prepare_startup_credentials()
        except Exception:
            # Roll back the dispatch swap so the handler is no worse off
            # than it would have been if start() had never been called.
            self._handler._dispatch_completed_transcript = self._original_dispatch
            self._original_dispatch = None
            self._on_completed = None
            raise

    async def feed_audio(self, frame: AudioFrame) -> None:
        """Forward one captured audio frame to the underlying listener.

        In standalone mode, consults the ``should_drop_frame`` callback
        first (Phase 5e.2 echo-guard). Host-coupled mode forwards
        unconditionally — the :class:`LocalSTTInputMixin.receive` path
        has its own echo-guard skip via ``_speaking_until``.
        """
        samples = frame.samples
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.int16)
        if self._standalone:
            if self._should_drop_frame is not None and self._should_drop_frame():
                return
            assert self._listener is not None, "feed_audio called before start()"
            await self._listener.feed_audio(frame.sample_rate, samples)
        else:
            await self._handler.receive((frame.sample_rate, samples))

    async def stop(self) -> None:
        """Tear down Moonshine state and restore any monkey-patched hook."""
        if self._standalone:
            if self._listener is not None:
                try:
                    await self._listener.stop()
                except Exception as exc:  # pragma: no cover — best-effort cleanup
                    logger.warning("MoonshineSTTAdapter listener stop raised: %s", exc)
                self._listener = None
        else:
            try:
                await self._handler.shutdown()
            except Exception as exc:  # pragma: no cover — best-effort cleanup
                logger.warning("MoonshineSTTAdapter shutdown raised: %s", exc)
            if self._original_dispatch is not None:
                self._handler._dispatch_completed_transcript = self._original_dispatch
                self._original_dispatch = None
        self._on_completed = None
