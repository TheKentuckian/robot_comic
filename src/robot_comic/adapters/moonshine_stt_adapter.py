"""MoonshineSTTAdapter: expose LocalSTTInputMixin as STTBackend.

The legacy ``LocalSTTInputMixin`` is a mixin baked into the response-handler
hierarchy (``LocalSTTLlamaElevenLabsHandler``, ``LocalSTTGeminiElevenLabsHandler``,
…). It owns the Moonshine transcriber + stream + listener, ingests audio via
``receive((sample_rate, ndarray))``, and dispatches completed transcripts
into the rest of the handler via ``_dispatch_completed_transcript(text)``.

For ``ComposablePipeline`` to drive what happens after a transcript completes,
the adapter has to intercept that dispatch. We do it by monkey-patching the
handler's bound ``_dispatch_completed_transcript`` to a thin wrapper that
calls the Protocol callback instead of the legacy LLM/TTS chain. The
original method is restored in ``stop()`` so the handler can be re-used.

## Audio frame conversion

``STTBackend.feed_audio`` receives an :class:`AudioFrame` (samples may be
``list`` or ``numpy.ndarray``). The legacy ``receive`` expects
``(sample_rate, np.ndarray[int16])``. The adapter coerces lists to a 16-bit
numpy array on the way through.
"""

from __future__ import annotations
import logging
from typing import Any

import numpy as np

from robot_comic.backends import AudioFrame, TranscriptCallback


logger = logging.getLogger(__name__)


class MoonshineSTTAdapter:
    """Adapter exposing ``LocalSTTInputMixin`` (on a host handler) as ``STTBackend``."""

    def __init__(self, handler: Any) -> None:
        """Wrap a handler instance that mixes in ``LocalSTTInputMixin``.

        ``handler`` must expose the mixin's surface: ``_prepare_startup_credentials``,
        ``receive``, ``shutdown``, and ``_dispatch_completed_transcript``.
        """
        self._handler = handler
        self._on_completed: TranscriptCallback | None = None
        self._original_dispatch: Any = None

    async def start(self, on_completed: TranscriptCallback) -> None:
        """Bind the transcript callback and initialise Moonshine state."""
        self._on_completed = on_completed
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
        """Convert the Protocol frame to the legacy receive() shape and forward."""
        samples = frame.samples
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.int16)
        await self._handler.receive((frame.sample_rate, samples))

    async def stop(self) -> None:
        """Tear down Moonshine state and restore the original dispatch method."""
        try:
            await self._handler.shutdown()
        except Exception as exc:  # pragma: no cover — best-effort cleanup
            logger.warning("MoonshineSTTAdapter shutdown raised: %s", exc)
        if self._original_dispatch is not None:
            self._handler._dispatch_completed_transcript = self._original_dispatch
            self._original_dispatch = None
        self._on_completed = None
