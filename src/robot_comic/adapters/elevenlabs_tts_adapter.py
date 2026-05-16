"""ElevenLabsTTSAdapter: expose ElevenLabs-shaped TTS handlers as TTSBackend.

The legacy ``_stream_tts_to_queue`` pushes PCM frames into
``self.output_queue`` as ``(sample_rate, np.ndarray)`` tuples. The Phase 1
``TTSBackend.synthesize`` Protocol yields :class:`AudioFrame` objects. The
adapter bridges them by:

1. Substituting the handler's ``output_queue`` with an internal temp queue
   for the duration of one ``synthesize`` call.
2. Running ``_stream_tts_to_queue`` as a task that pushes a sentinel value
   to the temp queue on completion (success OR failure).
3. The async generator consumes the temp queue, yielding an ``AudioFrame``
   per item until it sees the sentinel, then awaits the task to propagate
   any errors.
4. The handler's original ``output_queue`` is restored in a ``finally``
   block, including the early-exit path where the consumer abandons the
   generator partway through.

The queue substitution preserves the handler's per-response state
(``_response_audio_bytes``, ``_response_start_ts``, echo-guard accounting)
because the adapter delegates to ``_enqueue_audio_frame`` indirectly via
the legacy ``_stream_tts_to_queue`` — only the *destination* queue changes.

## Duck-typed handler surface (Phase 4c.3)

The constructor accepts any object satisfying the
:class:`_ElevenLabsCompatibleHandler` Protocol below — the four-member
duck-typed surface the adapter actually uses. Today that includes:

- ``ElevenLabsTTSResponseHandler`` (and subclasses).
- ``GeminiTextElevenLabsResponseHandler`` (diamond-MRO subclass of
  ``ElevenLabsTTSResponseHandler``).
- ``LlamaElevenLabsTTSResponseHandler`` (structural match, parallel
  inheritance chain — no nominal relation to
  ``ElevenLabsTTSResponseHandler``).

Broadening from the concrete class to a Protocol pays down the
``cast(Any, legacy)`` workaround introduced in Phase 4b for the llama
variant, and unblocks the Phase 4c.3 routing for
``(moonshine, elevenlabs, gemini)`` without a new adapter.

## First-audio marker (Phase 5a.2)

The Protocol-level ``synthesize(first_audio_marker=...)`` channel is wired:
on the first yielded frame the adapter appends ``time.monotonic()`` to
the caller-supplied list. This is **orthogonal** to the legacy
``_stream_tts_to_queue(first_audio_marker=...)`` parameter — that one is
a caller-prefilled start ts the callee uses for a delta calc against
``telemetry.record_tts_first_audio``; the new Protocol channel is an
empty list the callee fills with a fresh wallclock read so the
orchestrator can observe first-audio latency without subscribing to
internal telemetry events. The legacy channel still fires from
``_stream_tts_to_queue`` for ElevenLabs-specific dashboards.
"""

from __future__ import annotations
import time
import asyncio
import logging
from typing import Any, Protocol, AsyncIterator

from robot_comic.backends import AudioFrame


logger = logging.getLogger(__name__)


# Sentinel pushed to the temp queue when the streaming task completes so the
# yielding loop can wake up and exit. A bare ``object()`` works because we
# only ever compare with ``is``.
_STREAM_DONE = object()


class _ElevenLabsCompatibleHandler(Protocol):
    """Duck-typed surface ``ElevenLabsTTSAdapter`` needs on its wrapped handler.

    Captures only the four members the adapter actually touches:

    - ``_prepare_startup_credentials()`` — awaited from ``prepare()``.
    - ``output_queue`` — read + reassigned (swapped with a temp queue for
      the duration of one ``synthesize()`` call, then restored).
    - ``_stream_tts_to_queue(text, tags=...)`` — the streaming entry point.
    - ``_http`` — optional httpx client closed in ``shutdown()``.

    ``_http`` is typed ``Any`` so the adapter doesn't need to import
    ``httpx`` just for the closeable-resource contract. The only operation
    the adapter performs on it is ``await http.aclose()`` after a
    ``getattr(..., None)`` guard.

    Not ``@runtime_checkable`` — we don't ``isinstance``-check Protocol
    matches; mypy structural typing is the only consumer.
    """

    output_queue: asyncio.Queue[Any]
    _http: Any

    async def _prepare_startup_credentials(self) -> None: ...

    async def _stream_tts_to_queue(
        self,
        text: str,
        first_audio_marker: list[float] | None = None,
        tags: list[str] | None = None,
    ) -> bool: ...


class ElevenLabsTTSAdapter:
    """Adapter exposing ElevenLabs-shaped TTS handlers as ``TTSBackend``."""

    def __init__(self, handler: "_ElevenLabsCompatibleHandler") -> None:
        """Wrap a pre-constructed handler instance."""
        self._handler = handler

    async def prepare(self) -> None:
        """Initialise the underlying handler's httpx client + Gemini client."""
        await self._handler._prepare_startup_credentials()

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
        first_audio_marker: list[float] | None = None,
    ) -> AsyncIterator[AudioFrame]:
        """Stream PCM frames for *text* as :class:`AudioFrame` instances.

        Substitutes the handler's ``output_queue`` for the call's duration so
        the legacy push-to-queue code path becomes a yielding generator
        without any changes to ``_stream_tts_to_queue`` itself.

        Phase 5a.2: ``first_audio_marker`` (when non-None) receives a single
        ``time.monotonic()`` append on the first yielded frame so the
        orchestrator can record per-turn first-audio latency. Orthogonal to
        the legacy ``_stream_tts_to_queue(first_audio_marker=...)`` channel
        — that one is a caller-prefilled start ts the callee uses for a
        delta calc; this one is an empty list the callee fills with a
        fresh wallclock read.
        """
        temp_queue: asyncio.Queue[Any] = asyncio.Queue()
        original_queue = self._handler.output_queue
        self._handler.output_queue = temp_queue
        tags_list = list(tags) if tags else None
        _marker_appended = False

        async def _stream_and_signal() -> None:
            try:
                await self._handler._stream_tts_to_queue(text, tags=tags_list)
            finally:
                # Always push the sentinel so the consumer loop can exit
                # cleanly even on error.
                try:
                    await temp_queue.put(_STREAM_DONE)
                except Exception:  # pragma: no cover — best-effort
                    pass

        task = asyncio.create_task(_stream_and_signal(), name="elevenlabs-tts-adapter")
        try:
            while True:
                item = await temp_queue.get()
                if item is _STREAM_DONE:
                    break
                sample_rate, frame = item
                if first_audio_marker is not None and not _marker_appended:
                    first_audio_marker.append(time.monotonic())
                    _marker_appended = True
                yield AudioFrame(samples=frame, sample_rate=sample_rate)
            # Propagate any error from the streaming task.
            await task
        finally:
            # If the consumer abandoned the generator before draining,
            # cancel the streaming task so it doesn't leak.
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            self._handler.output_queue = original_queue

    async def shutdown(self) -> None:
        """Close the underlying handler's httpx client if open."""
        http = getattr(self._handler, "_http", None)
        if http is not None:
            try:
                await http.aclose()
            except Exception as exc:  # pragma: no cover — best-effort cleanup
                logger.warning("ElevenLabsTTSAdapter shutdown: aclose() raised: %s", exc)
            self._handler._http = None
