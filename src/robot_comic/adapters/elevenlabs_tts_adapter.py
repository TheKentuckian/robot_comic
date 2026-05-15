"""ElevenLabsTTSAdapter: expose ElevenLabsTTSResponseHandler as TTSBackend.

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

## Known gap

The legacy ``_stream_tts_to_queue`` accepts a ``first_audio_marker:
list[float] | None`` parameter for echo-guard / first-audio-latency
telemetry. The adapter doesn't surface it — Protocol-level ``synthesize``
has no analogous channel yet. Phase 4 will plumb it through
:class:`LLMResponse.metadata` alongside the ``tags`` work, at which point
the adapter can forward both.
"""

from __future__ import annotations
import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from robot_comic.backends import AudioFrame


if TYPE_CHECKING:
    from robot_comic.elevenlabs_tts import ElevenLabsTTSResponseHandler


logger = logging.getLogger(__name__)


# Sentinel pushed to the temp queue when the streaming task completes so the
# yielding loop can wake up and exit. A bare ``object()`` works because we
# only ever compare with ``is``.
_STREAM_DONE = object()


class ElevenLabsTTSAdapter:
    """Adapter exposing ``ElevenLabsTTSResponseHandler`` as ``TTSBackend``."""

    def __init__(self, handler: "ElevenLabsTTSResponseHandler") -> None:
        """Wrap a pre-constructed handler instance."""
        self._handler = handler

    async def prepare(self) -> None:
        """Initialise the underlying handler's httpx client + Gemini client."""
        await self._handler._prepare_startup_credentials()

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
    ) -> AsyncIterator[AudioFrame]:
        """Stream PCM frames for *text* as :class:`AudioFrame` instances.

        Substitutes the handler's ``output_queue`` for the call's duration so
        the legacy push-to-queue code path becomes a yielding generator
        without any changes to ``_stream_tts_to_queue`` itself.
        """
        temp_queue: asyncio.Queue[Any] = asyncio.Queue()
        original_queue = self._handler.output_queue
        self._handler.output_queue = temp_queue
        tags_list = list(tags) if tags else None

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
