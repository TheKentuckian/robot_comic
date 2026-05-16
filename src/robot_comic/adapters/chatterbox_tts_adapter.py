"""ChatterboxTTSAdapter: expose ChatterboxTTSResponseHandler as TTSBackend.

The legacy ``_synthesize_and_enqueue`` pushes PCM frames into
``self.output_queue`` as ``(sample_rate, np.ndarray)`` tuples (and the
occasional ``AdditionalOutputs`` error sentinel when synthesis fails). The
Phase 1 ``TTSBackend.synthesize`` Protocol yields :class:`AudioFrame`
objects. The adapter bridges them by:

1. Substituting the handler's ``output_queue`` with an internal temp queue
   for the duration of one ``synthesize`` call.
2. Running ``_synthesize_and_enqueue`` as a task that pushes a sentinel value
   to the temp queue on completion (success OR failure).
3. The async generator consumes the temp queue, yielding an ``AudioFrame``
   per ``(sample_rate, frame)`` tuple. Non-tuple items (e.g. fastrtc
   ``AdditionalOutputs`` sentinels emitted on synthesis failure) are
   dropped — see "Known gap" below.
4. The handler's original ``output_queue`` is restored in a ``finally``
   block, including the early-exit path where the consumer abandons the
   generator partway through.

The queue substitution preserves the handler's per-response state
(persona-driven exaggeration / cfg-weight, voice-clone reference, auto-gain
and target-dBFS knobs) because the adapter delegates to
``_synthesize_and_enqueue`` → ``_call_chatterbox_tts`` → ``_wav_to_pcm``
unchanged — only the *destination* queue changes.

## Auto-gain / target-dBFS

``ChatterboxTTSResponseHandler`` reads the gain knobs from ``config.py`` via
properties (``_auto_gain_enabled``, ``_target_dbfs``, ``_gain``). The
adapter delegates to ``_synthesize_and_enqueue`` so those properties keep
firing — operator runtime changes to ``REACHY_MINI_CHATTERBOX_*`` env vars
are picked up the next time the legacy handler reads them, just like
today.

## Known gaps

- **No tag forwarding into the chatterbox handler.** The chatterbox handler
  does not accept tags; per-segment delivery is driven by
  ``chatterbox_tag_translator.translate`` using the active persona. The
  adapter accepts ``tags`` for Protocol compliance and drops them — Phase
  5a.2 logs non-empty tags at DEBUG so future audits can spot when the
  orchestrator starts routing structured cues through this triple. A
  future PR may retrofit the legacy handler to read structured tags
  alongside the persona-driven path.
- **`AdditionalOutputs` items dropped.** When chatterbox can't synthesise
  (all retries failed), the legacy handler pushes a fastrtc
  ``AdditionalOutputs({"role": "assistant", "content": "[TTS error]"})``
  sentinel for UI surfacing. The Protocol has no metadata channel for
  these yet; the adapter drops them so they don't fail the
  ``isinstance(item, tuple)`` unpack. Mirrors the parallel gap in
  :class:`ElevenLabsTTSAdapter` — a future PR may plumb both through a
  ``TTSBackend`` event channel.

## First-audio marker (Phase 5a.2)

The Protocol-level ``synthesize(first_audio_marker=...)`` channel is wired:
on the first real audio frame (``(sample_rate, ndarray)`` tuple) the adapter
appends ``time.monotonic()`` to the caller-supplied list. ``AdditionalOutputs``
sentinels do not count — only PCM frames trigger the append.
"""

from __future__ import annotations
import time
import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from robot_comic.backends import AudioFrame


if TYPE_CHECKING:
    from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler


logger = logging.getLogger(__name__)


# Sentinel pushed to the temp queue when the streaming task completes so the
# yielding loop can wake up and exit. A bare ``object()`` works because we
# only ever compare with ``is``.
_STREAM_DONE = object()


class ChatterboxTTSAdapter:
    """Adapter exposing ``ChatterboxTTSResponseHandler`` as ``TTSBackend``."""

    def __init__(self, handler: "ChatterboxTTSResponseHandler") -> None:
        """Wrap a pre-constructed handler instance."""
        self._handler = handler

    async def prepare(self) -> None:
        """Initialise the underlying handler's httpx client + warm the voice model."""
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
        without any changes to ``_synthesize_and_enqueue`` itself.

        ``tags`` is accepted for Protocol compliance. The chatterbox handler
        derives per-segment delivery from the active persona via
        :func:`chatterbox_tag_translator.translate`, not from externally-supplied
        tags, so the param is dropped. Phase 5a.2: a non-empty ``tags``
        param is logged at DEBUG so future audits can spot when the
        orchestrator starts routing structured tags through this triple
        (the wire is open; the legacy handler just doesn't read from it).

        Phase 5a.2: ``first_audio_marker`` (when non-None) receives a
        single ``time.monotonic()`` append on the first real audio frame
        yielded. ``AdditionalOutputs`` sentinels (dropped) do not count.
        """
        if tags:
            logger.debug(
                "ChatterboxTTSAdapter: dropping delivery tags %r; legacy "
                "handler reads from active persona via "
                "chatterbox_tag_translator.translate",
                tags,
            )
        temp_queue: asyncio.Queue[Any] = asyncio.Queue()
        original_queue = self._handler.output_queue
        self._handler.output_queue = temp_queue
        _marker_appended = False

        async def _stream_and_signal() -> None:
            try:
                await self._handler._synthesize_and_enqueue(text)
            finally:
                # Always push the sentinel so the consumer loop can exit
                # cleanly even on error.
                try:
                    await temp_queue.put(_STREAM_DONE)
                except Exception:  # pragma: no cover — best-effort
                    pass

        task = asyncio.create_task(_stream_and_signal(), name="chatterbox-tts-adapter")
        try:
            while True:
                item = await temp_queue.get()
                if item is _STREAM_DONE:
                    break
                # Chatterbox emits two kinds of items:
                # 1. (sample_rate, frame) tuples — the normal PCM path.
                # 2. fastrtc.AdditionalOutputs(...) sentinels on synthesis
                #    failure. The Protocol has no channel for those yet
                #    (see module docstring) — drop them silently.
                if isinstance(item, tuple) and len(item) == 2:
                    sample_rate, frame = item
                    if first_audio_marker is not None and not _marker_appended:
                        first_audio_marker.append(time.monotonic())
                        _marker_appended = True
                    yield AudioFrame(samples=frame, sample_rate=sample_rate)
                # else: AdditionalOutputs-shaped sentinel — dropped.
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
                logger.warning("ChatterboxTTSAdapter shutdown: aclose() raised: %s", exc)
            self._handler._http = None

    # ------------------------------------------------------------------ #
    # Voice methods (Phase 5c.1) — thin forwards to the wrapped handler. #
    # ------------------------------------------------------------------ #

    async def get_available_voices(self) -> list[str]:
        """Forward to ``ChatterboxTTSResponseHandler.get_available_voices``.

        The legacy method HTTP-fetches the predefined-voices catalog from
        the running Chatterbox server (falling back to the current voice
        on error); the adapter has no opinion on that — it just bridges
        the Protocol surface.
        """
        return await self._handler.get_available_voices()

    def get_current_voice(self) -> str:
        """Forward to ``ChatterboxTTSResponseHandler.get_current_voice``."""
        return self._handler.get_current_voice()

    async def change_voice(self, voice: str) -> str:
        """Forward to ``ChatterboxTTSResponseHandler.change_voice``.

        Legacy contract: stores the value as the voice override, no
        validation against the predefined-voices catalog.
        """
        return await self._handler.change_voice(voice)
