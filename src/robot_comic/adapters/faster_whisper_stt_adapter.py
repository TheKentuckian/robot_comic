"""FasterWhisperSTTAdapter: faster-whisper batch STT as :class:`STTBackend`.

Phase 5f of the pipeline refactor — alternate to :class:`MoonshineSTTAdapter`
for the on-robot STT slot. Issue #387 and the
``2026-05-16-moonshine-reliability-alternates-sibling-daemon.md`` memo
recommend faster-whisper ``tiny.en`` (CTranslate2 int8) as the lowest-risk
alternate to Moonshine: ~2s cold-load (vs Moonshine's ~20s), mature C++
runtime with no equivalent of Moonshine's "rearm-N-then-die" stall pattern
(#314), and a clean batch-call API.

Architecture
------------

Faster-whisper is batch-only — it transcribes a complete utterance buffer
in one call, not a streaming pipeline. The adapter chunks streaming audio
frames into utterances via `silero-vad`:

1. ``feed_audio`` receives an :class:`AudioFrame` (int16 PCM, any sample
   rate). The adapter converts to float32 mono normalised to [-1, 1] and
   resamples to silero-vad's required 16 kHz target.
2. The audio is fed into a fixed-size chunk buffer; each 512-sample chunk
   is passed through :class:`silero_vad.VADIterator`.
3. On a ``{"start": …}`` event from the VAD iterator, the adapter fires
   ``on_speech_started()`` (if registered) and begins accumulating audio
   into an utterance buffer.
4. On a ``{"end": …}`` event, the accumulated buffer is submitted to
   ``WhisperModel.transcribe`` on a background thread (via
   :func:`asyncio.to_thread` so the asyncio loop keeps draining), and the
   joined segment text fires ``on_completed(text)``.

Partial transcripts
-------------------

Faster-whisper is batch — it does not emit partial transcripts mid-utterance.
``on_partial`` is therefore **never fired** by this adapter. This is a
documented limitation; the only consumer of ``on_partial`` today is the
orchestrator's decorative ``user_partial`` output-queue publish, which
already tolerates an adapter that opts out. A future revision could
sliding-window the buffer if partials become load-bearing.

Echo guard
----------

Mirrors :class:`MoonshineSTTAdapter`'s ``should_drop_frame`` callback shape
— when truthy, ``feed_audio`` short-circuits BEFORE any conversion / VAD
processing, so TTS-playback frames don't pollute the utterance buffer.

Dependencies
------------

This adapter requires the optional ``faster_whisper_stt`` extra:

.. code-block:: bash

    uv pip install -e .[faster_whisper_stt]

If the deps aren't installed, ``start()`` raises a clean
:class:`FasterWhisperSTTDependencyError` pointing the operator at the
install command — mirrors :class:`LocalSTTDependencyError` from
``local_stt_realtime.py``.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from robot_comic.backends import (
    AudioFrame,
    TranscriptCallback,
    SpeechStartedCallback,
)


logger = logging.getLogger(__name__)


# silero-vad operates on 512-sample chunks at 16 kHz (32 ms). The VAD
# iterator buffers smaller chunks internally but a fixed 512 makes the
# adapter's accounting straightforward.
_VAD_SAMPLE_RATE = 16000
_VAD_CHUNK_SAMPLES = 512


class FasterWhisperSTTDependencyError(RuntimeError):
    """Raised when the optional ``faster_whisper_stt`` extras aren't installed."""


class FasterWhisperSTTAdapter:
    """Adapter exposing faster-whisper batch STT as :class:`STTBackend`.

    Standalone-only — there is no host-coupled mode (unlike
    :class:`MoonshineSTTAdapter` which carried a transition shape for
    Phase 5e). Future STT backends follow this shape.
    """

    def __init__(
        self,
        *,
        should_drop_frame: Callable[[], bool] | None = None,
        model_name: str = "tiny.en",
        compute_type: str = "int8",
    ) -> None:
        """Capture echo-guard + model overrides; defer all model loads to ``start()``.

        ``should_drop_frame`` — when provided, called before every
        ``feed_audio`` frame is processed. Truthy return drops the frame.
        Intended to wire the same TTS-playback echo guard
        :class:`MoonshineSTTAdapter` uses (closure over
        ``host._speaking_until``).

        ``model_name`` / ``compute_type`` — passed straight through to
        :class:`faster_whisper.WhisperModel`. Defaults match the memo's
        recommendation (``tiny.en`` int8 quantised — ~75 MB, ~1-2s cold
        load on Pi 5).
        """
        self._should_drop_frame = should_drop_frame
        self._model_name = model_name
        self._compute_type = compute_type

        self._on_completed: TranscriptCallback | None = None
        self._on_speech_started: SpeechStartedCallback | None = None
        # on_partial intentionally NOT stored — faster-whisper is batch.

        self._model: Any = None
        self._vad_iterator: Any = None
        # Pending audio that has not yet been split into 512-sample chunks.
        self._chunk_remainder: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        # Audio accumulated between VAD start and end events (float32 @ 16 kHz).
        self._utterance_buffer: list[NDArray[np.float32]] = []
        self._in_utterance: bool = False

    async def start(
        self,
        on_completed: TranscriptCallback,
        on_partial: TranscriptCallback | None = None,
        on_speech_started: SpeechStartedCallback | None = None,
    ) -> None:
        """Load the model + VAD iterator and bind transcript callbacks.

        ``on_partial`` is accepted for Protocol compatibility but is
        **ignored** — faster-whisper is batch-only and does not produce
        partial transcripts. See the module docstring.
        """
        self._on_completed = on_completed
        self._on_speech_started = on_speech_started
        if on_partial is not None:
            logger.debug(
                "FasterWhisperSTTAdapter: on_partial provided but faster-whisper "
                "is batch-only; partial transcripts will never fire."
            )

        if self._model is not None:
            # Already started — rebind callbacks but skip the load.
            return

        try:
            await asyncio.to_thread(self._load_model_and_vad)
        except Exception:
            # Roll back state so a re-attempt isn't blocked.
            self._on_completed = None
            self._on_speech_started = None
            self._model = None
            self._vad_iterator = None
            raise

    def _load_model_and_vad(self) -> None:
        """Load faster-whisper + silero-vad. Synchronous — runs in a thread."""
        try:
            from faster_whisper import WhisperModel
        except Exception as e:  # pragma: no cover — covered via stubbed tests
            raise FasterWhisperSTTDependencyError(
                "faster-whisper STT requires the optional dependency group: "
                "install with `uv pip install -e .[faster_whisper_stt]`."
            ) from e

        try:
            from silero_vad import VADIterator, load_silero_vad
        except Exception as e:  # pragma: no cover — covered via stubbed tests
            raise FasterWhisperSTTDependencyError(
                "faster-whisper STT requires silero-vad: install with `uv pip install -e .[faster_whisper_stt]`."
            ) from e

        logger.info(
            "Loading faster-whisper STT: model=%s compute_type=%s",
            self._model_name,
            self._compute_type,
        )
        self._model = WhisperModel(
            self._model_name,
            device="cpu",
            compute_type=self._compute_type,
        )
        vad_model = load_silero_vad()
        self._vad_iterator = VADIterator(vad_model, sampling_rate=_VAD_SAMPLE_RATE)

    async def feed_audio(self, frame: AudioFrame) -> None:
        """Push one audio frame through the VAD chunker.

        Consults ``should_drop_frame`` first (echo-guard) and short-circuits
        before any conversion if truthy. Otherwise: convert to float32 @ 16
        kHz, append to the chunk-remainder buffer, and process every full
        512-sample chunk through the VAD iterator.
        """
        if self._should_drop_frame is not None and self._should_drop_frame():
            return
        if self._vad_iterator is None or self._model is None:
            # start() not called or load failed — drop silently.
            return

        audio_f32 = _to_float32_mono(frame.samples, frame.sample_rate)
        if audio_f32.size == 0:
            return

        # Append + slice into 512-sample chunks.
        self._chunk_remainder = np.concatenate([self._chunk_remainder, audio_f32])
        while self._chunk_remainder.size >= _VAD_CHUNK_SAMPLES:
            chunk = self._chunk_remainder[:_VAD_CHUNK_SAMPLES]
            self._chunk_remainder = self._chunk_remainder[_VAD_CHUNK_SAMPLES:]
            await self._process_vad_chunk(chunk)

    async def _process_vad_chunk(self, chunk: NDArray[np.float32]) -> None:
        """Push one 512-sample chunk through silero-vad; dispatch on start/end."""
        event: Any = None
        try:
            event = self._vad_iterator(chunk)
        except Exception as e:
            logger.debug("silero-vad iterator raised on chunk: %s", e)
            return

        if self._in_utterance:
            self._utterance_buffer.append(chunk)

        if event is None:
            return

        if "start" in event:
            self._in_utterance = True
            # Restart the buffer with this chunk (the VAD start landmark
            # is at the START of speech in this chunk; include it).
            self._utterance_buffer = [chunk]
            await self._fire_speech_started()
        elif "end" in event:
            # The "end" event lands AFTER the trailing silence so the buffer
            # already contains the closing samples. Flush.
            buffer = self._utterance_buffer
            self._utterance_buffer = []
            self._in_utterance = False
            await self._transcribe_and_dispatch(buffer)

    async def _fire_speech_started(self) -> None:
        cb = self._on_speech_started
        if cb is None:
            return
        try:
            await cb()
        except Exception:
            logger.exception("FasterWhisperSTTAdapter on_speech_started callback raised")

    async def _transcribe_and_dispatch(self, buffer: list[NDArray[np.float32]]) -> None:
        """Run transcribe on a thread and fire ``on_completed`` with the text."""
        if not buffer:
            return
        cb = self._on_completed
        if cb is None:
            return
        audio = np.concatenate(buffer)
        try:
            text = await asyncio.to_thread(self._transcribe_sync, audio)
        except Exception:
            logger.exception("faster-whisper transcribe raised")
            return
        text = text.strip()
        if not text:
            # Mirror the moonshine partial-empty-drop behaviour: empty
            # transcripts are uninteresting noise (e.g. silero-vad fired
            # on a brief noise spike with no speech content).
            return
        try:
            await cb(text)
        except Exception:
            logger.exception("FasterWhisperSTTAdapter on_completed callback raised")

    def _transcribe_sync(self, audio: NDArray[np.float32]) -> str:
        """Run a synchronous transcribe call; invoked from the to_thread worker."""
        assert self._model is not None, "_transcribe_sync called before start()"
        # faster-whisper's transcribe returns ``(segments_iter, info)``.
        # segments_iter is a generator — iterating it triggers the actual
        # decode work. We join the .text fields with spaces; that mirrors
        # what an operator would expect to see as the final transcript.
        segments, _info = self._model.transcribe(audio, language="en")
        parts: list[str] = []
        for segment in segments:
            text = getattr(segment, "text", "")
            if isinstance(text, str) and text:
                parts.append(text.strip())
        return " ".join(parts).strip()

    async def stop(self) -> None:
        """Release model + VAD state. Idempotent."""
        model = self._model
        vad = self._vad_iterator
        self._model = None
        self._vad_iterator = None
        self._on_completed = None
        self._on_speech_started = None
        self._utterance_buffer = []
        self._chunk_remainder = np.zeros(0, dtype=np.float32)
        self._in_utterance = False

        def _close() -> None:
            for obj, methods in (
                (vad, ("reset_states",)),
                (model, ("close",)),
            ):
                if obj is None:
                    continue
                for method_name in methods:
                    method = getattr(obj, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception as e:  # pragma: no cover — best-effort
                            logger.debug(
                                "FasterWhisperSTTAdapter %s during shutdown: %s",
                                method_name,
                                e,
                            )

        if model is not None or vad is not None:
            await asyncio.to_thread(_close)

    async def reset_per_session_state(self) -> None:
        """No per-session accumulator state in this adapter — no-op.

        Phase 5c.2 hook the orchestrator fires on persona switch. The
        :class:`MoonshineSTTAdapter` analogue also relies on the
        :class:`STTBackend` Protocol default (no-op). We declare the
        method explicitly so structural :class:`STTBackend` conformance
        is unambiguous.
        """
        return None


def _to_float32_mono(samples: Any, sample_rate: int) -> NDArray[np.float32]:
    """Convert a frame's samples to float32 mono @ 16 kHz, normalised [-1, 1].

    Mirrors the conversion :class:`MoonshineListener.feed_audio` performs,
    with two differences: (1) silero-vad wants float32 not int16, and (2)
    the target sample rate is always 16 kHz regardless of source.
    """
    arr = samples
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.int16)

    # Downmix stereo / multi-channel to mono.
    if arr.ndim == 2:
        if arr.shape[1] > arr.shape[0]:
            arr = arr.T
        if arr.shape[1] > 1:
            arr = arr[:, 0]

    # int16 → float32 normalised. Other input dtypes (already float?) get
    # coerced; we trust the AudioFrame contract that samples are int16.
    if arr.dtype != np.float32:
        if np.issubdtype(arr.dtype, np.integer):
            max_val = float(np.iinfo(arr.dtype).max)
            arr = arr.astype(np.float32) / max_val
        else:
            arr = arr.astype(np.float32)

    if sample_rate != _VAD_SAMPLE_RATE and arr.size > 0:
        # Deferred import: scipy is heavy and the unit tests stub the model
        # path so they don't reach this branch in the common case.
        from scipy.signal import resample

        target_len = int(round(arr.size * _VAD_SAMPLE_RATE / sample_rate))
        arr = resample(arr, target_len).astype(np.float32, copy=False)

    # Annotate the final type for mypy — the intermediate scipy.signal.resample
    # return is loosely typed but we know we coerced to float32 above.
    out: NDArray[np.float32] = arr.astype(np.float32, copy=False)
    return out
