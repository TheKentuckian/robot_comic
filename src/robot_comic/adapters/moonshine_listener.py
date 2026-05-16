"""Standalone Moonshine streaming-STT listener.

Phase 5e.1 of the pipeline refactor: extracts the *STT-only* concerns
from :class:`robot_comic.local_stt_realtime.LocalSTTInputMixin` into a
class that has no reference to a conversation-handler host. The mixin
survives unchanged for the existing host-coupled
:class:`robot_comic.adapters.moonshine_stt_adapter.MoonshineSTTAdapter`
path; the standalone listener is what subsequent 5e.* PRs wire into the
per-triple factory helpers after stripping the mixin out of the host.

This class is deliberately *narrow*: it owns the transcriber, stream,
listener bridge, audio ingestion, and #279 rearm recovery. It does NOT
own:

* ``deps`` / movement manager / pause controller / welcome gate /
  output queue / turn-span / echo guard / name-validation. Those are
  conversation-handler concerns the mixin currently bundles in; each
  belongs on the orchestrator or the adapter, not on the STT primitive.
* Heartbeat / ``MOONSHINE_DIAG`` instrumentation. The mixin has them
  today; the standalone listener does not. Subsequent PRs may port
  them when needed.

Threading model mirrors the mixin: Moonshine's transcriber fires
listener callbacks from its own worker thread; the listener schedules
the async callback onto the asyncio loop captured during
:meth:`start`.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable, Awaitable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from robot_comic import telemetry
from robot_comic.config import config
from robot_comic.local_stt_realtime import (
    LOCAL_STT_SAMPLE_RATE,
    LocalSTTDependencyError,
    prewarm_model_file,
    resolve_ort_model_path,
    _resolve_moonshine_arch,
)


logger = logging.getLogger(__name__)


# Event-kind constants. These match the strings the mixin schedules via
# ``_schedule_local_stt_event`` so any future consumer that wants to
# replace the mixin in-place can compare against the same vocabulary.
EVENT_STARTED = "started"
EVENT_PARTIAL = "partial"
EVENT_COMPLETED = "completed"
EVENT_ERROR = "error"


EventCallback = Callable[[str, str], Awaitable[None]]
"""Listener callback: ``async def on_event(kind: str, text: str) -> None``.

``kind`` is one of ``EVENT_STARTED`` / ``EVENT_PARTIAL`` /
``EVENT_COMPLETED`` / ``EVENT_ERROR``. ``text`` is the transcript so
far (or the error repr for ``EVENT_ERROR``). The callback is invoked on
the asyncio loop the listener was started on.
"""


class MoonshineListener:
    """Owns one Moonshine transcriber + streaming stream + listener bridge.

    Mirrors the STT-only pieces of :class:`LocalSTTInputMixin` without
    any reference to a conversation-handler host. See module docstring
    for the scope boundary.
    """

    def __init__(
        self,
        *,
        language: str | None = None,
        model_name: str | None = None,
        update_interval: float | None = None,
        cache_root: Path | None = None,
    ) -> None:
        """Capture Moonshine config overrides; falls back to ``config.LOCAL_STT_*``."""
        self._language = language
        self._model_name = model_name
        self._update_interval = update_interval
        self._cache_root = cache_root

        self._on_event: EventCallback | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._transcriber: Any = None
        self._stream: Any = None
        self._listener: Any = None
        self._listener_base_cls: Any = None
        self._effective_update_interval: float = 0.0

        # #279 — Moonshine streams have no public reset; after each
        # LineCompleted the stream silently stops emitting events. The
        # listener flags a rearm; the next ``feed_audio`` rebuilds the
        # stream before pushing the next frame.
        self._pending_rearm: bool = False

    async def start(self, on_event: EventCallback) -> None:
        """Capture the asyncio loop and build the streaming transcriber.

        Idempotent: calling ``start`` twice rebinds the callback but
        re-uses the existing transcriber + stream. Restart-after-stop is
        not supported; create a new instance instead.
        """
        self._on_event = on_event
        self._loop = asyncio.get_running_loop()
        if self._transcriber is None:
            await asyncio.to_thread(self._build_stream)

    async def feed_audio(self, sample_rate: int, samples: NDArray[np.int16]) -> None:
        """Push one captured audio frame into the streaming recognizer.

        Handles the #279 rearm-before-push fast path. Resamples to the
        Moonshine target sample rate when needed. Does NOT apply any
        echo-guard skip — that's a host concern for the consumer.
        """
        from fastrtc import audio_to_float32  # deferred: fastrtc pulls gradio at boot

        if self._pending_rearm:
            try:
                await asyncio.to_thread(self._rearm_stream)
            except Exception as e:
                logger.warning("Moonshine stream rearm failed: %s", e)
                # Clear the flag so we don't tight-loop on the same
                # failure; the next completion will retry.
                self._pending_rearm = False

        if self._stream is None:
            return

        audio_frame = samples
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        audio_float = audio_to_float32(audio_frame)

        target_sr = LOCAL_STT_SAMPLE_RATE
        if target_sr != sample_rate:
            audio_float = resample(
                audio_float,
                int(len(audio_frame) * target_sr / sample_rate),
            ).astype(np.float32, copy=False)

        try:
            self._stream.add_audio(audio_float.tolist(), target_sr)
        except Exception as e:
            logger.debug("Dropping local STT audio frame: %s", e)

    async def stop(self) -> None:
        """Tear down the stream + transcriber. Safe to call multiple times."""
        stream = self._stream
        transcriber = self._transcriber
        self._stream = None
        self._transcriber = None
        self._listener = None
        self._on_event = None

        def _close() -> None:
            for obj, methods in (
                (stream, ("stop", "close")),
                (transcriber, ("close",)),
            ):
                if obj is None:
                    continue
                for method_name in methods:
                    method = getattr(obj, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception as e:
                            logger.debug(
                                "Local STT %s failed during shutdown: %s",
                                method_name,
                                e,
                            )

        await asyncio.to_thread(_close)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_stream(self) -> None:
        """Load Moonshine, build transcriber + stream. Synchronous; runs in thread."""
        try:
            import moonshine_voice
            from moonshine_voice import (
                Transcriber,
                TranscriptEventListener,
                get_model_for_language,
            )
        except Exception as e:  # pragma: no cover — covered via unit tests without dep
            raise LocalSTTDependencyError(
                "Local STT requires the optional dependency group: install with "
                "`uv sync --extra local_stt` or `pip install -e .[local_stt]`."
            ) from e

        language = (self._language or getattr(config, "LOCAL_STT_LANGUAGE", None) or "en").strip().lower()
        model_name = self._model_name or str(getattr(config, "LOCAL_STT_MODEL", "tiny_streaming"))
        update_interval = float(
            self._update_interval
            if self._update_interval is not None
            else getattr(config, "LOCAL_STT_UPDATE_INTERVAL", 0.35)
        )
        requested_arch = _resolve_moonshine_arch(moonshine_voice, model_name)
        cache_root = (
            self._cache_root or Path(getattr(config, "LOCAL_STT_CACHE_DIR", "./cache/moonshine_voice")).expanduser()
        )

        try:
            model_path, model_arch = get_model_for_language(
                language,
                wanted_model_arch=requested_arch,
                cache_root=cache_root,
            )
        except TypeError:
            model_path, model_arch = get_model_for_language(language, cache_root=cache_root)
            model_arch = requested_arch or model_arch

        model_path, model_format = resolve_ort_model_path(Path(model_path))
        prewarm_model_file(model_path)

        logger.info(
            "Starting standalone Moonshine listener: language=%s model=%s format=%s "
            "update_interval=%.2fs cache=%s path=%s",
            language,
            model_name,
            model_format,
            update_interval,
            cache_root,
            model_path,
        )

        transcriber_path = model_path.parent if model_path.is_file() else model_path
        self._transcriber = Transcriber(
            model_path=str(transcriber_path),
            model_arch=model_arch,
            update_interval=update_interval,
        )
        self._effective_update_interval = update_interval
        self._listener_base_cls = TranscriptEventListener
        self._open_stream()

    def _open_stream(self) -> None:
        """Create + start a fresh stream on the current transcriber.

        Called once from ``_build_stream`` and again from
        ``_rearm_stream`` after each completed utterance.
        """
        transcriber = self._transcriber
        base_cls = self._listener_base_cls
        if transcriber is None or base_cls is None:
            return
        stream = transcriber.create_stream(update_interval=self._effective_update_interval)
        listener_cls = type(
            "StandaloneMoonshineListener",
            (_StandaloneListenerBridge, base_cls),
            {},
        )
        listener = listener_cls(self)
        stream.add_listener(listener)
        stream.start()
        self._stream = stream
        self._listener = listener
        self._pending_rearm = False

    def _rearm_stream(self) -> None:
        """Tear down the current stream and start a fresh one. Mirrors #279 path.

        Runs synchronously; the transcriber is preserved so the
        multi-hundred-millisecond ONNX model load is NOT repeated.
        """
        old_stream = self._stream
        self._stream = None
        self._listener = None
        if old_stream is not None:
            for method_name in ("stop", "close"):
                method = getattr(old_stream, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as e:
                        logger.debug("Moonshine stream %s during rearm: %s", method_name, e)
        if self._transcriber is None:
            # Shutdown raced with us; nothing to do.
            self._pending_rearm = False
            return
        self._open_stream()

    def _schedule_event(self, kind: str, text: str) -> None:
        """Schedule an event onto the asyncio loop captured in ``start``.

        Called from the Moonshine listener thread via the bridge below.
        """
        loop = self._loop
        cb = self._on_event
        if loop is None or loop.is_closed() or cb is None:
            return

        async def _fire() -> None:
            assert cb is not None  # narrow for mypy
            try:
                await cb(kind, text)
            except Exception:
                logger.exception("Standalone Moonshine event callback raised")

        loop.call_soon_threadsafe(lambda: asyncio.create_task(_fire()))


class _StandaloneListenerBridge:
    """Bridge Moonshine listener callbacks into the asyncio loop.

    Mounted dynamically over ``TranscriptEventListener`` (the moonshine
    base) by :meth:`MoonshineListener._open_stream`. The bridge
    converts each callback into an ``on_event(kind, text)`` invocation
    scheduled on the listener's asyncio loop.
    """

    def __init__(self, listener: MoonshineListener) -> None:
        self._listener = listener

    def on_line_started(self, event: Any) -> None:
        text = self._text_from_event(event)
        self._listener._schedule_event(EVENT_STARTED, text)

    def on_line_updated(self, event: Any) -> None:
        text = self._text_from_event(event)
        self._listener._schedule_event(EVENT_PARTIAL, text)

    def on_line_text_changed(self, event: Any) -> None:
        text = self._text_from_event(event)
        self._listener._schedule_event(EVENT_PARTIAL, text)

    def on_line_completed(self, event: Any) -> None:
        text = self._text_from_event(event)
        self._listener._schedule_event(EVENT_COMPLETED, text)
        # #279 — flag a rearm so the next ``feed_audio`` rebuilds the
        # stream before pushing the next frame. Same recovery the mixin
        # uses; see ``LocalSTTInputMixin.receive``.
        self._listener._pending_rearm = True

    def on_error(self, event: Any) -> None:
        err = getattr(event, "error", event)
        text = repr(err)
        logger.warning("Local STT error (standalone listener): %s", err)
        telemetry.inc_errors({"subsystem": "stt", "error_type": "stream_error"})
        self._listener._schedule_event(EVENT_ERROR, text)
        # A stream-level error wedges the C handle; rearm is the only
        # recovery. Mirrors the mixin's _MoonshineListener.on_error.
        self._listener._pending_rearm = True

    @staticmethod
    def _text_from_event(event: Any) -> str:
        line = getattr(event, "line", event)
        text = getattr(line, "text", "")
        return text if isinstance(text, str) else str(text)
