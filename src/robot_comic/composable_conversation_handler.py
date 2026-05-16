"""ConversationHandler ABC wrapper around ComposablePipeline (Phase 4a of #337).

Closes the surface gap between :class:`ComposablePipeline` and the existing
:class:`ConversationHandler` ABC so the factory (Phase 4b) can return either
interchangeably. Forwards voice / personality calls to a legacy TTS handler
held by reference — no Protocol churn.

Phase 4a deliberately leaves these lifecycle hooks unwired; each is a
follow-up PR between 4b and 4d:

    - telemetry.record_llm_duration       — wired (Hook #2)
    - boot-timeline supporting events (#321) — wired (Hook #3)
    - record_joke_history (llama_base.py:578-594) — wired (Hook #4) at
      ``ComposablePipeline._speak_assistant_text``
    - history_trim.trim_history_in_place — wired (Hook #5) at
      ``ComposablePipeline._run_llm_loop_and_speak``
    - _speaking_until echo-guard timestamps (elevenlabs_tts.py:471-473) —
      per-turn write site wired (Hook #1, PR #372); per-persona reset
      wired (Phase 5a.1) at ``apply_personality``
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable

from robot_comic.config import set_custom_profile
from robot_comic.prompts import get_session_instructions
from robot_comic.backends import AudioFrame as BackendAudioFrame
from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.composable_pipeline import ComposablePipeline
from robot_comic.conversation_handler import AudioFrame, HandlerOutput, ConversationHandler


logger = logging.getLogger(__name__)


class ComposableConversationHandler(ConversationHandler):
    """Wrap a :class:`ComposablePipeline` as a :class:`ConversationHandler`."""

    def __init__(
        self,
        pipeline: ComposablePipeline,
        *,
        tts_handler: ConversationHandler,
        deps: ToolDependencies,
        build: Callable[[], "ComposableConversationHandler"],
    ) -> None:
        """Store the pipeline, the legacy TTS handler, deps, and the rebuild closure."""
        self.pipeline = pipeline
        self._tts_handler = tts_handler
        self.deps = deps
        self._build = build
        # Backing slot for the _clear_queue property below. Assign before any
        # external code can read; see the setter for why we mirror onto the
        # underlying TTS handler.
        self.__clear_queue: Callable[[], None] | None = None

    @property
    def output_queue(self) -> asyncio.Queue[Any]:
        """Read-through to the pipeline's queue (what ``emit()`` drains)."""
        return self.pipeline.output_queue

    @output_queue.setter
    def output_queue(self, queue: asyncio.Queue[Any]) -> None:
        """Replace the pipeline's queue.

        ``LocalStream.clear_audio_queue`` does
        ``handler.output_queue = asyncio.Queue()`` to drop queued TTS frames
        on barge-in. The pipeline owns the read queue, so the assignment has
        to land there or the rebind is a no-op and ``emit()`` keeps draining
        the stale queue.
        """
        self.pipeline.output_queue = queue

    @property
    def _clear_queue(self) -> Callable[[], None] | None:
        """The queue-flush callback. Mirrors onto the wrapped TTS handler."""
        return self.__clear_queue

    @_clear_queue.setter
    def _clear_queue(self, callback: Callable[[], None] | None) -> None:
        """Mirror the queue-flush callback onto the underlying TTS handler.

        The ``LocalSTTInputMixin`` listener calls ``self._clear_queue`` on the
        factory-private host instance it is mixed into (e.g.
        ``_LocalSTTLlamaElevenLabsHost``); that instance is our
        ``_tts_handler``, not the wrapper. So when
        ``LocalStream.__init__`` does ``self.handler._clear_queue =
        self.clear_audio_queue`` on the wrapper, we forward the assignment to
        the legacy handler — otherwise barge-in stops flushing the player on
        the composable path.
        """
        self.__clear_queue = callback
        if getattr(self, "_tts_handler", None) is not None:
            self._tts_handler._clear_queue = callback

    def copy(self) -> "ComposableConversationHandler":
        """Build a fresh wrapper + pipeline via the injected factory closure.

        FastRTC clones the handler per peer; the new instance must own
        independent pipeline state (history, stop event, adapter sessions).
        """
        return self._build()

    async def start_up(self) -> None:
        """Delegate to the pipeline and then emit ``handler.start_up.complete``.

        Boot memo PR #383 (§"weird things to know" #2) and instrumentation
        audit PR #385 (§6 gap #3) flagged that the previous implementation
        emitted the event *before* awaiting ``pipeline.start_up()``, which
        meant the event timing reflected wrapper-entry rather than handler
        readiness — operators reading the row got an "entered start_up"
        signal labelled as "complete". This implementation fires the event
        once :meth:`ComposablePipeline.start_up` has finished preparing the
        STT/LLM/TTS adapters, which is the moment the handler is actually
        ready to accept audio.

        :meth:`ComposablePipeline.start_up` blocks until shutdown, so the
        emit is wrapped in ``try/finally``: the supporting-event row fires
        when the pipeline returns (either normal shutdown or an exception
        propagating out of ``prepare``/``start``). Without ``finally`` an
        exception in ``prepare`` would suppress the row entirely; downstream
        consumers expecting it would see a missing event rather than an
        early-exit signal.

        ``app.startup`` and ``welcome.wav.played`` are not emitted here; they
        fire from ``main.py`` and ``warmup_audio.py`` before any handler is
        built and are preserved on both factory paths.
        ``first_greeting.tts_first_audio`` fires from the TTS frame-enqueue
        sites; the ElevenLabs, Chatterbox and Gemini-TTS adapters preserve
        it via delegation.
        """
        try:
            await self.pipeline.start_up()
        finally:
            try:
                from robot_comic import telemetry as _telemetry
                from robot_comic.startup_timer import since_startup

                _telemetry.emit_supporting_event(
                    "handler.start_up.complete",
                    dur_ms=since_startup() * 1000,
                )
            except Exception:
                # Telemetry must never block boot — drop the row if emission
                # throws (import error, exporter wiring quirk, etc.). Matches
                # the ``try/except`` at the legacy emit site in
                # elevenlabs_tts.py.
                pass

    async def shutdown(self) -> None:
        """Delegate to :meth:`ComposablePipeline.shutdown`."""
        await self.pipeline.shutdown()

    async def receive(self, frame: AudioFrame) -> None:
        """Forward a captured input frame to the pipeline's STT backend.

        FastRTC delivers frames as ``(sample_rate, ndarray)`` per the
        ``ConversationHandler`` ABC; the pipeline's STT Protocol expects the
        :class:`backends.AudioFrame` dataclass. Convert at the boundary.
        """
        sample_rate, samples = frame
        await self.pipeline.feed_audio(BackendAudioFrame(samples=samples, sample_rate=sample_rate))

    async def emit(self) -> HandlerOutput:
        """Pull the next output item from the pipeline's output queue."""
        from fastrtc import wait_for_item  # deferred — fastrtc pulls gradio at boot

        return await wait_for_item(self.pipeline.output_queue)

    async def apply_personality(self, profile: str | None) -> str:
        """Switch personality, reset pipeline history, clear per-session state, re-seed system prompt.

        Persona switch is a hard cut on listening state: the wrapped TTS
        handler's echo-guard accumulators are cleared via
        :meth:`_reset_tts_per_session_state` so a stale ``_speaking_until``
        from an in-flight or just-finished playback does not bleed into the
        new persona's listening window. See spec
        ``docs/superpowers/specs/2026-05-16-phase-5a1-echo-guard-persona-reset.md``
        for the audit + rationale; Phase 4 lifecycle hook #1 (PR #372) wired
        the per-turn write site, this is the per-persona reset complement.

        Joke history is a cross-persona file
        (``~/.robot-comic/joke-history.json``) read in
        ``prompts.py:_append_joke_history`` — it intentionally persists
        across personality switches so each persona avoids the others'
        punchlines (legacy parity, see ``joke_history.format_for_prompt``
        comment "Pulls entries from ALL personas (cross-persona dedup)").
        """
        try:
            set_custom_profile(profile)
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"
        self.pipeline.reset_history(keep_system=False)
        self._reset_tts_per_session_state()
        self.pipeline._conversation_history.append({"role": "system", "content": get_session_instructions()})
        return f"Applied personality {profile!r}. Conversation history reset."

    def _reset_tts_per_session_state(self) -> None:
        """Clear per-session echo-guard accumulators on the wrapped TTS handler.

        Defensively guarded via :func:`hasattr` so handlers without echo-guard
        state (e.g. :class:`~robot_comic.gemini_tts.GeminiTTSResponseHandler`,
        which has no ``_speaking_until`` field) are a clean no-op.

        Three fields are reset, mirroring the per-turn reset site in
        ``ElevenLabsTTSResponseHandler._dispatch_completed_transcript_impl``
        (``elevenlabs_tts.py:558-560``) and the canned-opener reset at
        ``llama_base.py:182-184``:

        - ``_speaking_until`` — playback-deadline timestamp consulted by
          ``LocalSTTInputMixin._handle_local_stt_event`` (see
          ``local_stt_realtime.py:619``).
        - ``_response_start_ts`` / ``_response_audio_bytes`` — the byte-count
          accumulators that feed the next ``_speaking_until`` derivation in
          ``_enqueue_audio_frame`` (lifecycle hook #1, PR #372).

        ``_is_responding`` / ``_dispatch_in_flight`` on
        ``ElevenLabsTTSResponseHandler`` are intentionally NOT touched —
        they only guard the legacy ``_dispatch_completed_transcript`` path
        that the composable pipeline bypasses. Clearing them would mask
        bugs if the legacy dispatch ever re-engaged.
        """
        handler = getattr(self, "_tts_handler", None)
        if handler is None:
            return
        for field, value in (
            ("_speaking_until", 0.0),
            ("_response_start_ts", 0.0),
            ("_response_audio_bytes", 0),
        ):
            if hasattr(handler, field):
                setattr(handler, field, value)

    async def get_available_voices(self) -> list[str]:
        """Forward to the pipeline's TTS adapter (Phase 5c.1).

        Replaces the prior ``self._tts_handler.get_available_voices()``
        forward — the adapter now implements the voice-method surface
        directly via :class:`~robot_comic.backends.TTSBackend`'s Phase
        5c.1 Protocol extension, so the wrapper no longer needs to reach
        into the legacy handler for voice queries. ``self._tts_handler``
        is still consulted by :meth:`_reset_tts_per_session_state`
        (Phase 5c.2 / 5d will revisit).
        """
        return await self.pipeline.tts.get_available_voices()

    def get_current_voice(self) -> str:
        """Forward to the pipeline's TTS adapter (Phase 5c.1)."""
        return self.pipeline.tts.get_current_voice()

    async def change_voice(self, voice: str) -> str:
        """Forward to the pipeline's TTS adapter (Phase 5c.1)."""
        return await self.pipeline.tts.change_voice(voice)
