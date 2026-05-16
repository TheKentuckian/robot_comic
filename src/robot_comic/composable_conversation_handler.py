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
    - history_trim.trim_history_in_place
    - _speaking_until echo-guard timestamps (elevenlabs_tts.py:471-473)
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
        legacy ``LocalSTTLlamaElevenLabsHandler`` instance it is mixed into;
        that instance is our ``_tts_handler``, not the wrapper. So when
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
        """Emit ``handler.start_up.complete`` then delegate to the pipeline.

        Mirrors the legacy ``ElevenLabsTTSResponseHandler.start_up`` emit
        (#321 / #301) on the composable path: fire the supporting-event row
        *before* delegating to :meth:`ComposablePipeline.start_up`, which
        blocks until shutdown. Emitting after the delegate would only land
        the row on app shutdown — the same bug PR #337 already fixed on the
        legacy side.

        ``app.startup`` and ``welcome.wav.played`` are not emitted here; they
        fire from ``main.py`` and ``warmup_audio.py`` before any handler is
        built and are preserved on both factory paths.
        ``first_greeting.tts_first_audio`` fires from the TTS frame-enqueue
        sites; the ElevenLabs and Chatterbox adapters preserve it via
        delegation. The ``GeminiTTSAdapter`` gap is a separate follow-up.
        """
        try:
            from robot_comic import telemetry as _telemetry
            from robot_comic.startup_timer import since_startup

            _telemetry.emit_supporting_event(
                "handler.start_up.complete",
                dur_ms=since_startup() * 1000,
            )
        except Exception:
            # Telemetry must never block boot — drop the row if emission
            # throws (import error, exporter wiring quirk, etc.). Matches the
            # ``try/except`` at the legacy emit site in elevenlabs_tts.py.
            pass
        await self.pipeline.start_up()

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
        """Switch personality, reset pipeline history, re-seed system prompt."""
        # TODO(phase4-lifecycle): legacy handlers also clear per-session
        # echo-guard state on persona switch. Compose that in when the
        # echo-guard hook lands. Joke history is a cross-persona file
        # (``~/.robot-comic/joke-history.json``) read in
        # ``prompts.py:_append_joke_history`` — it intentionally persists
        # across personality switches so each persona avoids the others'
        # punchlines (legacy parity, see joke_history.format_for_prompt
        # comment "Pulls entries from ALL personas (cross-persona dedup)").
        try:
            set_custom_profile(profile)
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"
        self.pipeline.reset_history(keep_system=False)
        self.pipeline._conversation_history.append({"role": "system", "content": get_session_instructions()})
        return f"Applied personality {profile!r}. Conversation history reset."

    async def get_available_voices(self) -> list[str]:
        """Forward to the underlying TTS handler's voice catalog."""
        return await self._tts_handler.get_available_voices()

    def get_current_voice(self) -> str:
        """Forward to the underlying TTS handler's current-voice getter."""
        return self._tts_handler.get_current_voice()

    async def change_voice(self, voice: str) -> str:
        """Forward to the underlying TTS handler's voice switcher."""
        return await self._tts_handler.change_voice(voice)
