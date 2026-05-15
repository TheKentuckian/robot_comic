"""ConversationHandler ABC wrapper around ComposablePipeline (Phase 4a of #337).

Closes the surface gap between :class:`ComposablePipeline` and the existing
:class:`ConversationHandler` ABC so the factory (Phase 4b) can return either
interchangeably. Forwards voice / personality calls to a legacy TTS handler
held by reference — no Protocol churn.

Phase 4a deliberately leaves these lifecycle hooks unwired; each is a
follow-up PR between 4b and 4d:

    - telemetry.record_llm_duration
    - boot-timeline supporting events (#321)
    - record_joke_history (llama_base.py:553-568)
    - history_trim.trim_history_in_place
    - _speaking_until echo-guard timestamps (elevenlabs_tts.py:471-473)
"""

from __future__ import annotations
import logging
from typing import Callable

from robot_comic.composable_pipeline import ComposablePipeline
from robot_comic.conversation_handler import AudioFrame, ConversationHandler, HandlerOutput
from robot_comic.tools.core_tools import ToolDependencies


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
        self.pipeline = pipeline
        self._tts_handler = tts_handler
        self.deps = deps
        self._build = build
        self.output_queue = pipeline.output_queue
        self._clear_queue = None

    def copy(self) -> "ComposableConversationHandler":
        raise NotImplementedError

    async def start_up(self) -> None:
        """Delegate to :meth:`ComposablePipeline.start_up` — blocks until shutdown."""
        # TODO(phase4-lifecycle): emit the four boot-timeline supporting events
        # from #321 before delegating; composable mode currently drops them.
        await self.pipeline.start_up()

    async def shutdown(self) -> None:
        """Delegate to :meth:`ComposablePipeline.shutdown`."""
        await self.pipeline.shutdown()

    async def receive(self, frame: AudioFrame) -> None:
        """Forward a captured input frame to the pipeline's STT backend."""
        await self.pipeline.feed_audio(frame)

    async def emit(self) -> HandlerOutput:
        """Pull the next output item from the pipeline's output queue."""
        from fastrtc import wait_for_item  # deferred — fastrtc pulls gradio at boot

        return await wait_for_item(self.pipeline.output_queue)

    async def apply_personality(self, profile: str | None) -> str:
        raise NotImplementedError

    async def get_available_voices(self) -> list[str]:
        raise NotImplementedError

    def get_current_voice(self) -> str:
        raise NotImplementedError

    async def change_voice(self, voice: str) -> str:
        raise NotImplementedError
