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
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError

    async def receive(self, frame: AudioFrame) -> None:
        raise NotImplementedError

    async def emit(self) -> HandlerOutput:
        raise NotImplementedError

    async def apply_personality(self, profile: str | None) -> str:
        raise NotImplementedError

    async def get_available_voices(self) -> list[str]:
        raise NotImplementedError

    def get_current_voice(self) -> str:
        raise NotImplementedError

    async def change_voice(self, voice: str) -> str:
        raise NotImplementedError
