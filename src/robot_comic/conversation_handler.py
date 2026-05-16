from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import Any, TypeAlias
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from robot_comic.tools.core_tools import ToolDependencies


AudioFrame: TypeAlias = tuple[int, NDArray[np.int16]]
# Broad aliases — concrete handlers use fastrtc types directly.
HandlerOutput: TypeAlias = Any
QueueItem: TypeAlias = Any


class ConversationHandler(ABC):
    """FastRTC-shim ABC for realtime conversation backends.

    Defines lifecycle (``start_up``/``shutdown``), per-frame I/O
    (``receive``/``emit``), and per-peer cloning (``copy``) — the surface
    FastRTC's ``AsyncStreamHandler`` consumes. Voice methods
    (``get_available_voices``/``get_current_voice``/``change_voice``) live
    on :class:`~robot_comic.backends.TTSBackend` (Phase 5c.1) and
    persona-switch state surgery lives on
    :class:`~robot_comic.composable_pipeline.ComposablePipeline.apply_personality`
    (Phase 5c.2). Concrete handlers (the composable wrapper, the
    bundled-realtime handlers) still implement those methods directly for
    callers that reach for them via duck-typing
    (``headless_personality_ui.py`` REST routes); they are no longer
    ABC-enforced as of Phase 5d.

    See ``docs/superpowers/specs/2026-05-16-phase-5d-conversationhandler-abc-shrink.md``
    for the shrink rationale.

    ``deps`` stays on the ABC because ``console.py``'s ``_play_loop`` reads
    ``handler.deps.head_wobbler`` through the ABC type; ``_clear_queue``
    stays because ``LocalStream`` writes to it for barge-in plumbing.
    Both are documented "leaky abstraction" carry-forwards (exploration
    memo §2.4) deferred past 5d.
    """

    deps: ToolDependencies
    output_queue: asyncio.Queue[QueueItem]
    _clear_queue: Callable[[], None] | None

    @abstractmethod
    def copy(self) -> ConversationHandler:
        """Create a copy of the handler."""
        ...

    @abstractmethod
    async def start_up(self) -> None:
        """Start the realtime handler."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shut down the realtime handler."""
        ...

    @abstractmethod
    async def receive(self, frame: AudioFrame) -> None:
        """Receive an input audio frame."""
        ...

    @abstractmethod
    async def emit(self) -> HandlerOutput:
        """Emit the next output item."""
        ...
