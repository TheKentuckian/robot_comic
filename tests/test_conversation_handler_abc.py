"""Pin the post-Phase-5d ``ConversationHandler`` ABC contract.

Phase 5d shrinks the ABC to its FastRTC-shim role. Voice methods moved
onto :class:`~robot_comic.backends.TTSBackend` in Phase 5c.1; persona-switch
state surgery moved onto
:class:`~robot_comic.composable_pipeline.ComposablePipeline.apply_personality`
in Phase 5c.2. The four corresponding ``@abstractmethod`` declarations on
the ABC are now duplicates and must be removed.

This file pins the result: a subclass implementing only the five
remaining FastRTC-shim methods (``copy`` / ``start_up`` / ``shutdown``
/ ``receive`` / ``emit``) must instantiate cleanly, and the four
removed methods must not appear in ``__abstractmethods__``.

See ``docs/superpowers/specs/2026-05-16-phase-5d-conversationhandler-abc-shrink.md``.
"""

from __future__ import annotations
import asyncio

from robot_comic.conversation_handler import (
    AudioFrame,
    HandlerOutput,
    ConversationHandler,
)


class _MinimalShimHandler(ConversationHandler):
    """Implements only the five remaining FastRTC-shim abstract methods."""

    def __init__(self) -> None:
        # Type-level attributes are declared on the ABC; concrete assignments
        # live here so the runtime instance has them populated.
        self.deps = None  # type: ignore[assignment]
        self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()
        self._clear_queue = None

    def copy(self) -> "_MinimalShimHandler":
        return _MinimalShimHandler()

    async def start_up(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def receive(self, frame: AudioFrame) -> None:
        return None

    async def emit(self) -> HandlerOutput:
        return None


def test_minimal_shim_handler_instantiates_without_voice_methods() -> None:
    """A subclass that implements only the five FastRTC-shim methods must
    instantiate cleanly. Pre-5d this raised ``TypeError`` because
    ``apply_personality`` / ``get_available_voices`` / ``get_current_voice``
    / ``change_voice`` were marked ``@abstractmethod``."""
    handler = _MinimalShimHandler()
    assert isinstance(handler, ConversationHandler)


def test_voice_and_personality_methods_not_in_abstractmethods() -> None:
    """Phase 5d removed the four voice/personality ``@abstractmethod``
    declarations. If a future PR accidentally re-adds one, this test
    catches it before merge."""
    abstract_methods = ConversationHandler.__abstractmethods__
    for moved in (
        "apply_personality",
        "get_available_voices",
        "get_current_voice",
        "change_voice",
    ):
        assert moved not in abstract_methods, (
            f"{moved!r} was moved off the ABC in Phase 5c.1 / 5c.2 and "
            f"must not be re-added as @abstractmethod (Phase 5d shrink)."
        )


def test_fastrtc_shim_methods_remain_abstract() -> None:
    """The five FastRTC-shim methods stay abstract — they're the ABC's
    raison d'être after the shrink."""
    abstract_methods = ConversationHandler.__abstractmethods__
    for kept in ("copy", "start_up", "shutdown", "receive", "emit"):
        assert kept in abstract_methods, (
            f"{kept!r} is a FastRTC-shim method and must remain @abstractmethod "
            f"(Phase 5d shrink preserves the lifecycle + per-frame I/O surface)."
        )
