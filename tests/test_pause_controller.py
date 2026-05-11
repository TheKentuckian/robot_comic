"""Unit tests for the PauseController."""

from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock

import pytest

from robot_comic.pause import (
    PauseState,
    PauseController,
    TranscriptDisposition,
    _phrase_matches,
)
from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.local_stt_realtime import LocalSTTRealtimeHandler


def _make_controller(**overrides):
    """Build a controller with mock callbacks and return it alongside the mocks."""
    clear_move_queue = overrides.pop("clear_move_queue", MagicMock())
    on_shutdown = overrides.pop("on_shutdown", MagicMock())
    on_switch_requested = overrides.pop("on_switch_requested", MagicMock())
    return (
        PauseController(
            clear_move_queue=clear_move_queue,
            on_shutdown=on_shutdown,
            on_switch_requested=on_switch_requested,
            **overrides,
        ),
        clear_move_queue,
        on_shutdown,
        on_switch_requested,
    )


def test_active_transcripts_are_dispatched():
    """Normal user speech flows through to the backend unchanged."""
    controller, clear, shutdown, switch = _make_controller()
    assert controller.handle_transcript("hey what's up") is TranscriptDisposition.DISPATCH
    assert controller.state is PauseState.ACTIVE
    clear.assert_not_called()
    shutdown.assert_not_called()
    switch.assert_not_called()


def test_stop_word_clears_queue_and_enters_paused():
    """Hearing the stop word transitions to paused and clears the move queue."""
    controller, clear, shutdown, switch = _make_controller()
    result = controller.handle_transcript("System pause, please.")
    assert result is TranscriptDisposition.HANDLED
    assert controller.is_paused
    clear.assert_called_once()
    shutdown.assert_not_called()
    switch.assert_not_called()


def test_stop_word_match_is_punctuation_insensitive():
    """Trailing punctuation does not prevent stop-word detection."""
    controller, clear, _, _ = _make_controller()
    assert controller.handle_transcript("Robot pause!!!") is TranscriptDisposition.HANDLED
    assert controller.is_paused
    clear.assert_called_once()


def test_unrelated_words_inside_paused_are_still_handled():
    """While paused, off-menu transcripts are consumed and not dispatched."""
    controller, clear, shutdown, switch = _make_controller()
    controller.handle_transcript("system pause")
    clear.reset_mock()
    result = controller.handle_transcript("tell me a joke")
    assert result is TranscriptDisposition.HANDLED
    assert controller.is_paused
    clear.assert_not_called()
    shutdown.assert_not_called()
    switch.assert_not_called()


def test_continue_resumes_active():
    """A resume phrase returns the controller to the active state."""
    controller, _, shutdown, switch = _make_controller()
    controller.handle_transcript("system pause")
    assert controller.is_paused
    result = controller.handle_transcript("Continue.")
    assert result is TranscriptDisposition.HANDLED
    assert controller.state is PauseState.ACTIVE
    shutdown.assert_not_called()
    switch.assert_not_called()


def test_shutdown_invokes_callback_and_marks_state():
    """A shutdown phrase fires the callback exactly once and locks the state."""
    controller, _, shutdown, _ = _make_controller()
    controller.handle_transcript("system pause")
    result = controller.handle_transcript("Shut down now")
    assert result is TranscriptDisposition.HANDLED
    assert controller.state is PauseState.SHUTTING_DOWN
    shutdown.assert_called_once()


def test_switch_phrase_invokes_callback_and_stays_paused():
    """A switch phrase fires the optional callback but does not auto-resume."""
    controller, _, _, switch = _make_controller()
    controller.handle_transcript("system pause")
    result = controller.handle_transcript("switch comic")
    assert result is TranscriptDisposition.HANDLED
    assert controller.is_paused
    switch.assert_called_once()


def test_after_shutdown_further_transcripts_are_silently_handled():
    """Once shutting down, later transcripts are absorbed and do not re-trigger callbacks."""
    controller, clear, shutdown, switch = _make_controller()
    controller.handle_transcript("system pause")
    controller.handle_transcript("shutdown")
    shutdown.assert_called_once()
    shutdown.reset_mock()
    clear.reset_mock()
    switch.reset_mock()

    result = controller.handle_transcript("continue")
    assert result is TranscriptDisposition.HANDLED
    assert controller.state is PauseState.SHUTTING_DOWN
    shutdown.assert_not_called()
    clear.assert_not_called()
    switch.assert_not_called()


def test_clear_move_queue_exceptions_are_swallowed():
    """An exception from clear_move_queue must not prevent entering the paused state."""

    def bad_clear() -> None:
        raise RuntimeError("boom")

    controller, _, _, _ = _make_controller(clear_move_queue=bad_clear)
    result = controller.handle_transcript("system pause")
    assert result is TranscriptDisposition.HANDLED
    assert controller.is_paused


def test_empty_transcript_is_dispatched():
    """Empty or whitespace-only transcripts are passed through without state change."""
    controller, clear, _, _ = _make_controller()
    assert controller.handle_transcript("") is TranscriptDisposition.DISPATCH
    assert controller.handle_transcript("   ") is TranscriptDisposition.DISPATCH
    clear.assert_not_called()


def test_stop_phrase_must_be_whole_token_sequence():
    """The stop phrase must appear as a contiguous token sequence, not a substring."""
    controller, clear, _, _ = _make_controller(
        stop_phrases=("system pause",),
    )
    assert controller.handle_transcript("pausesystem now") is TranscriptDisposition.DISPATCH
    clear.assert_not_called()


def test_phrase_matches_helper():
    """Whole-word phrase matching is case- and punctuation-insensitive."""
    assert _phrase_matches("System pause!", ["system pause"]) == "system pause"
    assert _phrase_matches("Yo, system pause man", ["system pause"]) == "system pause"
    assert _phrase_matches("Nothing here", ["system pause"]) is None
    assert _phrase_matches("", ["system pause"]) is None


@pytest.mark.asyncio
async def test_local_stt_skips_dispatch_when_paused():
    """When PauseController returns HANDLED, the local STT path must NOT dispatch."""
    clear = MagicMock()
    shutdown = MagicMock()
    controller = PauseController(clear_move_queue=clear, on_shutdown=shutdown)

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        pause_controller=controller,
    )
    handler = LocalSTTRealtimeHandler(deps)
    handler._dispatch_completed_transcript = AsyncMock()  # type: ignore[method-assign]

    await handler._handle_local_stt_event("completed", "system pause")

    handler._dispatch_completed_transcript.assert_not_awaited()
    assert controller.is_paused
    clear.assert_called_once()


@pytest.mark.asyncio
async def test_local_stt_dispatches_when_not_handled():
    """Normal transcripts must still be dispatched to the response backend."""
    controller = PauseController(
        clear_move_queue=MagicMock(),
        on_shutdown=MagicMock(),
    )

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        pause_controller=controller,
    )
    handler = LocalSTTRealtimeHandler(deps)
    handler._dispatch_completed_transcript = AsyncMock()  # type: ignore[method-assign]

    await handler._handle_local_stt_event("completed", "hello there")

    handler._dispatch_completed_transcript.assert_awaited_once_with("hello there")
    assert not controller.is_paused
