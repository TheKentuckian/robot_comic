"""Tests for trigger-prefix stop-word robustness (issue #98).

Covers:
- "robot shutdown" / "robot pause" recognised as primary trigger (default prefix).
- "reachy shutdown" / "reachy pause" still recognised (backward compat).
- Custom prefix via REACHY_MINI_TRIGGER_PREFIX env var.
- Unrelated phrases are NOT recognised.
- Split-word variants ("shut down") work for both prefixes.
"""

from __future__ import annotations
import sys
from unittest.mock import MagicMock

import pytest

from robot_comic.pause import (
    PauseState,
    PauseController,
    TranscriptDisposition,
    _build_stop_phrases,
    _build_shutdown_phrases,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_controller(**overrides):
    """Return (controller, on_shutdown_mock) with sensible defaults."""
    clear = overrides.pop("clear_move_queue", MagicMock())
    on_shutdown = overrides.pop("on_shutdown", MagicMock())
    controller = PauseController(
        clear_move_queue=clear,
        on_shutdown=on_shutdown,
        **overrides,
    )
    return controller, on_shutdown, clear


def _controller_with_prefix(prefix: str, **overrides):
    """Build a PauseController pre-loaded with phrases for *prefix*."""
    stop = _build_stop_phrases(prefix)
    shutdown = _build_shutdown_phrases(prefix)
    return _make_controller(stop_phrases=stop, shutdown_phrases=shutdown, **overrides)


# ---------------------------------------------------------------------------
# Default-prefix "robot" tests
# ---------------------------------------------------------------------------


class TestRobotPrefixDefault:
    """The default primary prefix is 'robot'."""

    def test_robot_shutdown_is_recognised(self):
        """'robot shutdown' must trigger the shutdown callback."""
        controller, on_shutdown, _ = _controller_with_prefix("robot")
        result = controller.handle_transcript("robot shutdown")
        assert result is TranscriptDisposition.HANDLED
        assert controller.state is PauseState.SHUTTING_DOWN
        on_shutdown.assert_called_once()

    def test_robot_shutdown_split_word(self):
        """'robot shut down' (two words, split by Moonshine) must be recognised."""
        controller, on_shutdown, _ = _controller_with_prefix("robot")
        result = controller.handle_transcript("robot shut down")
        assert result is TranscriptDisposition.HANDLED
        assert controller.state is PauseState.SHUTTING_DOWN
        on_shutdown.assert_called_once()

    def test_robot_pause_is_recognised(self):
        """'robot pause' must enter the paused state."""
        controller, _, clear = _controller_with_prefix("robot")
        result = controller.handle_transcript("robot pause")
        assert result is TranscriptDisposition.HANDLED
        assert controller.is_paused
        clear.assert_called_once()

    def test_robot_paws_mishearing_is_recognised(self):
        """'robot paws' (common Moonshine mishearing) must pause the controller."""
        controller, _, clear = _controller_with_prefix("robot")
        result = controller.handle_transcript("robot paws")
        assert result is TranscriptDisposition.HANDLED
        assert controller.is_paused
        clear.assert_called_once()


# ---------------------------------------------------------------------------
# Backward-compat "reachy" prefix tests
# ---------------------------------------------------------------------------


class TestReachyBackwardCompat:
    """Legacy 'reachy' phrases are always recognised regardless of active prefix."""

    def test_reachy_shutdown_still_recognised(self):
        """'reachy shutdown' must work even when the primary prefix is 'robot'."""
        controller, on_shutdown, _ = _controller_with_prefix("robot")
        result = controller.handle_transcript("reachy shutdown")
        assert result is TranscriptDisposition.HANDLED
        assert controller.state is PauseState.SHUTTING_DOWN
        on_shutdown.assert_called_once()

    def test_reachy_shut_down_split_still_recognised(self):
        """'reachy shut down' (split-word) must still trigger shutdown."""
        controller, on_shutdown, _ = _controller_with_prefix("robot")
        result = controller.handle_transcript("reachy shut down")
        assert result is TranscriptDisposition.HANDLED
        assert controller.state is PauseState.SHUTTING_DOWN
        on_shutdown.assert_called_once()

    def test_reachy_pause_still_recognised(self):
        """'reachy pause' must still enter the paused state."""
        controller, _, clear = _controller_with_prefix("robot")
        result = controller.handle_transcript("reachy pause")
        assert result is TranscriptDisposition.HANDLED
        assert controller.is_paused
        clear.assert_called_once()

    def test_reachy_paws_mishearing_still_recognised(self):
        """'reachy paws' mishearing must still pause."""
        controller, _, clear = _controller_with_prefix("robot")
        result = controller.handle_transcript("reachy paws")
        assert result is TranscriptDisposition.HANDLED
        assert controller.is_paused
        clear.assert_called_once()


# ---------------------------------------------------------------------------
# Custom prefix via env var
# ---------------------------------------------------------------------------


class TestCustomPrefix:
    """REACHY_MINI_TRIGGER_PREFIX=comic adds comic-prefixed phrases."""

    def test_comic_shutdown_recognised_when_prefix_set(self):
        """'comic shutdown' is recognised when prefix='comic'."""
        controller, on_shutdown, _ = _controller_with_prefix("comic")
        result = controller.handle_transcript("comic shutdown")
        assert result is TranscriptDisposition.HANDLED
        assert controller.state is PauseState.SHUTTING_DOWN
        on_shutdown.assert_called_once()

    def test_comic_shut_down_split_recognised(self):
        """'comic shut down' (split-word) is recognised when prefix='comic'."""
        controller, on_shutdown, _ = _controller_with_prefix("comic")
        result = controller.handle_transcript("comic shut down")
        assert result is TranscriptDisposition.HANDLED
        assert controller.state is PauseState.SHUTTING_DOWN
        on_shutdown.assert_called_once()

    def test_comic_pause_recognised(self):
        """'comic pause' enters paused state when prefix='comic'."""
        controller, _, clear = _controller_with_prefix("comic")
        result = controller.handle_transcript("comic pause")
        assert result is TranscriptDisposition.HANDLED
        assert controller.is_paused
        clear.assert_called_once()

    def test_reachy_still_works_with_comic_prefix(self):
        """Legacy 'reachy shutdown' must still work when prefix='comic'."""
        controller, on_shutdown, _ = _controller_with_prefix("comic")
        result = controller.handle_transcript("reachy shutdown")
        assert result is TranscriptDisposition.HANDLED
        assert controller.state is PauseState.SHUTTING_DOWN
        on_shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Env-var reload tests
# ---------------------------------------------------------------------------


class TestEnvVarReload:
    """_build_* helpers use the env var when called at module reload time."""

    def test_build_shutdown_phrases_robot(self):
        phrases = _build_shutdown_phrases("robot")
        assert "robot shutdown" in phrases
        assert "robot shut down" in phrases
        assert "reachy shutdown" in phrases
        assert "reachy shut down" in phrases

    def test_build_shutdown_phrases_comic(self):
        phrases = _build_shutdown_phrases("comic")
        assert "comic shutdown" in phrases
        assert "comic shut down" in phrases
        assert "reachy shutdown" in phrases  # always included

    def test_build_stop_phrases_robot(self):
        phrases = _build_stop_phrases("robot")
        assert "robot pause" in phrases
        assert "robot paws" in phrases
        assert "reachy pause" in phrases  # always included

    def test_build_stop_phrases_reachy_deduplicates(self):
        """When prefix IS 'reachy', reachy entries appear only once."""
        phrases = _build_shutdown_phrases("reachy")
        assert phrases.count("reachy shutdown") == 1
        assert phrases.count("reachy shut down") == 1

    def test_env_var_respected_on_module_load(self, monkeypatch):
        """DEFAULT_SHUTDOWN_PHRASES uses the env var value set before import."""
        monkeypatch.setenv("REACHY_MINI_TRIGGER_PREFIX", "comic")
        # Remove cached module so it is re-evaluated with the new env var
        sys.modules.pop("robot_comic.pause", None)
        import robot_comic.pause as pause_mod

        try:
            assert "comic shutdown" in pause_mod.DEFAULT_SHUTDOWN_PHRASES
            assert "reachy shutdown" in pause_mod.DEFAULT_SHUTDOWN_PHRASES
        finally:
            # Restore original module
            sys.modules.pop("robot_comic.pause", None)
            import robot_comic.pause  # noqa: F401  re-cache with real env


# ---------------------------------------------------------------------------
# Negative tests — unrelated phrases are NOT recognised
# ---------------------------------------------------------------------------


class TestUnrelatedPhrases:
    """Phrases that don't match the trigger vocabulary must be dispatched normally."""

    @pytest.mark.parametrize(
        "transcript",
        [
            "please shut down",
            "shutdown",
            "just stop it",
            "free cheese",  # the observed Moonshine mishearing for "reachy"
            "okay stop the music",
            "turn off",
            "power off",
            "halt",
        ],
    )
    def test_unrelated_phrase_is_dispatched(self, transcript: str):
        controller, on_shutdown, clear = _controller_with_prefix("robot")
        result = controller.handle_transcript(transcript)
        assert result is TranscriptDisposition.DISPATCH, f"Expected DISPATCH for {transcript!r}"
        assert controller.state is PauseState.ACTIVE
        on_shutdown.assert_not_called()
        clear.assert_not_called()

    def test_partial_prefix_alone_is_not_a_stop_phrase(self):
        """Just saying 'robot' without a keyword must not trigger anything."""
        controller, _, clear = _controller_with_prefix("robot")
        result = controller.handle_transcript("robot")
        assert result is TranscriptDisposition.DISPATCH
        clear.assert_not_called()

    def test_reversed_word_order_is_not_recognised(self):
        """'shutdown robot' (wrong order) must not match 'robot shutdown'."""
        controller, on_shutdown, _ = _controller_with_prefix("robot")
        result = controller.handle_transcript("shutdown robot")
        assert result is TranscriptDisposition.DISPATCH
        on_shutdown.assert_not_called()
