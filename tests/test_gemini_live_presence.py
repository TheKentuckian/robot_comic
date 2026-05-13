"""Unit tests for PresenceMonitor and _is_question_turn helper."""

from __future__ import annotations
import asyncio
from typing import List, Tuple

import pytest

from robot_comic.gemini_live import _is_question_turn
from robot_comic.presence_monitor import PresenceState, PresenceMonitor


# ---------------------------------------------------------------------------
# PresenceMonitor tests
# ---------------------------------------------------------------------------


def _make_monitor(
    first_s: float = 0.01,
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
) -> Tuple[PresenceMonitor, List[Tuple[int, str]]]:
    """Return a (monitor, fired_probes) pair.  fired_probes accumulates calls."""
    fired: List[Tuple[int, str]] = []

    async def probe_cb(attempt: int, nudge: str) -> None:
        fired.append((attempt, nudge))

    monitor = PresenceMonitor(
        probe_callback=probe_cb,
        first_s=first_s,
        backoff_factor=backoff_factor,
        max_attempts=max_attempts,
    )
    return monitor, fired


class TestPresenceMonitorArm:
    @pytest.mark.asyncio
    async def test_arm_schedules_first_probe(self) -> None:
        """arm() fires the first probe after first_s."""
        monitor, fired = _make_monitor(first_s=0.01)
        assert monitor.state == PresenceState.IDLE

        monitor.arm()
        assert monitor.state == PresenceState.ARMED

        # Wait long enough for all probes to fire.
        await asyncio.sleep(0.5)
        assert len(fired) >= 1
        assert fired[0][0] == 1  # first probe is attempt=1

    @pytest.mark.asyncio
    async def test_arm_while_armed_is_noop(self) -> None:
        """arm() when already ARMED does not double-schedule."""
        monitor, fired = _make_monitor(first_s=0.05)
        monitor.arm()
        task_before = monitor._task

        monitor.arm()  # second call — must be no-op
        assert monitor._task is task_before  # same task object
        assert monitor.state == PresenceState.ARMED

        monitor.cancel()

    @pytest.mark.asyncio
    async def test_arm_from_paused_resets(self) -> None:
        """arm() when PAUSED resets to IDLE then starts a fresh cycle."""
        monitor, fired = _make_monitor(first_s=0.01, max_attempts=1)
        monitor.arm()

        # Wait for PAUSED state
        await asyncio.sleep(0.3)
        assert monitor.state == PresenceState.PAUSED

        # Re-arm should work
        monitor.arm()
        assert monitor.state == PresenceState.ARMED
        monitor.cancel()


class TestPresenceMonitorCancel:
    @pytest.mark.asyncio
    async def test_cancel_prevents_probe(self) -> None:
        """cancel() after arm() prevents the probe from firing."""
        monitor, fired = _make_monitor(first_s=0.05)
        monitor.arm()
        monitor.cancel()

        await asyncio.sleep(0.15)
        assert fired == []
        assert monitor.state == PresenceState.IDLE

    @pytest.mark.asyncio
    async def test_cancel_is_idempotent(self) -> None:
        """cancel() can be called multiple times without error."""
        monitor, _ = _make_monitor(first_s=0.05)
        monitor.cancel()  # cancel when IDLE
        monitor.cancel()  # cancel again
        monitor.arm()
        monitor.cancel()
        monitor.cancel()  # cancel after cancel

    @pytest.mark.asyncio
    async def test_on_user_activity_cancels(self) -> None:
        """on_user_activity() resets the monitor to IDLE."""
        monitor, fired = _make_monitor(first_s=0.05)
        monitor.arm()
        assert monitor.state == PresenceState.ARMED

        monitor.on_user_activity()
        assert monitor.state == PresenceState.IDLE

        await asyncio.sleep(0.15)
        assert fired == []


class TestPresenceMonitorBackoff:
    @pytest.mark.asyncio
    async def test_probe_delays_follow_exponential_backoff(self) -> None:
        """Probe n fires after first_s * factor^(n-1) cumulative delay."""
        # Use a small first_s so tests stay fast.  We can't measure exact delays
        # inside asyncio, but we verify all probes fire in order with the right
        # attempt numbers.
        monitor, fired = _make_monitor(first_s=0.01, max_attempts=3, backoff_factor=2.0)
        monitor.arm()

        # Total time: 0.01 + 0.02 + 0.04 = 0.07s; give plenty of margin.
        await asyncio.sleep(0.5)

        assert len(fired) == 3
        attempts = [a for a, _ in fired]
        assert attempts == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_reaches_paused_after_max_attempts(self) -> None:
        """After max_attempts probes the monitor enters PAUSED."""
        monitor, fired = _make_monitor(first_s=0.01, max_attempts=2, backoff_factor=2.0)
        monitor.arm()

        await asyncio.sleep(0.3)
        assert monitor.state == PresenceState.PAUSED
        assert len(fired) == 2

    @pytest.mark.asyncio
    async def test_no_probe_fires_after_paused(self) -> None:
        """No additional probes fire once PAUSED state is reached."""
        monitor, fired = _make_monitor(first_s=0.01, max_attempts=1)
        monitor.arm()

        await asyncio.sleep(0.3)
        assert monitor.state == PresenceState.PAUSED
        count_at_pause = len(fired)

        # Wait longer — should not fire more probes
        await asyncio.sleep(0.2)
        assert len(fired) == count_at_pause


class TestPresenceMonitorShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_stops_task(self) -> None:
        """shutdown() cancels the background task without raising."""
        monitor, fired = _make_monitor(first_s=0.1)
        monitor.arm()
        assert monitor.state == PresenceState.ARMED

        monitor.shutdown()
        await asyncio.sleep(0.2)
        assert fired == []  # nothing fired after shutdown


# ---------------------------------------------------------------------------
# _is_question_turn tests
# ---------------------------------------------------------------------------


class TestIsQuestionTurn:
    @pytest.mark.parametrize(
        "transcript,expected",
        [
            # Question mark
            ("Are you still there?", True),
            ("What do you think?", True),
            ("You okay?", True),
            # Interrogative words
            ("Who are you talking to", True),
            ("What is going on", True),
            ("Where did you go", True),
            ("When will you reply", True),
            ("Why are you silent", True),
            ("How do you feel", True),
            # Common question phrases
            ("Are you still with me", True),
            ("Do you have anything to say", True),
            ("Can you hear me", True),
            ("Would you like to continue", True),
            ("Did you understand", True),
            # Statements that are NOT questions
            ("I told you the story", False),
            ("The robot dances well", False),
            ("Let me know if you need help", False),
            ("I was just saying hello", False),
            # Edge cases
            ("", False),
            ("   ", False),
        ],
    )
    def test_classification(self, transcript: str, expected: bool) -> None:
        assert _is_question_turn(transcript) is expected, f"_is_question_turn({transcript!r}) expected {expected}"
