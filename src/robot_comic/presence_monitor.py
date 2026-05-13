"""Presence-backoff monitor for Gemini Live.

After the robot asks a question and the user stays silent, this module
schedules up to ``max_attempts`` re-prompt nudges on an exponential-backoff
cadence (first_s → first_s * factor → first_s * factor² → …).  On the last
attempt the monitor enters a silent-wait state: outbound logic is unchanged
but no further probes are sent.  Any user activity (transcript chunk, explicit
cancel) resets the monitor back to IDLE.

Design principles
-----------------
* One ``asyncio.Task`` owns the sleep; it is cancelled cleanly on reset or
  shutdown.
* The monitor is a pure asyncio object — no threads, no locks.  All public
  methods must be called from the same event loop.
* The caller provides a ``probe_callback`` coroutine that the monitor awaits
  when firing.  If the callback raises the error is logged and the monitor
  continues to the next back-off cycle.
* Session reconnects: the handler calls ``cancel()`` before tearing down a
  session, which drops any pending timer without firing.

States
------
IDLE       — no timer running; waiting to be armed.
ARMED      — user was silent; timer is running for the next probe.
PAUSED     — max_attempts exhausted; listening silently; no more probes.
"""

from __future__ import annotations
import enum
import asyncio
import logging
from collections.abc import Callable, Awaitable


logger = logging.getLogger(__name__)

# Re-prompt nudge text.  The model sees this as synthetic user content so it
# has context to understand why it is being asked to check in again.
_NUDGE_TEXTS = [
    "[user has been silent — gently check if they are still there]",
    "[user still silent — try a different angle or ask something else]",
    "[user has not responded — make one last playful attempt to re-engage]",
]


class PresenceState(enum.Enum):
    """State machine values for PresenceMonitor."""

    IDLE = "idle"
    ARMED = "armed"
    PAUSED = "paused"


class PresenceMonitor:
    """Exponential-backoff presence monitor for Gemini Live sessions.

    Parameters
    ----------
    probe_callback:
        Async callable invoked each time a probe fires.  Receives the attempt
        number (1-based) and the nudge text to send.
    first_s:
        Delay in seconds before the first probe.
    backoff_factor:
        Multiplier applied to the delay after each probe.
    max_attempts:
        Number of probes before entering PAUSED state.

    """

    def __init__(
        self,
        probe_callback: Callable[[int, str], Awaitable[None]],
        *,
        first_s: float = 10.0,
        backoff_factor: float = 2.0,
        max_attempts: int = 3,
    ) -> None:
        """Store config and initialize state to IDLE."""
        self._probe_callback = probe_callback
        self._first_s = first_s
        self._backoff_factor = backoff_factor
        self._max_attempts = max_attempts

        self._state: PresenceState = PresenceState.IDLE
        self._attempt: int = 0
        self._next_delay: float = first_s
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> PresenceState:
        """Current state of the presence monitor."""
        return self._state

    @property
    def attempt(self) -> int:
        """Number of probes fired so far in the current arm cycle."""
        return self._attempt

    def arm(self) -> None:
        """Start the presence timer after a question-turn completes.

        Idempotent: if already ARMED, this is a no-op (the existing timer
        continues).  If PAUSED, resets to IDLE first then arms.
        """
        if self._state == PresenceState.ARMED:
            return
        # Reset state for a fresh arm cycle.
        self._attempt = 0
        self._next_delay = self._first_s
        self._state = PresenceState.ARMED
        self._cancel_task()
        self._task = asyncio.create_task(self._run(), name="presence-monitor")
        logger.debug("PresenceMonitor armed (first_s=%.1f, max=%d)", self._first_s, self._max_attempts)

    def cancel(self) -> None:
        """Cancel the pending timer and return to IDLE (no probe fires)."""
        if self._state == PresenceState.IDLE:
            return
        self._cancel_task()
        prev = self._state
        self._state = PresenceState.IDLE
        self._attempt = 0
        self._next_delay = self._first_s
        logger.debug("PresenceMonitor cancelled (was %s)", prev.value)

    def on_user_activity(self) -> None:
        """Reset the monitor when the user speaks or sends any transcript chunk."""
        if self._state != PresenceState.IDLE:
            logger.debug("PresenceMonitor: user activity — resetting from %s", self._state.value)
        self.cancel()

    def shutdown(self) -> None:
        """Hard stop — called when the handler shuts down or session reconnects."""
        self._cancel_task()
        self._state = PresenceState.IDLE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cancel_task(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _run(self) -> None:
        """Background task: sleep → fire probe → reschedule or pause."""
        try:
            while self._state == PresenceState.ARMED and self._attempt < self._max_attempts:
                delay = self._next_delay
                logger.debug(
                    "PresenceMonitor sleeping %.1fs (attempt %d/%d)",
                    delay,
                    self._attempt + 1,
                    self._max_attempts,
                )
                await asyncio.sleep(delay)

                # Double check we weren't cancelled/reset while sleeping.
                if self._state != PresenceState.ARMED:
                    return

                self._attempt += 1
                nudge_idx = min(self._attempt - 1, len(_NUDGE_TEXTS) - 1)
                nudge = _NUDGE_TEXTS[nudge_idx]

                logger.info(
                    "PresenceMonitor probe %d/%d — sending nudge: %r",
                    self._attempt,
                    self._max_attempts,
                    nudge,
                )
                try:
                    await self._probe_callback(self._attempt, nudge)
                except Exception as exc:
                    logger.error("PresenceMonitor probe callback raised: %s", exc)

                # Schedule next probe at 2× delay, or enter PAUSED on last.
                if self._attempt >= self._max_attempts:
                    self._state = PresenceState.PAUSED
                    logger.info(
                        "PresenceMonitor: max_attempts=%d reached — entering silent-wait (PAUSED)",
                        self._max_attempts,
                    )
                    return

                self._next_delay *= self._backoff_factor

        except asyncio.CancelledError:
            logger.debug("PresenceMonitor task cancelled")
            raise
        except Exception as exc:
            logger.error("PresenceMonitor unexpected error: %s", exc)
            self._state = PresenceState.IDLE
