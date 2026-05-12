"""Pause controller for stop-word interruption and the paused-state menu.

The PauseController watches user transcripts as they complete. When a
stop-word phrase is detected it interrupts the active routine by clearing
the move queue (which drops any running dance/emotion primary move and
lets the breathing/rest pose resume) and enters a paused state. While
paused, subsequent transcripts are matched against a small keyword
grammar:

  * continue / resume          -> leave paused state
  * shutdown / power off       -> invoke the shutdown callback
  * switch / new comic         -> acknowledged (voice-driven persona
                                  selection is a follow-up)

The controller does *not* try to cancel the backend's in-progress LLM
response. Existing VAD interruption already cuts the LLM audio when the
user starts speaking, and a graceful menu prompt via TTS is left as a
follow-up to keep the first slice small.
"""

from __future__ import annotations
import re
import enum
import logging
import threading
from typing import Callable, Iterable, Optional, Sequence


logger = logging.getLogger(__name__)


DEFAULT_STOP_PHRASES: tuple[str, ...] = (
    "reachy pause",
)
DEFAULT_RESUME_PHRASES: tuple[str, ...] = (
    "continue",
    "resume",
    "keep going",
    "carry on",
)
DEFAULT_SHUTDOWN_PHRASES: tuple[str, ...] = (
    "reachy shutdown",
)
DEFAULT_SWITCH_PHRASES: tuple[str, ...] = (
    "switch",
    "switch comic",
    "switch comics",
    "new comic",
    "change comic",
    "change comics",
)


class TranscriptDisposition(enum.Enum):
    """Outcome of routing a completed user transcript through the controller."""

    DISPATCH = "dispatch"
    HANDLED = "handled"


class PauseState(enum.Enum):
    """High-level lifecycle state for the pause controller."""

    ACTIVE = "active"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"


_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for matching."""
    lowered = text.lower().strip()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", no_punct).strip()


def _phrase_matches(text: str, phrases: Iterable[str]) -> Optional[str]:
    """Return the first phrase that occurs as a whole-token match in text."""
    norm = _normalise(text)
    tokens = norm.split()
    for phrase in phrases:
        phrase_tokens = _normalise(phrase).split()
        if not phrase_tokens:
            continue
        for i in range(len(tokens) - len(phrase_tokens) + 1):
            if tokens[i : i + len(phrase_tokens)] == phrase_tokens:
                return phrase
    return None


class PauseController:
    """Routes user transcripts through a stop-word + paused-menu state machine."""

    def __init__(
        self,
        *,
        clear_move_queue: Callable[[], None],
        on_shutdown: Callable[[], None],
        on_switch_requested: Optional[Callable[[], None]] = None,
        on_pause_state_changed: Optional[Callable[[bool], None]] = None,
        stop_phrases: Sequence[str] = DEFAULT_STOP_PHRASES,
        resume_phrases: Sequence[str] = DEFAULT_RESUME_PHRASES,
        shutdown_phrases: Sequence[str] = DEFAULT_SHUTDOWN_PHRASES,
        switch_phrases: Sequence[str] = DEFAULT_SWITCH_PHRASES,
    ) -> None:
        """Build a controller wired to the supplied callbacks and phrase sets."""
        self._clear_move_queue = clear_move_queue
        self._on_shutdown = on_shutdown
        self._on_switch_requested = on_switch_requested
        self._on_pause_state_changed = on_pause_state_changed
        self._stop_phrases = tuple(stop_phrases)
        self._resume_phrases = tuple(resume_phrases)
        self._shutdown_phrases = tuple(shutdown_phrases)
        self._switch_phrases = tuple(switch_phrases)
        self._state = PauseState.ACTIVE
        self._lock = threading.Lock()

    @property
    def state(self) -> PauseState:
        """Return the current pause state (thread-safe snapshot)."""
        with self._lock:
            return self._state

    @property
    def is_paused(self) -> bool:
        """True iff the controller is currently in the paused state."""
        return self.state is PauseState.PAUSED

    def handle_transcript(self, transcript: str) -> TranscriptDisposition:
        """Inspect a completed user transcript and update state accordingly.

        Returns DISPATCH if the transcript should be forwarded to the LLM
        as normal, or HANDLED if the controller consumed it.
        """
        if not transcript or not transcript.strip():
            return TranscriptDisposition.DISPATCH

        with self._lock:
            state = self._state
            stop_phrases = self._stop_phrases
            resume_phrases = self._resume_phrases
            shutdown_phrases = self._shutdown_phrases
            switch_phrases = self._switch_phrases

        if state is PauseState.SHUTTING_DOWN:
            logger.debug("Pause controller already shutting down; ignoring transcript")
            return TranscriptDisposition.HANDLED

        # Shutdown phrases short-circuit from any state.
        shutdown_match = _phrase_matches(transcript, shutdown_phrases)
        if shutdown_match is not None:
            self._enter_shutdown(shutdown_match)
            return TranscriptDisposition.HANDLED

        if state is PauseState.ACTIVE:
            stop_match = _phrase_matches(transcript, stop_phrases)
            if stop_match is None:
                return TranscriptDisposition.DISPATCH
            self._enter_paused(stop_match)
            return TranscriptDisposition.HANDLED

        # PAUSED
        resume_match = _phrase_matches(transcript, resume_phrases)
        if resume_match is not None:
            self._enter_active(resume_match)
            return TranscriptDisposition.HANDLED

        switch_match = _phrase_matches(transcript, switch_phrases)
        if switch_match is not None:
            self._handle_switch(switch_match)
            return TranscriptDisposition.HANDLED

        logger.info(
            "Paused; ignoring transcript (say continue, reachy shutdown, or switch): %r",
            transcript,
        )
        return TranscriptDisposition.HANDLED

    def get_phrases(self) -> dict[str, tuple[str, ...]]:
        """Return a snapshot of the active phrase lists keyed by category."""
        with self._lock:
            return {
                "stop": self._stop_phrases,
                "resume": self._resume_phrases,
                "shutdown": self._shutdown_phrases,
                "switch": self._switch_phrases,
            }

    def update_phrases(
        self,
        *,
        stop: Sequence[str] | None = None,
        resume: Sequence[str] | None = None,
        shutdown: Sequence[str] | None = None,
        switch: Sequence[str] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Replace one or more phrase lists at runtime; returns the new snapshot.

        Each argument left as None preserves the current value. Pass an
        empty sequence to disable matching for that category entirely.
        """
        with self._lock:
            if stop is not None:
                self._stop_phrases = tuple(stop)
            if resume is not None:
                self._resume_phrases = tuple(resume)
            if shutdown is not None:
                self._shutdown_phrases = tuple(shutdown)
            if switch is not None:
                self._switch_phrases = tuple(switch)
            snapshot = {
                "stop": self._stop_phrases,
                "resume": self._resume_phrases,
                "shutdown": self._shutdown_phrases,
                "switch": self._switch_phrases,
            }
        logger.info(
            "Pause phrases updated (stop=%d, resume=%d, shutdown=%d, switch=%d)",
            len(snapshot["stop"]),
            len(snapshot["resume"]),
            len(snapshot["shutdown"]),
            len(snapshot["switch"]),
        )
        return snapshot

    def _enter_paused(self, matched_phrase: str) -> None:
        with self._lock:
            self._state = PauseState.PAUSED
        logger.info("Stop-word %r matched — entering paused state", matched_phrase)
        try:
            self._clear_move_queue()
        except Exception as e:
            logger.error("clear_move_queue raised during pause: %s", e)
        if self._on_pause_state_changed is not None:
            try:
                self._on_pause_state_changed(True)
            except Exception as e:
                logger.error("on_pause_state_changed(True) raised: %s", e)
        logger.info("Paused. Say 'continue', 'shutdown', or 'switch comic'.")

    def _enter_active(self, matched_phrase: str) -> None:
        with self._lock:
            self._state = PauseState.ACTIVE
        logger.info("Resume phrase %r matched — leaving paused state", matched_phrase)
        if self._on_pause_state_changed is not None:
            try:
                self._on_pause_state_changed(False)
            except Exception as e:
                logger.error("on_pause_state_changed(False) raised: %s", e)

    def _enter_shutdown(self, matched_phrase: str) -> None:
        with self._lock:
            self._state = PauseState.SHUTTING_DOWN
        logger.info("Shutdown phrase %r matched — invoking shutdown callback", matched_phrase)
        try:
            self._on_shutdown()
        except Exception as e:
            logger.error("on_shutdown raised: %s", e)

    def _handle_switch(self, matched_phrase: str) -> None:
        logger.info(
            "Switch phrase %r matched (voice-driven persona selection is a follow-up)",
            matched_phrase,
        )
        if self._on_switch_requested is not None:
            try:
                self._on_switch_requested()
            except Exception as e:
                logger.error("on_switch_requested raised: %s", e)
