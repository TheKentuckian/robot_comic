"""Persistent joke history for avoid-repeat prompt injection.

Stores the last N punchlines across sessions in ``~/.robot-comic/joke-history.json``
and appends a "don't repeat" section to the system prompt at each session start.

Enable/disable via the ``REACHY_MINI_JOKE_HISTORY_ENABLED`` env var (default: True).
"""

from __future__ import annotations
import os
import re
import json
import logging
import tempfile
from typing import Any
from pathlib import Path
from datetime import datetime, timezone


logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_DIR = Path.home() / ".robot-comic"
_DEFAULT_HISTORY_FILENAME = "joke-history.json"
_DEFAULT_MAX_ENTRIES = 50
_DEFAULT_RECENT_N = 10

# Split on sentence-ending punctuation, keeping the terminator attached.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def last_sentence_of(text: str) -> str:
    """Extract the trailing sentence from *text*.

    Splits on ``.``, ``!``, or ``?`` boundaries and returns the last
    non-empty segment.  Returns the whole string (stripped) when no
    sentence terminator is found, or an empty string when *text* is empty.
    """
    text = text.strip()
    if not text:
        return ""
    parts = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
    return parts[-1] if parts else text


def default_history_path() -> Path:
    """Return the default joke-history file path, creating the parent dir if needed."""
    try:
        _DEFAULT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create joke-history directory %s: %s", _DEFAULT_HISTORY_DIR, exc)
    return _DEFAULT_HISTORY_DIR / _DEFAULT_HISTORY_FILENAME


class JokeHistory:
    """FIFO store for recent punchlines with prompt-injection support.

    Args:
        path: Path to the JSON file used for persistence.
        max_entries: Maximum number of entries to keep (oldest are dropped).

    """

    def __init__(self, path: Path, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        """Initialise the store, loading any existing entries from *path*."""
        self._path = path
        self._max_entries = max_entries
        self._entries: list[dict[str, Any]] = self.load()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def load(self) -> list[dict[str, Any]]:
        """Load entries from disk.  Returns ``[]`` if the file is missing or unreadable."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            logger.warning("joke-history file %s has unexpected format; starting fresh", self._path)
        except Exception as exc:
            logger.warning("Could not read joke-history from %s: %s", self._path, exc)
        return []

    def save(self, entries: list[dict[str, Any]]) -> None:
        """Atomically write *entries* to disk (tmp file + rename)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent,
                prefix=".joke-history-",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(entries, fh, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self._path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.warning("Could not save joke-history to %s: %s", self._path, exc)

    # ------------------------------------------------------------------ #
    # Mutation                                                             #
    # ------------------------------------------------------------------ #

    def add(self, punchline: str, topic: str = "") -> None:
        """Append a new punchline entry and auto-save.

        Truncates the in-memory list to the last ``max_entries`` before saving.
        """
        punchline = punchline.strip()
        if not punchline:
            logger.debug("joke_history.add: empty punchline, skipping")
            return
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "punchline": punchline,
            "topic": topic.strip(),
        }
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        self.save(self._entries)

    # ------------------------------------------------------------------ #
    # Query / formatting                                                   #
    # ------------------------------------------------------------------ #

    def recent(self, n: int = _DEFAULT_RECENT_N) -> list[dict[str, Any]]:
        """Return the last *n* entries in chronological order (oldest first)."""
        return list(self._entries[-n:])

    def format_for_prompt(self, n: int = _DEFAULT_RECENT_N) -> str:
        """Return a formatted block ready to drop into the system prompt.

        Returns an empty string when there are no recent entries so callers
        can skip the section entirely.
        """
        entries = self.recent(n)
        if not entries:
            return ""
        lines = ["## RECENT JOKES (DO NOT REPEAT)", ""]
        lines.append("Recent jokes you've told (avoid repeating these themes/punchlines):")
        for entry in entries:
            punchline = entry.get("punchline", "").strip()
            if punchline:
                lines.append(f"- {punchline}")
        lines.append("")
        return "\n".join(lines)
