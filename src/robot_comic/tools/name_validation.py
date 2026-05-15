"""Validate that an LLM-supplied ``name`` was actually spoken by the user.

Issue #287: realtime LLMs occasionally hallucinate a name (e.g. operator says
"Hello" and the LLM calls ``greet identify name="John"``).  Without a guard,
that name flows into ``crowd_work update`` and gets persisted across sessions.

The guard is a simple word-boundary, case-insensitive search of the LLM-supplied
name across the most recent N user transcripts kept on
``ToolDependencies.recent_user_transcripts``.

This module is deliberately tiny and import-free beyond ``re`` so any tool can
use it without circular-import risk.
"""

from __future__ import annotations
import re
import logging
from typing import Sequence


logger = logging.getLogger(__name__)

# How many of the most recent user transcripts the handler should keep around
# for tools to consult.  Each handler appends to ``deps.recent_user_transcripts``
# and trims to this bound.  Five turns is enough headroom for the LLM to ask
# the user's name on one turn and identify them on the next without losing
# the introduction.
RECENT_USER_TRANSCRIPTS_MAXLEN = 5


def record_user_transcript(transcripts: list[str], transcript: str | None) -> None:
    """Append ``transcript`` (if non-empty) to ``transcripts`` and trim.

    Called by handlers on each finalised user turn.  ``transcripts`` is mutated
    in place so existing references on ``ToolDependencies`` keep working.
    """
    if not transcript:
        return
    stripped = transcript.strip()
    if not stripped:
        return
    transcripts.append(stripped)
    overflow = len(transcripts) - RECENT_USER_TRANSCRIPTS_MAXLEN
    if overflow > 0:
        del transcripts[:overflow]


def name_in_transcripts(name: str, transcripts: Sequence[str]) -> bool:
    """Return True when ``name`` appears as a word in any of ``transcripts``.

    Match is case-insensitive and word-boundary aware so:
      * "tony" matches "Hi I'm Tony"
      * "Tony" matches "tony here"
      * "Anton" does NOT match "Antonio" (word-boundary)
      * empty / whitespace-only names always return False
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return False
    # ``\b`` in Python's ``re`` is unicode-aware on str patterns, so accented
    # names work as expected.  ``re.escape`` defends against names containing
    # regex metacharacters (rare but possible — e.g. someone says "C-3PO").
    pattern = re.compile(rf"\b{re.escape(cleaned)}\b", re.IGNORECASE)
    return any(pattern.search(t) for t in transcripts if t)


def validate_name_or_warn(
    name: str,
    transcripts: Sequence[str],
    *,
    tool_name: str,
) -> bool:
    """Return True if ``name`` is supported by ``transcripts``; log a WARNING otherwise.

    The warning includes the rejected name and the transcripts that were
    checked, which is the key forensic data for issue #287 follow-ups.
    """
    if name_in_transcripts(name, transcripts):
        return True
    logger.warning(
        "%s: rejected name %r — not found in recent user transcripts (last %d checked: %r). "
        "Treating as if no name was provided (likely LLM hallucination, see #287).",
        tool_name,
        name,
        len(transcripts),
        list(transcripts),
    )
    return False


__all__ = (
    "RECENT_USER_TRANSCRIPTS_MAXLEN",
    "record_user_transcript",
    "name_in_transcripts",
    "validate_name_or_warn",
)
