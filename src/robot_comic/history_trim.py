"""Bounded conversation history helpers.

Long-running sessions (Gemini TTS, llama-server) accumulate every user/
assistant/tool turn in memory and feed the full transcript back to the LLM
on every request. Without a cap this eventually overflows the model's
context window or runs the token bill into the ground.

The trimmer keeps the last ``REACHY_MINI_MAX_HISTORY_TURNS`` *user* turns
plus everything attached to them (assistant replies, tool calls, tool
results). The system prompt is fed in separately by the caller so it is
never part of the history list and therefore never dropped.
"""

from __future__ import annotations
import os
import logging
from typing import Any


logger = logging.getLogger(__name__)


DEFAULT_MAX_HISTORY_TURNS: int = 20
MAX_HISTORY_TURNS_ENV: str = "REACHY_MINI_MAX_HISTORY_TURNS"


# Bracketed status markers that handlers inject into the assistant output queue
# for monitor visibility (e.g. "[Skipped TTS: tool-call limit reached]",
# "[TTS error — could not generate audio]", "[Gemini TTS rate-limited; ...]").
# These are NOT things the LLM actually said and must be filtered from the
# conversation history before being fed back to the model on the next turn —
# otherwise subsequent calls treat the status string as a real assistant turn.
# See issue #306. Match is defensive: the literal "[Skipped TTS:" prefix from
# PR #260, plus the related "[Skipped:", "[TTS error", "[ElevenLabs TTS rate-limited",
# "[Gemini TTS rate-limited", and generic "[error]" prefixes that share the
# same monitor-only intent.
_SYNTHETIC_STATUS_MARKER_PREFIXES: tuple[str, ...] = (
    "[Skipped TTS:",
    "[Skipped:",
    "[TTS error",
    "[ElevenLabs TTS rate-limited",
    "[Gemini TTS rate-limited",
    "[error]",
)


def is_synthetic_status_marker(content: str | None) -> bool:
    """Return True when ``content`` is a handler-injected status marker.

    Status markers are surfaced to the monitor via ``AdditionalOutputs`` so the
    operator sees the disposition (skipped TTS, rate-limit, TTS error, etc.),
    but they must not be appended to ``_conversation_history`` — the LLM should
    never see them on the next turn. Matches the bracketed prefixes listed in
    ``_SYNTHETIC_STATUS_MARKER_PREFIXES`` after stripping leading whitespace.
    Non-string or empty input returns False.
    """
    if not isinstance(content, str):
        return False
    stripped = content.lstrip()
    if not stripped:
        return False
    return stripped.startswith(_SYNTHETIC_STATUS_MARKER_PREFIXES)


def get_max_history_turns() -> int:
    """Resolve the configured per-session turn cap from the environment.

    Returns ``DEFAULT_MAX_HISTORY_TURNS`` for unset, empty, or invalid values,
    and ``0`` (disabled) only when explicitly configured as ``0``.
    """
    raw = os.getenv(MAX_HISTORY_TURNS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_HISTORY_TURNS
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default=%d",
            MAX_HISTORY_TURNS_ENV,
            raw,
            DEFAULT_MAX_HISTORY_TURNS,
        )
        return DEFAULT_MAX_HISTORY_TURNS
    if value < 0:
        logger.warning(
            "%s=%d is negative; using default=%d",
            MAX_HISTORY_TURNS_ENV,
            value,
            DEFAULT_MAX_HISTORY_TURNS,
        )
        return DEFAULT_MAX_HISTORY_TURNS
    return value


def trim_history_in_place(
    history: list[dict[str, Any]],
    *,
    role_key: str = "role",
    max_turns: int | None = None,
) -> int:
    """Drop the oldest user-turn groups until ``history`` has ``max_turns`` user turns.

    A "turn" here is one user message plus everything that came after it before
    the next user message (assistant reply, model tool-call turn, tool result
    turn). System messages are not expected in ``history`` — callers pass the
    system prompt separately — but if present they are preserved at the front.

    Returns the number of entries removed.
    """
    cap = max_turns if max_turns is not None else get_max_history_turns()
    if cap <= 0:
        return 0

    user_indices = [i for i, item in enumerate(history) if item.get(role_key) == "user"]
    if len(user_indices) <= cap:
        return 0

    drop_through = user_indices[-cap]
    leading_system: list[dict[str, Any]] = []
    for item in history[:drop_through]:
        if item.get(role_key) == "system":
            leading_system.append(item)
    new_history = leading_system + history[drop_through:]
    removed = len(history) - len(new_history)
    history[:] = new_history
    if removed > 0:
        logger.info(
            "Trimmed %d entries from conversation history (cap=%d user turns)",
            removed,
            cap,
        )
    return removed
