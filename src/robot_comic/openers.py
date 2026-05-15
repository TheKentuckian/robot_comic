"""Canned per-persona startup openers.

Issue #290: the original ``_send_startup_trigger`` dispatched a synthetic
``[conversation started]`` transcript through the full LLM pipeline so the
model would speak an in-character opener. Under load (especially Gemini's
free-tier quotas) that round-trip frequently tool-storms or returns an empty
STOP, leaving the robot mute at boot.

This module replaces that fragile LLM round-trip with a canned line read from
``profiles/<persona>/openers.txt`` (one line per opener). The handler enqueues
the chosen line directly to TTS and writes it to ``_conversation_history`` as
a ``model`` turn so subsequent LLM calls see what the robot "said".
"""

from __future__ import annotations
import random
import logging
from pathlib import Path

from robot_comic.config import DEFAULT_PROFILES_DIRECTORY, config


logger = logging.getLogger(__name__)

OPENERS_FILENAME = "openers.txt"

# Last-resort fallback when no openers file exists and no profile is active.
# Deliberately neutral — by design we do NOT fall back to the old LLM path
# because that fragility is exactly what #290 fixes.
_DEFAULT_FALLBACK_OPENER = "Well, hello there."


def _profile_openers_path(profile: str | None) -> Path | None:
    """Return the path to the openers file for *profile*, or None for defaults."""
    if not profile:
        return None
    try:
        return config.PROFILES_DIRECTORY / profile / OPENERS_FILENAME
    except Exception:  # pragma: no cover - defensive
        return None


def _read_openers(path: Path) -> list[str]:
    """Read non-empty, non-comment lines from *path*."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not read openers file %s: %s", path, exc)
        return []
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def load_openers(profile: str | None = None) -> list[str]:
    """Return the list of opener lines for *profile*.

    Lookup order:
      1. ``<PROFILES_DIRECTORY>/<profile>/openers.txt`` if ``profile`` set.
      2. ``<DEFAULT_PROFILES_DIRECTORY>/default/openers.txt`` as a baseline.

    Returns an empty list if no openers file is found.
    """
    candidates: list[Path] = []
    profile_path = _profile_openers_path(profile)
    if profile_path is not None:
        candidates.append(profile_path)
    # Always consult the bundled default as a safety net so brand-new external
    # profiles still get a sensible opener instead of the hardcoded fallback.
    candidates.append(DEFAULT_PROFILES_DIRECTORY / "default" / OPENERS_FILENAME)

    for path in candidates:
        if path.exists():
            lines = _read_openers(path)
            if lines:
                logger.debug("Loaded %d opener(s) from %s", len(lines), path)
                return lines
    return []


def get_canned_opener(profile: str | None = None) -> str:
    """Return a random opener for *profile*, or a hardcoded fallback.

    If *profile* is None, the currently-active profile from
    ``config.REACHY_MINI_CUSTOM_PROFILE`` is used.
    """
    if profile is None:
        profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    openers = load_openers(profile)
    if not openers:
        logger.warning(
            "No openers found for profile %r; using hardcoded fallback %r",
            profile,
            _DEFAULT_FALLBACK_OPENER,
        )
        return _DEFAULT_FALLBACK_OPENER
    return random.choice(openers)
