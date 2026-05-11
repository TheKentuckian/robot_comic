"""Helpers for persisting configurable PauseController phrases.

The pause stop-word, resume, shutdown, and switch phrase lists are
exposed via the headless admin UI. Persisted values live in an
instance-local JSON file next to startup_settings.json.

The schema is intentionally permissive: an unset list falls back to
the controller defaults defined in `pause.py`. This keeps backwards
compatibility — existing instances without a saved file simply use
the built-in defaults.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from dataclasses import dataclass

from robot_comic.pause import (
    DEFAULT_STOP_PHRASES,
    DEFAULT_RESUME_PHRASES,
    DEFAULT_SWITCH_PHRASES,
    DEFAULT_SHUTDOWN_PHRASES,
)


logger = logging.getLogger(__name__)

PAUSE_SETTINGS_FILENAME = "pause_settings.json"

MAX_PHRASES_PER_FIELD = 32
MAX_PHRASE_LENGTH = 80


@dataclass(frozen=True)
class PausePhraseSettings:
    """Instance-local pause-phrase overrides selected from the admin UI.

    Each field is either a tuple of normalised phrases or None to mean
    "use the controller defaults".
    """

    stop: tuple[str, ...] | None = None
    resume: tuple[str, ...] | None = None
    shutdown: tuple[str, ...] | None = None
    switch: tuple[str, ...] | None = None

    def resolved_stop(self) -> tuple[str, ...]:
        """Return the effective stop phrases (override or defaults)."""
        return self.stop if self.stop is not None else DEFAULT_STOP_PHRASES

    def resolved_resume(self) -> tuple[str, ...]:
        """Return the effective resume phrases (override or defaults)."""
        return self.resume if self.resume is not None else DEFAULT_RESUME_PHRASES

    def resolved_shutdown(self) -> tuple[str, ...]:
        """Return the effective shutdown phrases (override or defaults)."""
        return self.shutdown if self.shutdown is not None else DEFAULT_SHUTDOWN_PHRASES

    def resolved_switch(self) -> tuple[str, ...]:
        """Return the effective switch phrases (override or defaults)."""
        return self.switch if self.switch is not None else DEFAULT_SWITCH_PHRASES


def _settings_path(instance_path: str | Path | None) -> Path | None:
    """Return the pause-settings JSON path for an instance directory, or None."""
    if instance_path is None:
        return None
    return Path(instance_path) / PAUSE_SETTINGS_FILENAME


def _normalise_phrase(value: object) -> str | None:
    """Lower-case, collapse whitespace, and validate a single phrase."""
    if not isinstance(value, str):
        return None
    trimmed = " ".join(value.lower().split())
    if not trimmed or len(trimmed) > MAX_PHRASE_LENGTH:
        return None
    return trimmed


def _normalise_phrase_list(value: object) -> tuple[str, ...] | None:
    """Validate a phrase list and return a deduplicated tuple, or None."""
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    seen: set[str] = set()
    normalised: list[str] = []
    for item in value[:MAX_PHRASES_PER_FIELD]:
        phrase = _normalise_phrase(item)
        if phrase is None or phrase in seen:
            continue
        seen.add(phrase)
        normalised.append(phrase)
    return tuple(normalised)


def read_pause_settings(instance_path: str | Path | None) -> PausePhraseSettings:
    """Load pause phrase overrides from the instance directory."""
    settings_path = _settings_path(instance_path)
    if settings_path is None or not settings_path.exists():
        return PausePhraseSettings()

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read pause settings from %s: %s", settings_path, exc)
        return PausePhraseSettings()

    if not isinstance(payload, dict):
        logger.warning("Ignoring invalid pause settings payload from %s: %r", settings_path, payload)
        return PausePhraseSettings()

    return PausePhraseSettings(
        stop=_normalise_phrase_list(payload.get("stop")),
        resume=_normalise_phrase_list(payload.get("resume")),
        shutdown=_normalise_phrase_list(payload.get("shutdown")),
        switch=_normalise_phrase_list(payload.get("switch")),
    )


def write_pause_settings(
    instance_path: str | Path | None,
    settings: PausePhraseSettings,
) -> PausePhraseSettings:
    """Persist pause phrase overrides; return the canonicalised settings actually written."""
    settings_path = _settings_path(instance_path)
    if settings_path is None:
        return settings

    payload: dict[str, list[str]] = {}
    if settings.stop is not None:
        payload["stop"] = list(settings.stop)
    if settings.resume is not None:
        payload["resume"] = list(settings.resume)
    if settings.shutdown is not None:
        payload["shutdown"] = list(settings.shutdown)
    if settings.switch is not None:
        payload["switch"] = list(settings.switch)

    if not payload:
        try:
            settings_path.unlink()
        except FileNotFoundError:
            pass
        return PausePhraseSettings()

    settings_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return settings


def settings_from_payload(payload: object) -> PausePhraseSettings:
    """Build a PausePhraseSettings from an arbitrary JSON-like payload (e.g. a POST body)."""
    if not isinstance(payload, dict):
        return PausePhraseSettings()
    return PausePhraseSettings(
        stop=_normalise_phrase_list(payload.get("stop")),
        resume=_normalise_phrase_list(payload.get("resume")),
        shutdown=_normalise_phrase_list(payload.get("shutdown")),
        switch=_normalise_phrase_list(payload.get("switch")),
    )
