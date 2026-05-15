"""Helpers for persisting UI-selected startup profile and voice settings."""

from __future__ import annotations
import os
import json
import logging
from typing import Any
from pathlib import Path
from dataclasses import dataclass


logger = logging.getLogger(__name__)

STARTUP_SETTINGS_FILENAME = "startup_settings.json"

MOVEMENT_SPEED_MIN = 0.1
MOVEMENT_SPEED_MAX = 2.0

# Sentinel used by write_startup_settings to distinguish "caller did not pass
# this key" from "caller passed None to clear it". Lets callers patch a single
# field without clobbering the others on disk.
_UNSET: Any = object()


@dataclass(frozen=True)
class StartupSettings:
    """Instance-local startup profile/voice/movement settings selected from the UI."""

    profile: str | None = None
    voice: str | None = None
    movement_speed: float | None = None


def _normalize_optional_text(value: object) -> str | None:
    """Return a stripped string or None for empty/non-string values."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_optional_movement_speed(value: object) -> float | None:
    """Return a clamped float in [MOVEMENT_SPEED_MIN, MOVEMENT_SPEED_MAX] or None."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    f = float(value)
    if f != f:  # NaN
        return None
    return max(MOVEMENT_SPEED_MIN, min(MOVEMENT_SPEED_MAX, f))


def _startup_settings_path(instance_path: str | Path | None) -> Path | None:
    """Return the startup settings JSON path for an instance directory."""
    if instance_path is None:
        return None
    return Path(instance_path) / STARTUP_SETTINGS_FILENAME


def read_startup_settings(instance_path: str | Path | None) -> StartupSettings:
    """Read startup settings from an instance-local JSON file."""
    settings_path = _startup_settings_path(instance_path)
    if settings_path is None or not settings_path.exists():
        return StartupSettings()

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read startup settings from %s: %s", settings_path, exc)
        return StartupSettings()

    if not isinstance(payload, dict):
        logger.warning("Ignoring invalid startup settings payload from %s: %r", settings_path, payload)
        return StartupSettings()

    return StartupSettings(
        profile=_normalize_optional_text(payload.get("profile")),
        voice=_normalize_optional_text(payload.get("voice")),
        movement_speed=_normalize_optional_movement_speed(payload.get("movement_speed")),
    )


def write_startup_settings(
    instance_path: str | Path | None,
    *,
    profile: str | None | Any = _UNSET,
    voice: str | None | Any = _UNSET,
    movement_speed: float | None | Any = _UNSET,
) -> None:
    """Persist startup settings in an instance-local JSON file.

    Each keyword acts as a patch: keys left at _UNSET preserve their current
    on-disk value, so callers can update one field (e.g. movement_speed) without
    touching the others. Passing an explicit ``None`` clears that field.
    """
    settings_path = _startup_settings_path(instance_path)
    if settings_path is None:
        return

    existing = read_startup_settings(instance_path)
    new_profile = existing.profile if profile is _UNSET else _normalize_optional_text(profile)
    new_voice = existing.voice if voice is _UNSET else _normalize_optional_text(voice)
    new_movement_speed = (
        existing.movement_speed if movement_speed is _UNSET else _normalize_optional_movement_speed(movement_speed)
    )

    if new_profile is None and new_voice is None and new_movement_speed is None:
        try:
            settings_path.unlink()
        except FileNotFoundError:
            return
        return

    payload: dict[str, str | float] = {}
    if new_profile is not None:
        payload["profile"] = new_profile
    if new_voice is not None:
        payload["voice"] = new_voice
    if new_movement_speed is not None:
        payload["movement_speed"] = new_movement_speed

    settings_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


def load_startup_settings_into_runtime(instance_path: str | Path | None) -> StartupSettings:
    """Load instance-local startup settings when no explicit profile override is set."""
    from robot_comic.config import LOCKED_PROFILE, set_custom_profile

    if LOCKED_PROFILE is not None:
        return StartupSettings()

    settings_path = _startup_settings_path(instance_path)
    settings = read_startup_settings(instance_path)
    if settings_path is None or not settings_path.exists():
        if os.getenv("REACHY_MINI_CUSTOM_PROFILE"):
            return StartupSettings(voice=settings.voice)

    set_custom_profile(settings.profile)
    return settings
