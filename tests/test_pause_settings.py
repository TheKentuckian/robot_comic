"""Tests for the pause_settings persistence + payload parsing module."""

from __future__ import annotations
import json
from pathlib import Path

from robot_comic.pause import (
    DEFAULT_STOP_PHRASES,
    DEFAULT_RESUME_PHRASES,
    DEFAULT_SWITCH_PHRASES,
    DEFAULT_SHUTDOWN_PHRASES,
)
from robot_comic.pause_settings import (
    MAX_PHRASE_LENGTH,
    MAX_PHRASES_PER_FIELD,
    PAUSE_SETTINGS_FILENAME,
    PausePhraseSettings,
    read_pause_settings,
    write_pause_settings,
    settings_from_payload,
)


def test_read_missing_file_returns_empty_settings(tmp_path: Path) -> None:
    """If no settings file exists, reading returns the empty/defaults sentinel."""
    settings = read_pause_settings(tmp_path)
    assert settings == PausePhraseSettings()
    assert settings.resolved_stop() == DEFAULT_STOP_PHRASES
    assert settings.resolved_resume() == DEFAULT_RESUME_PHRASES
    assert settings.resolved_shutdown() == DEFAULT_SHUTDOWN_PHRASES
    assert settings.resolved_switch() == DEFAULT_SWITCH_PHRASES


def test_read_with_none_instance_path_returns_defaults() -> None:
    """Without an instance path there is no file to read; return defaults."""
    settings = read_pause_settings(None)
    assert settings == PausePhraseSettings()


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    """Writing settings then reading them back produces the same normalised values."""
    initial = PausePhraseSettings(
        stop=("System Pause", "Robot Stop"),
        resume=None,
        shutdown=("Power Off",),
        switch=("switch comic",),
    )
    written = write_pause_settings(tmp_path, initial)
    assert written == initial

    loaded = read_pause_settings(tmp_path)
    assert loaded.stop == ("system pause", "robot stop")
    assert loaded.shutdown == ("power off",)
    assert loaded.resume is None
    assert loaded.switch == ("switch comic",)
    assert loaded.resolved_resume() == DEFAULT_RESUME_PHRASES


def test_write_empty_settings_removes_file(tmp_path: Path) -> None:
    """Writing all-None settings deletes any existing JSON file."""
    settings_path = tmp_path / PAUSE_SETTINGS_FILENAME
    settings_path.write_text(json.dumps({"stop": ["sp"]}))
    assert settings_path.exists()

    result = write_pause_settings(tmp_path, PausePhraseSettings())
    assert result == PausePhraseSettings()
    assert not settings_path.exists()


def test_settings_from_payload_normalises_and_dedupes() -> None:
    """Payload values are lower-cased, whitespace-collapsed, deduped, and length-bounded."""
    payload = {
        "stop": ["  System Pause  ", "system pause", "Robot Stop"],
        "resume": [],
        "shutdown": None,
        "switch": ["Switch Comic"],
    }
    settings = settings_from_payload(payload)
    assert settings.stop == ("system pause", "robot stop")
    assert settings.resume == ()
    assert settings.shutdown is None
    assert settings.switch == ("switch comic",)


def test_settings_from_payload_rejects_non_dict() -> None:
    """A non-dict body decodes to all-None settings."""
    assert settings_from_payload(None) == PausePhraseSettings()
    assert settings_from_payload([1, 2, 3]) == PausePhraseSettings()


def test_phrases_too_long_are_dropped() -> None:
    """Single phrases longer than MAX_PHRASE_LENGTH are filtered out."""
    long_phrase = "x" * (MAX_PHRASE_LENGTH + 1)
    payload = {"stop": [long_phrase, "ok phrase"]}
    settings = settings_from_payload(payload)
    assert settings.stop == ("ok phrase",)


def test_phrase_count_is_capped() -> None:
    """Phrase lists are truncated to MAX_PHRASES_PER_FIELD entries."""
    payload = {"stop": [f"phrase {i}" for i in range(MAX_PHRASES_PER_FIELD + 5)]}
    settings = settings_from_payload(payload)
    assert settings.stop is not None
    assert len(settings.stop) == MAX_PHRASES_PER_FIELD


def test_invalid_json_file_falls_back_to_defaults(tmp_path: Path) -> None:
    """A corrupt JSON file is logged and treated as 'no settings'."""
    (tmp_path / PAUSE_SETTINGS_FILENAME).write_text("not-json{")
    settings = read_pause_settings(tmp_path)
    assert settings == PausePhraseSettings()


def test_non_dict_json_payload_falls_back_to_defaults(tmp_path: Path) -> None:
    """A JSON file containing a non-object value is ignored."""
    (tmp_path / PAUSE_SETTINGS_FILENAME).write_text("[1, 2, 3]")
    settings = read_pause_settings(tmp_path)
    assert settings == PausePhraseSettings()


def test_phrase_with_only_whitespace_is_dropped() -> None:
    """Empty/whitespace-only strings never enter the phrase list."""
    payload = {"stop": ["   ", "", "real phrase"]}
    settings = settings_from_payload(payload)
    assert settings.stop == ("real phrase",)
