"""Unit tests for the per-sentence Gemini TTS styling helpers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from robot_comic.gemini_tts import (
    DEFAULT_TTS_SYSTEM_INSTRUCTION,
    extract_delivery_tags,
    build_tts_system_instruction,
    load_profile_tts_instruction,
)


def test_extract_delivery_tags_picks_up_known_tags() -> None:
    """All recognised delivery tags in a sentence are returned in order."""
    sentence = "[annoyance] Oh, look at you. [short pause] [fast] You hockey puck!"
    assert extract_delivery_tags(sentence) == ["annoyance", "short pause", "fast"]


def test_extract_delivery_tags_deduplicates_in_order() -> None:
    """Repeated tags appear only once, in first-seen order."""
    sentence = "[fast] One. [fast] Two. [annoyance] Three. [fast] Four."
    assert extract_delivery_tags(sentence) == ["fast", "annoyance"]


def test_extract_delivery_tags_ignores_unknown_tags() -> None:
    """Tags not in the known delivery vocabulary are skipped."""
    sentence = "[whisper] secret [fast] yell"
    assert extract_delivery_tags(sentence) == ["fast"]


def test_extract_delivery_tags_no_tags_returns_empty() -> None:
    """Plain text returns an empty list."""
    assert extract_delivery_tags("Plain old sentence.") == []


def test_build_tts_system_instruction_no_tags_returns_base() -> None:
    """When there are no tags, the base instruction is returned unchanged."""
    base = "Base prompt."
    assert build_tts_system_instruction(base, []) == base


def test_build_tts_system_instruction_appends_cue_suffix() -> None:
    """Tag names are joined into a 'Delivery cues for this line:' suffix."""
    base = "Base prompt."
    result = build_tts_system_instruction(base, ["fast", "annoyance"])
    assert result.startswith(base)
    assert "Delivery cues for this line: fast, annoyance." in result


def test_load_profile_tts_instruction_no_profile_falls_back() -> None:
    """With no active custom profile, the default Brooklyn prompt is returned."""
    with patch("robot_comic.gemini_tts.config") as fake_config:
        fake_config.REACHY_MINI_CUSTOM_PROFILE = None
        assert load_profile_tts_instruction() == DEFAULT_TTS_SYSTEM_INSTRUCTION


def test_load_profile_tts_instruction_reads_profile_file(tmp_path: Path) -> None:
    """A profile's gemini_tts.txt overrides the default instruction."""
    profile_dir = tmp_path / "test_persona"
    profile_dir.mkdir()
    (profile_dir / "gemini_tts.txt").write_text("Whispered, conspiratorial.\n", encoding="utf-8")

    with patch("robot_comic.gemini_tts.config") as fake_config:
        fake_config.REACHY_MINI_CUSTOM_PROFILE = "test_persona"
        fake_config.PROFILES_DIRECTORY = tmp_path
        assert load_profile_tts_instruction() == "Whispered, conspiratorial."


def test_load_profile_tts_instruction_missing_file_falls_back(tmp_path: Path) -> None:
    """Profiles without a gemini_tts.txt fall back to the default."""
    (tmp_path / "test_persona").mkdir()
    with patch("robot_comic.gemini_tts.config") as fake_config:
        fake_config.REACHY_MINI_CUSTOM_PROFILE = "test_persona"
        fake_config.PROFILES_DIRECTORY = tmp_path
        assert load_profile_tts_instruction() == DEFAULT_TTS_SYSTEM_INSTRUCTION


def test_load_profile_tts_instruction_empty_file_falls_back(tmp_path: Path) -> None:
    """A whitespace-only gemini_tts.txt falls back to the default."""
    profile_dir = tmp_path / "test_persona"
    profile_dir.mkdir()
    (profile_dir / "gemini_tts.txt").write_text("   \n", encoding="utf-8")
    with patch("robot_comic.gemini_tts.config") as fake_config:
        fake_config.REACHY_MINI_CUSTOM_PROFILE = "test_persona"
        fake_config.PROFILES_DIRECTORY = tmp_path
        assert load_profile_tts_instruction() == DEFAULT_TTS_SYSTEM_INSTRUCTION


@pytest.mark.parametrize(
    "tag",
    ["FAST", "Fast", "fAsT"],
)
def test_extract_delivery_tags_is_case_insensitive(tag: str) -> None:
    """Tag matching ignores letter case and normalises to lowercase."""
    sentence = f"[{tag}] go go go"
    assert extract_delivery_tags(sentence) == ["fast"]
