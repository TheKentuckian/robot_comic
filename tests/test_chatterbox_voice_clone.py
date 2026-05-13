"""Unit tests for chatterbox_voice_clone.load_voice_clone_ref.

These tests exercise pure path resolution — no audio is loaded and no Chatterbox
server is required.
"""

from __future__ import annotations
from pathlib import Path

import pytest

from robot_comic.chatterbox_voice_clone import load_voice_clone_ref


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(tmp_path: Path, filename: str | None) -> Path:
    """Create a minimal profile directory, optionally containing *filename*."""
    profile_dir = tmp_path / "test_persona"
    profile_dir.mkdir()
    if filename:
        (profile_dir / filename).write_bytes(b"")  # empty placeholder
    return profile_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_wav_when_only_wav_exists(tmp_path: Path) -> None:
    """Should return the .wav path when only voice_clone_ref.wav is present."""
    profile_dir = _make_profile(tmp_path, "voice_clone_ref.wav")
    result = load_voice_clone_ref(profile_dir)
    assert result is not None
    assert result.name == "voice_clone_ref.wav"
    assert result.exists()


def test_falls_through_to_mp3_when_wav_absent(tmp_path: Path) -> None:
    """Should return the .mp3 path when .wav is absent but .mp3 is present."""
    profile_dir = _make_profile(tmp_path, "voice_clone_ref.mp3")
    result = load_voice_clone_ref(profile_dir)
    assert result is not None
    assert result.name == "voice_clone_ref.mp3"


def test_falls_through_to_flac(tmp_path: Path) -> None:
    """Should return the .flac path when neither .wav nor .mp3 is present."""
    profile_dir = _make_profile(tmp_path, "voice_clone_ref.flac")
    result = load_voice_clone_ref(profile_dir)
    assert result is not None
    assert result.name == "voice_clone_ref.flac"


def test_falls_through_to_ogg(tmp_path: Path) -> None:
    """Should return the .ogg path as the last fallback."""
    profile_dir = _make_profile(tmp_path, "voice_clone_ref.ogg")
    result = load_voice_clone_ref(profile_dir)
    assert result is not None
    assert result.name == "voice_clone_ref.ogg"


def test_returns_none_when_no_candidate_file(tmp_path: Path) -> None:
    """Should return None when no voice_clone_ref.* file exists."""
    profile_dir = _make_profile(tmp_path, filename=None)
    result = load_voice_clone_ref(profile_dir)
    assert result is None


def test_wav_preferred_over_mp3(tmp_path: Path) -> None:
    """Should prefer .wav over .mp3 when both are present."""
    profile_dir = _make_profile(tmp_path, "voice_clone_ref.wav")
    (profile_dir / "voice_clone_ref.mp3").write_bytes(b"")
    result = load_voice_clone_ref(profile_dir)
    assert result is not None
    assert result.name == "voice_clone_ref.wav"


def test_logs_info_on_found(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Should log an INFO message with persona name and file size when ref found."""
    import logging

    profile_dir = _make_profile(tmp_path, "voice_clone_ref.wav")
    with caplog.at_level(logging.INFO, logger="robot_comic.chatterbox_voice_clone"):
        load_voice_clone_ref(profile_dir)
    assert any("voice clone ref" in record.message for record in caplog.records)
    assert any("test_persona" in record.message for record in caplog.records)


def test_logs_info_on_not_found(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Should log an INFO message when no ref is found."""
    import logging

    profile_dir = _make_profile(tmp_path, filename=None)
    with caplog.at_level(logging.INFO, logger="robot_comic.chatterbox_voice_clone"):
        load_voice_clone_ref(profile_dir)
    assert any("no voice_clone_ref" in record.message for record in caplog.records)
