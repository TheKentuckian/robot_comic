"""Tests for the early welcome WAV dispatch in ``robot_comic.main``.

The early-play block runs at module import time (before any non-stdlib
imports) so the operator hears the welcome greeting within ~1s of
``systemctl start`` instead of waiting 5-15s for fastrtc/google.genai/etc to
load. These tests exercise ``_play_welcome_early`` directly with a mocked
``subprocess.Popen`` rather than re-importing the module — the function is a
pure stdlib block by design and that makes it cleanly unit-testable.
"""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Importing ``robot_comic.main`` triggers ``_play_welcome_early()`` at module
# load. Ensure the skip env var is set BEFORE the import so the test host's
# aplay (if any) is not invoked and no leftover env flag pollutes other tests.
os.environ.setdefault("REACHY_MINI_SKIP_EARLY_WELCOME", "1")

from robot_comic.main import _play_welcome_early  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_early_played_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean env so dispatch decisions are testable."""
    monkeypatch.delenv("REACHY_MINI_EARLY_WELCOME_PLAYED", raising=False)
    monkeypatch.delenv("REACHY_MINI_SKIP_EARLY_WELCOME", raising=False)
    monkeypatch.delenv("REACHY_MINI_ALSA_DEVICE", raising=False)


def _make_assets_with(tmp_path: Path, *files: str) -> Path:
    """Create an ``assets/welcome/<file>`` layout matching the package root.

    The early-play helper resolves assets via
    ``Path(main.__file__).parents[2] / "assets" / "welcome"``. The tests do
    not stub that path; instead they monkeypatch ``Path.is_file`` so the
    decision logic is exercised without depending on what's actually shipped
    in the repo's ``assets/`` dir.
    """
    welcome = tmp_path / "assets" / "welcome"
    welcome.mkdir(parents=True)
    for name in files:
        (welcome / name).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    return welcome


def test_dispatches_aplay_subprocess_when_wav_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: spawn aplay with -D + the welcome WAV, set env flag."""
    mock_popen = MagicMock()
    with patch("robot_comic.main.subprocess.Popen", mock_popen):
        # Pretend the legacy welcome.wav is present on disk.
        original_is_file = Path.is_file

        def _is_file(self: Path) -> bool:
            if self.name == "welcome_intro.wav":
                return False
            if self.name == "welcome.wav":
                return True
            return original_is_file(self)

        monkeypatch.setattr(Path, "is_file", _is_file)

        _play_welcome_early()

    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "aplay"
    assert cmd[1] == "-D"
    # Default device is the daemon's dmix sink.
    assert cmd[2] == "plug:reachymini_audio_sink"
    assert cmd[3] == "-q"
    assert cmd[-1].endswith("welcome.wav")
    # The wait() / communicate() methods must NOT be invoked — Popen is
    # supposed to return immediately so heavy imports run in parallel.
    assert not mock_popen.return_value.wait.called
    assert not mock_popen.return_value.communicate.called
    # Successful dispatch sets the dedup flag so warmup_audio.py skips later.
    assert os.environ.get("REACHY_MINI_EARLY_WELCOME_PLAYED") == "1"


def test_prefers_welcome_intro_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PR #311's split file is present, dispatch the intro variant."""
    mock_popen = MagicMock()
    with patch("robot_comic.main.subprocess.Popen", mock_popen):

        def _is_file(self: Path) -> bool:
            return self.name in {"welcome_intro.wav", "welcome.wav"}

        monkeypatch.setattr(Path, "is_file", _is_file)

        _play_welcome_early()

    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]
    assert cmd[-1].endswith("welcome_intro.wav"), "should prefer split intro file when present"


def test_honours_alsa_device_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``REACHY_MINI_ALSA_DEVICE`` overrides the default dmix sink."""
    monkeypatch.setenv("REACHY_MINI_ALSA_DEVICE", "default")
    mock_popen = MagicMock()
    with patch("robot_comic.main.subprocess.Popen", mock_popen):

        def _is_file(self: Path) -> bool:
            return self.name == "welcome.wav"

        monkeypatch.setattr(Path, "is_file", _is_file)

        _play_welcome_early()

    cmd = mock_popen.call_args[0][0]
    assert cmd[2] == "default"


def test_skip_env_disables_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``REACHY_MINI_SKIP_EARLY_WELCOME=1`` short-circuits the helper."""
    monkeypatch.setenv("REACHY_MINI_SKIP_EARLY_WELCOME", "1")
    with patch("robot_comic.main.subprocess.Popen") as mock_popen:
        _play_welcome_early()

    mock_popen.assert_not_called()
    assert "REACHY_MINI_EARLY_WELCOME_PLAYED" not in os.environ


def test_no_asset_skips_silently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing welcome WAVs → no spawn, no env flag, no exception."""
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with patch("robot_comic.main.subprocess.Popen") as mock_popen:
        _play_welcome_early()

    mock_popen.assert_not_called()
    assert "REACHY_MINI_EARLY_WELCOME_PLAYED" not in os.environ


def test_aplay_missing_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """No aplay on PATH (dev workstation) → silent FileNotFoundError catch."""

    def _is_file(self: Path) -> bool:
        return self.name == "welcome.wav"

    monkeypatch.setattr(Path, "is_file", _is_file)

    with patch(
        "robot_comic.main.subprocess.Popen",
        side_effect=FileNotFoundError("aplay"),
    ):
        # Must not raise.
        _play_welcome_early()

    # Failed dispatch must NOT set the dedup flag — the warmup_audio fallback
    # path still needs to fire.
    assert "REACHY_MINI_EARLY_WELCOME_PLAYED" not in os.environ


def test_other_oserror_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permission errors / arbitrary OSErrors do not crash boot."""

    def _is_file(self: Path) -> bool:
        return self.name == "welcome.wav"

    monkeypatch.setattr(Path, "is_file", _is_file)

    with patch(
        "robot_comic.main.subprocess.Popen",
        side_effect=PermissionError("denied"),
    ):
        _play_welcome_early()

    assert "REACHY_MINI_EARLY_WELCOME_PLAYED" not in os.environ


def test_warmup_audio_skips_when_early_flag_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The in-process ``play_warmup_wav`` must respect the early-played flag."""
    monkeypatch.setenv("REACHY_MINI_EARLY_WELCOME_PLAYED", "1")

    wav = tmp_path / "welcome.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    with (
        patch("robot_comic.warmup_audio._PLAYER_CMD", ["/usr/bin/aplay", "-q"]),
        patch("subprocess.Popen") as mock_popen,
        patch("robot_comic.startup_timer.log_checkpoint"),
    ):
        from robot_comic.warmup_audio import play_warmup_wav

        play_warmup_wav(wav)

    mock_popen.assert_not_called()


def test_emits_welcome_wav_played_after_successful_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a successful Popen, ``welcome.wav.played`` must be emitted (#337).

    Prior to this fix the in-process ``play_warmup_wav`` returned early once
    ``REACHY_MINI_EARLY_WELCOME_PLAYED=1`` was set, and the early-play helper
    in ``main.py`` never emitted the event itself — so the monitor's boot
    timeline lost the dispatch row entirely whenever the early-play path was
    taken (i.e. always in the default config).
    """

    def _is_file(self: Path) -> bool:
        return self.name == "welcome.wav"

    monkeypatch.setattr(Path, "is_file", _is_file)
    with (
        patch("robot_comic.main.subprocess.Popen") as mock_popen,
        patch("robot_comic.telemetry.emit_supporting_event") as emit,
    ):
        _play_welcome_early()

    mock_popen.assert_called_once()
    # The .played emit happens once, after the Popen returns.
    played_calls = [c for c in emit.call_args_list if c.args and c.args[0] == "welcome.wav.played"]
    assert len(played_calls) == 1, f"expected one welcome.wav.played emit, got {emit.call_args_list}"
    args, kwargs = played_calls[0]
    assert args[0] == "welcome.wav.played"
    assert "dur_ms" in kwargs
    # Dispatch is essentially instantaneous in a mocked Popen, but it must
    # be a non-negative float.
    assert isinstance(kwargs["dur_ms"], float) and kwargs["dur_ms"] >= 0


def test_no_welcome_wav_played_when_dispatch_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the early-play helper short-circuits (no asset / aplay missing), no event fires."""
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with patch("robot_comic.telemetry.emit_supporting_event") as emit:
        _play_welcome_early()
    played_calls = [c for c in emit.call_args_list if c.args and c.args[0] == "welcome.wav.played"]
    assert played_calls == []


def test_subprocess_call_is_nonblocking(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper must use Popen (non-blocking), never run/check_call/check_output.

    Regression guard: switching this to ``subprocess.run`` would defeat the
    whole point of the change — the welcome WAV would block heavy imports
    instead of overlapping with them.
    """

    def _is_file(self: Path) -> bool:
        return self.name == "welcome.wav"

    monkeypatch.setattr(Path, "is_file", _is_file)

    with (
        patch("robot_comic.main.subprocess.Popen") as mock_popen,
        patch.object(subprocess, "run") as mock_run,
        patch.object(subprocess, "check_call") as mock_check_call,
        patch.object(subprocess, "check_output") as mock_check_output,
    ):
        _play_welcome_early()

    mock_popen.assert_called_once()
    mock_run.assert_not_called()
    mock_check_call.assert_not_called()
    mock_check_output.assert_not_called()
