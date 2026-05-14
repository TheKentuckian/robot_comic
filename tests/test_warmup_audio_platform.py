"""Tests for warmup_audio platform-specific playback paths.

Covers:
- Linux player preference order: pw-play → paplay → aplay
- Windows winsound path
- macOS afplay path
- Missing WAV graceful no-op
- Blip WAV generation
- Blip playback dispatch (Linux + Windows)
"""

from __future__ import annotations
import io
import sys
import wave
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_wav_frames(wav_bytes: bytes) -> tuple[int, int, int, bytes]:
    """Return (n_channels, sample_width, framerate, raw_frames) for WAV bytes."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(wf.getnframes())


# ---------------------------------------------------------------------------
# _detect_player
# ---------------------------------------------------------------------------


class TestDetectPlayer:
    """_detect_player returns the correct player command list per platform."""

    def test_windows_returns_none(self) -> None:
        with patch.object(sys, "platform", "win32"):
            from importlib import reload

            import robot_comic.warmup_audio as wa

            reload(wa)
            # On Windows _detect_player should always return None
            assert wa._detect_player() is None

    def test_darwin_afplay_found(self) -> None:
        with patch.object(sys, "platform", "darwin"), patch("shutil.which", return_value="/usr/bin/afplay"):
            from robot_comic.warmup_audio import _detect_player

            result = _detect_player()
            assert result == ["/usr/bin/afplay"]

    def test_darwin_no_afplay(self) -> None:
        with patch.object(sys, "platform", "darwin"), patch("shutil.which", return_value=None):
            from robot_comic.warmup_audio import _detect_player

            result = _detect_player()
            assert result is None

    def test_linux_prefers_pw_play(self) -> None:
        with (
            patch.object(sys, "platform", "linux"),
            patch(
                "shutil.which",
                side_effect=lambda cmd: f"/usr/bin/{cmd}" if cmd == "pw-play" else None,
            ),
        ):
            from robot_comic.warmup_audio import _detect_player

            result = _detect_player()
            assert result == ["/usr/bin/pw-play"]

    def test_linux_falls_back_to_paplay(self) -> None:
        def _which(cmd: str) -> str | None:
            return f"/usr/bin/{cmd}" if cmd == "paplay" else None

        with patch.object(sys, "platform", "linux"), patch("shutil.which", side_effect=_which):
            from robot_comic.warmup_audio import _detect_player

            result = _detect_player()
            assert result == ["/usr/bin/paplay"]

    def test_linux_falls_back_to_aplay(self) -> None:
        def _which(cmd: str) -> str | None:
            return f"/usr/bin/{cmd}" if cmd == "aplay" else None

        with patch.object(sys, "platform", "linux"), patch("shutil.which", side_effect=_which):
            from robot_comic.warmup_audio import _detect_player

            result = _detect_player()
            # aplay is routed through `plug:reachymini_audio_sink` so the dmix
            # mixer can mediate access to /dev/snd/pcmC0D0p while the daemon
            # also has it mmap'd. See _detect_player() in warmup_audio.py for
            # the full rationale.
            assert result == ["/usr/bin/aplay", "-q", "-D", "plug:reachymini_audio_sink"]

    def test_linux_no_player_returns_none(self) -> None:
        with patch.object(sys, "platform", "linux"), patch("shutil.which", return_value=None):
            from robot_comic.warmup_audio import _detect_player

            result = _detect_player()
            assert result is None


# ---------------------------------------------------------------------------
# generate_blip_wav_bytes
# ---------------------------------------------------------------------------


class TestGenerateBlipWavBytes:
    """The blip generator produces a valid mono 16-bit PCM WAV."""

    def test_returns_bytes(self) -> None:
        from robot_comic.warmup_audio import generate_blip_wav_bytes

        result = generate_blip_wav_bytes()
        assert isinstance(result, bytes)
        assert len(result) > 44  # WAV header is 44 bytes minimum

    def test_valid_wav_structure(self) -> None:
        from robot_comic.warmup_audio import _BLIP_DURATION_S, _BLIP_SAMPLE_RATE, generate_blip_wav_bytes

        wav_bytes = generate_blip_wav_bytes()
        n_channels, sample_width, framerate, frames = _parse_wav_frames(wav_bytes)

        assert n_channels == 1, "expected mono"
        assert sample_width == 2, "expected 16-bit (2 bytes per sample)"
        assert framerate == _BLIP_SAMPLE_RATE

        expected_samples = int(_BLIP_SAMPLE_RATE * _BLIP_DURATION_S)
        actual_samples = len(frames) // sample_width
        # Allow ±1 sample for rounding
        assert abs(actual_samples - expected_samples) <= 1

    def test_samples_within_int16_range(self) -> None:
        from robot_comic.warmup_audio import generate_blip_wav_bytes

        wav_bytes = generate_blip_wav_bytes()
        _, _, _, frames = _parse_wav_frames(wav_bytes)
        n_samples = len(frames) // 2
        samples = struct.unpack(f"<{n_samples}h", frames)

        assert all(-32768 <= s <= 32767 for s in samples), "samples must be within int16 range"

    def test_not_silent(self) -> None:
        """The blip must actually contain non-zero audio data."""
        from robot_comic.warmup_audio import generate_blip_wav_bytes

        wav_bytes = generate_blip_wav_bytes()
        _, _, _, frames = _parse_wav_frames(wav_bytes)
        n_samples = len(frames) // 2
        samples = struct.unpack(f"<{n_samples}h", frames)

        assert any(s != 0 for s in samples), "blip must be non-silent"


# ---------------------------------------------------------------------------
# play_warmup_wav — missing file
# ---------------------------------------------------------------------------


class TestPlayWarmupWavMissingFile:
    """play_warmup_wav is a no-op when the WAV file does not exist."""

    def test_missing_wav_returns_without_spawn(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.wav"

        with (
            patch("robot_comic.warmup_audio._PLAYER_CMD", ["/usr/bin/pw-play"]),
            patch("subprocess.Popen") as mock_popen,
            patch("robot_comic.startup_timer.log_checkpoint"),
        ):
            from robot_comic.warmup_audio import play_warmup_wav

            play_warmup_wav(missing)

        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# play_warmup_wav — Linux subprocess path
# ---------------------------------------------------------------------------


class TestPlayWarmupWavLinux:
    """play_warmup_wav dispatches the correct subprocess on Linux."""

    def test_dispatches_pw_play(self, tmp_path: Path) -> None:
        wav = tmp_path / "welcome.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")  # minimal stub so is_file() passes

        with (
            patch.object(sys, "platform", "linux"),
            patch("robot_comic.warmup_audio._PLAYER_CMD", ["/usr/bin/pw-play"]),
            patch("subprocess.Popen") as mock_popen,
            patch("robot_comic.startup_timer.log_checkpoint"),
        ):
            from robot_comic.warmup_audio import play_warmup_wav

            play_warmup_wav(wav)

        mock_popen.assert_called_once()
        cmd_arg = mock_popen.call_args[0][0]
        assert cmd_arg[0] == "/usr/bin/pw-play"
        assert cmd_arg[-1] == str(wav)

    def test_no_player_logs_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        wav = tmp_path / "welcome.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        import robot_comic.warmup_audio as wa

        original_warned = wa._PLAYER_WARNED
        wa._PLAYER_WARNED = False
        try:
            with (
                patch.object(sys, "platform", "linux"),
                patch("robot_comic.warmup_audio._PLAYER_CMD", None),
                patch("subprocess.Popen") as mock_popen,
                patch("robot_comic.startup_timer.log_checkpoint"),
                patch.dict("os.environ", {"REACHY_MINI_WARMUP_BLIP_ENABLED": "0"}),
            ):
                import logging

                with caplog.at_level(logging.WARNING, logger="robot_comic.warmup_audio"):
                    wa.play_warmup_wav(wav)

            mock_popen.assert_not_called()
            assert any("No audio player available" in r.message for r in caplog.records)
        finally:
            wa._PLAYER_WARNED = original_warned


# ---------------------------------------------------------------------------
# play_warmup_wav — Windows winsound path
# ---------------------------------------------------------------------------


class TestPlayWarmupWavWindows:
    """play_warmup_wav uses winsound.PlaySound on Windows."""

    def test_dispatches_winsound(self, tmp_path: Path) -> None:
        wav = tmp_path / "welcome.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        mock_winsound = MagicMock()
        mock_winsound.SND_FILENAME = 0x00020000
        mock_winsound.SND_ASYNC = 0x00000001
        mock_winsound.SND_NODEFAULT = 0x00000002

        with (
            patch.object(sys, "platform", "win32"),
            patch.dict("sys.modules", {"winsound": mock_winsound}),
            patch("robot_comic.startup_timer.log_checkpoint"),
            patch.dict("os.environ", {"REACHY_MINI_WARMUP_BLIP_ENABLED": "0"}),
        ):
            import importlib

            import robot_comic.warmup_audio as wa

            importlib.reload(wa)
            wa.play_warmup_wav(wav)

        mock_winsound.PlaySound.assert_called_once()
        args, _ = mock_winsound.PlaySound.call_args
        assert args[0] == str(wav)
        # SND_ASYNC flag must be present in the flags bitmask
        assert args[1] & mock_winsound.SND_ASYNC

    def test_winsound_failure_does_not_raise(self, tmp_path: Path) -> None:
        wav = tmp_path / "welcome.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        mock_winsound = MagicMock()
        mock_winsound.SND_FILENAME = 0x00020000
        mock_winsound.SND_ASYNC = 0x00000001
        mock_winsound.SND_NODEFAULT = 0x00000002
        mock_winsound.PlaySound.side_effect = RuntimeError("audio device busy")

        with (
            patch.object(sys, "platform", "win32"),
            patch.dict("sys.modules", {"winsound": mock_winsound}),
            patch("robot_comic.startup_timer.log_checkpoint"),
            patch.dict("os.environ", {"REACHY_MINI_WARMUP_BLIP_ENABLED": "0"}),
        ):
            import importlib

            import robot_comic.warmup_audio as wa

            importlib.reload(wa)
            # Must not raise
            wa.play_warmup_wav(wav)


# ---------------------------------------------------------------------------
# play_warmup_blip — Linux and Windows
# ---------------------------------------------------------------------------


class TestPlayWarmupBlipLinux:
    """play_warmup_blip dispatches a temp WAV via subprocess on Linux."""

    def test_dispatches_subprocess(self) -> None:
        with (
            patch.object(sys, "platform", "linux"),
            patch("robot_comic.warmup_audio._PLAYER_CMD", ["/usr/bin/pw-play"]),
            patch("subprocess.Popen") as mock_popen,
            patch("atexit.register"),
        ):
            from robot_comic.warmup_audio import play_warmup_blip

            result = play_warmup_blip()

        assert result is True
        mock_popen.assert_called_once()
        cmd_arg = mock_popen.call_args[0][0]
        assert cmd_arg[0] == "/usr/bin/pw-play"
        assert cmd_arg[-1].endswith(".wav")

    def test_no_player_returns_false(self) -> None:
        with patch.object(sys, "platform", "linux"), patch("robot_comic.warmup_audio._PLAYER_CMD", None):
            from robot_comic.warmup_audio import play_warmup_blip

            result = play_warmup_blip()

        assert result is False


class TestPlayWarmupBlipWindows:
    """play_warmup_blip uses winsound on Windows."""

    def test_dispatches_winsound(self) -> None:
        mock_winsound = MagicMock()
        mock_winsound.SND_FILENAME = 0x00020000
        mock_winsound.SND_ASYNC = 0x00000001
        mock_winsound.SND_NODEFAULT = 0x00000002

        with (
            patch.object(sys, "platform", "win32"),
            patch.dict("sys.modules", {"winsound": mock_winsound}),
            patch("atexit.register"),
        ):
            import importlib

            import robot_comic.warmup_audio as wa

            importlib.reload(wa)
            result = wa.play_warmup_blip()

        assert result is True
        mock_winsound.PlaySound.assert_called_once()
        args, _ = mock_winsound.PlaySound.call_args
        assert args[0].endswith(".wav")
        assert args[1] & mock_winsound.SND_ASYNC


# ---------------------------------------------------------------------------
# Blip opt-in via env var in play_warmup_wav
# ---------------------------------------------------------------------------


class TestBlipEnvFlag:
    """play_warmup_wav calls play_warmup_blip only when the env flag is set."""

    def test_blip_called_when_enabled(self, tmp_path: Path) -> None:
        wav = tmp_path / "welcome.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        with (
            patch("robot_comic.warmup_audio.play_warmup_blip") as mock_blip,
            patch("robot_comic.warmup_audio._PLAYER_CMD", ["/usr/bin/pw-play"]),
            patch("subprocess.Popen"),
            patch("robot_comic.startup_timer.log_checkpoint"),
            patch.dict("os.environ", {"REACHY_MINI_WARMUP_BLIP_ENABLED": "1"}),
            patch.object(sys, "platform", "linux"),
        ):
            from robot_comic.warmup_audio import play_warmup_wav

            play_warmup_wav(wav)

        mock_blip.assert_called_once()

    def test_blip_not_called_when_disabled(self, tmp_path: Path) -> None:
        wav = tmp_path / "welcome.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        with (
            patch("robot_comic.warmup_audio.play_warmup_blip") as mock_blip,
            patch("robot_comic.warmup_audio._PLAYER_CMD", ["/usr/bin/pw-play"]),
            patch("subprocess.Popen"),
            patch("robot_comic.startup_timer.log_checkpoint"),
            patch.dict("os.environ", {"REACHY_MINI_WARMUP_BLIP_ENABLED": "0"}),
            patch.object(sys, "platform", "linux"),
        ):
            from robot_comic.warmup_audio import play_warmup_wav

            play_warmup_wav(wav)

        mock_blip.assert_not_called()

    def test_blip_not_called_by_default(self, tmp_path: Path) -> None:
        wav = tmp_path / "welcome.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        env_without_flag = {
            k: v for k, v in __import__("os").environ.items() if k != "REACHY_MINI_WARMUP_BLIP_ENABLED"
        }

        with (
            patch("robot_comic.warmup_audio.play_warmup_blip") as mock_blip,
            patch("robot_comic.warmup_audio._PLAYER_CMD", ["/usr/bin/pw-play"]),
            patch("subprocess.Popen"),
            patch("robot_comic.startup_timer.log_checkpoint"),
            patch.dict("os.environ", env_without_flag, clear=True),
            patch.object(sys, "platform", "linux"),
        ):
            from robot_comic.warmup_audio import play_warmup_wav

            play_warmup_wav(wav)

        mock_blip.assert_not_called()
