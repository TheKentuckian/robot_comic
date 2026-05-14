"""Tests for the welcome.wav.completed daemon-thread helper (#324).

PR #321 emitted a ``welcome.wav.played`` supporting-event span on Popen
dispatch — that span measures only the ~5-10 ms subprocess spawn cost, not
the ~2-3 s of actual audible playback. Issue #324 adds a daemon thread that
``wait()``s on the Popen object and emits ``welcome.wav.completed`` with the
real wall-clock window plus ``aplay.exit_code`` / ``aplay.command``
attributes, so the operator can see on the monitor whether the player
actually produced sound or exited non-zero.

The original dispatch span is preserved (operator preference); these tests
exercise only the new completion path.
"""

from __future__ import annotations
import time
import subprocess
from typing import Any, cast
from unittest.mock import MagicMock, patch

from robot_comic.warmup_audio import _wait_and_emit_completion


def _as_popen(p: Any) -> subprocess.Popen[bytes]:
    """Cast a fake Popen to satisfy mypy; runtime behaviour is unchanged."""
    return cast(subprocess.Popen[bytes], p)


# ---------------------------------------------------------------------------
# Fake Popen
# ---------------------------------------------------------------------------


class _FakePopen:
    """Minimal Popen stand-in covering ``poll`` / ``wait`` / ``returncode``."""

    def __init__(self, *, exit_code: int = 0, wait_delay: float = 0.0, already_done: bool = False) -> None:
        self._exit_code = exit_code
        self._wait_delay = wait_delay
        self._already_done = already_done
        self.returncode: int | None = exit_code if already_done else None
        self.wait_called = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        self.wait_called = True
        if self._wait_delay:
            time.sleep(self._wait_delay)
        self.returncode = self._exit_code
        return self._exit_code


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def test_returns_none_when_popen_is_none() -> None:
    """Failed-spawn / no-op call site path: helper must not crash."""
    with patch("robot_comic.telemetry.emit_supporting_event") as mock_emit:
        result = _wait_and_emit_completion(None, ["aplay", "x.wav"])
    assert result is None
    mock_emit.assert_not_called()


def test_thread_is_daemon() -> None:
    """The wait thread MUST be daemon so it doesn't block shutdown."""
    popen = _FakePopen(exit_code=0, wait_delay=0.0)
    with patch("robot_comic.telemetry.emit_supporting_event"):
        thread = _wait_and_emit_completion(_as_popen(popen),["aplay", "x.wav"])
    assert thread is not None, "Live Popen should produce a thread"
    assert thread.daemon is True, "Completion-wait thread must be daemon"
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "Thread should exit promptly after wait() returns"


def test_emits_completion_span_with_expected_attrs_on_exit_zero() -> None:
    """Successful playback: span carries dur_ms + exit_code=0 + command."""
    popen = _FakePopen(exit_code=0)
    captured: dict[str, Any] = {}

    def _spy(name: str, dur_ms: float | None = None, extra_attrs: dict[str, Any] | None = None) -> None:
        captured["name"] = name
        captured["dur_ms"] = dur_ms
        captured["attrs"] = extra_attrs or {}

    cmd = ["aplay", "-D", "plug:reachymini_audio_sink", "-q", "/tmp/welcome.wav"]
    started_at = time.monotonic()

    with patch("robot_comic.telemetry.emit_supporting_event", side_effect=_spy):
        thread = _wait_and_emit_completion(_as_popen(popen),cmd, started_at=started_at)
        assert thread is not None
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert captured["name"] == "welcome.wav.completed"
    assert captured["dur_ms"] is not None
    assert captured["dur_ms"] >= 0.0
    assert captured["attrs"]["aplay.exit_code"] == "0"
    assert captured["attrs"]["aplay.command"] == " ".join(cmd)
    assert popen.wait_called is True


def test_captures_nonzero_exit_code() -> None:
    """aplay failure path: span carries the non-zero exit_code for debug."""
    popen = _FakePopen(exit_code=2)
    captured: dict[str, Any] = {}

    def _spy(name: str, dur_ms: float | None = None, extra_attrs: dict[str, Any] | None = None) -> None:
        captured["name"] = name
        captured["attrs"] = extra_attrs or {}

    with patch("robot_comic.telemetry.emit_supporting_event", side_effect=_spy):
        thread = _wait_and_emit_completion(_as_popen(popen),["aplay", "missing.wav"])
        assert thread is not None
        thread.join(timeout=2.0)

    assert captured["name"] == "welcome.wav.completed"
    assert captured["attrs"]["aplay.exit_code"] == "2"


def test_already_exited_popen_emits_synchronously_without_thread() -> None:
    """When Popen has already exited by call time, no thread is started but
    the completion span still fires synchronously."""
    popen = _FakePopen(exit_code=0, already_done=True)
    captured: dict[str, Any] = {}

    def _spy(name: str, dur_ms: float | None = None, extra_attrs: dict[str, Any] | None = None) -> None:
        captured["name"] = name
        captured["dur_ms"] = dur_ms
        captured["attrs"] = extra_attrs or {}

    with patch("robot_comic.telemetry.emit_supporting_event", side_effect=_spy):
        result = _wait_and_emit_completion(
            _as_popen(popen),
            ["aplay", "x.wav"],
            started_at=time.monotonic(),
        )

    assert result is None, "Already-exited Popen must not spawn a thread"
    assert popen.wait_called is False, "wait() must not be called on already-exited Popen"
    assert captured["name"] == "welcome.wav.completed"
    assert captured["attrs"]["aplay.exit_code"] == "0"


def test_telemetry_failure_is_swallowed() -> None:
    """If emit_supporting_event raises inside the thread, no crash propagates."""
    popen = _FakePopen(exit_code=0)
    with patch(
        "robot_comic.telemetry.emit_supporting_event",
        side_effect=RuntimeError("telemetry exploded"),
    ):
        thread = _wait_and_emit_completion(_as_popen(popen),["aplay", "x.wav"])
        assert thread is not None
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "Thread must exit cleanly even on telemetry failure"


# ---------------------------------------------------------------------------
# Integration with call sites
# ---------------------------------------------------------------------------


def test_play_warmup_wav_wires_completion_helper(monkeypatch: Any, tmp_path: Any) -> None:
    """``play_warmup_wav`` POSIX path must call the completion helper."""
    # Force the POSIX branch (skip Windows winsound).
    monkeypatch.setattr("robot_comic.warmup_audio.sys.platform", "linux")
    monkeypatch.setattr("robot_comic.warmup_audio._PLAYER_CMD", ["/usr/bin/aplay", "-q"])
    monkeypatch.delenv("REACHY_MINI_EARLY_WELCOME_PLAYED", raising=False)
    monkeypatch.delenv("REACHY_MINI_WARMUP_BLIP_ENABLED", raising=False)

    wav = tmp_path / "welcome.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    fake_popen_obj = MagicMock()
    fake_popen_obj.poll.return_value = None  # still running

    with (
        patch("robot_comic.warmup_audio.subprocess.Popen", return_value=fake_popen_obj) as mock_popen,
        patch("robot_comic.warmup_audio._wait_and_emit_completion") as mock_wait,
        patch("robot_comic.startup_timer.log_checkpoint"),
        patch("robot_comic.telemetry.emit_supporting_event"),
    ):
        from robot_comic.warmup_audio import play_warmup_wav

        play_warmup_wav(wav)

    mock_popen.assert_called_once()
    mock_wait.assert_called_once()
    # The first positional arg must be the Popen object so the helper can wait on it.
    assert mock_wait.call_args[0][0] is fake_popen_obj
