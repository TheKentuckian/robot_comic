"""Tests for ``welcome.wav.completed`` daemon-thread emission (issue #324).

The helper under test, :func:`warmup_audio._wait_and_emit_completion`, spawns
a background thread that ``wait()``s on a Popen and then emits a
``welcome.wav.completed`` supporting-event span via ``telemetry``. These tests
drive it with subprocess doubles to assert on:

- thread lifecycle (daemon flag, return value, no-op paths)
- span shape (attrs include ``aplay.exit_code`` + ``aplay.command``)
- already-exited Popens trigger synchronous emission, not a thread
- telemetry failure is swallowed (helper is best-effort)
"""

from __future__ import annotations
import sys
import time
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from robot_comic import warmup_audio


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakePopen:
    """Minimal stand-in for ``subprocess.Popen`` that controls ``wait`` timing."""

    def __init__(
        self,
        exit_code: int = 0,
        wait_delay: float = 0.0,
        already_exited: bool = False,
    ) -> None:
        self._exit_code = exit_code
        self._wait_delay = wait_delay
        self._already_exited = already_exited
        self.returncode: int | None = exit_code if already_exited else None

    def poll(self) -> int | None:
        return self._exit_code if self._already_exited else None

    def wait(self) -> int:
        if self._wait_delay:
            time.sleep(self._wait_delay)
        self.returncode = self._exit_code
        return self._exit_code


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_none_when_popen_is_none() -> None:
    result = warmup_audio._wait_and_emit_completion(None, ["aplay", "-q", "x.wav"])
    assert result is None


def test_returns_daemon_thread_for_live_popen() -> None:
    proc = _FakePopen(exit_code=0, wait_delay=0.05)
    with patch("robot_comic.telemetry.emit_supporting_event"):
        thread = warmup_audio._wait_and_emit_completion(
            proc,  # type: ignore[arg-type]
            ["aplay", "-q", "x.wav"],
            started_at=time.monotonic(),
        )
    assert thread is not None
    assert isinstance(thread, threading.Thread)
    assert thread.daemon is True
    thread.join(timeout=2)
    assert not thread.is_alive(), "wait-thread did not finish within timeout"


def test_already_exited_popen_emits_synchronously_no_thread() -> None:
    proc = _FakePopen(exit_code=0, already_exited=True)
    with patch("robot_comic.telemetry.emit_supporting_event") as emit:
        thread = warmup_audio._wait_and_emit_completion(
            proc,  # type: ignore[arg-type]
            ["aplay", "-q", "x.wav"],
            started_at=time.monotonic(),
        )
    assert thread is None
    assert emit.call_count == 1, "sync path must still emit the completion event"


def test_thread_emits_expected_span_attrs_on_zero_exit() -> None:
    proc = _FakePopen(exit_code=0)
    with patch("robot_comic.telemetry.emit_supporting_event") as emit:
        thread = warmup_audio._wait_and_emit_completion(
            proc,  # type: ignore[arg-type]
            ["aplay", "-q", "/tmp/welcome.wav"],
            started_at=time.monotonic(),
        )
        assert thread is not None
        thread.join(timeout=2)

    emit.assert_called_once()
    args, kwargs = emit.call_args
    assert args[0] == "welcome.wav.completed"
    assert "dur_ms" in kwargs and kwargs["dur_ms"] >= 0
    assert "extra_attrs" in kwargs
    attrs = kwargs["extra_attrs"]
    assert attrs["aplay.exit_code"] == "0"
    assert "aplay -q /tmp/welcome.wav" in attrs["aplay.command"]


def test_thread_captures_nonzero_exit_code() -> None:
    proc = _FakePopen(exit_code=1)
    with patch("robot_comic.telemetry.emit_supporting_event") as emit:
        thread = warmup_audio._wait_and_emit_completion(
            proc,  # type: ignore[arg-type]
            ["aplay", "-q", "/tmp/welcome.wav"],
            started_at=time.monotonic(),
        )
        assert thread is not None
        thread.join(timeout=2)

    emit.assert_called_once()
    _args, kwargs = emit.call_args
    assert kwargs["extra_attrs"]["aplay.exit_code"] == "1"


def test_telemetry_failure_is_swallowed() -> None:
    """Helper must never raise; telemetry exceptions are debug-logged."""
    proc = _FakePopen(exit_code=0)
    with patch(
        "robot_comic.telemetry.emit_supporting_event",
        side_effect=RuntimeError("forced"),
    ):
        thread = warmup_audio._wait_and_emit_completion(
            proc,  # type: ignore[arg-type]
            ["aplay", "x.wav"],
            started_at=time.monotonic(),
        )
        assert thread is not None
        thread.join(timeout=2)
    # If we got here without raising, the swallowing worked.


def test_wait_raising_does_not_emit_event() -> None:
    """A Popen.wait() that raises must skip the emission, not propagate."""
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.side_effect = OSError("simulated wait failure")
    with patch("robot_comic.telemetry.emit_supporting_event") as emit:
        thread = warmup_audio._wait_and_emit_completion(
            proc,
            ["aplay", "x.wav"],
            started_at=time.monotonic(),
        )
        assert thread is not None
        thread.join(timeout=2)
    emit.assert_not_called()


def test_started_at_default_is_thread_local_now() -> None:
    """When ``started_at`` is omitted, dur_ms is small and non-negative."""
    proc = _FakePopen(exit_code=0)
    with patch("robot_comic.telemetry.emit_supporting_event") as emit:
        thread = warmup_audio._wait_and_emit_completion(
            proc,  # type: ignore[arg-type]
            ["aplay", "x.wav"],
        )
        assert thread is not None
        thread.join(timeout=2)
    _args, kwargs = emit.call_args
    assert kwargs["dur_ms"] >= 0
    assert kwargs["dur_ms"] < 5_000  # tens of ms in practice; 5s is generous


def test_dispatch_single_wav_triggers_completion_on_posix(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a successful POSIX dispatch must hand off to the helper."""
    if sys.platform == "win32":
        pytest.skip("POSIX dispatch path doesn't run on Windows")

    wav = tmp_path / "welcome.wav"
    wav.write_bytes(b"")  # is_file() check only

    fake_proc = _FakePopen(exit_code=0)
    monkeypatch.setattr(warmup_audio, "_PLAYER_CMD", ["aplay", "-q"])

    with (
        patch("subprocess.Popen", return_value=fake_proc) as popen_mock,
        patch.object(warmup_audio, "_wait_and_emit_completion") as helper,
    ):
        result = warmup_audio._dispatch_single_wav(wav)

    assert result is True
    popen_mock.assert_called_once()
    helper.assert_called_once()
    helper_args, helper_kwargs = helper.call_args
    assert helper_args[0] is fake_proc
    # command argv is the second positional
    assert helper_args[1][0] == "aplay"
    assert "started_at" in helper_kwargs
