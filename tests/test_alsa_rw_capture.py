"""Unit tests for AlsaRwCapture.

These tests never spawn a real arecord process — Popen is stubbed.  This
keeps them portable to macOS/Windows CI.  The on-Pi field test covers
real-arecord behaviour and lives outside the automated suite.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pytest

from robot_comic.audio_input.alsa_rw_capture import AlsaRwCapture


class _FakeStdout:
    """File-like wrapper that returns scripted byte chunks then EOF."""

    def __init__(self, chunks: list[bytes]):
        self._chunks: list[bytes] = list(chunks)
        self._buffer = b""

    def read(self, n: int) -> bytes:
        # Behave like a non-blocking pipe: feed at most one scripted chunk per
        # call.  AlsaRwCapture buffers across calls to handle short reads.
        if not self._buffer and self._chunks:
            self._buffer = self._chunks.pop(0)
        if not self._buffer:
            return b""
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def close(self) -> None:
        pass


class _FakePopen:
    """Stub for subprocess.Popen — captures args and serves scripted bytes."""

    instances: list["_FakePopen"] = []

    def __init__(self, cmd: list[str], *args: Any, **kwargs: Any) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = _FakeStdout(self._scripted_chunks)
        self.stderr = _FakeStdout([])
        self.terminated = False
        self.killed = False
        self._return_code: int | None = None
        type(self).instances.append(self)

    # Class-level slot the test overrides per case.
    _scripted_chunks: list[bytes] = []

    def poll(self) -> int | None:
        return self._return_code

    def terminate(self) -> None:
        self.terminated = True
        self._return_code = 0

    def kill(self) -> None:
        self.killed = True
        self._return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        self._return_code = self._return_code if self._return_code is not None else 0
        return self._return_code


@pytest.fixture
def fake_popen(monkeypatch):
    _FakePopen.instances = []
    _FakePopen._scripted_chunks = []
    monkeypatch.setattr("robot_comic.audio_input.alsa_rw_capture.subprocess.Popen", _FakePopen)
    yield _FakePopen


def test_start_on_non_linux_raises(monkeypatch):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "darwin",
    )
    cap = AlsaRwCapture()
    with pytest.raises(RuntimeError, match="Linux-only"):
        cap.start()


def test_start_invokes_arecord_with_expected_args(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture(device="reachymini_audio_src", sample_rate=16000, channels=2)
    cap.start()
    assert len(fake_popen.instances) == 1
    cmd = fake_popen.instances[0].cmd
    assert cmd[0] == "arecord"
    assert "-D" in cmd and "reachymini_audio_src" in cmd
    assert "-r" in cmd and "16000" in cmd
    assert "-c" in cmd and "2" in cmd
    assert "-f" in cmd and "S16_LE" in cmd
    assert "-t" in cmd and "raw" in cmd
    # arecord defaults to RW-interleaved access mode.  Passing -M would
    # enable MMAP — the broken mode this module exists to bypass — so
    # the flag MUST be absent from the spawn command.
    assert "-M" not in cmd


def test_get_audio_sample_returns_none_when_buffer_empty(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    fake_popen._scripted_chunks = []  # no bytes available
    cap = AlsaRwCapture(frame_samples=4, channels=2)
    cap.start()
    assert cap.get_audio_sample() is None


def test_get_audio_sample_returns_full_frame(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    # Build a known 4-frame stereo S16_LE buffer: 4 * 2 * 2 = 16 bytes.
    samples = np.array(
        [[100, -100], [200, -200], [300, -300], [400, -400]],
        dtype=np.int16,
    )
    fake_popen._scripted_chunks = [samples.tobytes()]
    cap = AlsaRwCapture(frame_samples=4, channels=2)
    cap.start()
    frame = cap.get_audio_sample()
    assert frame is not None
    assert frame.dtype == np.int16
    assert frame.shape == (4, 2)
    np.testing.assert_array_equal(frame, samples)


def test_get_audio_sample_handles_short_reads(monkeypatch, fake_popen):
    """Two short reads should combine into one full frame."""
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    samples = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        dtype=np.int16,
    )
    raw = samples.tobytes()  # 16 bytes
    # Split into two chunks: 7 bytes, then 9 bytes.
    fake_popen._scripted_chunks = [raw[:7], raw[7:]]
    cap = AlsaRwCapture(frame_samples=4, channels=2)
    cap.start()
    # First call: only 7 bytes available, frame not complete.
    assert cap.get_audio_sample() is None
    # Second call: remaining 9 bytes arrive, frame completes.
    frame = cap.get_audio_sample()
    assert frame is not None
    np.testing.assert_array_equal(frame, samples)


def test_sample_rate_property_returns_configured_value(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture(sample_rate=24000)
    cap.start()
    assert cap.sample_rate == 24000


def test_stop_terminates_subprocess(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture()
    cap.start()
    cap.stop()
    assert fake_popen.instances[0].terminated is True


def test_stop_before_start_is_noop(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture()
    cap.stop()  # must not raise
    assert fake_popen.instances == []


def test_double_start_raises(monkeypatch, fake_popen):
    monkeypatch.setattr(
        "robot_comic.audio_input.alsa_rw_capture.sys.platform",
        "linux",
    )
    cap = AlsaRwCapture()
    cap.start()
    with pytest.raises(RuntimeError, match="already started"):
        cap.start()
