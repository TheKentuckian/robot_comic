"""Unit tests for LocalStream._build_audio_source selection logic."""

from __future__ import annotations
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from robot_comic.console import LocalStream, _DaemonAudioSource


def _fake_robot(sample_rate: int = 16000, frame: np.ndarray | None = None) -> MagicMock:
    robot = MagicMock()
    robot.media.get_input_audio_samplerate.return_value = sample_rate
    robot.media.get_audio_sample.return_value = frame
    return robot


def test_daemon_audio_source_delegates_to_robot_media():
    frame = np.zeros((256, 2), dtype=np.int16)
    robot = _fake_robot(sample_rate=24000, frame=frame)
    src = _DaemonAudioSource(robot)
    assert src.sample_rate == 24000
    out = src.get_audio_sample()
    assert out is frame
    robot.media.get_audio_sample.assert_called_once()


def test_daemon_audio_source_returns_none_when_robot_returns_none():
    robot = _fake_robot(frame=None)
    src = _DaemonAudioSource(robot)
    assert src.get_audio_sample() is None


def test_daemon_audio_source_start_stop_are_noops():
    robot = _fake_robot()
    src = _DaemonAudioSource(robot)
    src.start()  # must not raise
    src.stop()  # must not raise


def test_build_audio_source_returns_daemon_when_path_is_daemon(monkeypatch):
    import robot_comic.console as console_mod

    monkeypatch.setattr(console_mod.config, "AUDIO_CAPTURE_PATH", "daemon")
    robot = _fake_robot()
    stream = LocalStream.__new__(LocalStream)  # bypass __init__
    stream._robot = robot
    src = stream._build_audio_source()
    assert isinstance(src, _DaemonAudioSource)


def test_build_audio_source_returns_alsa_rw_when_path_is_alsa_rw(monkeypatch):
    import robot_comic.console as console_mod
    from robot_comic.audio_input import AlsaRwCapture

    monkeypatch.setattr(console_mod.config, "AUDIO_CAPTURE_PATH", "alsa_rw")
    robot = _fake_robot()
    stream = LocalStream.__new__(LocalStream)
    stream._robot = robot
    src = stream._build_audio_source()
    assert isinstance(src, AlsaRwCapture)


def test_build_audio_source_falls_back_to_daemon_when_robot_is_none(monkeypatch):
    """Sim mode constructs LocalStream(robot=None) for the admin UI only.

    record_loop never runs in sim mode, but _build_audio_source must
    tolerate robot=None so __init__ doesn't blow up.
    """
    import robot_comic.console as console_mod

    monkeypatch.setattr(console_mod.config, "AUDIO_CAPTURE_PATH", "daemon")
    stream = LocalStream.__new__(LocalStream)
    stream._robot = None
    src = stream._build_audio_source()
    # In sim mode we still need *something* to satisfy the type, but it
    # must not call robot.media.* (because robot is None).  None is
    # acceptable — record_loop's assert catches misuse loudly in real mode.
    assert src is None


@pytest.mark.asyncio
async def test_record_loop_tolerates_audio_source_nulled_during_shutdown():
    """close() nulls _audio_source before setting _stop_event; record_loop
    must snapshot the source per-iteration so the racing read can't raise
    AttributeError on shutdown.
    """
    frame = np.zeros((256,), dtype=np.int16)
    source = MagicMock()
    source.sample_rate = 16000
    source.get_audio_sample.return_value = frame

    stream = LocalStream.__new__(LocalStream)
    stream._robot = MagicMock()
    stream.handler = MagicMock()
    stream.handler.receive = AsyncMock()
    stream._audio_source = source
    stream._stop_event = threading.Event()

    async def kill_source_then_stop() -> None:
        # Let record_loop iterate a few times, then mimic close()'s order:
        # null the source first, then set the stop event.
        await asyncio.sleep(0.01)
        stream._audio_source = None
        await asyncio.sleep(0.005)
        stream._stop_event.set()

    await asyncio.gather(stream.record_loop(), kill_source_then_stop())
    assert stream.handler.receive.await_count > 0
