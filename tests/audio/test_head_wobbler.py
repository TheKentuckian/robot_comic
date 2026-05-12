"""Regression tests for the audio-driven head wobble behaviour."""

import math
import time
import base64
import threading
from typing import Any, List, Tuple
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from robot_comic.audio.head_wobbler import HeadWobbler


def _make_audio_chunk(duration_s: float = 0.3, frequency_hz: float = 220.0) -> str:
    """Generate a base64-encoded mono PCM16 sine wave."""
    pcm = _make_pcm(duration_s=duration_s, frequency_hz=frequency_hz, sample_rate=24000)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def _make_pcm(
    duration_s: float = 0.3,
    frequency_hz: float = 220.0,
    sample_rate: int = 24000,
) -> NDArray[np.int16]:
    """Generate a mono PCM16 sine wave at the requested sample rate."""
    sample_count = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, sample_count, endpoint=False)
    wave = 0.6 * np.sin(2 * math.pi * frequency_hz * t)
    return np.clip(wave * np.iinfo(np.int16).max, -32768, 32767).astype(np.int16)


def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Poll `predicate` until true or timeout."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _start_wobbler() -> Tuple[HeadWobbler, List[Tuple[float, Tuple[float, float, float, float, float, float]]]]:
    captured: List[Tuple[float, Tuple[float, float, float, float, float, float]]] = []

    def capture(offsets: Tuple[float, float, float, float, float, float]) -> None:
        captured.append((time.time(), offsets))

    wobbler = HeadWobbler(set_speech_offsets=capture)
    wobbler.start()
    return wobbler, captured


def test_reset_drops_pending_offsets() -> None:
    """Reset should stop prior wobble and restore neutral speech offsets."""
    wobbler, captured = _start_wobbler()
    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert _wait_for(lambda: len(captured) > 0), "wobbler did not emit initial offsets"

        pre_reset_count = len(captured)
        wobbler.reset()
        assert _wait_for(lambda: len(captured) == pre_reset_count + 1), "reset did not emit neutral offsets"
        assert captured[-1][1] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        time.sleep(0.3)
        assert len(captured) == pre_reset_count + 1, "offsets continued after reset without new audio"
    finally:
        wobbler.stop()


def test_reset_allows_future_offsets() -> None:
    """After reset, fresh audio must still produce wobble offsets."""
    wobbler, captured = _start_wobbler()
    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert _wait_for(lambda: len(captured) > 0), "wobbler did not emit initial offsets"

        wobbler.reset()
        pre_second_count = len(captured)

        wobbler.feed(_make_audio_chunk(duration_s=0.35, frequency_hz=440.0))
        assert _wait_for(lambda: len(captured) > pre_second_count), "no offsets after reset"
        assert wobbler._thread is not None and wobbler._thread.is_alive()
    finally:
        wobbler.stop()


def test_request_reset_after_current_audio_handles_16khz_chunks() -> None:
    """Queued reset should wait for 16 kHz audio to finish before restoring neutral offsets."""
    wobbler, captured = _start_wobbler()
    try:
        pcm = _make_pcm(duration_s=0.35, sample_rate=16000)
        wobbler.feed_pcm(pcm.reshape(1, -1), 16000)
        assert _wait_for(lambda: any(offsets != (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _, offsets in captured))

        wobbler.request_reset_after_current_audio()
        assert _wait_for(lambda: len(captured) > 0 and captured[-1][1] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    finally:
        wobbler.stop()


def test_speed_factor_scales_offset_amplitude() -> None:
    """speed_factor_getter values should scale the magnitude of emitted offsets."""

    def make_wobbler(
        getter: Callable[[], float] | None,
    ) -> Tuple[HeadWobbler, List[Tuple[float, float, float, float, float, float]]]:
        captured: List[Tuple[float, float, float, float, float, float]] = []

        def capture(offsets: Tuple[float, float, float, float, float, float]) -> None:
            captured.append(offsets)

        wob = HeadWobbler(set_speech_offsets=capture, speed_factor_getter=getter)
        wob.start()
        return wob, captured

    full, full_captured = make_wobbler(getter=lambda: 1.0)
    half, half_captured = make_wobbler(getter=lambda: 0.5)
    try:
        # Feed identical audio to both wobblers.
        chunk = _make_audio_chunk(duration_s=0.5, frequency_hz=220.0)
        full.feed(chunk)
        half.feed(chunk)
        assert _wait_for(lambda: len(full_captured) >= 5 and len(half_captured) >= 5)

        def peak_abs(offsets_list: List[Tuple[float, float, float, float, float, float]]) -> float:
            return max(max(abs(v) for v in o) for o in offsets_list)

        full_peak = peak_abs(full_captured)
        half_peak = peak_abs(half_captured)
        # half should be roughly half the amplitude of full; allow loose tolerance
        # because the two wobblers may not align perfectly in time.
        assert full_peak > 0
        assert half_peak < full_peak * 0.75
    finally:
        full.stop()
        half.stop()


def test_speed_factor_clamped() -> None:
    """Out-of-range speed_factor values should be clamped to [0.1, 2.0]."""
    captured: List[Tuple[float, float, float, float, float, float]] = []

    def capture(offsets: Tuple[float, float, float, float, float, float]) -> None:
        captured.append(offsets)

    huge, tiny = 999.0, -1.0
    factors = iter([huge, tiny])

    wob = HeadWobbler(set_speech_offsets=capture, speed_factor_getter=lambda: next(factors, 1.0))
    wob.start()
    try:
        wob.feed(_make_audio_chunk(duration_s=0.3))
        assert _wait_for(lambda: len(captured) >= 2)
        # Should not crash; values stay bounded by 2x the natural amplitude.
        peak = max(max(abs(v) for v in o) for o in captured)
        assert peak < 1.0  # natural offsets are tiny; 2x cap keeps them well under 1 unit
    finally:
        wob.stop()


def test_reset_during_inflight_chunk_keeps_worker(monkeypatch: Any) -> None:
    """Simulate reset during chunk processing to ensure the worker survives."""
    wobbler, captured = _start_wobbler()
    ready = threading.Event()
    release = threading.Event()

    original_feed = wobbler.sway.feed

    def blocking_feed(pcm, sr):  # type: ignore[no-untyped-def]
        ready.set()
        release.wait(timeout=2.0)
        return original_feed(pcm, sr)

    monkeypatch.setattr(wobbler.sway, "feed", blocking_feed)

    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert ready.wait(timeout=1.0), "worker thread did not dequeue audio"

        wobbler.reset()
        release.set()

        # Allow the worker to finish processing the first chunk (which should be discarded)
        time.sleep(0.1)

        assert wobbler._thread is not None and wobbler._thread.is_alive(), "worker thread died after reset"

        pre_second = len(captured)
        wobbler.feed(_make_audio_chunk(duration_s=0.35, frequency_hz=440.0))
        assert _wait_for(lambda: len(captured) > pre_second), "no offsets emitted after in-flight reset"
        assert wobbler._thread.is_alive()
    finally:
        wobbler.stop()
