"""Tests for CameraWorker lazy initialization (issue #323).

The boot greeting does not need vision, so the ~5 s gstreamer pipeline cost
is deferred from the systemd cold-start path to the first camera-touching
tool call. These tests pin the new contract:

* Constructing a CameraWorker does NOT spawn the capture thread.
* ``get_latest_frame()`` lazy-starts the worker on first call.
* ``set_head_tracking_enabled(True)`` lazy-starts the worker.
* Repeated calls reuse the same worker (no duplicate threads).
* Explicit ``start()`` is still idempotent and safe.
"""

from __future__ import annotations
import time
import threading
from unittest.mock import MagicMock

import numpy as np

from robot_comic.camera_worker import CameraWorker


def _make_worker() -> CameraWorker:
    """Build a CameraWorker with a mocked ReachyMini that returns a fake frame."""
    robot = MagicMock()
    # A 1x1 BGR frame is enough — we only care about thread lifecycle here.
    robot.media.get_frame.return_value = np.zeros((1, 1, 3), dtype=np.uint8)
    return CameraWorker(robot, head_tracker=None)


def _stop(worker: CameraWorker) -> None:
    """Best-effort shutdown so test runs don't leak daemon threads."""
    worker._stop_event.set()
    if worker._thread is not None:
        worker._thread.join(timeout=2.0)


def test_construction_does_not_start_thread() -> None:
    """The gstreamer pipeline must not fire just because we built a worker.

    This is the load-bearing assertion for the boot-time win: deps init is
    allowed to instantiate the CameraWorker, but the capture loop is held
    until something actually wants a frame.
    """
    worker = _make_worker()
    try:
        assert worker._thread is None
        assert not worker.reachy_mini.media.get_frame.called
    finally:
        _stop(worker)


def test_get_latest_frame_lazy_starts_worker() -> None:
    """First ``get_latest_frame()`` call is the trigger for camera init."""
    worker = _make_worker()
    try:
        assert worker._thread is None

        worker.get_latest_frame()

        assert worker._thread is not None
        assert worker._thread.is_alive()

        # Loop should be actually running — wait briefly for the first
        # ``media.get_frame`` call so we know the gstreamer path is touched.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if worker.reachy_mini.media.get_frame.called:
                break
            time.sleep(0.01)
        assert worker.reachy_mini.media.get_frame.called
    finally:
        _stop(worker)


def test_repeated_get_latest_frame_reuses_same_worker() -> None:
    """Subsequent reads must not spawn additional threads."""
    worker = _make_worker()
    try:
        worker.get_latest_frame()
        first_thread = worker._thread
        assert first_thread is not None

        for _ in range(5):
            worker.get_latest_frame()

        assert worker._thread is first_thread
    finally:
        _stop(worker)


def test_enable_head_tracking_lazy_starts_worker() -> None:
    """Enabling head tracking is also a vision intent → start the worker."""
    worker = _make_worker()
    try:
        assert worker._thread is None

        worker.set_head_tracking_enabled(True)

        assert worker._thread is not None
        assert worker._thread.is_alive()
    finally:
        _stop(worker)


def test_disable_head_tracking_does_not_start_worker() -> None:
    """``set_head_tracking_enabled(False)`` on a never-started worker must
    not pay the gstreamer cost — disable is not a vision-using intent.
    """
    worker = _make_worker()
    try:
        worker.set_head_tracking_enabled(False)
        assert worker._thread is None
    finally:
        _stop(worker)


def test_ensure_started_is_idempotent_and_thread_safe() -> None:
    """Concurrent callers must spawn exactly one capture thread."""
    worker = _make_worker()
    try:
        results: list[bool] = []
        ready = threading.Event()

        def call() -> None:
            ready.wait()
            results.append(worker.ensure_started())

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        ready.set()
        for t in threads:
            t.join(timeout=2.0)

        # Exactly one caller should have observed the cold start.
        assert results.count(True) == 1
        assert results.count(False) == len(threads) - 1
        assert worker._thread is not None
        assert worker._thread.is_alive()
    finally:
        _stop(worker)


def test_explicit_start_is_idempotent() -> None:
    """A double-``start()`` (legacy callers) must not spawn two threads."""
    worker = _make_worker()
    try:
        worker.start()
        first_thread = worker._thread
        assert first_thread is not None

        worker.start()

        assert worker._thread is first_thread
    finally:
        _stop(worker)


def test_stop_on_never_started_worker_is_safe() -> None:
    """``stop()`` on a worker that lazy-init never armed should not raise.

    Boot shutdown calls ``camera_worker.stop()`` unconditionally; if no tool
    ever used the camera that path must remain harmless.
    """
    worker = _make_worker()
    # Should not raise even though no thread was ever started.
    worker.stop()
    assert worker._thread is None
