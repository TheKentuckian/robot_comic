"""Tests for the deferred-emit queue in ``robot_comic.telemetry`` (issue #337).

The early-play path and the welcome-WAV completion daemon thread both call
``emit_supporting_event`` before ``telemetry.init()`` runs in ``run()``. Without
queueing, those events resolve against OTel's no-op default tracer and are
silently dropped — so the boot timeline shows only ``app.startup`` even though
four other events were "emitted".

These tests pin the queue behaviour: events emitted before ``init()`` are
buffered; once ``init()`` runs they are flushed in order so the monitor TUI
gets all of them.
"""

from __future__ import annotations
from unittest.mock import patch

import pytest

from robot_comic import telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry_state() -> None:
    """Each test starts with init() not-yet-run state."""
    telemetry._initialized = False
    telemetry._pending_supporting.clear()
    yield
    telemetry._initialized = False
    telemetry._pending_supporting.clear()


def test_emit_before_init_is_queued_not_emitted() -> None:
    """Pre-init calls go to the pending list, never to ``get_tracer``."""
    with patch.object(telemetry, "get_tracer") as tracer_factory:
        telemetry.emit_supporting_event("welcome.wav.played", dur_ms=42.5)
    tracer_factory.assert_not_called()
    assert len(telemetry._pending_supporting) == 1
    name, dur_ms, extra = telemetry._pending_supporting[0]
    assert name == "welcome.wav.played"
    assert dur_ms == 42.5
    assert extra == {}


def test_emit_before_init_preserves_extra_attrs() -> None:
    """Queued entry carries ``extra_attrs`` so completion events keep their shape."""
    telemetry.emit_supporting_event(
        "welcome.wav.completed",
        dur_ms=3210.0,
        extra_attrs={"aplay.exit_code": "0", "aplay.command": "aplay -q x.wav"},
    )
    assert len(telemetry._pending_supporting) == 1
    _name, _dur, extra = telemetry._pending_supporting[0]
    assert extra == {"aplay.exit_code": "0", "aplay.command": "aplay -q x.wav"}


def test_init_flushes_queue_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """All queued events are emitted on ``init()`` in the order they were queued."""
    # Pre-init: queue three events.
    telemetry.emit_supporting_event("app.startup", dur_ms=5057.0)
    telemetry.emit_supporting_event("welcome.wav.played", dur_ms=42.5)
    telemetry.emit_supporting_event(
        "welcome.wav.completed",
        dur_ms=3210.0,
        extra_attrs={"aplay.exit_code": "0"},
    )
    assert len(telemetry._pending_supporting) == 3

    # init() runs — patch _init_otel/_init_instruments to avoid real OTel setup,
    # but exercise the flush path by stubbing ROBOT_INSTRUMENTATION so ENABLED
    # is True.
    monkeypatch.setenv("ROBOT_INSTRUMENTATION", "trace")
    emitted: list[tuple[str, float | None, dict]] = []

    def _capture(name: str, dur_ms: float | None, extra_attrs: dict) -> None:
        emitted.append((name, dur_ms, dict(extra_attrs or {})))

    with (
        patch.object(telemetry, "_init_otel"),
        patch.object(telemetry, "_init_instruments"),
        patch.object(telemetry, "_emit_supporting_event_now", side_effect=_capture),
    ):
        telemetry.init()

    assert [e[0] for e in emitted] == [
        "app.startup",
        "welcome.wav.played",
        "welcome.wav.completed",
    ]
    assert emitted[2][2] == {"aplay.exit_code": "0"}
    # Queue is drained after flush.
    assert telemetry._pending_supporting == []


def test_emit_after_init_bypasses_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once ``_initialized`` is True, events go directly to the tracer path."""
    monkeypatch.setenv("ROBOT_INSTRUMENTATION", "trace")
    with (
        patch.object(telemetry, "_init_otel"),
        patch.object(telemetry, "_init_instruments"),
    ):
        telemetry.init()

    with patch.object(telemetry, "_emit_supporting_event_now") as emit_now:
        telemetry.emit_supporting_event("turn.complete", dur_ms=1234.0)

    emit_now.assert_called_once()
    args, _kwargs = emit_now.call_args
    assert args[0] == "turn.complete"
    assert args[1] == 1234.0
    assert telemetry._pending_supporting == []


def test_init_when_instrumentation_disabled_still_drains_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ROBOT_INSTRUMENTATION is unset, init() still marks initialized and
    drains the queue (events become no-ops via the default tracer) so subsequent
    calls don't pile up forever."""
    monkeypatch.delenv("ROBOT_INSTRUMENTATION", raising=False)
    telemetry.emit_supporting_event("app.startup", dur_ms=5057.0)
    assert len(telemetry._pending_supporting) == 1

    telemetry.init()

    assert telemetry._initialized is True
    assert telemetry._pending_supporting == []


def test_queue_emit_is_thread_safe() -> None:
    """Concurrent emits from many threads do not lose or corrupt entries."""
    import threading

    n_threads = 16
    per_thread = 50
    barrier = threading.Barrier(n_threads)

    def _emit_many(thread_id: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            telemetry.emit_supporting_event(
                f"thread.{thread_id}",
                dur_ms=float(i),
            )

    threads = [threading.Thread(target=_emit_many, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(telemetry._pending_supporting) == n_threads * per_thread
