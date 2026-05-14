import time as _time_mod
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.local_stt_realtime import LocalSTTRealtimeHandler


@pytest.mark.asyncio
async def test_local_stt_completion_sends_text_turn_and_queues_response() -> None:
    """A finalized local transcript should become a text turn in the realtime session."""
    movement_manager = MagicMock()
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    handler = LocalSTTRealtimeHandler(deps)

    item = SimpleNamespace(create=AsyncMock())
    handler.connection = SimpleNamespace(conversation=SimpleNamespace(item=item))
    handler._safe_response_create = AsyncMock()  # type: ignore[method-assign]

    await handler._handle_local_stt_event("completed", "tell me a quick joke")

    movement_manager.set_listening.assert_called_with(False)
    item.create.assert_awaited_once()
    sent_item = item.create.await_args.kwargs["item"]
    assert sent_item["role"] == "user"
    assert sent_item["content"][0]["type"] == "input_text"
    assert sent_item["content"][0]["text"] == "tell me a quick joke"
    handler._safe_response_create.assert_awaited_once()

    output = await asyncio.wait_for(handler.output_queue.get(), timeout=1.0)
    assert isinstance(output, AdditionalOutputs)
    assert output.args[0] == {"role": "user", "content": "tell me a quick joke"}


@pytest.mark.asyncio
async def test_local_stt_receive_resamples_and_feeds_stream() -> None:
    """Mic frames should be normalized and sent to Moonshine's local stream."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)
    stream = MagicMock()
    handler._local_stt_stream = stream

    audio = np.arange(240, dtype=np.int16)
    await handler.receive((24000, audio))

    stream.add_audio.assert_called_once()
    samples, sample_rate = stream.add_audio.call_args.args
    assert sample_rate == 16000
    assert isinstance(samples, list)
    assert len(samples) == 160


def _make_heartbeat_handler():
    """Build a minimal LocalSTTInputMixin-like object with _heartbeat initialized."""

    class _Stub:
        def __init__(self):
            self._heartbeat = {
                "state": "idle",
                "last_event": None,
                "last_text": "",
                "last_event_at": _time_mod.monotonic(),
                "audio_frames": 0,
            }
            self._local_loop = None
            self._local_stt_stream = MagicMock()
            self._heartbeat_future = None

        def _log_heartbeat(self):
            import logging

            h = self._heartbeat
            age = _time_mod.monotonic() - h["last_event_at"]
            logging.getLogger("robot_comic.local_stt_realtime").info(
                "[Moonshine] state=%s  last_event=%s  age=%.1fs  frames=%d  text=%r",
                h["state"],
                h["last_event"],
                age,
                h["audio_frames"],
                (h["last_text"] or "")[:40],
            )

    return _Stub()


def test_heartbeat_dict_has_required_keys():
    obj = _make_heartbeat_handler()
    assert "state" in obj._heartbeat
    assert "last_event_at" in obj._heartbeat
    assert "audio_frames" in obj._heartbeat
    assert obj._heartbeat["state"] == "idle"


def test_log_heartbeat_emits_info(caplog):
    import logging

    obj = _make_heartbeat_handler()
    with caplog.at_level(logging.INFO, logger="robot_comic.local_stt_realtime"):
        obj._log_heartbeat()
    assert any("Moonshine" in r.message for r in caplog.records)


# --- Dedup / startup-grace tests for the real handler's _log_heartbeat ------


def _fresh_handler():
    """Construct a real LocalSTTRealtimeHandler with no I/O wired up.

    All `_log_heartbeat` cares about is the `_heartbeat` dict, so we can
    poke that directly. This exercises the real mixin code (not a stub).
    """
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return LocalSTTRealtimeHandler(deps)


def test_heartbeat_dedupes_identical_state_text_to_debug(caplog):
    """Repeated identical (state, text) heartbeats should be DEBUG, not INFO."""
    import logging

    handler = _fresh_handler()
    handler._heartbeat.update(
        {
            "state": "completed",
            "last_event": "completed",
            "last_text": "to see how we're doing for it's",
            "last_event_at": _time_mod.monotonic(),
            "audio_frames": 100,
        }
    )

    # First emit: changed signal vs. initial None → INFO.
    with caplog.at_level(logging.DEBUG, logger="robot_comic.local_stt_realtime"):
        handler._log_heartbeat()
        # Bump frame count to simulate ongoing audio while still in completed.
        handler._heartbeat["audio_frames"] = 200
        handler._log_heartbeat()
        handler._heartbeat["audio_frames"] = 300
        handler._log_heartbeat()

    moonshine_records = [r for r in caplog.records if "Moonshine] state=" in r.message]
    assert len(moonshine_records) == 3, [r.levelname for r in moonshine_records]
    assert moonshine_records[0].levelno == logging.INFO
    assert moonshine_records[1].levelno == logging.DEBUG
    assert moonshine_records[2].levelno == logging.DEBUG


def test_heartbeat_changed_state_re_emits_info(caplog):
    """A change in state or text should immediately re-emit at INFO."""
    import logging

    handler = _fresh_handler()
    handler._heartbeat.update(
        {
            "state": "partial",
            "last_event": "partial",
            "last_text": "hello",
            "last_event_at": _time_mod.monotonic(),
            "audio_frames": 1,
        }
    )

    with caplog.at_level(logging.DEBUG, logger="robot_comic.local_stt_realtime"):
        handler._log_heartbeat()  # INFO (first)
        handler._log_heartbeat()  # DEBUG (dedup)
        handler._heartbeat["state"] = "completed"
        handler._heartbeat["last_event"] = "completed"
        handler._heartbeat["last_text"] = "hello world"
        handler._log_heartbeat()  # INFO (state+text changed)

    moonshine_records = [r for r in caplog.records if "Moonshine] state=" in r.message]
    assert [r.levelno for r in moonshine_records] == [logging.INFO, logging.DEBUG, logging.INFO]


def test_heartbeat_repeats_info_periodically_for_liveness(caplog, monkeypatch):
    """Even on identical (state, text), an INFO line should re-emit every ~30s."""
    import logging

    handler = _fresh_handler()
    base = 1000.0

    times = iter([base, base + 1.0, base + 31.0])
    monkeypatch.setattr(
        "robot_comic.local_stt_realtime.time.monotonic",
        lambda: next(times),
    )

    handler._heartbeat.update(
        {
            "state": "completed",
            "last_event": "completed",
            "last_text": "stuck text",
            "last_event_at": base - 5.0,
            "audio_frames": 10,
        }
    )

    with caplog.at_level(logging.DEBUG, logger="robot_comic.local_stt_realtime"):
        handler._log_heartbeat()  # t=base → INFO (first)
        handler._log_heartbeat()  # t=base+1 → DEBUG (dedup)
        handler._log_heartbeat()  # t=base+31 → INFO (interval elapsed)

    moonshine_records = [r for r in caplog.records if "Moonshine] state=" in r.message]
    assert [r.levelno for r in moonshine_records] == [logging.INFO, logging.DEBUG, logging.INFO]


def test_idle_stall_warning_suppressed_during_startup_grace(caplog, monkeypatch):
    """The 'thread-lock or model stall' warning should NOT fire within ~10s of first audio."""
    import logging

    handler = _fresh_handler()
    base = 5000.0

    # Pretend first audio arrived 3s ago (well inside the grace window) but the
    # last event was 12s ago (which would normally trigger the warning).
    handler._heartbeat.update(
        {
            "state": "idle",
            "last_event": None,
            "last_text": "",
            "last_event_at": base - 12.0,
            "audio_frames": 5,
            "first_audio_at": base - 3.0,
        }
    )
    monkeypatch.setattr("robot_comic.local_stt_realtime.time.monotonic", lambda: base)

    with caplog.at_level(logging.WARNING, logger="robot_comic.local_stt_realtime"):
        handler._log_heartbeat()

    stall_records = [r for r in caplog.records if "thread-lock or model stall" in r.message]
    assert stall_records == [], "stall warning fired during startup grace window"


def test_idle_stall_warning_fires_after_startup_grace(caplog, monkeypatch):
    """Past the grace window, an idle stall *should* still warn."""
    import logging

    handler = _fresh_handler()
    base = 6000.0

    handler._heartbeat.update(
        {
            "state": "idle",
            "last_event": None,
            "last_text": "",
            "last_event_at": base - 12.0,
            "audio_frames": 5,
            "first_audio_at": base - 60.0,
        }
    )
    monkeypatch.setattr("robot_comic.local_stt_realtime.time.monotonic", lambda: base)

    with caplog.at_level(logging.WARNING, logger="robot_comic.local_stt_realtime"):
        handler._log_heartbeat()

    stall_records = [r for r in caplog.records if "thread-lock or model stall" in r.message]
    assert len(stall_records) == 1


def test_idle_stall_warning_suppressed_when_no_audio_seen_yet(caplog, monkeypatch):
    """Before the first audio frame, the original guard (audio_frames > 0) still skips."""
    import logging

    handler = _fresh_handler()
    base = 7000.0

    handler._heartbeat.update(
        {
            "state": "idle",
            "last_event": None,
            "last_text": "",
            "last_event_at": base - 12.0,
            "audio_frames": 0,
            "first_audio_at": None,
        }
    )
    monkeypatch.setattr("robot_comic.local_stt_realtime.time.monotonic", lambda: base)

    with caplog.at_level(logging.WARNING, logger="robot_comic.local_stt_realtime"):
        handler._log_heartbeat()

    stall_records = [r for r in caplog.records if "thread-lock or model stall" in r.message]
    assert stall_records == []


# -- #279: stream rearm after completion -----------------------------------


def test_listener_on_line_completed_sets_pending_stream_rearm():
    """on_line_completed must flag a stream rebuild so subsequent turns transcribe."""
    from robot_comic.local_stt_realtime import _MoonshineListener

    handler = MagicMock()
    handler._heartbeat = {
        "state": "idle",
        "last_event": None,
        "last_text": "",
        "last_event_at": _time_mod.monotonic(),
    }
    handler._pending_stream_rearm = False
    listener = _MoonshineListener(handler)

    event = SimpleNamespace(line=SimpleNamespace(text="hello world"))
    listener.on_line_completed(event)

    assert handler._pending_stream_rearm is True
    handler._schedule_local_stt_event.assert_called_once_with("completed", "hello world")


def test_listener_on_error_also_sets_pending_stream_rearm():
    """A stream-level error leaves the C handle wedged; rearm is the only recovery."""
    from robot_comic.local_stt_realtime import _MoonshineListener

    handler = MagicMock()
    handler._pending_stream_rearm = False
    listener = _MoonshineListener(handler)

    listener.on_error(SimpleNamespace(error=RuntimeError("boom")))

    assert handler._pending_stream_rearm is True


def test_rearm_local_stt_stream_recreates_stream_on_same_transcriber():
    """Rearm must stop+close the old stream and call create_stream on the transcriber."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)

    old_stream = MagicMock()
    new_stream = MagicMock()
    transcriber = MagicMock()
    transcriber.create_stream.return_value = new_stream

    handler._local_stt_stream = old_stream
    handler._local_stt_transcriber = transcriber
    handler._local_stt_listener = MagicMock()
    handler._local_stt_update_interval = 0.35

    class _DummyBase:
        def on_line_started(self, event):
            pass

        def on_line_updated(self, event):
            pass

        def on_line_text_changed(self, event):
            pass

        def on_line_completed(self, event):
            pass

        def on_error(self, event):
            pass

    handler._local_stt_listener_base_cls = _DummyBase
    handler._pending_stream_rearm = True

    handler._rearm_local_stt_stream()

    old_stream.stop.assert_called_once()
    old_stream.close.assert_called_once()
    transcriber.create_stream.assert_called_once_with(update_interval=0.35)
    new_stream.add_listener.assert_called_once()
    new_stream.start.assert_called_once()
    assert handler._local_stt_stream is new_stream
    assert handler._local_stt_listener is not None
    assert handler._pending_stream_rearm is False


def test_rearm_local_stt_stream_noop_when_transcriber_already_gone():
    """If shutdown already nulled the transcriber, rearm must not blow up."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)

    old_stream = MagicMock()
    handler._local_stt_stream = old_stream
    handler._local_stt_transcriber = None
    handler._local_stt_listener = MagicMock()
    handler._pending_stream_rearm = True

    handler._rearm_local_stt_stream()

    old_stream.stop.assert_called_once()
    old_stream.close.assert_called_once()
    assert handler._local_stt_stream is None
    assert handler._pending_stream_rearm is False


@pytest.mark.asyncio
async def test_receive_rearms_stream_before_pushing_audio_when_flag_set():
    """When the rearm flag is set, the next receive() rebuilds before add_audio."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)

    original_stream = MagicMock()
    handler._local_stt_stream = original_stream
    handler._pending_stream_rearm = True
    rebuilt_stream = MagicMock()

    def _fake_rearm():
        handler._local_stt_stream = rebuilt_stream
        handler._pending_stream_rearm = False

    handler._rearm_local_stt_stream = _fake_rearm  # type: ignore[method-assign]

    audio = np.zeros(160, dtype=np.int16)
    await handler.receive((16000, audio))

    original_stream.add_audio.assert_not_called()
    rebuilt_stream.add_audio.assert_called_once()
    assert handler._pending_stream_rearm is False


@pytest.mark.asyncio
async def test_receive_no_rearm_when_flag_clear():
    """The fast path: no rebuild when no completion has happened."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = LocalSTTRealtimeHandler(deps)
    stream = MagicMock()
    handler._local_stt_stream = stream
    handler._pending_stream_rearm = False
    handler._rearm_local_stt_stream = MagicMock()  # type: ignore[method-assign]

    audio = np.zeros(160, dtype=np.int16)
    await handler.receive((16000, audio))

    handler._rearm_local_stt_stream.assert_not_called()  # type: ignore[attr-defined]
    stream.add_audio.assert_called_once()
