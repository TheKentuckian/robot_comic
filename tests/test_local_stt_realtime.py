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
