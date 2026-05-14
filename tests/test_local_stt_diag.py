"""Tests for the MOONSHINE_DIAG=1 instrumentation path (issue #314).

The diag instrumentation is operator-deployed via the MOONSHINE_DIAG=1 env
var. These tests pin down two contracts:

1. When MOONSHINE_DIAG is unset, no [MOONSHINE_DIAG] log records are emitted
   anywhere — zero impact on production.
2. When MOONSHINE_DIAG=1, each of the documented instrumentation points
   emits a recognizable log entry.
"""

from __future__ import annotations
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from robot_comic.tools.core_tools import ToolDependencies
from robot_comic.local_stt_realtime import (
    LocalSTTRealtimeHandler,
    _diag_enabled,
    _MoonshineListener,
)


_LOGGER_NAME = "robot_comic.local_stt_realtime"
_DIAG_MARKER = "[MOONSHINE_DIAG]"


def _fresh_handler() -> LocalSTTRealtimeHandler:
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return LocalSTTRealtimeHandler(deps)


def _diag_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _DIAG_MARKER in r.getMessage()]


# ---------------------------------------------------------------------------
# Silence contract: MOONSHINE_DIAG unset → no diag log records anywhere.
# ---------------------------------------------------------------------------


def test_diag_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env var must be unset by default — _diag_enabled() returns False."""
    monkeypatch.delenv("MOONSHINE_DIAG", raising=False)
    assert _diag_enabled() is False


def test_diag_disabled_when_value_is_other(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the exact string '1' enables diag — '0', 'true', '' all stay off."""
    for value in ("0", "true", "True", "yes", "", "on"):
        monkeypatch.setenv("MOONSHINE_DIAG", value)
        assert _diag_enabled() is False, f"value={value!r} should not enable diag"


@pytest.mark.asyncio
async def test_diag_silent_in_receive_when_env_unset(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """receive() must emit zero [MOONSHINE_DIAG] records when the env var is unset."""
    monkeypatch.delenv("MOONSHINE_DIAG", raising=False)
    handler = _fresh_handler()
    handler._local_stt_stream = MagicMock()

    audio = np.arange(240, dtype=np.int16)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        await handler.receive((24000, audio))
        await handler.receive((24000, audio))
        await handler.receive((24000, audio))

    assert _diag_records(caplog) == []


def test_diag_silent_in_listener_callbacks_when_env_unset(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listener's _diag_log_callback must emit nothing when diag is off."""
    monkeypatch.delenv("MOONSHINE_DIAG", raising=False)
    handler = _fresh_handler()
    listener = _MoonshineListener(handler)
    event = SimpleNamespace(line=SimpleNamespace(text="hi"))

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        listener.on_line_started(event)
        listener.on_line_updated(event)
        listener.on_line_text_changed(event)
        listener.on_line_completed(event)
        listener.on_error(SimpleNamespace(error="boom"))

    assert _diag_records(caplog) == []


def test_diag_silent_in_periodic_state_when_env_unset(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_diag_log_periodic_state must be a no-op when diag is off."""
    monkeypatch.delenv("MOONSHINE_DIAG", raising=False)
    handler = _fresh_handler()
    handler._local_stt_stream = MagicMock()

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        handler._diag_log_periodic_state()

    assert _diag_records(caplog) == []


# ---------------------------------------------------------------------------
# Active contract: MOONSHINE_DIAG=1 → each instrumentation point emits.
# ---------------------------------------------------------------------------


def test_diag_listener_callbacks_emit_when_enabled(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All four MoonshineListener callbacks must emit one diag line each."""
    monkeypatch.setenv("MOONSHINE_DIAG", "1")
    handler = _fresh_handler()
    handler._local_stt_stream = MagicMock()
    listener = _MoonshineListener(handler)
    event = SimpleNamespace(line=SimpleNamespace(text="hello"))

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        listener.on_line_started(event)
        listener.on_line_updated(event)
        listener.on_line_text_changed(event)
        listener.on_line_completed(event)
        listener.on_error(SimpleNamespace(error="boom"))

    diag = [r.getMessage() for r in _diag_records(caplog)]
    # Five callback fires → at least five diag records.
    assert any("callback=on_line_started" in m for m in diag), diag
    assert any("callback=on_line_updated" in m for m in diag), diag
    assert any("callback=on_line_text_changed" in m for m in diag), diag
    assert any("callback=on_line_completed" in m for m in diag), diag
    assert any("callback=on_error" in m for m in diag), diag


@pytest.mark.asyncio
async def test_diag_add_audio_logs_first_n_calls_when_enabled(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """receive() must emit a detailed add_audio log for the first N frames."""
    monkeypatch.setenv("MOONSHINE_DIAG", "1")
    handler = _fresh_handler()
    handler._local_stt_stream = MagicMock()
    audio = np.arange(240, dtype=np.int16)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        await handler.receive((24000, audio))
        await handler.receive((24000, audio))

    add_audio_records = [m for m in (r.getMessage() for r in _diag_records(caplog)) if "add_audio call#" in m]
    assert len(add_audio_records) == 2, add_audio_records
    # Confirm the diag log surfaces the dtype/shape detail needed for hyp 3.
    msg = add_audio_records[0]
    assert "input_sample_rate=24000" in msg
    assert "sample_rate_arg=16000" in msg
    assert "resampled_len=160" in msg
    assert "input_dtype=int16" in msg


@pytest.mark.asyncio
async def test_diag_add_audio_logging_capped_at_limit(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """add_audio detail logs must cap at _DIAG_ADD_AUDIO_LOG_LIMIT (no journal flood)."""
    from robot_comic.local_stt_realtime import _DIAG_ADD_AUDIO_LOG_LIMIT

    monkeypatch.setenv("MOONSHINE_DIAG", "1")
    handler = _fresh_handler()
    handler._local_stt_stream = MagicMock()
    audio = np.arange(240, dtype=np.int16)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        for _ in range(_DIAG_ADD_AUDIO_LOG_LIMIT + 5):
            await handler.receive((24000, audio))

    add_audio_records = [m for m in (r.getMessage() for r in _diag_records(caplog)) if "add_audio call#" in m]
    assert len(add_audio_records) == _DIAG_ADD_AUDIO_LOG_LIMIT


def test_diag_periodic_state_emits_when_enabled(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_diag_log_periodic_state must emit a state-dump line with key fields."""
    monkeypatch.setenv("MOONSHINE_DIAG", "1")
    handler = _fresh_handler()
    handler._local_stt_stream = MagicMock()
    handler._local_stt_listener = MagicMock()
    handler._heartbeat["audio_frames"] = 42

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        handler._diag_log_periodic_state()

    diag = [r.getMessage() for r in _diag_records(caplog)]
    assert len(diag) == 1, diag
    msg = diag[0]
    assert "periodic" in msg
    assert "audio_frames=42" in msg
    assert "listener_id=" in msg
    assert "stream_id=" in msg
    assert "pending_rearm=False" in msg


@pytest.mark.asyncio
async def test_diag_add_audio_exception_logged_when_enabled(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception from stream.add_audio must surface a diag log line."""
    monkeypatch.setenv("MOONSHINE_DIAG", "1")
    handler = _fresh_handler()

    failing_stream = MagicMock()
    failing_stream.add_audio.side_effect = RuntimeError("boom")
    handler._local_stt_stream = failing_stream
    audio = np.arange(240, dtype=np.int16)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        await handler.receive((24000, audio))

    raises = [m for m in (r.getMessage() for r in _diag_records(caplog)) if "add_audio raised" in m]
    assert len(raises) == 1, raises
    assert "err_type=RuntimeError" in raises[0]


def test_diag_enabled_helper_reads_env_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """_diag_enabled() must read os.environ live so operators can toggle at runtime."""
    monkeypatch.setenv("MOONSHINE_DIAG", "1")
    assert _diag_enabled() is True
    monkeypatch.setenv("MOONSHINE_DIAG", "0")
    assert _diag_enabled() is False
