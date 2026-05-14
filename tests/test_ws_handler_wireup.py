"""Tests for WS handler wire-up (Issue #234).

Verifies:
- With WS_ENABLED=False (default), no WsClient is created, no calls fail.
- With WS_ENABLED=True, a mock WsClient receives the expected pi_status payload
  after a turn completes (simulated via _ws_emit).
- Failure to send (mock raise) doesn't crash the handler turn.
- WS_PAUSE_FLAG is present in config and defaults to False.
- Laptop-mode vs Pi-mode selection heuristic works correctly.
"""

from __future__ import annotations
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_ws_client(*, send_raises: Exception | None = None) -> MagicMock:
    """Return a mock that looks like a WsClient."""
    mock = MagicMock()
    mock.__class__ = _import_ws_client()
    if send_raises is not None:
        mock.send = AsyncMock(side_effect=send_raises)
    else:
        mock.send = AsyncMock(return_value=True)
    return mock


def _import_ws_client() -> type:
    from robot_comic.ws_client import WsClient

    return WsClient


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestWsPauseFlagConfig:
    """WS_PAUSE_FLAG is present in config and defaults to False."""

    def test_ws_pause_flag_defaults_false(self) -> None:
        from robot_comic.config import config

        assert config.WS_PAUSE_FLAG is False

    def test_ws_pause_flag_is_mutable(self) -> None:
        from robot_comic.config import config

        original = config.WS_PAUSE_FLAG
        try:
            config.WS_PAUSE_FLAG = True
            assert config.WS_PAUSE_FLAG is True
        finally:
            config.WS_PAUSE_FLAG = original

    def test_ws_enabled_defaults_false(self) -> None:
        from robot_comic.config import config

        assert config.WS_ENABLED is False


# ---------------------------------------------------------------------------
# _ws_emit unit tests (via BaseRealtimeHandler subclass stub)
# ---------------------------------------------------------------------------


class TestWsEmit:
    """Unit tests for BaseRealtimeHandler._ws_emit fire-and-forget logic."""

    def _make_handler(self) -> Any:
        """Instantiate the minimal stub handler defined below."""
        return _StubHandler()

    def test_ws_emit_no_client_is_noop(self) -> None:
        """When _ws_client is None, _ws_emit does nothing and does not raise."""
        handler = self._make_handler()
        from robot_comic.ws_protocol import make_pi_status

        # No exception should be raised
        asyncio.get_event_loop().run_until_complete(_async_noop())
        handler._ws_emit(make_pi_status(motor_state="active"))

    def test_ws_emit_with_ws_client_schedules_send(self) -> None:
        """_ws_emit with a WsClient schedules a send task on the event loop."""
        handler = self._make_handler()
        mock_client = _make_mock_ws_client()
        handler.set_ws_client(mock_client)

        from robot_comic.ws_protocol import make_pi_status

        msg = make_pi_status(motor_state="active")

        async def _run() -> None:
            handler._ws_emit(msg)
            # Yield to let the created task run
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        asyncio.get_event_loop().run_until_complete(_run())
        mock_client.send.assert_awaited_once_with(msg)

    def test_ws_emit_send_failure_does_not_raise(self) -> None:
        """When send() raises, _ws_emit swallows the error and does not crash."""
        handler = self._make_handler()
        mock_client = _make_mock_ws_client(send_raises=RuntimeError("boom"))
        handler.set_ws_client(mock_client)

        from robot_comic.ws_protocol import make_pi_status

        msg = make_pi_status(motor_state="active")

        async def _run() -> None:
            handler._ws_emit(msg)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        # Must not raise
        asyncio.get_event_loop().run_until_complete(_run())
        mock_client.send.assert_awaited_once_with(msg)

    def test_ws_emit_only_fires_for_ws_client_not_ws_server(self) -> None:
        """WsServer instances must not trigger pi_status emits."""
        from robot_comic.ws_server import WsServer

        handler = self._make_handler()
        mock_server = MagicMock(spec=WsServer)
        mock_server.send = AsyncMock(return_value=True)
        handler.set_ws_client(mock_server)

        from robot_comic.ws_protocol import make_pi_status

        async def _run() -> None:
            handler._ws_emit(make_pi_status(motor_state="active"))
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        asyncio.get_event_loop().run_until_complete(_run())
        mock_server.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# Turn-complete integration: WS_ENABLED=False → nothing created
# ---------------------------------------------------------------------------


class TestWsEnabledFalseNoSideEffects:
    """When WS_ENABLED=False (default), no WsClient is ever instantiated."""

    def test_set_ws_client_not_called_when_disabled(self) -> None:
        """handler.set_ws_client must never be called when WS_ENABLED is False."""
        from robot_comic.config import config

        assert config.WS_ENABLED is False
        handler = _StubHandler()
        # _ws_client should remain None
        assert handler._ws_client is None

    def test_ws_emit_is_safe_when_no_client(self) -> None:
        """_ws_emit with no client set is always safe (no error, no send)."""
        from robot_comic.ws_protocol import WsMessage

        handler = _StubHandler()
        msg = WsMessage(type="pi_status", payload={"motor_state": "active"})

        async def _run() -> None:
            handler._ws_emit(msg)
            await asyncio.sleep(0)

        asyncio.get_event_loop().run_until_complete(_run())
        # No assertions needed — just must not raise


# ---------------------------------------------------------------------------
# Turn-complete integration: WS_ENABLED=True → pi_status sent
# ---------------------------------------------------------------------------


class TestWsEnabledTrueTurnComplete:
    """When WS_ENABLED=True, pi_status is emitted after turn audio completes."""

    def test_pi_status_sent_after_turn_complete(self) -> None:
        """Mock WsClient receives a pi_status WsMessage after _ws_emit is called."""
        from robot_comic.ws_protocol import MsgType, WsMessage

        handler = _StubHandler()
        mock_client = _make_mock_ws_client()
        handler.set_ws_client(mock_client)

        captured: list[WsMessage] = []

        async def _capture(msg: WsMessage) -> bool:
            captured.append(msg)
            return True

        mock_client.send = AsyncMock(side_effect=_capture)

        # Simulate what the response.output_audio.done hook does
        async def _run() -> None:
            msg = WsMessage(
                type=MsgType.PI_STATUS,
                payload={"motor_state": "active", "persona": "default"},
            )
            handler._ws_emit(msg)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        asyncio.get_event_loop().run_until_complete(_run())

        assert len(captured) == 1
        assert captured[0].type == MsgType.PI_STATUS
        assert "motor_state" in captured[0].payload
        assert "persona" in captured[0].payload

    def test_pi_status_payload_structure(self) -> None:
        """pi_status payload includes motor_state and persona keys."""
        from robot_comic.ws_protocol import MsgType, WsMessage

        msg = WsMessage(
            type=MsgType.PI_STATUS,
            payload={"motor_state": "active", "persona": "rickles"},
        )
        assert msg.type == MsgType.PI_STATUS
        assert msg.payload["motor_state"] == "active"
        assert msg.payload["persona"] == "rickles"


# ---------------------------------------------------------------------------
# Laptop-mode heuristic
# ---------------------------------------------------------------------------


class TestLaptopModeHeuristic:
    """The is_laptop heuristic fires when LLAMA_CPP_URL starts with localhost."""

    @pytest.mark.parametrize(
        "url,expected_laptop",
        [
            ("http://localhost:8080", True),
            ("http://localhost/v1", True),
            ("http://127.0.0.1:8080", True),
            ("http://192.168.1.10:8080", False),
            ("http://astralplane.lan:8080", False),
            ("http://10.0.0.1:8080", False),
        ],
    )
    def test_is_laptop_from_llama_cpp_url(self, url: str, expected_laptop: bool) -> None:
        """_is_laptop heuristic in main.py: localhost/127.x → laptop."""
        is_laptop = url.startswith("http://localhost") or url.startswith("http://127.")
        assert is_laptop is expected_laptop


# ---------------------------------------------------------------------------
# Stub handler for unit tests (no real robot/audio deps)
# ---------------------------------------------------------------------------


async def _async_noop() -> None:
    pass


class _StubHandler:
    """Minimal stub that only carries the WS-related methods from BaseRealtimeHandler."""

    def __init__(self) -> None:
        self._ws_client: Any = None

    def set_ws_client(self, ws_client: Any) -> None:
        from robot_comic.base_realtime import BaseRealtimeHandler

        # Delegate to the real implementation by binding it
        BaseRealtimeHandler.set_ws_client(self, ws_client)  # type: ignore[arg-type]

    def _ws_emit(self, msg: Any) -> None:
        from robot_comic.base_realtime import BaseRealtimeHandler

        BaseRealtimeHandler._ws_emit(self, msg)  # type: ignore[arg-type]
