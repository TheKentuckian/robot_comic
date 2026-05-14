"""Tests for the WebSocket Pi ↔ laptop channel.

Covers:
- Protocol serialisation/deserialisation.
- Server accepts a client connection.
- Message envelope round-trip (type and payload survive the wire).
- Keepalive: server auto-responds to ping with pong.
- Client reconnects with back-off after the server closes the connection.

All tests use loopback (127.0.0.1) and port 0 (OS-assigned) so they never
conflict with each other or with a running app instance.
"""

from __future__ import annotations
import json
import asyncio

import pytest
from websockets.asyncio.client import connect

from robot_comic.ws_client import _BACKOFF_BASE, WsClient
from robot_comic.ws_server import WsServer
from robot_comic.ws_protocol import (
    MsgType,
    WsMessage,
    make_ping,
    make_pong,
    make_pi_log,
    make_pi_event,
    make_pi_status,
    make_laptop_event,
    make_laptop_status,
    make_laptop_command,
)


# ---------------------------------------------------------------------------
# Protocol: serialisation
# ---------------------------------------------------------------------------


class TestWsMessage:
    """Unit tests for :class:`~robot_comic.ws_protocol.WsMessage`."""

    def test_to_json_produces_envelope(self) -> None:
        """to_json produces a valid JSON string with type and payload keys."""
        msg = WsMessage(type="test_type", payload={"key": "value"})
        raw = msg.to_json()
        data = json.loads(raw)
        assert data["type"] == "test_type"
        assert data["payload"] == {"key": "value"}

    def test_from_json_round_trip(self) -> None:
        """A message serialised then deserialised preserves type and payload."""
        original = WsMessage(type="pi_status", payload={"battery": 85.5, "motor_state": "idle"})
        recovered = WsMessage.from_json(original.to_json())
        assert recovered.type == original.type
        assert recovered.payload == original.payload

    def test_from_json_missing_payload_defaults_to_empty_dict(self) -> None:
        """from_json accepts an envelope without a payload key."""
        raw = json.dumps({"type": "ping"})
        msg = WsMessage.from_json(raw)
        assert msg.type == "ping"
        assert msg.payload == {}

    def test_from_json_invalid_json_raises(self) -> None:
        """from_json raises ValueError on non-JSON input."""
        with pytest.raises(ValueError, match="invalid JSON"):
            WsMessage.from_json("not json at all")

    def test_from_json_non_object_raises(self) -> None:
        """from_json raises ValueError when the JSON root is not an object."""
        with pytest.raises(ValueError, match="expected JSON object"):
            WsMessage.from_json("[1, 2, 3]")

    def test_from_json_missing_type_raises(self) -> None:
        """from_json raises ValueError when the type field is absent."""
        with pytest.raises(ValueError, match="missing or non-string"):
            WsMessage.from_json(json.dumps({"payload": {}}))

    def test_from_json_empty_type_raises(self) -> None:
        """from_json raises ValueError when the type field is an empty string."""
        with pytest.raises(ValueError, match="missing or non-string"):
            WsMessage.from_json(json.dumps({"type": "", "payload": {}}))

    def test_from_json_non_dict_payload_raises(self) -> None:
        """from_json raises ValueError when payload is not a JSON object."""
        with pytest.raises(ValueError, match="'payload' must be a JSON object"):
            WsMessage.from_json(json.dumps({"type": "ping", "payload": [1, 2]}))


# ---------------------------------------------------------------------------
# Protocol: convenience constructors
# ---------------------------------------------------------------------------


class TestConvenienceConstructors:
    """Tests for the make_* factory functions."""

    def test_make_ping(self) -> None:
        """make_ping produces a message with type PING."""
        assert make_ping().type == MsgType.PING

    def test_make_pong(self) -> None:
        """make_pong produces a message with type PONG."""
        assert make_pong().type == MsgType.PONG

    def test_make_pi_status_full(self) -> None:
        """make_pi_status includes battery and motor_state in the payload."""
        msg = make_pi_status(battery=72.3, motor_state="active")
        assert msg.type == MsgType.PI_STATUS
        assert msg.payload["battery"] == pytest.approx(72.3)
        assert msg.payload["motor_state"] == "active"

    def test_make_pi_status_empty(self) -> None:
        """make_pi_status with no args produces an empty payload."""
        msg = make_pi_status()
        assert msg.payload == {}

    def test_make_pi_event(self) -> None:
        """make_pi_event includes the event name and extra kwargs."""
        msg = make_pi_event("emergency_stop", severity="critical")
        assert msg.type == MsgType.PI_EVENT
        assert msg.payload["event"] == "emergency_stop"
        assert msg.payload["severity"] == "critical"

    def test_make_pi_log(self) -> None:
        """make_pi_log includes message and level."""
        msg = make_pi_log("systemd: started", level="info")
        assert msg.type == MsgType.PI_LOG
        assert msg.payload["message"] == "systemd: started"
        assert msg.payload["level"] == "info"

    def test_make_laptop_status(self) -> None:
        """make_laptop_status includes model_loaded and gpu_available."""
        msg = make_laptop_status(model_loaded=True, gpu_available=True)
        assert msg.type == MsgType.LAPTOP_STATUS
        assert msg.payload["model_loaded"] is True
        assert msg.payload["gpu_available"] is True

    def test_make_laptop_event(self) -> None:
        """make_laptop_event includes the event name."""
        msg = make_laptop_event("restart_pending", eta_s=5)
        assert msg.type == MsgType.LAPTOP_EVENT
        assert msg.payload["event"] == "restart_pending"

    def test_make_laptop_command(self) -> None:
        """make_laptop_command includes the command name and kwargs."""
        msg = make_laptop_command("wake_up")
        assert msg.type == MsgType.LAPTOP_COMMAND
        assert msg.payload["command"] == "wake_up"


# ---------------------------------------------------------------------------
# Server: accepts connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_accepts_client_connection() -> None:
    """The server accepts a WebSocket client connection on loopback."""
    server = WsServer(host="127.0.0.1", port=0)
    await server.start()
    try:
        port = server.port
        assert port > 0
        async with connect(f"ws://127.0.0.1:{port}"):
            # Just connecting is sufficient; no assertion needed beyond no exception.
            pass
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Server: message round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_receives_and_handler_called() -> None:
    """A message sent from a client is received and dispatched to the handler."""
    received: list[WsMessage] = []

    server = WsServer(host="127.0.0.1", port=0)
    server.on_message(lambda msg, _ws: received.append(msg))
    await server.start()
    try:
        port = server.port
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(make_pi_status(battery=50.0, motor_state="idle").to_json())
            # Give the server a moment to process the message.
            await asyncio.sleep(0.05)
    finally:
        await server.stop()

    assert len(received) == 1
    assert received[0].type == MsgType.PI_STATUS
    assert received[0].payload["battery"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_server_broadcast_reaches_client() -> None:
    """WsServer.broadcast sends a message to the connected client."""
    server = WsServer(host="127.0.0.1", port=0)
    await server.start()
    try:
        port = server.port
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            # Wait for the server's internal connection tracking to register us.
            await asyncio.sleep(0.05)
            cmd = make_laptop_command("wake_up")
            await server.broadcast(cmd)
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = WsMessage.from_json(str(raw))
            assert msg.type == MsgType.LAPTOP_COMMAND
            assert msg.payload["command"] == "wake_up"
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Server: keepalive — ping triggers pong
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_responds_to_ping_with_pong() -> None:
    """The server auto-replies with a pong when it receives a ping."""
    server = WsServer(host="127.0.0.1", port=0)
    await server.start()
    try:
        port = server.port
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(make_ping().to_json())
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = WsMessage.from_json(str(raw))
            assert msg.type == MsgType.PONG
    finally:
        await server.stop()


# ---------------------------------------------------------------------------
# Client: connects and receives messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_connects_and_receives_broadcast() -> None:
    """WsClient connects to the server and receives a broadcast message."""
    server = WsServer(host="127.0.0.1", port=0)
    await server.start()
    received: list[WsMessage] = []
    try:
        port = server.port
        client = WsClient(server_host="127.0.0.1", port=port)
        client.on_message(lambda msg: received.append(msg))
        await client.start()
        connected = await client.wait_connected(timeout=3.0)
        assert connected, "client did not connect within timeout"

        # Give server connection tracking time to register the client.
        await asyncio.sleep(0.05)
        await server.broadcast(make_laptop_status(model_loaded=True))
        await asyncio.sleep(0.1)
        await client.stop()
    finally:
        await server.stop()

    assert len(received) == 1
    assert received[0].type == MsgType.LAPTOP_STATUS
    assert received[0].payload.get("model_loaded") is True


@pytest.mark.asyncio
async def test_client_sends_message_to_server() -> None:
    """WsClient.send delivers a message to the server handler."""
    received: list[WsMessage] = []
    server = WsServer(host="127.0.0.1", port=0)
    server.on_message(lambda msg, _ws: received.append(msg))
    await server.start()
    try:
        port = server.port
        client = WsClient(server_host="127.0.0.1", port=port)
        await client.start()
        connected = await client.wait_connected(timeout=3.0)
        assert connected

        ok = await client.send(make_pi_log("hello from pi"))
        assert ok is True
        await asyncio.sleep(0.1)
        await client.stop()
    finally:
        await server.stop()

    pi_logs = [m for m in received if m.type == MsgType.PI_LOG]
    assert len(pi_logs) == 1
    assert pi_logs[0].payload["message"] == "hello from pi"


# ---------------------------------------------------------------------------
# Client: reconnect after server closes connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_reconnects_after_server_closes() -> None:
    """WsClient reconnects automatically after the server drops the connection."""
    # Start the server, let the client connect, then stop/restart the server.
    server = WsServer(host="127.0.0.1", port=0)
    await server.start()
    port = server.port

    client = WsClient(server_host="127.0.0.1", port=port)
    await client.start()
    connected = await client.wait_connected(timeout=3.0)
    assert connected, "initial connect failed"

    # Tear down the server to force a disconnect.
    await server.stop()
    await asyncio.sleep(0.1)

    # Bring the server back on the same port.
    server2 = WsServer(host="127.0.0.1", port=port)
    await server2.start()
    try:
        # Client should reconnect within a couple of back-off cycles.
        reconnected = await client.wait_connected(timeout=_BACKOFF_BASE * 4)
        assert reconnected, "client did not reconnect after server restart"
    finally:
        await client.stop()
        await server2.stop()


@pytest.mark.asyncio
async def test_client_send_returns_false_when_not_connected() -> None:
    """WsClient.send returns False when there is no active connection."""
    client = WsClient(server_host="127.0.0.1", port=19999)
    # Do not start the client — no connection attempt.
    result = await client.send(make_ping())
    assert result is False
