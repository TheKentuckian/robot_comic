"""WebSocket protocol definitions for the Pi ↔ laptop channel.

Message envelope::

    {"type": <str>, "payload": <obj>}

All messages use this envelope.  The ``type`` field selects the message kind;
``payload`` carries message-specific data (can be an empty dict ``{}`` when
there is nothing meaningful to carry).

No authentication and no TLS in v1 — both endpoints are assumed to be on
the same trusted LAN.
"""

from __future__ import annotations
import json
from typing import Any
from dataclasses import field, dataclass


# ---------------------------------------------------------------------------
# Canonical type strings
# ---------------------------------------------------------------------------


class MsgType:
    """String constants for the ``type`` field of every WS envelope."""

    # Pi → laptop
    PI_STATUS = "pi_status"
    PI_EVENT = "pi_event"
    PI_LOG = "pi_log"

    # Laptop → Pi
    LAPTOP_STATUS = "laptop_status"
    LAPTOP_EVENT = "laptop_event"
    LAPTOP_COMMAND = "laptop_command"

    # Both directions
    PING = "ping"
    PONG = "pong"


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------


@dataclass
class WsMessage:
    """A single JSON-over-WS envelope message."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise the message to a JSON string."""
        return json.dumps({"type": self.type, "payload": self.payload})

    @classmethod
    def from_json(cls, raw: str) -> "WsMessage":
        """Deserialise a JSON string into a :class:`WsMessage`.

        Raises :class:`ValueError` when the string is not valid JSON or does
        not conform to the ``{"type": ..., "payload": ...}`` envelope.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ws_protocol: invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"ws_protocol: expected JSON object, got {type(data).__name__}")

        msg_type = data.get("type")
        if not isinstance(msg_type, str) or not msg_type:
            raise ValueError("ws_protocol: missing or non-string 'type' field")

        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError(f"ws_protocol: 'payload' must be a JSON object, got {type(payload).__name__}")

        return cls(type=msg_type, payload=payload)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def make_ping() -> WsMessage:
    """Create a keepalive ``ping`` message."""
    return WsMessage(type=MsgType.PING)


def make_pong() -> WsMessage:
    """Create a keepalive ``pong`` reply."""
    return WsMessage(type=MsgType.PONG)


def make_pi_status(battery: float | None = None, motor_state: str | None = None) -> WsMessage:
    """Create a ``pi_status`` message.

    Args:
        battery: Battery percentage (0–100) or ``None`` if unknown.
        motor_state: A short string describing motor state, e.g. ``"idle"``.

    """
    payload: dict[str, Any] = {}
    if battery is not None:
        payload["battery"] = battery
    if motor_state is not None:
        payload["motor_state"] = motor_state
    return WsMessage(type=MsgType.PI_STATUS, payload=payload)


def make_pi_event(event: str, **kwargs: Any) -> WsMessage:
    """Create a ``pi_event`` message.

    Args:
        event: Event name, e.g. ``"emergency_stop"``.
        **kwargs: Additional key/value pairs included in the payload.

    """
    payload = {"event": event, **kwargs}
    return WsMessage(type=MsgType.PI_EVENT, payload=payload)


def make_pi_log(message: str, level: str = "info") -> WsMessage:
    """Create a ``pi_log`` message forwarding a journalctl line.

    Args:
        message: Log line text.
        level: Syslog severity string, e.g. ``"info"``, ``"warning"``.

    """
    return WsMessage(type=MsgType.PI_LOG, payload={"message": message, "level": level})


def make_laptop_status(model_loaded: bool | None = None, gpu_available: bool | None = None) -> WsMessage:
    """Create a ``laptop_status`` message.

    Args:
        model_loaded: Whether the LLM is ready to serve requests.
        gpu_available: Whether a GPU was detected.

    """
    payload: dict[str, Any] = {}
    if model_loaded is not None:
        payload["model_loaded"] = model_loaded
    if gpu_available is not None:
        payload["gpu_available"] = gpu_available
    return WsMessage(type=MsgType.LAPTOP_STATUS, payload=payload)


def make_laptop_event(event: str, **kwargs: Any) -> WsMessage:
    """Create a ``laptop_event`` message.

    Args:
        event: Event name, e.g. ``"restart_pending"``.
        **kwargs: Additional key/value pairs included in the payload.

    """
    payload = {"event": event, **kwargs}
    return WsMessage(type=MsgType.LAPTOP_EVENT, payload=payload)


def make_laptop_command(command: str, **kwargs: Any) -> WsMessage:
    """Create a ``laptop_command`` message.

    Args:
        command: Command name, e.g. ``"wake_up"``, ``"pause"``.
        **kwargs: Additional key/value pairs included in the payload.

    """
    payload = {"command": command, **kwargs}
    return WsMessage(type=MsgType.LAPTOP_COMMAND, payload=payload)
