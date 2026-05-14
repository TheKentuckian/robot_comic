"""WebSocket client (Pi side) for the Pi ↔ laptop channel.

The client connects to the laptop's :mod:`robot_comic.ws_server` and
exchanges JSON-over-WS messages.  It reconnects automatically with
exponential back-off when the server is unreachable or the connection drops.
A heartbeat ``ping`` is sent every 30 s while connected.

Configuration (all optional, read from environment via ``config.py``)::

    REACHY_MINI_WS_ENABLED=true         # enable the channel (default: false)
    REACHY_MINI_WS_SERVER_HOST=<host>   # laptop hostname/IP (default: "localhost")
    REACHY_MINI_WS_PORT=8765            # server port (default: 8765)

Security note: v1 has **no authentication** and **no TLS**.  Both endpoints
are assumed to be on the same trusted LAN.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable

from websockets.protocol import State
from websockets.asyncio.client import ClientConnection, connect

from robot_comic.ws_protocol import MsgType, WsMessage, make_ping, make_pong


logger = logging.getLogger(__name__)

# Back-off bounds (seconds)
_BACKOFF_BASE = 1.0
_BACKOFF_MAX = 30.0
_HEARTBEAT_INTERVAL = 30.0  # seconds between keepalive pings


class WsClient:
    """Async WebSocket client that runs on the Pi.

    Automatically reconnects with exponential back-off when the connection is
    lost.  A heartbeat ``ping`` is sent every :data:`_HEARTBEAT_INTERVAL`
    seconds while connected.

    Typical usage::

        client = WsClient(server_host="astralplane.lan", port=8765)
        await client.start()
        await client.send(make_pi_status(battery=87.5, motor_state="idle"))
        …
        await client.stop()

    Or as an async context manager::

        async with WsClient(server_host="astralplane.lan") as client:
            await client.send(make_pi_status(battery=87.5))
    """

    def __init__(
        self,
        server_host: str = "localhost",
        port: int = 8765,
    ) -> None:
        """Initialise the client without connecting.

        Args:
            server_host: Hostname or IP of the laptop running :class:`~robot_comic.ws_server.WsServer`.
            port: TCP port the server is listening on.

        """
        self._server_host = server_host
        self._port = port
        self._uri = f"ws://{server_host}:{port}"
        self._stop_event: asyncio.Event = asyncio.Event()
        self._connected_event: asyncio.Event = asyncio.Event()
        self._ws: ClientConnection | None = None
        self._task: asyncio.Task[None] | None = None
        self._handlers: list[Callable[[WsMessage], Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return ``True`` when there is an active WebSocket connection."""
        return self._ws is not None and self._ws.state is State.OPEN

    def on_message(self, handler: Callable[[WsMessage], Any]) -> None:
        """Register a callback invoked for every inbound message.

        Args:
            handler: Called with the parsed :class:`~robot_comic.ws_protocol.WsMessage`.
                May be a plain function or a coroutine function.

        """
        self._handlers.append(handler)

    async def send(self, msg: WsMessage) -> bool:
        """Send *msg* to the server.

        Returns:
            ``True`` on success, ``False`` when not connected or send failed.

        """
        ws = self._ws
        if ws is None or ws.state is not State.OPEN:
            logger.debug("ws_client: send skipped — not connected")
            return False
        try:
            await ws.send(msg.to_json())
            return True
        except Exception as exc:
            logger.warning("ws_client: send failed: %s", exc)
            return False

    async def wait_connected(self, timeout: float = 5.0) -> bool:
        """Wait until the client has successfully connected.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            ``True`` if connected within *timeout*, ``False`` otherwise.

        """
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def start(self) -> None:
        """Start the background reconnect/receive loop."""
        self._stop_event.clear()
        self._task = asyncio.ensure_future(self._run_loop())
        logger.info("ws_client: started — targeting %s", self._uri)

    async def stop(self) -> None:
        """Signal the client to stop and wait for the background task."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("ws_client: stopped")

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "WsClient":
        """Start the client and return self."""
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop the client."""
        await self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Run the reconnect loop until :meth:`stop` is called."""
        backoff = _BACKOFF_BASE
        while not self._stop_event.is_set():
            try:
                async with connect(self._uri) as ws:
                    self._ws = ws
                    self._connected_event.set()
                    backoff = _BACKOFF_BASE  # reset on successful connect
                    logger.info("ws_client: connected to %s", self._uri)
                    await self._session(ws)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._ws = None
                self._connected_event.clear()
                if self._stop_event.is_set():
                    return
                logger.warning(
                    "ws_client: connection lost/failed (%s). Retrying in %.1f s…",
                    exc,
                    backoff,
                )
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                    return  # stop_event was set during wait
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, _BACKOFF_MAX)

        self._ws = None
        self._connected_event.clear()

    async def _session(self, ws: ClientConnection) -> None:
        """Handle a single connected session: receive + heartbeat."""
        heartbeat_task = asyncio.ensure_future(self._heartbeat(ws))
        try:
            async for raw in ws:
                if not isinstance(raw, str):
                    continue
                await self._dispatch(raw)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _heartbeat(self, ws: ClientConnection) -> None:
        """Send periodic ``ping`` keepalives while connected."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            try:
                await ws.send(make_ping().to_json())
                logger.debug("ws_client: ping sent")
            except Exception:
                return  # connection gone; let _session handle it

    async def _dispatch(self, raw: str) -> None:
        """Parse a raw frame and invoke registered message handlers."""
        try:
            msg = WsMessage.from_json(raw)
        except ValueError as exc:
            logger.warning("ws_client: malformed message from server: %s", exc)
            return

        # Auto-respond to server-initiated ping.
        if msg.type == MsgType.PING:
            ws = self._ws
            if ws is not None:
                try:
                    await ws.send(make_pong().to_json())
                except Exception:
                    pass
            return

        for handler in self._handlers:
            try:
                result = handler(msg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("ws_client: unhandled error in message handler")


# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------


async def start_ws_client_if_enabled() -> WsClient | None:
    """Start the WS client when ``REACHY_MINI_WS_ENABLED`` is set.

    Returns the running :class:`WsClient` instance, or ``None`` when the
    feature is disabled.  Callers are responsible for stopping the client
    (call :meth:`WsClient.stop` during shutdown).
    """
    from robot_comic.config import config  # local import to avoid circular dep

    enabled: bool = getattr(config, "WS_ENABLED", False)
    if not enabled:
        logger.debug("ws_client: disabled (REACHY_MINI_WS_ENABLED not set)")
        return None

    server_host: str = getattr(config, "WS_SERVER_HOST", "localhost")
    port: int = getattr(config, "WS_PORT", 8765)
    client = WsClient(server_host=server_host, port=port)
    await client.start()
    return client
