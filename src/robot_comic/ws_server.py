"""WebSocket server (laptop side) for the Pi ↔ laptop channel.

The server accepts one or more connections from the Pi (or any authorised
client on the same LAN) and exchanges JSON-over-WS messages using the
envelope defined in :mod:`robot_comic.ws_protocol`.

Configuration (all optional, read from environment via ``config.py``)::

    REACHY_MINI_WS_ENABLED=true   # enable the channel (default: false)
    REACHY_MINI_WS_PORT=8765      # TCP port to listen on (default: 8765)

Security note: v1 has **no authentication** and **no TLS**.  Restrict this
port to your LAN at the network/firewall level.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable

from websockets.asyncio.server import ServerConnection, serve

from robot_comic.ws_protocol import MsgType, WsMessage, make_pong


logger = logging.getLogger(__name__)

# Sent to every connected client when the server shuts down.
_SHUTDOWN_MSG = WsMessage(type="server_shutdown", payload={"reason": "server stopping"}).to_json()


class WsServer:
    """Async WebSocket server that runs on the laptop.

    Typical usage::

        server = WsServer(host="0.0.0.0", port=8765)
        await server.start()
        # … use server.broadcast(msg) …
        await server.stop()

    Or use the async context manager::

        async with WsServer() as server:
            await server.broadcast(make_laptop_status(model_loaded=True))

    Handlers registered via :meth:`on_message` receive every inbound message
    (from any connected client) and can reply or take action.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        """Initialise the server without starting it.

        Args:
            host: Interface to bind to.  ``"0.0.0.0"`` means all interfaces.
            port: TCP port.  Pass ``0`` to let the OS pick a free port (useful
                in tests); retrieve the actual port from :attr:`port` after
                calling :meth:`start`.

        """
        self._host = host
        self._port = port
        self._server: Any = None  # websockets.Server instance once started
        self._clients: set[ServerConnection] = set()
        self._handlers: list[Callable[[WsMessage, ServerConnection], Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def port(self) -> int:
        """The actual TCP port the server is listening on.

        Returns the configured port before :meth:`start` is called.  After
        calling :meth:`start` with port ``0`` this returns the OS-assigned
        port.
        """
        if self._server is not None:
            sockets = self._server.sockets
            if sockets:
                return int(sockets[0].getsockname()[1])
        return self._port

    def on_message(self, handler: Callable[[WsMessage, ServerConnection], Any]) -> None:
        """Register a callback invoked for every inbound message.

        Args:
            handler: Called with ``(message, connection)``.  May be a plain
                function or a coroutine function.

        """
        self._handlers.append(handler)

    async def broadcast(self, msg: WsMessage) -> None:
        """Send *msg* to all currently connected clients.

        Silently drops sends to clients whose connections have already closed.
        """
        if not self._clients:
            return
        raw = msg.to_json()
        for ws in set(self._clients):
            try:
                await ws.send(raw)
            except Exception:
                logger.debug("ws_server: broadcast failed for one client (connection dropped)")

    async def start(self) -> None:
        """Start listening for incoming WebSocket connections."""
        self._server = await serve(
            self._handle_connection,
            self._host,
            self._port,
            ping_interval=None,  # we do application-level heartbeats
        )
        actual_port = self.port
        logger.info("ws_server: listening on %s:%d", self._host, actual_port)

    async def stop(self) -> None:
        """Shut down the server and close all active connections."""
        if self._server is None:
            return
        for ws in set(self._clients):
            try:
                await ws.send(_SHUTDOWN_MSG)
                await ws.close()
            except Exception:
                pass
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        logger.info("ws_server: stopped")

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "WsServer":
        """Start the server and return self."""
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop the server."""
        await self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle the lifetime of a single client connection."""
        self._clients.add(ws)
        peer = ws.remote_address
        logger.info("ws_server: client connected from %s", peer)
        try:
            async for raw in ws:
                if not isinstance(raw, str):
                    # Binary frames are not used in the control channel; ignore.
                    continue
                await self._dispatch(raw, ws)
        except Exception as exc:
            logger.debug("ws_server: connection from %s closed: %s", peer, exc)
        finally:
            self._clients.discard(ws)
            logger.info("ws_server: client disconnected from %s", peer)

    async def _dispatch(self, raw: str, ws: ServerConnection) -> None:
        """Parse and dispatch a raw text frame to registered handlers."""
        try:
            msg = WsMessage.from_json(raw)
        except ValueError as exc:
            logger.warning("ws_server: malformed message from %s: %s", ws.remote_address, exc)
            return

        # Auto-respond to ping keepalives.
        if msg.type == MsgType.PING:
            try:
                await ws.send(make_pong().to_json())
            except Exception:
                pass
            return

        for handler in self._handlers:
            try:
                result = handler(msg, ws)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("ws_server: unhandled error in message handler")


# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------


async def start_ws_server_if_enabled() -> WsServer | None:
    """Start the WS server when ``REACHY_MINI_WS_ENABLED`` is set.

    Returns the running :class:`WsServer` instance, or ``None`` when the
    feature is disabled.  Callers are responsible for stopping the server
    (call :meth:`WsServer.stop` during shutdown).
    """
    from robot_comic.config import config  # local import to avoid circular dep

    enabled: bool = getattr(config, "WS_ENABLED", False)
    if not enabled:
        logger.debug("ws_server: disabled (REACHY_MINI_WS_ENABLED not set)")
        return None

    port: int = getattr(config, "WS_PORT", 8765)
    server = WsServer(port=port)
    await server.start()
    return server
