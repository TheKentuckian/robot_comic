# WebSocket Protocol — Pi ↔ Laptop (v1)

A bidirectional WebSocket channel lets the Pi (Reachy controller) and the
laptop (LLM / TTS server) push events cheaply without polling.

> **Status**: v1 scaffold — the channel is **opt-in** and orthogonal to the
> existing HTTP API.  Handler wire-in (streaming ASR finals, Opus audio) is
> deferred to a follow-up PR.

---

## Security

v1 makes **no security guarantees** beyond what your LAN provides:

- No authentication.
- No TLS (plain `ws://`).

Restrict port `8765` to your LAN at the network/firewall level.  Both
endpoints are assumed to be trusted machines.

---

## Enabling the channel

Set the environment variable on **both** the Pi and the laptop:

```bash
REACHY_MINI_WS_ENABLED=true
```

When the variable is absent or `false`, neither the server nor the client
attempts to use the channel — zero overhead.

Additional knobs:

| Variable | Side | Default | Description |
|---|---|---|---|
| `REACHY_MINI_WS_PORT` | Both | `8765` | TCP port |
| `REACHY_MINI_WS_SERVER_HOST` | Pi (client) | `localhost` | Laptop hostname or IP |

---

## Message envelope

Every frame is a UTF-8 encoded JSON object:

```json
{"type": "<string>", "payload": {}}
```

- `type` — selects the message kind (see tables below).
- `payload` — message-specific data; always a JSON object (may be `{}`).

Binary frames are reserved for a future audio streaming extension.

---

## Message types

### Pi → Laptop

| `type` | Description | Payload fields |
|---|---|---|
| `pi_status` | Periodic status report | `battery` (float, 0–100), `motor_state` (string) |
| `pi_event` | One-shot event from the robot | `event` (string, e.g. `"emergency_stop"`) + optional extras |
| `pi_log` | Forward a journalctl line | `message` (string), `level` (string, e.g. `"info"`) |

### Laptop → Pi

| `type` | Description | Payload fields |
|---|---|---|
| `laptop_status` | Periodic status report | `model_loaded` (bool), `gpu_available` (bool) |
| `laptop_event` | One-shot event from the laptop | `event` (string, e.g. `"restart_pending"`) + optional extras |
| `laptop_command` | Command for the robot to act on | `command` (string, e.g. `"wake_up"`, `"pause"`) + optional extras |

### Both directions

| `type` | Description | Payload |
|---|---|---|
| `ping` | Keepalive request | `{}` |
| `pong` | Keepalive reply | `{}` |

The server auto-replies to `ping` with `pong`.  The client sends a `ping`
every 30 s while connected and also auto-replies to server-initiated `ping`.

---

## Example exchanges

### Status update (Pi → Laptop)

```json
{"type": "pi_status", "payload": {"battery": 87.5, "motor_state": "idle"}}
```

### Command (Laptop → Pi)

```json
{"type": "laptop_command", "payload": {"command": "wake_up"}}
```

### Emergency stop (Pi → Laptop)

```json
{"type": "pi_event", "payload": {"event": "emergency_stop", "severity": "critical"}}
```

### Keepalive

```
Client → Server:  {"type": "ping", "payload": {}}
Server → Client:  {"type": "pong", "payload": {}}
```

---

## Reconnect behaviour

The Pi-side client (`ws_client.py`) reconnects with exponential back-off on
any disconnect:

- Base delay: 1 s.
- Doubles on each failed attempt, capped at 30 s.
- Resets to 1 s after a successful reconnect.

---

## Implementation

| File | Role |
|---|---|
| `src/robot_comic/ws_protocol.py` | Message types, envelope (de)serialisation, factory helpers |
| `src/robot_comic/ws_server.py` | Server (runs on laptop) |
| `src/robot_comic/ws_client.py` | Client (runs on Pi), reconnect logic |
| `tests/test_ws_protocol.py` | Unit + integration tests (loopback, port 0) |

---

## Roadmap

- **v2**: ASR final frames (`{"type": "asr_final", "text": "..."}`) + binary
  Opus audio frames Pi → Laptop → Pi (sentence pipeline).
- **v3**: TLS + token authentication.
