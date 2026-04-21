#!/usr/bin/env python3
"""Text-only realtime probe for the speech-to-speech backend.

This lets you exercise the same session allocator, resolved system prompt,
and voice selection that the conversation app would use, without requiring
the robot audio stack.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import websockets


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reachy_mini_conversation_app.config import DEFAULT_VOICE, config
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_voice
from reachy_mini_conversation_app.tools.core_tools import get_tool_specs


def _pcm(rate: int) -> dict[str, Any]:
    return {"type": "audio/pcm", "rate": rate}


def _session_audio_format_for_provider(rate: int) -> dict[str, Any] | None:
    """Return a session.update format block, or None to omit it."""
    if config.BACKEND_PROVIDER == "speech-to-speech" and rate == 16000:
        return None
    return _pcm(rate)


def add_model_query_param(ws_url: str) -> str:
    """Mirror the conversation app's realtime connect query."""
    parsed = urlsplit(ws_url)
    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_items.setdefault("model", config.MODEL_NAME)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_items),
            parsed.fragment,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-only latency probe for the speech-to-speech realtime endpoint.",
    )
    parser.add_argument(
        "--text",
        default="Hey, how are you doing? Answer in one short sentence.",
        help="User text to send as the turn.",
    )
    parser.add_argument(
        "--session-url",
        default=os.getenv("S2S_REALTIME_SESSION_URL") or getattr(config, "S2S_REALTIME_SESSION_URL", None),
        help="Session allocation URL. Defaults to S2S_REALTIME_SESSION_URL from env/config.",
    )
    parser.add_argument(
        "--authorization",
        default=os.getenv("S2S_AUTHORIZATION"),
        help="Optional Authorization header value for the session allocator.",
    )
    parser.add_argument(
        "--print-events",
        action="store_true",
        help="Print every non-audio realtime event as raw JSON.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the fully resolved system prompt before connecting.",
    )
    parser.add_argument(
        "--show-session-config",
        action="store_true",
        help="Print the exact session.update payload the probe sends.",
    )
    return parser.parse_args()


async def allocate_session(session_url: str, authorization: str | None) -> dict[str, Any]:
    headers = {}
    if authorization:
        headers["Authorization"] = authorization

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(session_url, headers=headers)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Session allocator returned non-object payload: {payload!r}")
    return payload


async def main() -> None:
    args = parse_args()

    if not args.session_url:
        raise SystemExit(
            "Missing session URL. Set S2S_REALTIME_SESSION_URL or pass --session-url."
        )

    instructions = get_session_instructions()
    voice = get_session_voice(default=DEFAULT_VOICE)

    if args.show_prompt:
        print("=== Prompt ===")
        print(instructions)
        print("=== End Prompt ===")
        print()

    input_audio: dict[str, Any] = {
        "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
        "turn_detection": {"type": "server_vad", "interrupt_response": True},
    }
    input_format = _session_audio_format_for_provider(16000)
    if input_format is not None:
        input_audio["format"] = input_format

    output_audio: dict[str, Any] = {
        "voice": voice,
    }
    output_format = _session_audio_format_for_provider(16000)
    if output_format is not None:
        output_audio["format"] = output_format

    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": instructions,
            "audio": {
                "input": input_audio,
                "output": output_audio,
            },
            "tools": get_tool_specs(),
            "tool_choice": "auto",
        },
    }

    if args.show_session_config:
        print("=== Session Update Payload ===")
        print(json.dumps(session_update, indent=2, ensure_ascii=True))
        print("=== End Session Update Payload ===")
        print()

    print(f"provider: {config.BACKEND_PROVIDER}")
    print(f"voice: {voice}")
    print(f"prompt_chars: {len(instructions)}")
    print(f"tool_count: {len(get_tool_specs())}")

    t0 = time.perf_counter()
    allocation = await allocate_session(args.session_url, args.authorization)
    t1 = time.perf_counter()
    connect_url = allocation.get("connect_url")
    session_id = allocation.get("session_id")

    if not isinstance(connect_url, str) or not connect_url:
        raise SystemExit(f"Allocator returned no valid connect_url: {allocation!r}")

    connect_url = add_model_query_param(connect_url)

    print(f"allocated session: {session_id or '<unknown>'}")
    print(f"allocation_time_ms: {(t1 - t0) * 1000:.0f}")
    print(f"model: {config.MODEL_NAME}")

    async with websockets.connect(
        connect_url,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:
        await websocket.send(json.dumps(session_update))

        await websocket.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": args.text}],
                    },
                }
            )
        )
        await websocket.send(json.dumps({"type": "response.create"}))

        turn_start = time.perf_counter()
        first_audio_at: float | None = None

        while True:
            raw = await websocket.recv()
            now = time.perf_counter()

            if isinstance(raw, bytes):
                continue

            event = json.loads(raw)
            event_type = str(event.get("type") or "").strip()

            if args.print_events and event_type != "response.output_audio.delta":
                print(json.dumps(event, ensure_ascii=True))

            if event_type == "session.created":
                continue

            if event_type == "response.created":
                print(f"response_created_ms: {(now - turn_start) * 1000:.0f}")
                continue

            if event_type == "response.output_audio.delta":
                if first_audio_at is None:
                    first_audio_at = now
                    print(f"first_audio_delta_ms: {(now - turn_start) * 1000:.0f}")
                continue

            if event_type == "response.output_audio_transcript.done":
                transcript = str(event.get("transcript") or "").strip()
                if transcript:
                    print(f"assistant: {transcript}")
                continue

            if event_type == "error":
                print("error:")
                print(json.dumps(event, indent=2, ensure_ascii=True))
                break

            if event_type == "response.done":
                response = event.get("response")
                status = ""
                if isinstance(response, dict):
                    status = str(response.get("status") or "").strip()
                print(f"response_done_ms: {(now - turn_start) * 1000:.0f}")
                if status:
                    print(f"response_status: {status}")
                break


if __name__ == "__main__":
    asyncio.run(main())
