# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install dependencies
uv sync --extra all_vision       # Include vision extras (MediaPipe, SmolVLM2, YOLO)

uv run ruff check . --fix        # Lint and auto-fix
uv run ruff format .             # Format code
uv run mypy --pretty --show-error-codes  # Type check

uv run pytest tests/ -v          # Run all tests
uv run pytest tests/test_openai_realtime.py -v  # Run a single test file
uv run pytest tests/ -k "test_name" -v          # Run a specific test by name

reachy-mini-conversation-app     # Run the app (console/headless mode)
reachy-mini-conversation-app --gradio  # Run with Gradio web UI
reachy-mini-conversation-app --debug   # Verbose logging
```

## Architecture

This is a conversational voice app for the **Reachy Mini robot**, implementing a comedian persona (Don Rickles-style). The key abstraction layers are:

### Handler (Voice Backend)
All realtime voice handlers inherit from `BaseRealtimeHandler` (`base_realtime.py`). Concrete backends:
- `huggingface_realtime.py` — Hugging Face (default)
- `openai_realtime.py` — OpenAI Realtime API
- `gemini_live.py` — Gemini Live
- `local_stt_realtime.py` — Local Moonshine STT

Backend is selected via `BACKEND_PROVIDER` env var in `config.py`.

### Movement Manager (`moves.py`)
Runs a 60 Hz control loop in a background thread. Distinguishes:
- **Primary moves** — exclusive (dances, explicit emotion poses, breathing) queued via `MoveQueue`
- **Secondary moves** — additive on top of primary (head wobble, speech-reactive sway)

Move classes inherit from `Move` and compose world-space offsets.

### Tool System (`tools/`)
LLM-callable tools (dance, camera, head tracking, emotions, etc.) are Python classes inheriting from `Tool`. Tool specs live in `tool_constants.py`. The `BackgroundToolManager` queues and dispatches tool calls from the handler without blocking audio streaming.

### Profiles / Personality System (`profiles/`)
Each profile directory contains:
- `instructions.txt` — system prompt / persona definition
- `tools.txt` — enabled tools list
- Optional `.py` files — custom tool implementations

External profiles can be loaded from `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY`. The `LOCKED_PROFILE` config option pins the app to a single persona. Profile packaging into the wheel is handled by `BuildPyWithProfiles` in `setup.py`.

### Camera & Vision (`camera_worker.py`, `vision/`)
Camera runs in a dedicated thread. Head tracking is pluggable: `mediapipe` (in-process, default) or `yolo` (subprocess). Local vision uses SmolVLM2; backend vision uses the LLM provider's vision API.

### Audio (`audio/`)
`HeadWobbler` produces speech-reactive secondary head motion. `startup_settings.json` persists UI profile/voice selections across restarts.

## Key Conventions

- **Threading model**: MovementManager, CameraWorker, and HeadWobbler each own a dedicated thread. The main thread drives the asyncio event loop for audio streaming.
- **Startup config**: Persisted to `startup_settings.json`; reloaded on app restart from the Gradio UI.
- **Config refresh**: `config.py` centralizes all env-var reading; backends call `refresh()` at startup.
- **Cross-platform**: All code must work on Linux, macOS, and Windows.
- **Git branches**: `feat/<issue>-<desc>`, `fix/<issue>-<desc>`. `main` is the release branch; PRs require CI (lint + types + tests) to pass.
