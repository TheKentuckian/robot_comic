# TDD plan — Phase 5e.1 Moonshine STT adapter decouple

Spec: `docs/superpowers/specs/2026-05-16-phase-5e-1-moonshine-adapter-decouple.md`.
Branch: `claude/phase-5e-1-moonshine-adapter-decouple`.

Steps are RED → GREEN → REFACTOR. Conventional commits all scoped
`phase-5e-1`.

## Step 1 — Skeleton + standalone-mode adapter test (RED)

Create `tests/adapters/test_moonshine_listener.py` and
`src/robot_comic/adapters/moonshine_listener.py` as an empty stub. Add
the standalone-mode adapter tests to `tests/adapters/test_moonshine_stt_adapter.py`
that exercise `MoonshineSTTAdapter()` with `handler=None`. Run the new
tests. They should ImportError / fail.

Commit: `test(phase-5e-1): add failing standalone MoonshineSTTAdapter tests`.

## Step 2 — Implement `MoonshineListener` (GREEN/RED)

Implement `MoonshineListener` in
`src/robot_comic/adapters/moonshine_listener.py`. Mirror the mixin's
stream-only pieces:

- `__init__` — capture config (language, model, update_interval,
  cache_root). Use `config` defaults when None.
- `start(on_event)` — capture asyncio loop, run
  `_build_stream` in a worker thread (mirrors mixin's
  `asyncio.to_thread(_build_local_stt_stream)`).
- `_build_stream` — load `moonshine_voice`, get model, prewarm,
  construct `Transcriber`, call `_open_stream`.
- `_open_stream` — create stream, create dynamic listener subclass
  bridging to `on_event`, attach + start. Reset `_pending_rearm`.
- `_rearm_stream` — stop/close old stream, recreate on same
  transcriber. Mirror mixin lines 463-501.
- `feed_audio(sample_rate, samples)` — handle rearm flag, resample if
  needed, push `add_audio(payload, target_sr)`. No echo-guard skip.
- `stop()` — stop/close stream + transcriber. Set sentinels to None.

The `_MoonshineListener` subclass in this file fires:
`on_event("started", text)`, `on_event("partial", text)`,
`on_event("completed", text)`, `on_event("error", repr(err))`. Schedule
on the captured asyncio loop via `loop.call_soon_threadsafe`.

Run new listener tests until GREEN.

Commit: `feat(phase-5e-1): add standalone MoonshineListener (no host-handler)`.

## Step 3 — Wire standalone mode on `MoonshineSTTAdapter` (GREEN)

In `src/robot_comic/adapters/moonshine_stt_adapter.py`:

- Make `handler` argument default to `None`.
- Add `self._listener: MoonshineListener | None = None`.
- In `start()`: branch on `self._handler is None`.
  - Standalone: create `MoonshineListener()`, call its `start` with a
    bridge that filters `kind == "completed"`. Surface the callback
    invocation.
  - Host-coupled: existing behaviour verbatim.
- In `feed_audio()`: branch the same way. Standalone → forward to
  listener. Host-coupled → existing `handler.receive((sr, samples))`.
- In `stop()`: branch. Standalone → call `listener.stop()`. Host-coupled
  → existing `handler.shutdown()` + dispatch-restore.

Run all adapter tests; both new and existing must be GREEN.

Commit: `feat(phase-5e-1): add standalone construction shape to MoonshineSTTAdapter`.

## Step 4 — Export + docs (GREEN)

Add `MoonshineListener` to `src/robot_comic/adapters/__init__.py`
`__all__` and import. Update the `MoonshineSTTAdapter` module docstring
to describe the two construction shapes.

Run the full test suite. All green; existing factory + integration tests
are untouched.

Commit: `docs(phase-5e-1): document two MoonshineSTTAdapter construction shapes`.

## Step 5 — Lint / format / type / test (verification)

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
/venvs/apps_venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/adapters/moonshine_stt_adapter.py \
    src/robot_comic/adapters/moonshine_listener.py \
    src/robot_comic/local_stt_realtime.py
/venvs/apps_venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

Fix any issues; new commits per fix. Then push.

## Step 6 — Push

```
git push -u origin claude/phase-5e-1-moonshine-adapter-decouple
```

Do NOT open PR. Manager opens it.
