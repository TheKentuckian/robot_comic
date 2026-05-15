# Plan — Lifecycle Hook #3 (boot-timeline supporting events on composable path)

**Spec:** `docs/superpowers/specs/2026-05-16-lifecycle-boot-timeline-events.md`
**Branch:** `claude/lifecycle-boot-timeline-events`

## Strategy

Two TDD tasks, each one failing test → minimum patch → green commit.

## Task 1 — emit `handler.start_up.complete` before delegating to pipeline

**Failing test (a):** `test_start_up_emits_handler_start_up_complete_before_delegating`

In `tests/test_composable_conversation_handler.py`, add a test that:

1. Builds a wrapper with `_make_wrapper()`.
2. Sets `wrapper.pipeline.start_up` to an `AsyncMock` that records whether
   `telemetry.emit_supporting_event` was called *before* it was awaited.
3. Patches `robot_comic.telemetry.emit_supporting_event`.
4. Awaits `wrapper.start_up()`.
5. Asserts:
   - `emit_supporting_event` was called with `name="handler.start_up.complete"`
     and a numeric `dur_ms` kwarg.
   - `pipeline.start_up` was awaited.
   - The emit happened before the pipeline await (via a recording side-effect
     on the pipeline mock).

Expect this to fail today: the wrapper delegates straight to the pipeline
without any emit.

**Minimum patch:**

In `src/robot_comic/composable_conversation_handler.py:start_up`, replace the
TODO + bare delegate body with:

```python
async def start_up(self) -> None:
    """Emit ``handler.start_up.complete`` then delegate to the pipeline.

    Mirrors the legacy ``ElevenLabsTTSResponseHandler.start_up`` emit so the
    monitor's boot-timeline lane closes with a ``handler.start_up.complete``
    row on the composable path too (#321 / #337 deferred lifecycle hook).
    """
    try:
        from robot_comic.startup_timer import since_startup
        from robot_comic import telemetry as _telemetry

        _telemetry.emit_supporting_event(
            "handler.start_up.complete",
            dur_ms=since_startup() * 1000,
        )
    except Exception:
        # Telemetry must never block boot — drop the row if emission throws
        # (import error, exporter wiring quirk, etc.).
        pass
    await self.pipeline.start_up()
```

Commit message:
```
feat(composable-handler): emit handler.start_up.complete before delegating (#337)
```

## Task 2 — emit-failure must not break delegation

**Failing test (b):** `test_start_up_emit_failure_does_not_break_pipeline_delegation`

In the same test file, add a test that:

1. Builds a wrapper.
2. Patches `robot_comic.telemetry.emit_supporting_event` to raise
   `RuntimeError("export wiring broken")`.
3. Awaits `wrapper.start_up()` — expects it to return normally.
4. Asserts `pipeline.start_up` was still awaited once.

If the Task 1 patch already wraps the emit in `try/except Exception:` (it does),
this test should pass without code changes. Commit only if a regression
emerged or to lock the behaviour.

Since Task 1 already lands the `try/except`, this is a regression-only test.
Single commit:

```
test(composable-handler): pin emit-failure resilience on start_up (#337)
```

## Verification (final commit, only if anything moved)

From repo root:

```
uvx ruff@0.12.0 check
uvx ruff@0.12.0 format --check
.venv/Scripts/mypy --pretty src/robot_comic/composable_conversation_handler.py tests/test_composable_conversation_handler.py
.venv/Scripts/pytest tests/ -q
```

If `tests/vision/test_local_vision.py` collection fails locally with a
tokenizers/transformers version clash, rerun with
`--ignore=tests/vision/test_local_vision.py`. CI does not have this issue.
