# TDD plan — Lifecycle Hook #4 `record_joke_history`

**Branch:** `claude/lifecycle-record-joke-history`
**Spec:** `docs/superpowers/specs/2026-05-16-lifecycle-record-joke-history.md`

Each task lands as one commit. Tests come first, implementation second,
docs / cleanup last.

## Task 1 — failing tests for `joke_history.record_joke_history`

Add tests in `tests/test_joke_history.py` for a not-yet-existing helper
`record_joke_history(response_text, *, persona=None)`:

- `test_record_joke_history_disabled_skips_extraction_and_add`
- `test_record_joke_history_empty_text_skips_extraction`
- `test_record_joke_history_success_path_calls_add_with_punchline_topic_persona`
- `test_record_joke_history_none_punchline_skips_add`
- `test_record_joke_history_extraction_exception_swallowed`
- `test_record_joke_history_add_exception_swallowed`
- `test_record_joke_history_persona_defaults_to_config_custom_profile`

Mock `extract_punchline_via_llm` and `JokeHistory.add` via
`monkeypatch.setattr` on the module. Use `tmp_path` for any path-side
work.

Commit message: `test(joke-history): add failing tests for record_joke_history helper (#337)`

## Task 2 — implement `record_joke_history` helper

In `src/robot_comic/joke_history.py`, add a public async function:

```python
async def record_joke_history(
    response_text: str,
    *,
    persona: str | None = None,
) -> None:
    """Capture a punchline + topic from *response_text* into joke history.

    Best-effort: returns silently if the feature is disabled, the text is
    empty, the extraction fails, or the file write fails. Mirrors the
    legacy capture in ``llama_base.py`` and ``gemini_tts.py`` but owns
    its own httpx client lifecycle so the orchestrator doesn't need to.
    """
```

Body:

- Import config locally to avoid circular at module load (matches the
  existing pattern in `extract_punchline_via_llm`).
- Early-return on `not response_text` or `not
  getattr(config, "JOKE_HISTORY_ENABLED", True)`.
- Resolve `persona = persona if persona is not None else
  getattr(config, "REACHY_MINI_CUSTOM_PROFILE", "") or ""`.
- `try` block:
  - `async with httpx.AsyncClient() as http: extracted = await
    extract_punchline_via_llm(response_text, http)`.
  - `punchline = extracted.get("punchline") if extracted else None`
  - if `punchline`: `JokeHistory(default_history_path()).add(punchline,
    topic=extracted.get("topic", "") or "", persona=persona)`.
- `except Exception as exc: logger.debug("joke_history capture failed: %s",
  exc)`.

Tests from Task 1 turn green.

Commit message: `feat(joke-history): add record_joke_history helper for composable path (#337)`

## Task 3 — failing tests for `ComposablePipeline` orchestrator hook

Add tests in `tests/test_composable_pipeline.py`:

- `test_record_joke_history_called_for_non_empty_assistant_text` — script
  one `LLMResponse(text="That's the joke!")`, monkeypatch
  `composable_pipeline.record_joke_history` to a recorder, assert called
  once with that text.
- `test_record_joke_history_not_called_for_empty_text` — empty text path
  must not invoke the helper (matches legacy `if response_text:` guard).
- `test_record_joke_history_not_called_for_tool_only_rounds` — script
  a tool round followed by an assistant text round; the helper fires
  exactly once, on the assistant text round, not on the tool round.

The first test will fail until Task 4 lands the call. The third test
exercises the "fires *after* final speak round, not per LLM round"
property documented in the spec.

Commit message: `test(composable-pipeline): assert record_joke_history fires on final speak round (#337)`

## Task 4 — wire `record_joke_history` into `ComposablePipeline`

In `src/robot_comic/composable_pipeline.py`:

- Add `from robot_comic.joke_history import record_joke_history` at the
  module top.
- In `_speak_assistant_text`, immediately after the empty-text guard
  (`if not text.strip(): ... return`) and before the
  `self._conversation_history.append(...)` line, call:

  ```python
  # Lifecycle Hook #4 (#337): capture punchline + topic into joke history
  # so the next session's system-prompt builder can include them in the
  # "RECENT JOKES (DO NOT REPEAT)" section. Best-effort, swallows errors
  # inside the helper. Legacy parity site: llama_base.py:578-594 /
  # gemini_tts.py:380-394.
  await record_joke_history(text)
  ```

Tests from Task 3 turn green.

Commit message: `feat(composable-pipeline): call record_joke_history after final assistant text (#337)`

## Task 5 — refresh the lifecycle TODO comment

In `src/robot_comic/composable_conversation_handler.py`, the module
docstring's deferred-hooks list and the inline `TODO(phase4-lifecycle)`
in `apply_personality` both reference `record_joke_history` as
unhandled. Update both to point at the new orchestrator hook (a comment-
only change; no behaviour change). This keeps future readers from
re-investigating the same question.

Commit message: `docs(composable-handler): mark record_joke_history hook as wired (#337)`

## Task 6 — pre-push verification

Run from repo root:

```
uvx ruff@0.12.0 check
uvx ruff@0.12.0 format --check
.venv/Scripts/python -m mypy --pretty src/robot_comic/joke_history.py \
                                     src/robot_comic/composable_pipeline.py \
                                     src/robot_comic/composable_conversation_handler.py
.venv/Scripts/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

All four green before push.

## Task 7 — push and draft PR

- Push `claude/lifecycle-record-joke-history` to `origin`.
- Draft PR with the body shape from the manager brief (background,
  decision, scope, acceptance criteria, anything-weird).
- Do NOT click "create"; the manager session opens the PR after
  reviewing the diff.
