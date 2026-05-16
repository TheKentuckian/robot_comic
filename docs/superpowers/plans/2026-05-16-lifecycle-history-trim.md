# TDD plan — Lifecycle Hook #5 `history_trim.trim_history_in_place`

**Branch:** `claude/lifecycle-history-trim`
**Spec:** `docs/superpowers/specs/2026-05-16-lifecycle-history-trim.md`

Each task lands as one commit. Tests come first, implementation
second, docs / cleanup last.

## Task 1 — failing tests for `ComposablePipeline` orchestrator trim hook

Add tests in `tests/test_composable_pipeline.py` under a new
"Lifecycle Hook #5 — trim_history_in_place" section:

- `test_trim_history_called_once_per_user_turn_before_llm_call` —
  monkeypatch `composable_pipeline.trim_history_in_place` to a
  recorder; one transcript → one LLM call → assert recorder fired
  exactly once and BEFORE the LLM `chat()` call (use call-order
  capture, e.g. timestamp lists or a shared sentinel list).
- `test_trim_history_uses_orchestrator_history_list` — assert the
  argument the recorder receives is identical (by `is`) to
  `pipeline._conversation_history`. This proves the trim operates on
  the orchestrator's list, not a copy.
- `test_trim_history_called_once_per_turn_not_per_tool_round` — script
  a tool round followed by an assistant text round; assert the trim
  recorder fired exactly once (not twice). Mirrors the legacy
  per-user-turn cadence.
- `test_trim_history_cap_respected_across_user_turns` — end-to-end
  test: set `REACHY_MINI_MAX_HISTORY_TURNS=2` via
  `monkeypatch.setenv`, drive three user transcripts through the
  pipeline (no monkeypatch of `trim_history_in_place` — exercise the
  real helper), assert
  `len([m for m in pipeline.conversation_history if m["role"] == "user"]) == 2`
  after the third turn.

These tests will fail until Task 2 lands the call.

Commit message: `test(composable-pipeline): assert trim_history_in_place fires per user turn (#337)`

## Task 2 — wire `trim_history_in_place` into `ComposablePipeline`

In `src/robot_comic/composable_pipeline.py`:

- Add `from robot_comic.history_trim import trim_history_in_place` at
  the module top.
- In `_run_llm_loop_and_speak`, immediately at the top of the method
  body and before the `for _round in range(self.max_tool_rounds):`
  loop, call:

  ```python
  # Lifecycle Hook #5 (#337): cap the conversation history at
  # ``REACHY_MINI_MAX_HISTORY_TURNS`` user turns so long sessions don't
  # blow the model's context window or run the token bill into the
  # ground. Once-per-user-turn cadence matches the legacy sites in
  # ``_dispatch_completed_transcript``. Legacy parity:
  # llama_base.py:506, gemini_tts.py:365, elevenlabs_tts.py:565.
  trim_history_in_place(self._conversation_history)
  ```

Tests from Task 1 turn green.

Commit message: `feat(composable-pipeline): trim conversation history before each LLM loop (#337)`

## Task 3 — refresh the lifecycle TODO comment

In `src/robot_comic/composable_conversation_handler.py`, the module
docstring's deferred-hooks list (lines 8–17) still lists
`history_trim.trim_history_in_place` as unhandled. Update that single
line to "wired (Hook #5) at `ComposablePipeline._run_llm_loop_and_speak`"
matching the format already used for Hooks #2–#4.

Commit message: `docs(composable-handler): mark history_trim hook as wired (#337)`

## Task 4 — pre-push verification

Run from repo root:

```
uvx ruff@0.12.0 check
uvx ruff@0.12.0 format --check
.venv/Scripts/python -m mypy --pretty src/robot_comic/composable_pipeline.py \
                                     src/robot_comic/composable_conversation_handler.py
.venv/Scripts/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

All four green before push.

## Task 5 — push and draft PR

- Push `claude/lifecycle-history-trim` to `origin`.
- Draft PR with the body shape from the manager brief (background,
  decision, scope, acceptance criteria, anything-weird).
- Do NOT click "create"; the manager session opens the PR after
  reviewing the diff.
