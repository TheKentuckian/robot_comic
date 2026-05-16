# Phase 5d — TDD execution plan

**Spec:** `docs/superpowers/specs/2026-05-16-phase-5d-conversationhandler-abc-shrink.md`
**Branch:** `claude/phase-5d-conversationhandler-abc-shrink`
**Scope tag for commits:** `phase-5d`

## Commit sequence (TDD)

### Commit 1 — Test pinning the new ABC contract (RED)

Add `tests/test_conversation_handler_abc.py`:

- Define a minimal subclass implementing only the five remaining
  abstract methods (`copy`, `start_up`, `shutdown`, `receive`, `emit`)
  PLUS the three field annotations (`deps`, `output_queue`,
  `_clear_queue`).
- Assert it instantiates without `TypeError`.
- Assert that `apply_personality`, `get_available_voices`,
  `get_current_voice`, `change_voice` are NOT in
  `ConversationHandler.__abstractmethods__`.

Before the source change this test fails because those four methods are
still in `__abstractmethods__`.

`test(phase-5d): pin shrunken ConversationHandler ABC contract`

### Commit 2 — Drop the four abstract methods (GREEN)

Edit `src/robot_comic/conversation_handler.py`:

- Remove `@abstractmethod` decls for `apply_personality`,
  `get_available_voices`, `get_current_voice`, `change_voice`.
- Update the class docstring to reflect the FastRTC-shim role and
  point at the spec.

The test from commit 1 turns green. The 20 existing
`test_composable_conversation_handler.py` tests stay green (wrapper
still implements those methods as forwarders).

`refactor(phase-5d): shrink ConversationHandler ABC to FastRTC-shim role`

### Commit 3 — Caller migration to duck-typing (mypy fix)

Edit `src/robot_comic/headless_personality_ui.py`:

- Wrap the four direct calls (`handler.apply_personality`,
  `handler.get_available_voices`, `handler.get_current_voice`,
  `handler.change_voice`) in `getattr` lookups with safe fallbacks.
- Preserve runtime behavior identically — concrete handlers always
  expose the methods today.

`refactor(phase-5d): reach voice/personality methods via getattr in headless UI`

## Verification matrix

After each commit, the following must hold:

| Check | Commit 1 (RED) | Commit 2 (GREEN) | Commit 3 |
|---|---|---|---|
| `tests/test_conversation_handler_abc.py` | FAIL (expected) | PASS | PASS |
| `tests/test_composable_conversation_handler.py` | PASS | PASS | PASS |
| `tests/test_handler_factory*.py` (all variants) | PASS | PASS | PASS |
| `mypy src/robot_comic/conversation_handler.py` | PASS | PASS | PASS |
| `mypy src/robot_comic/headless_personality_ui.py` | PASS | FAIL (expected) | PASS |
| `mypy src/robot_comic/composable_conversation_handler.py` | PASS | PASS | PASS |
| `mypy src/robot_comic/base_realtime.py` | PASS | PASS | PASS |
| `ruff check .` from repo root | PASS | PASS | PASS |
| Full pytest suite (xdist flakes excepted) | PASS | PASS | PASS |

## Out-of-band verification (after all commits)

- `uvx ruff@0.12.0 check .` from repo root.
- `uvx ruff@0.12.0 format --check .` from repo root.
- `mypy --pretty --show-error-codes src/robot_comic/conversation_handler.py
  src/robot_comic/composable_conversation_handler.py
  src/robot_comic/base_realtime.py
  src/robot_comic/headless_personality_ui.py`.
- `python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py`.

## Rollback plan

Each commit is independently revertable. Commit 1 alone is dead test
code (the test fails). Commit 2 alone breaks mypy in headless UI.
Commit 3 alone is a no-op refactor. All three together complete the
shrink.

## Push contract

- `git push -u origin claude/phase-5d-conversationhandler-abc-shrink`
- Do NOT open PR. Manager session will open the PR referencing #391.
