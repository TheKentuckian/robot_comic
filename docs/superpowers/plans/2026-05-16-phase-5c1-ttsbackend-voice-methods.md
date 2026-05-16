# Phase 5c.1 — TDD Plan

Spec: `docs/superpowers/specs/2026-05-16-phase-5c1-ttsbackend-voice-methods.md`

Conventional-commit scope: `phase-5c-1`.

## Task 1 — Add failing contract tests for new voice methods

Extend `tests/adapters/test_tts_backend_contract.py` with four new
parametric tests:

- `test_get_available_voices_returns_list_of_strings`
- `test_get_current_voice_returns_string`
- `test_change_voice_returns_resolved_id`
- `test_change_voice_to_unknown_voice_does_not_raise`

At this point the tests will fail because:

1. The stubs (`_ElevenLabsStub`, `_ChatterboxStub`, `_GeminiStub`) don't
   implement the methods.
2. The adapters don't implement the methods.

Also extend each stub with the three voice methods that match the
legacy handlers' contract (`async list[str]`, `sync str`, `async str`).
After stub updates, only the adapter side fails — that's the bug under
test.

Commit: `test(phase-5c-1): add failing TTSBackend voice-method contract tests`

## Task 2 — Extend `TTSBackend` Protocol with default-impl voice methods

Add the three methods to `TTSBackend` in `src/robot_comic/backends.py`,
each with a `raise NotImplementedError(...)` default body.

Verify the existing `test_mock_tts_satisfies_protocol` still passes by
extending `_MockTTS` in `tests/test_backends_protocols.py` with the
three methods (the simplest valid forms).

Commit: `feat(phase-5c-1): extend TTSBackend Protocol with voice methods + NotImplementedError defaults`

## Task 3 — Implement voice forwarding on each TTS adapter

For each of `ElevenLabsTTSAdapter`, `ChatterboxTTSAdapter`,
`GeminiTTSAdapter`:

- Add three forward methods (`get_available_voices`, `get_current_voice`,
  `change_voice`) that delegate to `self._handler`.
- Extend the duck-typed handler Protocols (`_ElevenLabsCompatibleHandler`,
  `_GeminiTTSCompatibleHandler`) with the three method declarations.

At this point all four new contract tests should pass for all three
adapters.

Commit: `feat(phase-5c-1): implement voice methods on TTS adapters (forward to handler)`

## Task 4 — Update wrapper to forward through pipeline.tts

Update `composable_conversation_handler.py`:

- `get_available_voices` → `self.pipeline.tts.get_available_voices()`
- `get_current_voice` → `self.pipeline.tts.get_current_voice()`
- `change_voice` → `self.pipeline.tts.change_voice(voice)`

Update the existing wrapper tests in
`tests/test_composable_conversation_handler.py`
(`test_get_current_voice_delegates`,
`test_get_available_voices_delegates`,
`test_change_voice_delegates`) to mock `wrapper.pipeline.tts` instead of
`wrapper._tts_handler`.

Keep `self._tts_handler` reference — still consulted by
`_reset_tts_per_session_state` in `apply_personality`.

Commit: `refactor(phase-5c-1): route wrapper voice methods through pipeline.tts adapter`

## Task 5 — Lint / format / typecheck / test gate

```bash
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/Scripts/python.exe -m mypy --pretty --show-error-codes \
    src/robot_comic/backends.py \
    src/robot_comic/composable_conversation_handler.py \
    src/robot_comic/adapters/
.venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

All four must be green before push.

## Task 6 — Push

`git push -u origin claude/phase-5c-1-ttsbackend-voice-methods`. No PR
opened (manager does that).
