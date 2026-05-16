# Phase 4e — Implementation plan

**Spec:** `docs/superpowers/specs/2026-05-16-phase-4e-legacy-deletion.md`
**Branch:** `claude/phase-4e-legacy-deletion`

## Strategy

One commit per logical unit. Each commit leaves the suite green so
`git bisect` works. Commits are ordered so the factory rewrite lands
*before* the legacy class deletions — the factory holds the only
in-`src/` references to the deleted classes, so cutting the import edges
first means the deletions become file-local once the factory is
updated.

## Tasks

### Task 1 — Spec + plan

Land the spec and TDD plan at `docs/superpowers/{specs,plans}/2026-05-16-phase-4e-legacy-deletion.md`.

### Task 2 — Factory: compose the mixin inline, drop the `FACTORY_PATH` dial

- `src/robot_comic/handler_factory.py`:
  - Define five factory-private host classes at module scope:
    `_LocalSTTLlamaElevenLabsHost`, `_LocalSTTLlamaChatterboxHost`,
    `_LocalSTTGeminiChatterboxHost`, `_LocalSTTGeminiElevenLabsHost`,
    `_LocalSTTGeminiTTSHost`.
    Each is `(LocalSTTInputMixin, <surviving response handler base>)`.
  - Rewrite the five `_build_composable_*` helpers to instantiate the
    host class. Drop the `LocalSTT*Handler` imports.
  - Delete the entire `if config.FACTORY_PATH == FACTORY_PATH_COMPOSABLE:`
    branching from the moonshine block. The composable path becomes the
    only path; the per-triple dispatch falls straight through to the
    `_build_composable_*` helper.
  - Delete the now-unused legacy import sites (`from
    robot_comic.chatterbox_tts import LocalSTTChatterboxHandler`, etc.).
  - Delete the `FACTORY_PATH_LEGACY` / `FACTORY_PATH_COMPOSABLE` imports.
  - Refresh the module docstring's "Supported (input, output) → handler"
    matrix to say `ComposableConversationHandler` (with the underlying
    *ResponseHandler base) for every composable triple. Refresh the
    `_SUPPORTED_MATRIX_DOC` operator-facing message similarly.
- `src/robot_comic/config.py`:
  - Delete `FACTORY_PATH_ENV`, `FACTORY_PATH_LEGACY`,
    `FACTORY_PATH_COMPOSABLE`, `FACTORY_PATH_CHOICES`,
    `DEFAULT_FACTORY_PATH`, `_normalize_factory_path()`.
  - Delete the `FACTORY_PATH: str = _normalize_factory_path(...)` instance
    attr.
  - Delete the refresh hook in `refresh_runtime_config_from_env`.
- `.env.example` — drop the `REACHY_MINI_FACTORY_PATH=composable` block.
- `tests/test_config_factory_path.py` — delete (the dial is gone).
- `tests/test_handler_factory_factory_path.py` — delete (the dial is
  gone; rewritten coverage lives in the per-triple tests below).
- `tests/test_handler_factory.py`:
  - Drop the autouse `_pin_factory_path_legacy` fixture.
  - Update each moonshine combo test to assert
    `isinstance(result, ComposableConversationHandler)` and that
    `result._tts_handler` is an instance of the expected legacy
    *ResponseHandler base.
- `tests/test_handler_factory_llama_llm.py` — same.
- `tests/test_handler_factory_gemini_llm.py` — same.
- `tests/test_handler_factory_pipeline_mode.py` — drop the autouse pin;
  composable assertions land here too.

**Acceptance**: factory + config + per-triple tests pass; `git grep
"FACTORY_PATH\|REACHY_MINI_FACTORY_PATH"` is empty in `src/`. Legacy
`LocalSTT*Handler` classes still exist (deleted in Task 3).

### Task 3 — Delete the legacy concrete classes

- `src/robot_comic/llama_elevenlabs_tts.py` — delete
  `LocalSTTLlamaElevenLabsHandler`. Drop unused `LocalSTTInputMixin`
  import if the rest of the file no longer uses it.
- `src/robot_comic/chatterbox_tts.py` — delete `LocalSTTChatterboxHandler`.
- `src/robot_comic/elevenlabs_tts.py` — delete
  `LocalSTTGeminiElevenLabsHandler` + alias `LocalSTTElevenLabsHandler`.
- `src/robot_comic/gemini_tts.py` — delete `LocalSTTGeminiTTSHandler`.
- `src/robot_comic/gemini_text_handlers.py` — delete
  `GeminiTextChatterboxHandler` and `GeminiTextElevenLabsHandler`. Keep
  the `*ResponseHandler` diamond bases (they are required by the
  composable Gemini-text builders).
- `src/robot_comic/llama_gemini_tts.py` — delete
  `LocalSTTLlamaGeminiTTSHandler` (orphan). Keep
  `LlamaGeminiTTSResponseHandler`.

Each deletion is verified by running the impacted test file *after* its
import is rewired in the same commit (test sites enumerated below):

- `tests/test_llama_base.py` (chatterbox handler → response handler).
- `tests/test_llama_streaming.py` (same).
- `tests/test_llama_health_check.py` (same).
- `tests/test_llm_warmup.py` (same).
- `tests/test_history_trim.py` (same, 3 sites).
- `tests/test_echo_suppression.py` (same, 2 sites).
- `tests/test_startup_opener.py` (same).
- `tests/test_llama_gemini_tts.py`
  (`LocalSTTLlamaGeminiTTSHandler` → `LlamaGeminiTTSResponseHandler`).
- `tests/integration/test_chatterbox_smoke.py` (compose mixin inline).
- `tests/integration/test_elevenlabs_smoke.py` (same).
- `tests/integration/test_gemini_text_smoke.py` (same).
- `tests/integration/test_llama_elevenlabs_smoke.py` (same).
- `tests/integration/test_handler_factory_smoke.py` (expected-class-name
  strings update to `ComposableConversationHandler`).

Stale references in docstrings (adapters, `backends.py`,
`composable_conversation_handler.py`, `gemini_text_base.py`) are
updated to mention the surviving classes.

**Acceptance**: `git grep "LocalSTTLlamaElevenLabsHandler\|LocalSTTChatterboxHandler\|LocalSTTGeminiElevenLabsHandler\|LocalSTTGeminiTTSHandler\|GeminiTextChatterboxHandler\|GeminiTextElevenLabsHandler\|LocalSTTLlamaGeminiTTSHandler\|LocalSTTElevenLabsHandler"` returns nothing under `src/`. Full pytest green.

### Task 4 — `PIPELINE_REFACTOR.md` status update

Mark Sub-phase 4e ✅ Done with `#TBD` for the PR number (manager fixes on
merge).

## Verification (final)

- `git grep "LocalSTTLlamaElevenLabsHandler\|LocalSTTChatterboxHandler\|LocalSTTGeminiElevenLabsHandler\|LocalSTTGeminiTTSHandler\|GeminiTextChatterboxHandler\|GeminiTextElevenLabsHandler\|LocalSTTLlamaGeminiTTSHandler\|LocalSTTElevenLabsHandler" src/` → empty
- `git grep "FACTORY_PATH\|REACHY_MINI_FACTORY_PATH" src/` → empty
- `uvx ruff@0.12.0 check` → green
- `uvx ruff@0.12.0 format --check` → green
- `.venv/Scripts/mypy --pretty src/robot_comic/handler_factory.py src/robot_comic/config.py` → green
- `.venv/Scripts/pytest tests/ -q --ignore=tests/vision/test_local_vision.py` → green
