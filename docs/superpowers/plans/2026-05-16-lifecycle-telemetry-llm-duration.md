# TDD plan — Lifecycle Hook #2 — `telemetry.record_llm_duration`

**Branch:** `claude/lifecycle-telemetry-llm-duration`
**Spec:** `docs/superpowers/specs/2026-05-16-lifecycle-telemetry-llm-duration.md`
**Date:** 2026-05-16

## Sequence

One commit per task. Each task is a failing test → minimum patch → green
commit. Order chosen so the smallest blast-radius adapter (Llama) lands
first; the gemini-text and bundled-gemini adapters follow with the same
pattern.

## Task 1 — `LlamaLLMAdapter` records duration on success path

**RED.** Add `test_chat_records_llm_duration_on_success` to
`tests/adapters/test_llama_llm_adapter.py`. Monkeypatch
`robot_comic.adapters.llama_llm_adapter.telemetry.record_llm_duration`
with a recorder lambda; call `adapter.chat(...)`; assert one record with
`{"gen_ai.system": "llama_cpp", "gen_ai.operation.name": "chat"}` and a
positive duration.

**GREEN.** In `LlamaLLMAdapter.chat()`:
1. Add `import time` and `from robot_comic import telemetry` at module top.
2. Wrap the `_call_llm` call in `t = time.perf_counter()` /
   `try: ... finally: telemetry.record_llm_duration(time.perf_counter() - t,
   {"gen_ai.system": "llama_cpp", "gen_ai.operation.name": "chat"})`.

**Commit:** `feat(adapters): record llm duration on LlamaLLMAdapter.chat (#337)`

## Task 2 — `LlamaLLMAdapter` records duration on exception path

**RED.** Add `test_chat_records_llm_duration_on_exception`. Use the
exception-raising stub already in the test file; assert the recorder was
called exactly once and the exception still propagates.

**GREEN.** Existing `try / finally` from Task 1 already covers this. If
the test fails it means the patch landed wrong; otherwise commit a
test-only delta.

**Commit:** `test(adapters): cover llm duration on LlamaLLMAdapter.chat exception (#337)`

## Task 3 — `GeminiLLMAdapter` records duration on success path

**RED.** Add `test_chat_records_llm_duration_on_success` to
`tests/adapters/test_gemini_llm_adapter.py`. Same pattern as Task 1.
`gen_ai.system="gemini"`.

**GREEN.** Same `time.perf_counter()` / `try / finally` shape in
`GeminiLLMAdapter.chat()`.

**Commit:** `feat(adapters): record llm duration on GeminiLLMAdapter.chat (#337)`

## Task 4 — `GeminiLLMAdapter` records duration on exception path

**RED.** Add `test_chat_records_llm_duration_on_exception`. Assert one
record on exception path; exception propagates.

**GREEN.** Existing `finally` from Task 3 covers it; test-only commit.

**Commit:** `test(adapters): cover llm duration on GeminiLLMAdapter.chat exception (#337)`

## Task 5 — `GeminiBundledLLMAdapter` records duration on success path

**RED.** Add `test_chat_records_llm_duration_on_success` to
`tests/adapters/test_gemini_bundled_llm_adapter.py`. Same pattern; this
wraps `_run_llm_with_tools()` rather than `_call_llm`.
`gen_ai.system="gemini"`.

**GREEN.** Same shape inside `GeminiBundledLLMAdapter.chat()`.

**Commit:** `feat(adapters): record llm duration on GeminiBundledLLMAdapter.chat (#337)`

## Task 6 — `GeminiBundledLLMAdapter` records duration on exception path

**RED.** Same test pattern with the exception-raising stub already in
the test file.

**GREEN.** Existing `finally` covers; test-only commit.

**Commit:** `test(adapters): cover llm duration on GeminiBundledLLMAdapter.chat exception (#337)`

## Task 7 — verification

Run from the repo root:

```
uvx ruff@0.12.0 check
uvx ruff@0.12.0 format --check
.venv/bin/mypy --pretty src/robot_comic/adapters/llama_llm_adapter.py \
                       src/robot_comic/adapters/gemini_llm_adapter.py \
                       src/robot_comic/adapters/gemini_bundled_llm_adapter.py
.venv/bin/pytest tests/ -q
```

All four must be green. Push branch.
