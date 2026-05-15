# Phase 4c.5 Implementation Plan — GeminiTTSAdapter + GeminiBundledLLMAdapter + factory routing

**Goal:** Ship `GeminiTTSAdapter` and `GeminiBundledLLMAdapter` and route `(moonshine, gemini_tts)` through `ComposableConversationHandler` when `REACHY_MINI_FACTORY_PATH=composable`. Default `legacy` is unchanged.

**Architecture:** Two new adapter files. One new factory helper. One new branch inside the `(moonshine, gemini_tts)` block. Eleven `GeminiTTSAdapter` unit tests + nine `GeminiBundledLLMAdapter` unit tests + five factory-dispatch tests.

**Tech Stack:** Python 3.12, pytest-asyncio, `unittest.mock`, existing `_fake_cls` patching style from 4b/4c.*.

**Spec:** `docs/superpowers/specs/2026-05-15-phase-4c5-gemini-tts-adapter.md`

---

## File Map

| File | Role |
|------|------|
| `src/robot_comic/adapters/gemini_tts_adapter.py` | NEW — TTS adapter + `_GeminiTTSCompatibleHandler` Protocol |
| `src/robot_comic/adapters/gemini_bundled_llm_adapter.py` | NEW — LLM adapter wrapping `_run_llm_with_tools` |
| `src/robot_comic/adapters/__init__.py` | EDIT — export both new adapters |
| `src/robot_comic/handler_factory.py` | EDIT — composable gate in the gemini_tts block + new `_build_composable_gemini_tts` helper |
| `tests/adapters/test_gemini_tts_adapter.py` | NEW — adapter unit tests |
| `tests/adapters/test_gemini_bundled_llm_adapter.py` | NEW — adapter unit tests |
| `tests/test_handler_factory_factory_path.py` | EDIT — new Phase 4c.5 dispatch tests |
| `PIPELINE_REFACTOR.md` | EDIT — status table: 4c.5 ✅, 4c umbrella ✅ |

---

## TDD task breakdown

Five tasks, each one failing test set → minimum implementation → green → commit.

### Task 1 — `GeminiTTSAdapter`

**Files:**
- NEW `src/robot_comic/adapters/gemini_tts_adapter.py`
- NEW `tests/adapters/test_gemini_tts_adapter.py`
- EDIT `src/robot_comic/adapters/__init__.py`

- [ ] **Step 1: Write the failing tests** (eleven tests per spec test plan):
  - `test_prepare_calls_handler_prepare`
  - `test_synthesize_yields_audio_frames_for_one_sentence`
  - `test_synthesize_yields_audio_frames_for_multiple_sentences`
  - `test_synthesize_strips_gemini_tags_from_spoken_text`
  - `test_synthesize_inserts_silence_for_short_pause_tag`
  - `test_synthesize_forwards_delivery_tags_to_system_instruction`
  - `test_synthesize_skips_sentence_when_tts_returns_none`
  - `test_synthesize_with_empty_text_yields_nothing`
  - `test_synthesize_ignores_protocol_tags_arg`
  - `test_shutdown_is_noop`
  - `test_adapter_satisfies_tts_backend_protocol`

  Stub handler exposes `_prepare_startup_credentials`, `_call_tts_with_retry`, and uses the **real** `GeminiTTSResponseHandler._pcm_to_frames` static (it's a pure numpy splitter; importing it doesn't hit the network).

  Run: `.venv/Scripts/python -m pytest tests/adapters/test_gemini_tts_adapter.py -q`. Expect `ImportError`.

- [ ] **Step 2: Implement the adapter**:
  - Define `_GeminiTTSCompatibleHandler` Protocol (`output_queue`, `_conversation_history`, `_client`, `_prepare_startup_credentials`, `_run_llm_with_tools`, `_call_tts_with_retry`).
  - `GeminiTTSAdapter.synthesize` replicates the per-sentence loop from `gemini_tts.py:396-418`: `split_sentences` → `strip_gemini_tags` → `extract_delivery_tags` → optional silence for `[short pause]` → `_call_tts_with_retry(spoken, system_instruction=…)` → `_pcm_to_frames` → yield `AudioFrame`.
  - `shutdown` is a no-op.
  - Export from `src/robot_comic/adapters/__init__.py`.

  Run: green.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/adapters/gemini_tts_adapter.py \
        src/robot_comic/adapters/__init__.py \
        tests/adapters/test_gemini_tts_adapter.py
git commit -m "feat(adapters): Phase 4c.5 — GeminiTTSAdapter wraps _call_tts_with_retry (#337)"
```

---

### Task 2 — `GeminiBundledLLMAdapter`

**Files:**
- NEW `src/robot_comic/adapters/gemini_bundled_llm_adapter.py`
- NEW `tests/adapters/test_gemini_bundled_llm_adapter.py`
- EDIT `src/robot_comic/adapters/__init__.py`

- [ ] **Step 1: Write the failing tests** (nine tests per spec):
  - `test_prepare_calls_handler_prepare`
  - `test_chat_returns_llmresponse_with_text_and_no_tool_calls`
  - `test_chat_swaps_history_for_duration_of_call`
  - `test_chat_restores_history_on_exception`
  - `test_chat_converts_orchestrator_messages_to_gemini_shape`
  - `test_chat_ignores_tools_arg`
  - `test_chat_empty_history_passes_empty_list`
  - `test_shutdown_is_noop`
  - `test_adapter_satisfies_llm_backend_protocol`

  Stub handler records its `_conversation_history` snapshot at the moment `_run_llm_with_tools` runs and returns a canned string.

  Run: expect ImportError.

- [ ] **Step 2: Implement the adapter**:
  - `chat` swaps `self._handler._conversation_history` with `_orchestrator_messages_to_gemini(messages)` for the call, restores on `finally`.
  - Returns `LLMResponse(text=..., tool_calls=())`.
  - `prepare` calls `_prepare_startup_credentials`. `shutdown` no-op.
  - Helper `_orchestrator_messages_to_gemini(messages)` translates per the spec table (user→user, assistant→model, system→user-primer, tool→skip).
  - Export from `__init__.py`.

  Run: green.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/adapters/gemini_bundled_llm_adapter.py \
        src/robot_comic/adapters/__init__.py \
        tests/adapters/test_gemini_bundled_llm_adapter.py
git commit -m "feat(adapters): Phase 4c.5 — GeminiBundledLLMAdapter wraps _run_llm_with_tools (#337)"
```

---

### Task 3 — Factory legacy-path regression guard for `(moonshine, gemini_tts)`

The point: pin that `FACTORY_PATH=legacy` (default) keeps `LocalSTTGeminiTTSHandler`. Should already pass on green main because the factory ignores `FACTORY_PATH` for this triple today — but having it as an explicit test prevents regressions later.

- [ ] **Step 1: Write the test**

```python
def test_legacy_path_returns_legacy_handler_for_gemini_tts(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_LEGACY)
    fake = _fake_cls("LocalSTTGeminiTTSHandler")
    with patch("robot_comic.gemini_tts.LocalSTTGeminiTTSHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE, AUDIO_OUTPUT_GEMINI_TTS, mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )
    assert isinstance(result, fake)
```

Run: green (already passes on main).

- [ ] **Step 2: No code changes needed.**

- [ ] **Step 3: Commit**

```
git add tests/test_handler_factory_factory_path.py
git commit -m "test(factory): Phase 4c.5 — pin legacy path for (moonshine, gemini_tts) (#337)"
```

---

### Task 4 — Composable-path factory wiring + dispatch tests

**Files:**
- EDIT `src/robot_comic/handler_factory.py` — composable gate + helper.
- EDIT `tests/test_handler_factory_factory_path.py` — four new composable-path tests.

- [ ] **Step 1: Write the four failing tests**:
  - `test_composable_path_returns_wrapper_for_gemini_tts`
  - `test_composable_path_wires_three_adapters_for_gemini_tts`
  - `test_composable_path_seeds_system_prompt_for_gemini_tts`
  - `test_composable_path_copy_constructs_fresh_legacy_for_gemini_tts`

  Pattern from 4c.4 section. Patch `robot_comic.gemini_tts.LocalSTTGeminiTTSHandler` with `_fake_cls`. Set `FACTORY_PATH=FACTORY_PATH_COMPOSABLE`.

  Run: expect failures — factory still returns the fake legacy directly.

- [ ] **Step 2: Implement the gate + helper**:
  - In `handler_factory.py`, the `(moonshine, gemini_tts)` block becomes:
    ```python
    if output_backend == AUDIO_OUTPUT_GEMINI_TTS:
        if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
            logger.info(...)
            return _build_composable_gemini_tts(**handler_kwargs)
        from robot_comic.gemini_tts import LocalSTTGeminiTTSHandler
        ...
    ```
  - Add `_build_composable_gemini_tts(**handler_kwargs)` at the bottom of `handler_factory.py`, mirroring `_build_composable_gemini_elevenlabs`. Uses `MoonshineSTTAdapter`, `GeminiBundledLLMAdapter`, `GeminiTTSAdapter` from `robot_comic.adapters`; constructs one `LocalSTTGeminiTTSHandler(**handler_kwargs)` and wraps it three times.

  Run: green.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/handler_factory.py tests/test_handler_factory_factory_path.py
git commit -m "feat(factory): Phase 4c.5 — composable routing for (moonshine, gemini_tts) (#337)"
```

---

### Task 5 — PIPELINE_REFACTOR.md status flip

- [ ] **Step 1: Edit `PIPELINE_REFACTOR.md`** to flip:
  - Row 4c.5 → ✅ Done with PR placeholder (manager fills in PR number on merge).
  - Row 4c umbrella → ✅ Done (5/5 triples).

- [ ] **Step 2: Commit**

```
git add PIPELINE_REFACTOR.md
git commit -m "docs(pipeline-refactor): mark 4c.5 + 4c umbrella done (#337)"
```

---

## Final verification (from repo root)

```
uvx ruff@0.12.0 check
uvx ruff@0.12.0 format --check
.venv/Scripts/python -m mypy --pretty src/robot_comic/adapters/gemini_tts_adapter.py src/robot_comic/adapters/gemini_bundled_llm_adapter.py src/robot_comic/handler_factory.py
.venv/Scripts/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

All four must be green before push. If ruff or mypy or pytest fails locally, fix and amend the relevant commit (do NOT skip).

## Push + return

```
git push -u origin claude/phase-4c5-gemini-tts-adapter
```

Then return the structured summary per the briefing's return contract.
