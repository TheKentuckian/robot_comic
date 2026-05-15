# Phase 4c.2 Implementation Plan — GeminiLLMAdapter + factory routing

**Goal:** Ship `GeminiLLMAdapter` and route `(moonshine, chatterbox, gemini)` through `ComposableConversationHandler` when `REACHY_MINI_FACTORY_PATH=composable`. Default `legacy` is unchanged.

**Architecture:** One new adapter file. One new factory helper. One new composable gate inside the existing gemini-LLM + chatterbox-output arm. Twelve adapter unit tests + five factory-dispatch tests.

**Tech Stack:** Python 3.12, pytest-asyncio, `unittest.mock`, existing `_fake_cls` patching style from 4b/4c.1.

**Spec:** `docs/superpowers/specs/2026-05-15-phase-4c2-gemini-llm-adapter.md`

---

## File Map

| File | Role |
|------|------|
| `src/robot_comic/adapters/gemini_llm_adapter.py` | NEW — mirrors `llama_llm_adapter.py` |
| `src/robot_comic/adapters/__init__.py` | EDIT — export `GeminiLLMAdapter` |
| `src/robot_comic/handler_factory.py` | EDIT — new composable gate inside the gemini+chatterbox arm + `_build_composable_gemini_chatterbox` helper |
| `tests/adapters/test_gemini_llm_adapter.py` | NEW — adapter unit tests (mirrors llama adapter tests) |
| `tests/test_handler_factory_factory_path.py` | EDIT — new gemini+chatterbox dispatch tests |

---

## TDD task breakdown

Three tasks, each one failing test set → minimum implementation → green → commit.

### Task 1 — `GeminiLLMAdapter` happy-path + tool-call conversion + history swap

**Files:**
- New: `src/robot_comic/adapters/gemini_llm_adapter.py`
- New: `tests/adapters/test_gemini_llm_adapter.py`
- Edit: `src/robot_comic/adapters/__init__.py`

- [ ] **Step 1: Write the failing tests** (twelve tests in one file, mirroring `test_llama_llm_adapter.py` task-for-task).

Run: `.venv/Scripts/python -m pytest tests/adapters/test_gemini_llm_adapter.py -q`

Expected: `ImportError` / `ModuleNotFoundError` — the adapter doesn't exist yet.

- [ ] **Step 2: Implement the adapter** by copying `llama_llm_adapter.py` and tweaking:
  - Rename class `LlamaLLMAdapter` → `GeminiLLMAdapter`.
  - Update `TYPE_CHECKING` import: `from robot_comic.gemini_text_base import GeminiTextResponseHandler`.
  - Update the constructor annotation: `handler: "GeminiTextResponseHandler"`.
  - Update the module docstring to describe the Gemini-specific surface (still wraps `_call_llm`; tool-call shape already converted to llama-server shape inside `gemini_llm.py::call_completion` — converter is byte-identical to llama's; document this).
  - Update log message identifiers to read `GeminiLLMAdapter` instead of `LlamaLLMAdapter`.
  - Keep `_convert_tool_call` as a module-level helper (avoid cross-adapter import).
  - Export the class from `src/robot_comic/adapters/__init__.py` and add it to `__all__` in alphabetical order.

Run: green.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/adapters/gemini_llm_adapter.py \
        src/robot_comic/adapters/__init__.py \
        tests/adapters/test_gemini_llm_adapter.py
git commit -m "feat(adapters): Phase 4c.2 — GeminiLLMAdapter wraps _call_llm (#337)"
```

---

### Task 2 — Factory legacy-path regression guard for `(moonshine, chatterbox, gemini)`

The point of this task is to pin that `FACTORY_PATH=legacy` keeps the existing `GeminiTextChatterboxHandler` for the gemini-chatterbox triple. Should already pass on green main because the factory's existing `LLM_BACKEND_GEMINI` + `AUDIO_OUTPUT_CHATTERBOX` branch returns `GeminiTextChatterboxHandler` unconditionally.

- [ ] **Step 1: Write the test**

Add to `tests/test_handler_factory_factory_path.py`:

```python
def test_legacy_path_returns_legacy_handler_for_gemini_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``FACTORY_PATH=legacy`` (default) keeps today's GeminiTextChatterboxHandler."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_LEGACY)

    fake = _fake_cls("GeminiTextChatterboxHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextChatterboxHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake)
```

Run: green (regression-guard for today's behaviour).

- [ ] **Step 2: No implementation needed.**

- [ ] **Step 3: Commit**

```
git add tests/test_handler_factory_factory_path.py
git commit -m "test(factory): pin legacy-path gemini+chatterbox dispatch (#337)"
```

---

### Task 3 — Composable branch for `(moonshine, chatterbox, gemini)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_handler_factory_factory_path.py`:

```python
def test_composable_path_returns_wrapper_for_gemini_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("GeminiTextChatterboxHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextChatterboxHandler", fake_legacy):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, ComposableConversationHandler)
    assert isinstance(result.pipeline, ComposablePipeline)
    assert isinstance(result._tts_handler, fake_legacy)


def test_composable_path_wires_three_adapters_for_gemini_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod
    from robot_comic.adapters import (
        GeminiLLMAdapter, MoonshineSTTAdapter, ChatterboxTTSAdapter,
    )

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("GeminiTextChatterboxHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextChatterboxHandler", fake_legacy):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    pipe = result.pipeline
    assert isinstance(pipe.stt, MoonshineSTTAdapter)
    assert isinstance(pipe.llm, GeminiLLMAdapter)
    assert isinstance(pipe.tts, ChatterboxTTSAdapter)
    assert pipe.stt._handler is pipe.llm._handler
    assert pipe.llm._handler is pipe.tts._handler
    assert pipe.stt._handler is result._tts_handler


def test_composable_path_seeds_system_prompt_for_gemini_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)
    monkeypatch.setattr(
        "robot_comic.prompts.get_session_instructions",
        lambda: "TEST INSTRUCTIONS",
    )

    fake_legacy = _fake_cls("GeminiTextChatterboxHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextChatterboxHandler", fake_legacy):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert result.pipeline._conversation_history[0] == {
        "role": "system",
        "content": "TEST INSTRUCTIONS",
    }


def test_composable_path_copy_constructs_fresh_legacy_for_gemini_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("GeminiTextChatterboxHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextChatterboxHandler", fake_legacy):
        original = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )
        copy = original.copy()

    assert copy is not original
    assert copy._tts_handler is not original._tts_handler
    assert copy.pipeline is not original.pipeline
```

Run: all four fail — the factory still returns `GeminiTextChatterboxHandler` for both legacy and composable paths.

- [ ] **Step 2: Implement the composable branch** in `handler_factory.py`:

Inside the `LLM_BACKEND_GEMINI` arm (around line 229), at the start of the `if output_backend == AUDIO_OUTPUT_CHATTERBOX:` block, prepend the composable check:

```python
if output_backend == AUDIO_OUTPUT_CHATTERBOX:
    if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
        logger.info(
            "HandlerFactory: selecting ComposableConversationHandler "
            "(%s → %s, llm=%s, factory_path=composable)",
            input_backend, output_backend, LLM_BACKEND_GEMINI,
        )
        return _build_composable_gemini_chatterbox(**handler_kwargs)
    from robot_comic.gemini_text_handlers import GeminiTextChatterboxHandler

    logger.info(
        "HandlerFactory: selecting GeminiTextChatterboxHandler (%s → %s, llm=%s)",
        input_backend, output_backend, LLM_BACKEND_GEMINI,
    )
    return GeminiTextChatterboxHandler(**handler_kwargs)
```

Add the helper at the bottom of the module next to `_build_composable_llama_chatterbox`:

```python
def _build_composable_gemini_chatterbox(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, chatterbox, gemini) pipeline.

    Builds a legacy ``GeminiTextChatterboxHandler`` (the adapters delegate
    into it), wraps it with the three Phase 3/4 adapters, composes them into
    a ``ComposablePipeline`` seeded with the current session instructions,
    and returns a ``ComposableConversationHandler`` whose ``build`` closure
    re-runs the same construction. FastRTC's ``copy()`` per-peer cloning
    invokes the closure for fresh state on each new peer.
    """
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        GeminiLLMAdapter,
        MoonshineSTTAdapter,
        ChatterboxTTSAdapter,
    )
    from robot_comic.gemini_text_handlers import GeminiTextChatterboxHandler
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        legacy = GeminiTextChatterboxHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = GeminiLLMAdapter(legacy)
        tts = ChatterboxTTSAdapter(legacy)
        pipeline = ComposablePipeline(
            stt, llm, tts,
            system_prompt=get_session_instructions(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=legacy,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()
```

Run: the four new tests pass; Task 2 legacy-guard still passes; the existing `test_composable_path_with_gemini_llm_unchanged` (elevenlabs+gemini) still passes — it patches `GeminiTextElevenLabsHandler` for the elevenlabs output, which 4c.2 does not touch.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/handler_factory.py tests/test_handler_factory_factory_path.py
git commit -m "feat(factory): wire composable path for (moonshine, chatterbox, gemini) (#337)"
```

---

## Pre-push checklist (from worktree / repo root)

```
uvx ruff@0.12.0 check
uvx ruff@0.12.0 format --check
.venv/Scripts/mypy --pretty src/robot_comic/adapters/gemini_llm_adapter.py \
                            src/robot_comic/handler_factory.py
.venv/Scripts/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

If any step is red, fix locally before pushing.

Sanity-check main's CI is green on the current tip commit before opening the PR.

## PR creation

After all three commits land on `claude/phase-4c2-gemini-llm-adapter`:

```
git push -u origin claude/phase-4c2-gemini-llm-adapter
```

Then the manager opens the PR with the body summarising the spec.

## Risks

Covered in the spec under "Risks" — duplicate `_convert_tool_call` helper (consolidate in 4e), system-prompt double-seed (examined and ruled out), diamond MRO (resolves correctly via legacy handler's explicit override). None are blocking.

## After-merge follow-ups (out of scope for 4c.2)

- 4c.3: `(moonshine, elevenlabs, gemini)` — reuse `GeminiLLMAdapter` + `ElevenLabsTTSAdapter`.
- 4c.4–5: remaining triples.
- Lifecycle hooks: per-PR rollout per the operating manual.
