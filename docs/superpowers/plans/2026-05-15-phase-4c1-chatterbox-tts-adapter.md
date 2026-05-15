# Phase 4c.1 Implementation Plan — ChatterboxTTSAdapter + factory routing

**Goal:** Ship `ChatterboxTTSAdapter` and route `(moonshine, chatterbox, llama)` through `ComposableConversationHandler` when `REACHY_MINI_FACTORY_PATH=composable`. Default `legacy` is unchanged.

**Architecture:** One new adapter file. One new factory helper. One new branch inside the existing llama-LLM gate. Twelve adapter unit tests + five factory-dispatch tests.

**Tech Stack:** Python 3.12, pytest-asyncio, `unittest.mock`, existing `_fake_cls` patching style from 4b.

**Spec:** `docs/superpowers/specs/2026-05-15-phase-4c1-chatterbox-tts-adapter.md`

---

## File Map

| File | Role |
|------|------|
| `src/robot_comic/adapters/chatterbox_tts_adapter.py` | NEW — mirrors `elevenlabs_tts_adapter.py` |
| `src/robot_comic/adapters/__init__.py` | EDIT — export `ChatterboxTTSAdapter` |
| `src/robot_comic/handler_factory.py` | EDIT — new llama+chatterbox composable gate + `_build_composable_llama_chatterbox` helper |
| `tests/adapters/test_chatterbox_tts_adapter.py` | NEW — adapter unit tests (mirrors elevenlabs adapter tests) |
| `tests/test_handler_factory_factory_path.py` | EDIT — new chatterbox dispatch tests; shrink the "other triples unchanged" parametrisation |

---

## TDD task breakdown

Four tasks, each one failing test set → minimum implementation → green → commit.

### Task 1 — `ChatterboxTTSAdapter` happy-path + queue isolation

**Files:**
- New: `src/robot_comic/adapters/chatterbox_tts_adapter.py`
- New: `tests/adapters/test_chatterbox_tts_adapter.py`
- Edit: `src/robot_comic/adapters/__init__.py`

- [ ] **Step 1: Write the failing tests** (twelve tests in one file, all the cases from the spec).

Run: `.venv/Scripts/python -m pytest tests/adapters/test_chatterbox_tts_adapter.py -q`

Expected: `ImportError` / `ModuleNotFoundError` — the adapter doesn't exist yet.

- [ ] **Step 2: Implement the adapter** by copying the elevenlabs adapter and tweaking:
  - Replace `_stream_tts_to_queue(text, tags=tags_list)` with `_synthesize_and_enqueue(text)` (no tags, no first_audio_marker).
  - Add `tags` to the `synthesize` signature for Protocol compliance, but drop them silently.
  - Add the `AdditionalOutputs`-drop branch in the consume loop (only yield 2-tuples).
  - Update module docstring to describe the chatterbox-specific surface.
  - Export the class from `src/robot_comic/adapters/__init__.py`.

Run: green.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/adapters/chatterbox_tts_adapter.py \
        src/robot_comic/adapters/__init__.py \
        tests/adapters/test_chatterbox_tts_adapter.py
git commit -m "feat(adapters): Phase 4c.1 — ChatterboxTTSAdapter wraps _synthesize_and_enqueue (#337)"
```

---

### Task 2 — Factory legacy-path regression guard for `(moonshine, chatterbox, llama)`

The point of this task is to pin that `FACTORY_PATH=legacy` keeps the existing `LocalSTTChatterboxHandler` for the chatterbox triple. Should already pass on green main because the factory ignores `FACTORY_PATH` for chatterbox today.

- [ ] **Step 1: Write the test**

Add to `tests/test_handler_factory_factory_path.py`:

```python
def test_legacy_path_returns_legacy_handler_for_llama_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``FACTORY_PATH=legacy`` (default) keeps today's chatterbox handler."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_LEGACY)

    fake = _fake_cls("LocalSTTChatterboxHandler")
    with patch("robot_comic.chatterbox_tts.LocalSTTChatterboxHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake)
```

Run: green (regression-guard test for today's behaviour).

- [ ] **Step 2: No implementation needed.**

- [ ] **Step 3: Commit**

```
git add tests/test_handler_factory_factory_path.py
git commit -m "test(factory): pin legacy-path llama+chatterbox dispatch (#337)"
```

---

### Task 3 — Composable branch for `(moonshine, chatterbox, llama)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_handler_factory_factory_path.py`:

```python
def test_composable_path_returns_wrapper_for_llama_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("LocalSTTChatterboxHandler")
    with patch("robot_comic.chatterbox_tts.LocalSTTChatterboxHandler", fake_legacy):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, ComposableConversationHandler)
    assert isinstance(result.pipeline, ComposablePipeline)
    assert isinstance(result._tts_handler, fake_legacy)


def test_composable_path_wires_three_adapters_for_llama_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod
    from robot_comic.adapters import (
        LlamaLLMAdapter, MoonshineSTTAdapter, ChatterboxTTSAdapter,
    )

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("LocalSTTChatterboxHandler")
    with patch("robot_comic.chatterbox_tts.LocalSTTChatterboxHandler", fake_legacy):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_CHATTERBOX,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    pipe = result.pipeline
    assert isinstance(pipe.stt, MoonshineSTTAdapter)
    assert isinstance(pipe.llm, LlamaLLMAdapter)
    assert isinstance(pipe.tts, ChatterboxTTSAdapter)
    assert pipe.stt._handler is pipe.llm._handler
    assert pipe.llm._handler is pipe.tts._handler
    assert pipe.stt._handler is result._tts_handler


def test_composable_path_seeds_system_prompt_for_llama_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)
    monkeypatch.setattr(
        "robot_comic.prompts.get_session_instructions",
        lambda: "TEST INSTRUCTIONS",
    )

    fake_legacy = _fake_cls("LocalSTTChatterboxHandler")
    with patch("robot_comic.chatterbox_tts.LocalSTTChatterboxHandler", fake_legacy):
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


def test_composable_path_copy_constructs_fresh_legacy_for_chatterbox(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("LocalSTTChatterboxHandler")
    with patch("robot_comic.chatterbox_tts.LocalSTTChatterboxHandler", fake_legacy):
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

Run: all four fail — the factory still returns `LocalSTTChatterboxHandler`.

- [ ] **Step 2: Implement the composable branch** in `handler_factory.py`:

In the `LLM_BACKEND_LLAMA` arm (after the elevenlabs `if output_backend == AUDIO_OUTPUT_ELEVENLABS:` block), add:

```python
if output_backend == AUDIO_OUTPUT_CHATTERBOX:
    if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
        logger.info(
            "HandlerFactory: selecting ComposableConversationHandler "
            "(%s → %s, llm=%s, factory_path=composable)",
            input_backend, output_backend, LLM_BACKEND_LLAMA,
        )
        return _build_composable_llama_chatterbox(**handler_kwargs)
    # Else: fall through to the legacy LocalSTTChatterboxHandler branch
    # at the bottom of the moonshine-input block.
```

Add the helper at module bottom (next to `_build_composable_llama_elevenlabs`):

```python
def _build_composable_llama_chatterbox(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, chatterbox, llama) pipeline.

    Builds a legacy ``LocalSTTChatterboxHandler`` (the adapters delegate
    into it), wraps it with the three Phase 3 adapters, composes them into
    a ``ComposablePipeline`` seeded with the current session instructions,
    and returns a ``ComposableConversationHandler`` whose ``build`` closure
    re-runs the same construction. FastRTC's ``copy()`` per-peer cloning
    invokes the closure for fresh state on each new peer.
    """
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        LlamaLLMAdapter,
        MoonshineSTTAdapter,
        ChatterboxTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.chatterbox_tts import LocalSTTChatterboxHandler
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        legacy = LocalSTTChatterboxHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = LlamaLLMAdapter(legacy)
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

Run: the four new tests pass; Task 2 legacy-guard still passes.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/handler_factory.py tests/test_handler_factory_factory_path.py
git commit -m "feat(factory): wire composable path for (moonshine, chatterbox, llama) (#337)"
```

---

### Task 4 — Shrink the "other triples unchanged" parametrisation

The existing `test_composable_path_only_affects_llama_elevenlabs` parametrisation lists `AUDIO_OUTPUT_CHATTERBOX` as one of the "still goes through legacy" rows. Now that chatterbox is migrated, that row is wrong — drop it and rename the test to reflect the broadened scope.

- [ ] **Step 1: Update the test**

Rename `test_composable_path_only_affects_llama_elevenlabs` → `test_composable_path_other_triples_remain_legacy` and remove the `AUDIO_OUTPUT_CHATTERBOX` row from the parameter list (leaving gemini_tts, openai_realtime, hf).

Run: green.

- [ ] **Step 2: No new implementation.**

- [ ] **Step 3: Commit**

```
git add tests/test_handler_factory_factory_path.py
git commit -m "test(factory): drop chatterbox from 'legacy-only' parametrisation post-4c.1 (#337)"
```

---

## Pre-push checklist

```
# From repo root, NEVER per-file:
.venv/Scripts/ruff check
.venv/Scripts/ruff format --check
.venv/Scripts/mypy --pretty src/robot_comic/adapters/chatterbox_tts_adapter.py \
                            src/robot_comic/handler_factory.py
.venv/Scripts/python -m pytest tests/ -q
```

If any step is red, fix locally before pushing.

Sanity-check main's CI is green on the current tip commit before opening the PR.

## PR creation

After all four commits land on `claude/phase-4c1-chatterbox-tts-adapter`:

```
git push -u origin claude/phase-4c1-chatterbox-tts-adapter
```

Then the manager opens the PR with the body summarising the spec.

## Risks

Covered in the spec under "Risks" — `AdditionalOutputs` error-sentinel drop, voice-clone reference path staleness, `_warmup_tts` running at `prepare()`. None are blocking.

## After-merge follow-ups (out of scope for 4c.1)

- 4c.2: `GeminiLLMAdapter` + `(moonshine, chatterbox, gemini)`.
- 4c.3–5: remaining triples.
- Lifecycle hooks (telemetry, boot-timeline, joke history, history trim, echo-guard) — per-hook PRs.
