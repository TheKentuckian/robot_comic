# Phase 4c.4 Implementation Plan — Composable routing for `(moonshine, elevenlabs, gemini-fallback)`

**Goal:** Route the gemini-fallback elevenlabs dispatch arm through `ComposableConversationHandler` under `FACTORY_PATH=composable`, reusing the existing 4c.3 helper. Default `legacy` unchanged.

**Architecture:** No new adapter. No new helper. Only a composable-path gate inside the outer `(moonshine, elevenlabs)` fallthrough arm of `handler_factory.py` that calls the existing `_build_composable_gemini_elevenlabs` from 4c.3. Five new factory-dispatch tests pin both branches.

**Tech Stack:** Python 3.12, pytest-asyncio, `unittest.mock`, existing `_fake_cls` patching style from 4b/4c.1/4c.2/4c.3.

**Spec:** `docs/superpowers/specs/2026-05-15-phase-4c4-gemini-fallback-elevenlabs-routing.md`

---

## File Map

| File | Role |
|------|------|
| `src/robot_comic/handler_factory.py` | EDIT — prepend composable gate inside the outer-`if input_backend == AUDIO_INPUT_MOONSHINE` arm's `output_backend == AUDIO_OUTPUT_ELEVENLABS` block. Reuses `_build_composable_gemini_elevenlabs` from 4c.3 unchanged. |
| `tests/test_handler_factory_factory_path.py` | EDIT — append Phase 4c.4 section with five new tests. |

---

## TDD task breakdown

Two tasks, each one failing test set → minimum implementation → green → commit.

### Task 1 — Legacy-path regression guard for the gemini-fallback dispatch

The point: pin `FACTORY_PATH=legacy` + `LLM_BACKEND` set to a non-default sentinel still returns the existing `LocalSTTGeminiElevenLabsHandler`. Should already pass on green main because the factory unconditionally returns `LocalSTTGeminiElevenLabsHandler` at line 314 today when `LLM_BACKEND` is neither `llama` nor `gemini`.

- [ ] **Step 1: Write the test** — append to `tests/test_handler_factory_factory_path.py` in a new "Phase 4c.4" section:

```python
# ---------------------------------------------------------------------------
# Phase 4c.4 — (moonshine, elevenlabs, gemini-fallback) composable path
# ---------------------------------------------------------------------------
#
# The "gemini-fallback" arm is the outer ``input_backend == AUDIO_INPUT_MOONSHINE``
# fallthrough in handler_factory.py (lines ~293-322). It is reached when
# LLM_BACKEND is neither "llama" nor "gemini" — i.e. a typo, an unrecognised
# value, or an empty string after .strip().lower(). In production this path
# is effectively unreachable because the default LLM_BACKEND is "llama"; the
# tests below use the sentinel "unknown" to deliberately trigger fallthrough.


def test_legacy_path_returns_legacy_handler_for_gemini_fallback_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``FACTORY_PATH=legacy`` + unknown LLM_BACKEND keeps LocalSTTGeminiElevenLabsHandler."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", "unknown")
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_LEGACY)

    fake = _fake_cls("LocalSTTGeminiElevenLabsHandler")
    with patch("robot_comic.elevenlabs_tts.LocalSTTGeminiElevenLabsHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake)
```

Run: `/d/Projects/robot_comic/.venv/Scripts/python.exe -m pytest tests/test_handler_factory_factory_path.py -q -k gemini_fallback`

Expected: green (regression-guard for today's behaviour).

- [ ] **Step 2: No implementation needed.**

- [ ] **Step 3: Commit**

```
git add tests/test_handler_factory_factory_path.py
git commit -m "test(factory): pin legacy-path gemini-fallback elevenlabs dispatch (#337)"
```

---

### Task 2 — Composable branch for the gemini-fallback arm

- [ ] **Step 1: Write the failing tests** — append four more tests to the Phase 4c.4 section in `tests/test_handler_factory_factory_path.py`:

```python
def test_composable_path_returns_wrapper_for_gemini_fallback_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``FACTORY_PATH=composable`` + gemini-fallback dispatch → ComposableConversationHandler.

    The composable path consolidates onto GeminiTextElevenLabsHandler (the
    _call_llm-capable host) so the GeminiLLMAdapter from 4c.2 can wrap it.
    LocalSTTGeminiElevenLabsHandler lacks _call_llm and is therefore not
    used in the composable path. Phase 4e will retire the legacy class.
    """
    from robot_comic import config as cfg_mod
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", "unknown")
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("GeminiTextElevenLabsHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextElevenLabsHandler", fake_legacy):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, ComposableConversationHandler)
    assert isinstance(result.pipeline, ComposablePipeline)
    assert isinstance(result._tts_handler, fake_legacy)


def test_composable_path_wires_three_adapters_for_gemini_fallback_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """All three adapters wrap the same single GeminiTextElevenLabsHandler instance."""
    from robot_comic import config as cfg_mod
    from robot_comic.adapters import (
        GeminiLLMAdapter,
        MoonshineSTTAdapter,
        ElevenLabsTTSAdapter,
    )

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", "unknown")
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("GeminiTextElevenLabsHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextElevenLabsHandler", fake_legacy):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    pipe = result.pipeline
    assert isinstance(pipe.stt, MoonshineSTTAdapter)
    assert isinstance(pipe.llm, GeminiLLMAdapter)
    assert isinstance(pipe.tts, ElevenLabsTTSAdapter)
    # All three adapters share the same legacy handler instance.
    assert pipe.stt._handler is pipe.llm._handler
    assert pipe.llm._handler is pipe.tts._handler
    assert pipe.stt._handler is result._tts_handler


def test_composable_path_seeds_system_prompt_for_gemini_fallback_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """The pipeline's system prompt is sourced from prompts.get_session_instructions."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", "unknown")
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)
    monkeypatch.setattr(
        "robot_comic.prompts.get_session_instructions",
        lambda: "TEST INSTRUCTIONS",
    )

    fake_legacy = _fake_cls("GeminiTextElevenLabsHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextElevenLabsHandler", fake_legacy):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert result.pipeline._conversation_history[0] == {
        "role": "system",
        "content": "TEST INSTRUCTIONS",
    }


def test_composable_path_copy_constructs_fresh_legacy_for_gemini_fallback_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """copy() must produce an independent wrapper + fresh GeminiTextElevenLabsHandler."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", "unknown")
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("GeminiTextElevenLabsHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextElevenLabsHandler", fake_legacy):
        original = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )
        copy = original.copy()

    assert copy is not original
    assert copy._tts_handler is not original._tts_handler
    assert copy.pipeline is not original.pipeline
```

Run: `/d/Projects/robot_comic/.venv/Scripts/python.exe -m pytest tests/test_handler_factory_factory_path.py -q -k gemini_fallback`

Expected: the legacy guard from Task 1 still passes; the four new composable tests **fail** because the factory still returns `LocalSTTGeminiElevenLabsHandler` for this dispatch even when `FACTORY_PATH=composable`.

- [ ] **Step 2: Implement the composable gate** in `handler_factory.py`.

Inside the outer `if input_backend == AUDIO_INPUT_MOONSHINE:` arm at line 293, in the `if output_backend == AUDIO_OUTPUT_ELEVENLABS:` block at line 314, prepend the composable check before the import + `return LocalSTTGeminiElevenLabsHandler(**handler_kwargs)`:

```python
if output_backend == AUDIO_OUTPUT_ELEVENLABS:
    # Phase 4c.4 (#337): the gemini-fallback dispatch arm is reached
    # when LLM_BACKEND is neither "llama" nor "gemini". For the
    # composable path we route through the same builder used by the
    # LLM_BACKEND=gemini arm (4c.3) because the underlying triple is
    # the same (moonshine + elevenlabs + Gemini-API LLM), just reached
    # through a different dispatch condition. The composable path's
    # GeminiLLMAdapter requires a handler with _call_llm
    # (GeminiTextElevenLabsHandler); LocalSTTGeminiElevenLabsHandler
    # uses _run_llm_with_tools instead and can't host that adapter.
    # Phase 4e will retire LocalSTTGeminiElevenLabsHandler.
    if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
        logger.info(
            "HandlerFactory: selecting ComposableConversationHandler "
            "(%s → %s, llm=gemini-fallback, factory_path=composable)",
            input_backend,
            output_backend,
        )
        return _build_composable_gemini_elevenlabs(**handler_kwargs)
    from robot_comic.elevenlabs_tts import LocalSTTGeminiElevenLabsHandler

    logger.info(
        "HandlerFactory: selecting LocalSTTGeminiElevenLabsHandler (%s → %s)",
        input_backend,
        output_backend,
    )
    return LocalSTTGeminiElevenLabsHandler(**handler_kwargs)
```

No new helper. `_build_composable_gemini_elevenlabs` already exists from 4c.3 at `handler_factory.py:492-537`.

Run: the four new composable tests pass; legacy guard still passes; existing tests unchanged.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/handler_factory.py tests/test_handler_factory_factory_path.py
git commit -m "feat(factory): Phase 4c.4 — composable routing for (moonshine, elevenlabs, gemini-fallback) (#337)"
```

---

## Pre-push checklist (from worktree / repo root)

```
uvx ruff@0.12.0 check
uvx ruff@0.12.0 format --check
/d/Projects/robot_comic/.venv/Scripts/mypy --pretty src/robot_comic/handler_factory.py
/d/Projects/robot_comic/.venv/Scripts/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

If any step is red, fix locally before pushing.

(`.clone/` and stray `*.m4a` files in the worktree root are user-private artifacts not tracked by git; ruff sees them locally but CI won't. CI is clean of those.)

Sanity-check main's CI is green on the current tip commit before opening the PR.

## PR creation

After both commits land on `claude/phase-4c4-gemini-fallback-elevenlabs-routing`:

```
git push -u origin claude/phase-4c4-gemini-fallback-elevenlabs-routing
```

Then the manager opens the PR with the body summarising the spec.

## Risks

Covered in the spec under "Risks". The biggest is the deliberate class-swap (composable path constructs `GeminiTextElevenLabsHandler` instead of `LocalSTTGeminiElevenLabsHandler`) — pinned by the test asserting `_tts_handler is GeminiTextElevenLabsHandler` under composable, and the test asserting `_tts_handler is LocalSTTGeminiElevenLabsHandler` under legacy.

## After-merge follow-ups (out of scope for 4c.4)

- 4c.5: build `GeminiTTSAdapter` for the gemini-bundled triple.
- Lifecycle hooks: per-PR rollout per the operating manual.
- Phase 4e: delete `LocalSTTGeminiElevenLabsHandler` + the gemini-fallback dispatch fallthrough.
