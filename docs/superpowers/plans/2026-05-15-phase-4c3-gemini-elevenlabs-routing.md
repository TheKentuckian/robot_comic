# Phase 4c.3 Implementation Plan — Composable routing for `(moonshine, elevenlabs, gemini)`

**Goal:** Broaden `ElevenLabsTTSAdapter`'s constructor annotation to a Protocol; route `(moonshine, elevenlabs, gemini)` through `ComposableConversationHandler` under `REACHY_MINI_FACTORY_PATH=composable`. Default `legacy` unchanged.

**Architecture:** No new adapter. One Protocol definition + annotation broadening in `elevenlabs_tts_adapter.py`. One new factory helper + composable gate in `handler_factory.py`. Five new factory-dispatch tests + one new adapter unit test + a small cleanup of `# type: ignore[arg-type]` in the adapter tests.

**Tech Stack:** Python 3.12, pytest-asyncio, `unittest.mock`, existing `_fake_cls` patching style from 4b/4c.1/4c.2.

**Spec:** `docs/superpowers/specs/2026-05-15-phase-4c3-gemini-elevenlabs-routing.md`

---

## File Map

| File | Role |
|------|------|
| `src/robot_comic/adapters/elevenlabs_tts_adapter.py` | EDIT — Protocol + annotation broadening |
| `src/robot_comic/handler_factory.py` | EDIT — new composable gate inside the gemini+elevenlabs arm + `_build_composable_gemini_elevenlabs` helper |
| `tests/adapters/test_elevenlabs_tts_adapter.py` | EDIT — add duck-typed gemini-shape acceptance test; remove now-redundant `# type: ignore[arg-type]` comments |
| `tests/test_handler_factory_factory_path.py` | EDIT — new gemini+elevenlabs dispatch tests; delete the stale `test_composable_path_with_gemini_llm_unchanged` |

---

## TDD task breakdown

Three tasks, each one failing test set → minimum implementation → green → commit.

### Task 1 — Broaden `ElevenLabsTTSAdapter` annotation via Protocol

**Files:**
- Edit: `src/robot_comic/adapters/elevenlabs_tts_adapter.py`
- Edit: `tests/adapters/test_elevenlabs_tts_adapter.py`

- [ ] **Step 1: Write the failing test** — add `test_adapter_accepts_duck_typed_gemini_elevenlabs_handler_shape` to `test_elevenlabs_tts_adapter.py`. Builds a stub class that does NOT inherit from `_StubElevenLabsHandler` (proves structural typing) and exposes the four Protocol members the adapter needs. Constructs an adapter and runs one `prepare()` → `synthesize()` → `shutdown()` cycle.

```python
@pytest.mark.asyncio
async def test_adapter_accepts_duck_typed_gemini_elevenlabs_handler_shape() -> None:
    """Phase 4c.3: the adapter accepts any handler matching the Protocol surface.

    Mimics ``GeminiTextElevenLabsResponseHandler``'s diamond-MRO shape:
    exposes ``_prepare_startup_credentials``, ``output_queue``,
    ``_stream_tts_to_queue``, and ``_http`` without inheriting from
    ``ElevenLabsTTSResponseHandler``.
    """

    class _DuckGeminiElevenLabs:
        def __init__(self) -> None:
            self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
            self._http: Any = None
            self.prepare_called = False
            self.streamed_text: str | None = None

        async def _prepare_startup_credentials(self) -> None:
            self.prepare_called = True

        async def _stream_tts_to_queue(
            self,
            text: str,
            first_audio_marker: list[float] | None = None,
            tags: list[str] | None = None,
        ) -> bool:
            self.streamed_text = text
            await self.output_queue.put((24000, [42]))
            return True

    handler = _DuckGeminiElevenLabs()
    adapter = ElevenLabsTTSAdapter(handler)
    await adapter.prepare()
    assert handler.prepare_called is True

    out = [frame async for frame in adapter.synthesize("hello, world")]
    assert len(out) == 1
    assert out[0].sample_rate == 24000
    assert handler.streamed_text == "hello, world"

    await adapter.shutdown()
```

No `# type: ignore` needed — once the Protocol broadening lands, the adapter accepts the duck-typed stub directly.

Run: `.venv/Scripts/python.exe -m pytest tests/adapters/test_elevenlabs_tts_adapter.py -q`

Expected: the new test fails mypy/runtime as a *type-construction* test when run through mypy — actually it'll pass at runtime today (Python doesn't enforce annotations); mypy is what fails. So also run mypy:

```
.venv/Scripts/mypy --pretty tests/adapters/test_elevenlabs_tts_adapter.py
```

Expected: mypy reports `Argument 1 to "ElevenLabsTTSAdapter" has incompatible type "_DuckGeminiElevenLabs"; expected "ElevenLabsTTSResponseHandler"`.

- [ ] **Step 2: Implement the Protocol + broaden the annotation.**

Edit `src/robot_comic/adapters/elevenlabs_tts_adapter.py`:

1. Remove the `TYPE_CHECKING` block that imports `ElevenLabsTTSResponseHandler`.
2. Add `Protocol` to the `typing` import.
3. Define `_ElevenLabsCompatibleHandler` Protocol at module scope (after the `_STREAM_DONE` sentinel) with the four members. Module-private (leading underscore). No `@runtime_checkable`.
4. Change the constructor annotation: `handler: "_ElevenLabsCompatibleHandler"`.
5. Update the module docstring to note the Protocol broadening (a one-paragraph addition pointing at the duck-typed surface and naming the satisfiers).

Also clean up the existing `# type: ignore[arg-type]` comments in `test_elevenlabs_tts_adapter.py` (two of them: in the `_SlowHandler` test and the Protocol-conformance test). With `warn_unused_ignores = true` mypy will flag them as unused once the Protocol is in place.

Run pytest + mypy. Both green.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/adapters/elevenlabs_tts_adapter.py \
        tests/adapters/test_elevenlabs_tts_adapter.py
git commit -m "refactor(adapters): broaden ElevenLabsTTSAdapter annotation via Protocol (#337)"
```

---

### Task 2 — Factory legacy-path regression guard for `(moonshine, elevenlabs, gemini)`

The point: pin `FACTORY_PATH=legacy` keeps the existing `GeminiTextElevenLabsHandler` for the gemini+elevenlabs triple. Should already pass on green main because the factory unconditionally returns `GeminiTextElevenLabsHandler` there today.

- [ ] **Step 1: Write the test** — add to `tests/test_handler_factory_factory_path.py`, in the new "Phase 4c.3" section (after the 4c.2 block):

```python
# ---------------------------------------------------------------------------
# Phase 4c.3 — (moonshine, elevenlabs, gemini) composable path
# ---------------------------------------------------------------------------


def test_legacy_path_returns_legacy_handler_for_gemini_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``FACTORY_PATH=legacy`` (default) keeps today's GeminiTextElevenLabsHandler."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_LEGACY)

    fake = _fake_cls("GeminiTextElevenLabsHandler")
    with patch("robot_comic.gemini_text_handlers.GeminiTextElevenLabsHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
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
git commit -m "test(factory): pin legacy-path gemini+elevenlabs dispatch (#337)"
```

---

### Task 3 — Composable branch for `(moonshine, elevenlabs, gemini)`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_handler_factory_factory_path.py`:

```python
def test_composable_path_returns_wrapper_for_gemini_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """``FACTORY_PATH=composable`` + gemini-elevenlabs triple → ComposableConversationHandler."""
    from robot_comic import config as cfg_mod
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
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


def test_composable_path_wires_three_adapters_for_gemini_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """All three adapters wrap the same single GeminiTextElevenLabsHandler instance."""
    from robot_comic import config as cfg_mod
    from robot_comic.adapters import (
        GeminiLLMAdapter,
        MoonshineSTTAdapter,
        ElevenLabsTTSAdapter,
    )

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
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
    assert pipe.stt._handler is pipe.llm._handler
    assert pipe.llm._handler is pipe.tts._handler
    assert pipe.stt._handler is result._tts_handler


def test_composable_path_seeds_system_prompt_for_gemini_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """The pipeline's system prompt is sourced from prompts.get_session_instructions."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
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


def test_composable_path_copy_constructs_fresh_legacy_for_gemini_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """copy() must produce an independent wrapper + fresh GeminiTextElevenLabsHandler."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
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

Also DELETE the now-stale `test_composable_path_with_gemini_llm_unchanged` test (it currently asserts elevenlabs+gemini stays on legacy under `FACTORY_PATH=composable`, which is what this PR negates). The new `test_legacy_path_returns_legacy_handler_for_gemini_elevenlabs` covers the same property for the default-legacy path; the four composable tests above cover the composable side.

Run: the four new tests fail (factory still returns legacy in both paths); the four deleted-and-replaced cases work because the deletion happens in this commit.

- [ ] **Step 2: Implement the composable branch** in `handler_factory.py`:

Inside the `LLM_BACKEND_GEMINI` arm (the `if output_backend == AUDIO_OUTPUT_ELEVENLABS:` block around line 254), prepend the composable check:

```python
if output_backend == AUDIO_OUTPUT_ELEVENLABS:
    # Phase 4c.3 (#337): gemini+elevenlabs is routed through
    # ComposableConversationHandler when FACTORY_PATH=composable.
    # Default FACTORY_PATH=legacy keeps the existing
    # GeminiTextElevenLabsHandler selection below.
    if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
        logger.info(
            "HandlerFactory: selecting ComposableConversationHandler "
            "(%s → %s, llm=%s, factory_path=composable)",
            input_backend,
            output_backend,
            LLM_BACKEND_GEMINI,
        )
        return _build_composable_gemini_elevenlabs(**handler_kwargs)
    from robot_comic.gemini_text_handlers import GeminiTextElevenLabsHandler

    logger.info(
        "HandlerFactory: selecting GeminiTextElevenLabsHandler (%s → %s, llm=%s)",
        input_backend,
        output_backend,
        LLM_BACKEND_GEMINI,
    )
    return GeminiTextElevenLabsHandler(**handler_kwargs)
```

Add the helper at the bottom of the module next to `_build_composable_gemini_chatterbox`:

```python
def _build_composable_gemini_elevenlabs(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, elevenlabs, gemini) pipeline.

    Builds a legacy ``GeminiTextElevenLabsHandler`` (the adapters delegate
    into it), wraps it with the three Phase 3/4 adapters, composes them into
    a ``ComposablePipeline`` seeded with the current session instructions,
    and returns a ``ComposableConversationHandler`` whose ``build`` closure
    re-runs the same construction. FastRTC's ``copy()`` per-peer cloning
    invokes the closure for fresh state on each new peer.

    The ElevenLabs TTS half is shared with Phase 4b's llama variant; the
    LLM half is the same ``GeminiLLMAdapter`` from Phase 4c.2.  No new
    adapter is introduced.
    """
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        GeminiLLMAdapter,
        MoonshineSTTAdapter,
        ElevenLabsTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.gemini_text_handlers import GeminiTextElevenLabsHandler
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        legacy = GeminiTextElevenLabsHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = GeminiLLMAdapter(legacy)
        tts = ElevenLabsTTSAdapter(legacy)
        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
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

Bonus cleanup: with the Protocol in place, the `cast(Any, legacy)` in `_build_composable_llama_elevenlabs` can be removed. Try removing it; if mypy is happy, also drop the `cast` import if it becomes unused. If mypy complains, leave the cast and follow up in a separate PR.

Run: the four new tests pass; legacy guard still passes; existing tests unchanged.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/handler_factory.py tests/test_handler_factory_factory_path.py
git commit -m "feat(factory): Phase 4c.3 — composable routing for (moonshine, elevenlabs, gemini) (#337)"
```

---

## Pre-push checklist (from worktree / repo root)

```
uvx ruff@0.12.0 check
uvx ruff@0.12.0 format --check
.venv/Scripts/mypy --pretty src/robot_comic/adapters/elevenlabs_tts_adapter.py src/robot_comic/handler_factory.py
.venv/Scripts/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

If any step is red, fix locally before pushing.

(`.clone/` and stray `*.m4a` files in the worktree root are user-private artifacts not tracked by git; ruff sees them locally but CI won't. Local lint runs with `--exclude .clone` for sanity; CI is clean.)

Sanity-check main's CI is green on the current tip commit before opening the PR.

## PR creation

After all three commits land on `claude/phase-4c3-gemini-elevenlabs-routing`:

```
git push -u origin claude/phase-4c3-gemini-elevenlabs-routing
```

Then the manager opens the PR with the body summarising the spec.

## Risks

Covered in the spec under "Risks". The biggest is the `# type: ignore[arg-type]` cleanup in `test_elevenlabs_tts_adapter.py` becoming "unused ignore" under `warn_unused_ignores = true` — handled in the same step as the Protocol broadening.

## After-merge follow-ups (out of scope for 4c.3)

- 4c.4: `LocalSTTGeminiElevenLabsHandler` — same shape, reuses the broadened `ElevenLabsTTSAdapter`.
- 4c.5: build `GeminiTTSAdapter` for the gemini-bundled triple.
- Lifecycle hooks: per-PR rollout per the operating manual.
