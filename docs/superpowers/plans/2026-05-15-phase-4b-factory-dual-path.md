# Phase 4b Implementation Plan — Factory Dual Path

**Goal:** Wire `handler_factory.py` to return `ComposableConversationHandler` for the `(moonshine, llama, elevenlabs)` triple behind a new `REACHY_MINI_FACTORY_PATH=legacy|composable` dial (default `legacy`). Fix the two barge-in wiring gaps in the 4a wrapper so the composable path is functionally equivalent for `_clear_queue`.

**Architecture:** One config dial. One new private helper in the factory. One new branch in the existing `(moonshine, elevenlabs, llama)` block. Two property/setter pairs on the wrapper. Three new test files / file additions.

**Tech Stack:** Python 3.12, pytest-asyncio, `unittest.mock`, existing `_fake_cls` patching style from `tests/test_handler_factory_llama_llm.py`.

**Spec:** `docs/superpowers/specs/2026-05-15-phase-4b-factory-dual-path.md`

---

## File Map

| File | Role |
|------|------|
| `src/robot_comic/config.py` | EDIT — add `FACTORY_PATH_*` constants, `_normalize_factory_path`, `Config.FACTORY_PATH`, refresh hook |
| `src/robot_comic/handler_factory.py` | EDIT — gate the existing llama+elevenlabs path on `FACTORY_PATH`; add `_build_composable_llama_elevenlabs` helper |
| `src/robot_comic/composable_conversation_handler.py` | EDIT — convert `output_queue` and `_clear_queue` to property/setter pairs |
| `tests/test_config_factory_path.py` | NEW — dial normalisation + env-var read |
| `tests/test_handler_factory_factory_path.py` | NEW — dispatch matrix under both `FACTORY_PATH` values |
| `tests/test_composable_conversation_handler.py` | APPEND — five new tests for the property/setter fixes |
| `.env.example` | EDIT — one-line addition documenting the new dial |

No changes to `composable_pipeline.py`, the adapters, the ABC, or `main.py`.

---

## TDD task breakdown

Seven tasks. Each is one failing test → minimum implementation → green → one commit. Same cadence as 4a's seven commits.

### Task 1 — Config dial constants + normaliser

**Files:**
- Edit: `src/robot_comic/config.py`
- New: `tests/test_config_factory_path.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_factory_path.py
from __future__ import annotations
import importlib
import logging

import pytest

from robot_comic import config as cfg


def test_constants_defined() -> None:
    assert cfg.FACTORY_PATH_ENV == "REACHY_MINI_FACTORY_PATH"
    assert cfg.FACTORY_PATH_LEGACY == "legacy"
    assert cfg.FACTORY_PATH_COMPOSABLE == "composable"
    assert cfg.FACTORY_PATH_CHOICES == ("legacy", "composable")
    assert cfg.DEFAULT_FACTORY_PATH == "legacy"


def test_normalize_default_when_unset() -> None:
    assert cfg._normalize_factory_path(None) == "legacy"
    assert cfg._normalize_factory_path("") == "legacy"
    assert cfg._normalize_factory_path("   ") == "legacy"


def test_normalize_known_values() -> None:
    assert cfg._normalize_factory_path("composable") == "composable"
    assert cfg._normalize_factory_path("COMPOSABLE") == "composable"
    assert cfg._normalize_factory_path("  legacy  ") == "legacy"


def test_normalize_invalid_falls_back_to_legacy_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="robot_comic.config"):
        result = cfg._normalize_factory_path("hybrid")
    assert result == "legacy"
    assert any("hybrid" in record.message for record in caplog.records)
```

Run: `.venv/bin/pytest tests/test_config_factory_path.py -v`

Expected: `AttributeError` — the constants don't exist yet.

- [ ] **Step 2: Implement**

Add to `src/robot_comic/config.py` immediately after the `PIPELINE_MODE_*` definitions (~line 327):

```python
# ---------------------------------------------------------------------------
# Factory path dial — Phase 4b of #337. Selects whether HandlerFactory returns
# the legacy concrete handler classes (default, today's behaviour) or the
# new ComposableConversationHandler wrapper around a ComposablePipeline.
# Only affects the (moonshine, llama, elevenlabs) triple in 4b; other triples
# are migrated in 4c.
# ---------------------------------------------------------------------------

FACTORY_PATH_ENV = "REACHY_MINI_FACTORY_PATH"
FACTORY_PATH_LEGACY = "legacy"
FACTORY_PATH_COMPOSABLE = "composable"
FACTORY_PATH_CHOICES: tuple[str, ...] = (FACTORY_PATH_LEGACY, FACTORY_PATH_COMPOSABLE)
DEFAULT_FACTORY_PATH = FACTORY_PATH_LEGACY


def _normalize_factory_path(value: str | None) -> str:
    """Validate REACHY_MINI_FACTORY_PATH; fall back to legacy on unknowns."""
    candidate = (value or "").strip().lower()
    if not candidate:
        return DEFAULT_FACTORY_PATH
    if candidate in FACTORY_PATH_CHOICES:
        return candidate
    logger.warning(
        "Invalid %s=%r. Expected one of: %s. Falling back to %r.",
        FACTORY_PATH_ENV,
        value,
        ", ".join(FACTORY_PATH_CHOICES),
        DEFAULT_FACTORY_PATH,
    )
    return DEFAULT_FACTORY_PATH
```

Run test. Expected: green.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/config.py tests/test_config_factory_path.py
git commit -m "feat(config): FACTORY_PATH dial constants + normaliser (#337)"
```

---

### Task 2 — `Config.FACTORY_PATH` field + refresh hook

- [ ] **Step 1: Add failing test**

Append to `tests/test_config_factory_path.py`:

```python
def test_config_field_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cfg.FACTORY_PATH_ENV, raising=False)
    cfg.refresh_runtime_config_from_env()
    assert cfg.config.FACTORY_PATH == "legacy"


def test_config_field_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cfg.FACTORY_PATH_ENV, "composable")
    cfg.refresh_runtime_config_from_env()
    assert cfg.config.FACTORY_PATH == "composable"


def test_config_field_invalid_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cfg.FACTORY_PATH_ENV, "nope")
    cfg.refresh_runtime_config_from_env()
    assert cfg.config.FACTORY_PATH == "legacy"
```

Run: `AttributeError: 'Config' object has no attribute 'FACTORY_PATH'`.

- [ ] **Step 2: Implement**

In `Config` class body (after the `PIPELINE_MODE` block at line 1106):

```python
FACTORY_PATH: str = _normalize_factory_path(os.getenv(FACTORY_PATH_ENV))
```

In `refresh_runtime_config_from_env()` (after the `PIPELINE_MODE` refresh at line 1285):

```python
config.FACTORY_PATH = _normalize_factory_path(os.getenv(FACTORY_PATH_ENV))
```

Run test. Expected: green.

- [ ] **Step 3: `.env.example` documentation**

Append to `.env.example`:

```
# Phase 4 pipeline refactor (#337). Set to "composable" to route the
# (moonshine, llama, elevenlabs) triple through the new ComposablePipeline.
# Default "legacy" preserves today's concrete handler classes.
REACHY_MINI_FACTORY_PATH=legacy
```

- [ ] **Step 4: Commit**

```
git add src/robot_comic/config.py tests/test_config_factory_path.py .env.example
git commit -m "feat(config): Config.FACTORY_PATH field + refresh + .env.example doc (#337)"
```

---

### Task 3 — Factory legacy-path preservation (regression guard)

The point of this task is to prove that `FACTORY_PATH=legacy` (the default) keeps returning the existing concrete classes for *every* triple — i.e., that we don't accidentally break anything before adding the new branch.

- [ ] **Step 1: Write the test**

```python
# tests/test_handler_factory_factory_path.py
"""Phase 4b: dispatch behaviour under REACHY_MINI_FACTORY_PATH."""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from robot_comic.config import (
    FACTORY_PATH_COMPOSABLE,
    FACTORY_PATH_LEGACY,
    LLM_BACKEND_LLAMA,
    LLM_BACKEND_GEMINI,
    AUDIO_INPUT_HF,
    AUDIO_INPUT_MOONSHINE,
    AUDIO_OUTPUT_HF,
    AUDIO_OUTPUT_CHATTERBOX,
    AUDIO_OUTPUT_ELEVENLABS,
    AUDIO_OUTPUT_GEMINI_TTS,
    AUDIO_OUTPUT_GEMINI_LIVE,
    AUDIO_INPUT_GEMINI_LIVE,
    AUDIO_OUTPUT_OPENAI_REALTIME,
    AUDIO_INPUT_OPENAI_REALTIME,
    PIPELINE_MODE_COMPOSABLE,
    PIPELINE_MODE_HF_REALTIME,
    PIPELINE_MODE_OPENAI_REALTIME,
    PIPELINE_MODE_GEMINI_LIVE,
)
from robot_comic.handler_factory import HandlerFactory


@pytest.fixture()
def mock_deps() -> MagicMock:
    return MagicMock(name="ToolDependencies")


def _fake_cls(name: str):
    class _Fake:
        def __init__(self, *a, **kw):
            pass
    _Fake.__name__ = name
    return _Fake


def test_legacy_path_returns_legacy_handler_for_llama_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod
    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_LEGACY)

    fake = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch("robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake)
```

Run: this test should already pass on green main because `FACTORY_PATH` is set via `monkeypatch.setattr` to `"legacy"` and the existing factory ignores the attribute. (If it doesn't pass, fix monkeypatch / import path before moving on.)

Expected: green out of the gate.

- [ ] **Step 2: No implementation needed**

This is a guard test that captures pre-4b behaviour for posterity.

- [ ] **Step 3: Commit**

```
git add tests/test_handler_factory_factory_path.py
git commit -m "test(factory): pin legacy-path llama+elevenlabs dispatch (#337)"
```

---

### Task 4 — Composable branch in factory (the meat of 4b)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_handler_factory_factory_path.py`:

```python
def test_composable_path_returns_wrapper_for_llama_elevenlabs(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod
    from robot_comic.composable_conversation_handler import ComposableConversationHandler
    from robot_comic.composable_pipeline import ComposablePipeline

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch(
        "robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler",
        fake_legacy,
    ):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, ComposableConversationHandler)
    assert isinstance(result.pipeline, ComposablePipeline)
    assert isinstance(result._tts_handler, fake_legacy)


def test_composable_path_wires_three_adapters(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod
    from robot_comic.adapters import (
        ElevenLabsTTSAdapter, LlamaLLMAdapter, MoonshineSTTAdapter,
    )

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch(
        "robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler",
        fake_legacy,
    ):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    pipe = result.pipeline
    assert isinstance(pipe.stt, MoonshineSTTAdapter)
    assert isinstance(pipe.llm, LlamaLLMAdapter)
    assert isinstance(pipe.tts, ElevenLabsTTSAdapter)
    # All three adapters wrap the same legacy instance.
    assert pipe.stt._handler is pipe.llm._handler is pipe.tts._handler
    assert pipe.stt._handler is result._tts_handler


def test_composable_path_seeds_system_prompt(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)
    monkeypatch.setattr(
        "robot_comic.prompts.get_session_instructions",
        lambda: "TEST INSTRUCTIONS",
    )

    fake_legacy = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch(
        "robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler",
        fake_legacy,
    ):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            AUDIO_OUTPUT_ELEVENLABS,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    history = result.pipeline._conversation_history
    assert history[0] == {"role": "system", "content": "TEST INSTRUCTIONS"}


def test_composable_path_copy_constructs_fresh_legacy(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake_legacy = _fake_cls("LocalSTTLlamaElevenLabsHandler")
    with patch(
        "robot_comic.llama_elevenlabs_tts.LocalSTTLlamaElevenLabsHandler",
        fake_legacy,
    ):
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

Run: all four fail — the factory still returns `LocalSTTLlamaElevenLabsHandler` regardless of `FACTORY_PATH`.

- [ ] **Step 2: Implement the composable branch**

In `src/robot_comic/handler_factory.py`, modify the `LLM_BACKEND_LLAMA` + `AUDIO_OUTPUT_ELEVENLABS` block (currently `lines 186-196`):

```python
if _llm_backend == LLM_BACKEND_LLAMA:
    if output_backend == AUDIO_OUTPUT_ELEVENLABS:
        if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
            logger.info(
                "HandlerFactory: selecting ComposableConversationHandler "
                "(%s → %s, llm=%s, factory_path=composable)",
                input_backend, output_backend, LLM_BACKEND_LLAMA,
            )
            return _build_composable_llama_elevenlabs(**handler_kwargs)
        from robot_comic.llama_elevenlabs_tts import LocalSTTLlamaElevenLabsHandler
        logger.info(
            "HandlerFactory: selecting LocalSTTLlamaElevenLabsHandler (%s → %s, llm=%s)",
            input_backend, output_backend, LLM_BACKEND_LLAMA,
        )
        return LocalSTTLlamaElevenLabsHandler(**handler_kwargs)
```

Add the helper at module bottom (after the `HandlerFactory` class):

```python
def _build_composable_llama_elevenlabs(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, llama, elevenlabs) pipeline.

    Builds the legacy handler instance (the adapters delegate into it),
    wraps it with the three Phase 3 adapters, composes them into a
    ComposablePipeline seeded with the current session instructions, and
    returns a ComposableConversationHandler whose ``build`` closure re-runs
    the same construction. FastRTC's ``copy()`` per-peer cloning invokes
    the closure for fresh state on each new peer.
    """
    from robot_comic.adapters import (
        ElevenLabsTTSAdapter,
        LlamaLLMAdapter,
        MoonshineSTTAdapter,
    )
    from robot_comic.composable_conversation_handler import (
        ComposableConversationHandler,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.llama_elevenlabs_tts import LocalSTTLlamaElevenLabsHandler
    from robot_comic.prompts import get_session_instructions

    def _build() -> ComposableConversationHandler:
        legacy = LocalSTTLlamaElevenLabsHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = LlamaLLMAdapter(legacy)
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

Add `FACTORY_PATH_COMPOSABLE`, `FACTORY_PATH_LEGACY` to the existing `from robot_comic.config import (...)` block.

Run: the four composable tests pass; the Task 3 legacy guard still passes; existing `test_handler_factory_llama_llm.py` still passes.

Expected: green.

- [ ] **Step 3: Commit**

```
git add src/robot_comic/handler_factory.py tests/test_handler_factory_factory_path.py
git commit -m "feat(factory): wire composable path for (moonshine, llama, elevenlabs) (#337)"
```

---

### Task 5 — Other triples remain on legacy under `FACTORY_PATH=composable`

The dial is scoped to one triple in 4b. Pin that.

- [ ] **Step 1: Write the failing tests**

Append:

```python
@pytest.mark.parametrize(
    "output_backend, target_class",
    [
        (AUDIO_OUTPUT_CHATTERBOX, "LocalSTTChatterboxHandler"),
        (AUDIO_OUTPUT_GEMINI_TTS, "LocalSTTGeminiTTSHandler"),
        (AUDIO_OUTPUT_OPENAI_REALTIME, "LocalSTTOpenAIRealtimeHandler"),
        (AUDIO_OUTPUT_HF, "LocalSTTHuggingFaceRealtimeHandler"),
    ],
)
def test_composable_path_only_affects_llama_elevenlabs(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
    output_backend: str,
    target_class: str,
) -> None:
    """Even with FACTORY_PATH=composable, other Moonshine triples flow through legacy."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_LLAMA)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    module_by_output = {
        AUDIO_OUTPUT_CHATTERBOX: "robot_comic.chatterbox_tts",
        AUDIO_OUTPUT_GEMINI_TTS: "robot_comic.gemini_tts",
        AUDIO_OUTPUT_OPENAI_REALTIME: "robot_comic.local_stt_realtime",
        AUDIO_OUTPUT_HF: "robot_comic.local_stt_realtime",
    }
    fake = _fake_cls(target_class)
    with patch(f"{module_by_output[output_backend]}.{target_class}", fake):
        result = HandlerFactory.build(
            AUDIO_INPUT_MOONSHINE,
            output_backend,
            mock_deps,
            pipeline_mode=PIPELINE_MODE_COMPOSABLE,
        )

    assert isinstance(result, fake)


@pytest.mark.parametrize(
    "pipeline_mode, input_b, output_b, module, target_class",
    [
        (PIPELINE_MODE_HF_REALTIME, AUDIO_INPUT_HF, AUDIO_OUTPUT_HF,
         "robot_comic.huggingface_realtime", "HuggingFaceRealtimeHandler"),
        (PIPELINE_MODE_OPENAI_REALTIME, AUDIO_INPUT_OPENAI_REALTIME, AUDIO_OUTPUT_OPENAI_REALTIME,
         "robot_comic.openai_realtime", "OpenaiRealtimeHandler"),
        (PIPELINE_MODE_GEMINI_LIVE, AUDIO_INPUT_GEMINI_LIVE, AUDIO_OUTPUT_GEMINI_LIVE,
         "robot_comic.gemini_live", "GeminiLiveHandler"),
    ],
)
def test_composable_path_ignored_in_bundled_realtime_modes(
    monkeypatch: pytest.MonkeyPatch,
    mock_deps: MagicMock,
    pipeline_mode: str, input_b: str, output_b: str,
    module: str, target_class: str,
) -> None:
    from robot_comic import config as cfg_mod
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

    fake = _fake_cls(target_class)
    with patch(f"{module}.{target_class}", fake):
        result = HandlerFactory.build(
            input_b, output_b, mock_deps, pipeline_mode=pipeline_mode,
        )
    assert isinstance(result, fake)


def test_composable_path_with_gemini_llm_unchanged(
    monkeypatch: pytest.MonkeyPatch, mock_deps: MagicMock
) -> None:
    """FACTORY_PATH=composable + LLM_BACKEND=gemini → legacy GeminiTextElevenLabsHandler (4c will migrate)."""
    from robot_comic import config as cfg_mod

    monkeypatch.setattr(cfg_mod.config, "LLM_BACKEND", LLM_BACKEND_GEMINI)
    monkeypatch.setattr(cfg_mod.config, "FACTORY_PATH", FACTORY_PATH_COMPOSABLE)

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

Run: should already pass if Task 4 implementation correctly scoped the new branch to `LLM_BACKEND_LLAMA + AUDIO_OUTPUT_ELEVENLABS` only. If any fails, the new branch is too eager — go back and narrow.

Expected: green.

- [ ] **Step 2: No new implementation expected**

If anything's red, fix the factory branch's gating.

- [ ] **Step 3: Commit**

```
git add tests/test_handler_factory_factory_path.py
git commit -m "test(factory): pin other triples + bundled modes unchanged under composable (#337)"
```

---

### Task 6 — Wrapper `_clear_queue` propagation

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_composable_conversation_handler.py`:

```python
def test_clear_queue_assignment_propagates_to_tts_handler() -> None:
    wrapper = _make_wrapper()
    cb = lambda: None
    wrapper._clear_queue = cb
    assert wrapper._clear_queue is cb
    assert wrapper._tts_handler._clear_queue is cb


def test_clear_queue_assignment_handles_none() -> None:
    wrapper = _make_wrapper()
    wrapper._clear_queue = lambda: None
    wrapper._clear_queue = None
    assert wrapper._clear_queue is None
    assert wrapper._tts_handler._clear_queue is None
```

Run: failures — current wrapper stores `_clear_queue` as plain attribute, doesn't propagate.

- [ ] **Step 2: Implement**

Replace the plain `self._clear_queue = None` in `composable_conversation_handler.py:__init__` with a property/setter pair below the constructor:

```python
def __init__(
    self,
    pipeline: ComposablePipeline,
    *,
    tts_handler: ConversationHandler,
    deps: ToolDependencies,
    build: Callable[[], "ComposableConversationHandler"],
) -> None:
    """Store the pipeline, the legacy TTS handler, deps, and the rebuild closure."""
    self.pipeline = pipeline
    self._tts_handler = tts_handler
    self.deps = deps
    self._build = build
    # Use the private slot directly so the setter's tts_handler dereference
    # is well-defined (no half-built object during __init__).
    self.__clear_queue: Callable[[], None] | None = None

@property
def _clear_queue(self) -> Callable[[], None] | None:  # type: ignore[override]
    """The queue-flush callback. Mirrors onto the wrapped TTS handler."""
    return self.__clear_queue

@_clear_queue.setter
def _clear_queue(self, callback: Callable[[], None] | None) -> None:
    """Mirror the queue-flush callback onto the underlying TTS handler.

    The LocalSTTInputMixin listener calls ``self._clear_queue`` on the
    legacy ``LocalSTTLlamaElevenLabsHandler`` instance it's mixed into;
    that instance is our ``_tts_handler``, not us. So when LocalStream
    sets ``handler._clear_queue = handler.clear_audio_queue`` on the
    wrapper, we have to forward the assignment to the legacy handler
    or barge-in stops flushing on the composable path.
    """
    self.__clear_queue = callback
    if getattr(self, "_tts_handler", None) is not None:
        self._tts_handler._clear_queue = callback
```

(Property name mangling note: writing `self.__clear_queue` inside the class translates to `self._ComposableConversationHandler__clear_queue` — that's intentional, it avoids ABC clash with the inherited `_clear_queue` annotation.)

Remove the old `self.output_queue = pipeline.output_queue` / `self._clear_queue = None` lines from `__init__`; they'll be re-added in Task 7 (for `output_queue`).

Run tests. Expected: green for the two new tests; existing 4a tests for `_make_wrapper()` still green (the helper sets `_tts_handler` as a MagicMock, which accepts arbitrary attribute writes).

- [ ] **Step 3: Commit**

```
git add src/robot_comic/composable_conversation_handler.py tests/test_composable_conversation_handler.py
git commit -m "fix(wrapper): forward _clear_queue assignment to underlying TTS handler (#337)"
```

---

### Task 7 — Wrapper `output_queue` redirection through pipeline

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_composable_conversation_handler.py`:

```python
def test_output_queue_getter_returns_pipeline_queue() -> None:
    wrapper = _make_wrapper()
    new_queue: asyncio.Queue = asyncio.Queue()
    wrapper.pipeline.output_queue = new_queue
    assert wrapper.output_queue is new_queue


def test_output_queue_setter_replaces_pipeline_queue() -> None:
    wrapper = _make_wrapper()
    fresh: asyncio.Queue = asyncio.Queue()
    wrapper.output_queue = fresh
    assert wrapper.pipeline.output_queue is fresh


@pytest.mark.asyncio
async def test_emit_reads_from_replaced_queue_after_clear() -> None:
    """After clear_audio_queue rebinds output_queue, emit() reads from the new queue."""
    wrapper = _make_wrapper()
    # Put a stale frame on the original queue; it must NOT be returned.
    await wrapper.pipeline.output_queue.put("stale")
    # Simulate console.clear_audio_queue.
    wrapper.output_queue = asyncio.Queue()
    # Put a fresh frame on the new queue.
    await wrapper.output_queue.put("fresh")
    result = await asyncio.wait_for(wrapper.emit(), timeout=1.0)
    assert result == "fresh"
```

Run: failures — current wrapper's `output_queue` is a plain attribute.

- [ ] **Step 2: Implement**

Below the `_clear_queue` setter in `composable_conversation_handler.py`:

```python
@property
def output_queue(self) -> asyncio.Queue[Any]:  # type: ignore[override]
    """Read-through to the pipeline's queue (what emit() drains)."""
    return self.pipeline.output_queue

@output_queue.setter
def output_queue(self, queue: asyncio.Queue[Any]) -> None:
    """Replace the pipeline's queue.

    LocalStream.clear_audio_queue does ``handler.output_queue = asyncio.Queue()``
    to drop queued TTS frames on barge-in. The pipeline owns the read queue,
    so the assignment has to land there or the rebind is a no-op.
    """
    self.pipeline.output_queue = queue
```

Add `import asyncio` and `from typing import Any` if not already present (re-check existing imports).

Remove the now-stale `self.output_queue = pipeline.output_queue` assignment from `__init__` (the getter replaces it).

Run tests. Expected: green for all three new tests; existing 4a `test_emit_pulls_from_output_queue` still green (the test puts an item on `wrapper.pipeline.output_queue` which the getter now reads through — same observable behaviour).

- [ ] **Step 3: Commit**

```
git add src/robot_comic/composable_conversation_handler.py tests/test_composable_conversation_handler.py
git commit -m "fix(wrapper): redirect output_queue read/write through pipeline (#337)"
```

---

## Pre-push checklist

Before the PR:

```
# From repo root, NEVER per-file:
.venv/bin/ruff check
.venv/bin/ruff format --check
.venv/bin/mypy --pretty src/robot_comic/composable_conversation_handler.py \
                       src/robot_comic/handler_factory.py \
                       src/robot_comic/config.py
.venv/bin/pytest tests/ -q
```

If any step is red, fix locally before pushing.

Sanity-check main's CI is green on the current tip commit before opening the PR. If main is red, open a small fix PR first (per the 2026-05-15 feedback memory). The fix PR for #354's ruff format issue (PR #356) is already merged, so main *should* be green — but verify.

## PR creation

After all seven commits land on `claude/phase-4b-factory-dual-path-TsCMu`:

```
git push -u origin claude/phase-4b-factory-dual-path-TsCMu
```

Then create the PR with body summarising the spec's design + success criteria + a "test plan" checklist for the operator to run on a robot:

- [ ] Default `REACHY_MINI_FACTORY_PATH=legacy` still works end-to-end (regression check).
- [ ] `export REACHY_MINI_FACTORY_PATH=composable` on a `(moonshine, llama, elevenlabs)` profile — confirm `journalctl` says `selecting ComposableConversationHandler`.
- [ ] Conversation works end-to-end; persona switching works (4a's `apply_personality` path).
- [ ] Voice switching works (the forwarded calls).
- [ ] Barge-in: start the bot talking, interrupt with a new utterance, confirm audio cuts off (the two property fixes are the precondition for this).
- [ ] Flip back to `legacy`, confirm nothing regressed.

No stacking. No additional PRs ride alongside 4b. Squash-merge with `--delete-branch` once CI is green and operator validates.

## Risks

- **Adapter `start` swaps `_dispatch_completed_transcript`.** `MoonshineSTTAdapter.start` monkey-patches the legacy handler's dispatch method. If the legacy handler's mixin tries to call its *original* dispatch elsewhere (it doesn't today, but watch for that), the patch would have to be reconsidered. Tracked in adapter docstrings; no change needed in 4b.
- **`copy()` cost.** Every WebRTC peer constructs a fresh `LocalSTTLlamaElevenLabsHandler` — the same cost as today's legacy `Stream(handler=…)` cloning. No new cost from the wrapper itself.
- **System-prompt staleness.** `system_prompt` is captured at construction. Persona switches mid-session call `apply_personality()`, which re-seeds via 4a's code. Persona switches across `copy()` calls re-read `get_session_instructions()` because the `_build` closure runs fresh. This matches the legacy behaviour.

## After-merge follow-ups (out of scope for 4b)

- Telemetry: `telemetry.record_llm_duration` from the adapter, separate PR.
- Boot-timeline supporting events (#321): from `start_up()`, separate PR.
- Joke history capture (`llama_base.py:553-568`): from the adapter or pipeline, separate PR.
- History trim (`history_trim.trim_history_in_place`): from the pipeline's `_run_llm_loop_and_speak`, separate PR.
- Echo-guard timestamps (`elevenlabs_tts.py:471-473`): already lives on the legacy handler and survives via delegation — confirm no PR needed.

4c picks up the Chatterbox / Gemini-text triples after these settle.
