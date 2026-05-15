# Phase 4b — Factory Dual Path Behind `REACHY_MINI_FACTORY_PATH`

**Date:** 2026-05-15
**Scope:** New env-var dial in `config.py`; new composable branch inside `handler_factory.py`; two `ComposableConversationHandler` wrapper fixes around barge-in / queue rebinding; tests under `tests/`.
**Epic:** #337 — Pipeline refactor (Option C, incremental retirement)
**Predecessors:** Phase 4a (#355 / c8de597) — `ComposableConversationHandler` wrapper landed but unrouted.
**Successor:** Phase 4c — expand to remaining composable triples (Chatterbox / Gemini text / etc.) and build `ChatterboxTTSAdapter` + `GeminiLLMAdapter`.

## Background

`ComposableConversationHandler` exists on `main` but no factory path returns it. Every operator session still flows through the legacy handler classes (`LocalSTTLlamaElevenLabsHandler`, `GeminiTextChatterboxHandler`, etc.) because `handler_factory.py`'s composable branch builds those directly. 4a's wrapper has zero production exposure; 4b is the first sub-phase where the wrapper is actually returned by the factory and reaches a `LocalStream` / FastRTC `Stream`.

Once the wrapper is routed, two latent gaps from 4a become real:

1. **Barge-in `_clear_queue` is dropped on the floor.** `console.py:261` sets `self.handler._clear_queue = self.clear_audio_queue` on the wrapper. `LocalSTTInputMixin.listener` (`local_stt_realtime.py:538`) calls `self._clear_queue` on the *legacy* `LocalSTTLlamaElevenLabsHandler` instance that lives inside the adapter chain — that instance never receives the callback assignment, so barge-in stops flushing the player.
2. **`output_queue` rebinding skips the pipeline.** `clear_audio_queue` does `self.handler.output_queue = asyncio.Queue()`. In the wrapper that rebinds the `output_queue` *attribute* but `emit()` reads `pipeline.output_queue` directly — so barge-in leaves stale TTS frames queued for emission even on the rare paths that do reach the callback.

Both are barge-in correctness, not the deferred lifecycle-hook gaps (telemetry / boot timeline / joke history / history trim / echo guard). They have to land in 4b — otherwise the moment an operator flips `REACHY_MINI_FACTORY_PATH=composable` on a robot, "stop talking when I talk" silently breaks.

## Goal

Add a top-level config dial `REACHY_MINI_FACTORY_PATH=legacy|composable` (default `legacy`). When set to `composable` *and* the resolved triple is `(moonshine, llama, elevenlabs)`, the factory returns a `ComposableConversationHandler` wrapping the legacy `LocalSTTLlamaElevenLabsHandler` (via the three adapters) plus a `build` closure that re-runs the same construction. Every other configuration — every other triple, every other `pipeline_mode`, and any value other than `composable` — flows through the existing branches unchanged. Fix the two barge-in wiring gaps in the wrapper itself so the composable path is functionally equivalent to the legacy path for barge-in.

## Out of scope (deferred to later sub-phases)

| Item | Sub-phase |
|------|-----------|
| Any triple other than `(moonshine, llama, elevenlabs)` under `composable` mode | 4c |
| `ChatterboxTTSAdapter`, `GeminiLLMAdapter`, `GeminiTTSAdapter` | 4c |
| Sibling `HybridRealtimePipeline` class for `LocalSTT*RealtimeHandler` pair | 4c-tris |
| Flip default to `composable` | 4d |
| Delete legacy handlers + orphan `LocalSTTLlamaGeminiTTSHandler` + test rewrites | 4e |
| Retire `BACKEND_PROVIDER` / `LOCAL_STT_RESPONSE_BACKEND` config dials | 4f |
| Plumbing the deferred lifecycle hooks (telemetry, boot-timeline supporting events, joke history, history trim, echo-guard timestamps) through the wrapper | Per-hook follow-up PRs between 4b and 4d |

The barge-in fixes (`_clear_queue` propagation + `output_queue` redirection) are explicitly *not* deferred — they ride 4b because 4b is the first PR that makes the composable path operator-reachable.

## Design

### 1. New config dial

```python
# src/robot_comic/config.py — near PIPELINE_MODE definitions (~line 316)

FACTORY_PATH_ENV = "REACHY_MINI_FACTORY_PATH"
FACTORY_PATH_LEGACY = "legacy"
FACTORY_PATH_COMPOSABLE = "composable"
FACTORY_PATH_CHOICES: tuple[str, ...] = (FACTORY_PATH_LEGACY, FACTORY_PATH_COMPOSABLE)
DEFAULT_FACTORY_PATH = FACTORY_PATH_LEGACY


def _normalize_factory_path(value: str | None) -> str:
    candidate = (value or "").strip().lower()
    if not candidate:
        return DEFAULT_FACTORY_PATH
    if candidate in FACTORY_PATH_CHOICES:
        return candidate
    logger.warning(
        "Invalid %s=%r. Expected one of: %s. Falling back to %r.",
        FACTORY_PATH_ENV, value, ", ".join(FACTORY_PATH_CHOICES), DEFAULT_FACTORY_PATH,
    )
    return DEFAULT_FACTORY_PATH
```

Read into `Config` at class body:

```python
FACTORY_PATH: str = _normalize_factory_path(os.getenv(FACTORY_PATH_ENV))
```

Refresh in `refresh_runtime_config_from_env()`:

```python
config.FACTORY_PATH = _normalize_factory_path(os.getenv(FACTORY_PATH_ENV))
```

### 2. Factory routing change

`handler_factory.py`'s `(moonshine, elevenlabs, llama)` branch (currently `lines 186-196`) gets one new gate:

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
        # Default: legacy concrete handler.
        from robot_comic.llama_elevenlabs_tts import LocalSTTLlamaElevenLabsHandler
        logger.info(
            "HandlerFactory: selecting LocalSTTLlamaElevenLabsHandler (%s → %s, llm=%s)",
            input_backend, output_backend, LLM_BACKEND_LLAMA,
        )
        return LocalSTTLlamaElevenLabsHandler(**handler_kwargs)
```

The new private helper:

```python
def _build_composable_llama_elevenlabs(**handler_kwargs: Any) -> "ComposableConversationHandler":
    """Build (moonshine, llama, elevenlabs) as a ComposablePipeline wrapper.

    Constructs the legacy LocalSTTLlamaElevenLabsHandler (the adapters need a
    concrete handler to delegate into), wraps it with the three Phase 3
    adapters, composes them into a ComposablePipeline seeded with the current
    session instructions, and returns the wrapper. The injected ``build``
    closure re-runs the same construction so FastRTC's ``copy()`` per-peer
    cloning gets fresh state on every call.
    """
    from robot_comic.adapters import (
        ElevenLabsTTSAdapter, LlamaLLMAdapter, MoonshineSTTAdapter,
    )
    from robot_comic.composable_conversation_handler import ComposableConversationHandler
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.llama_elevenlabs_tts import LocalSTTLlamaElevenLabsHandler
    from robot_comic.prompts import get_session_instructions

    def _build() -> ComposableConversationHandler:
        legacy = LocalSTTLlamaElevenLabsHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = LlamaLLMAdapter(legacy)
        tts = ElevenLabsTTSAdapter(legacy)
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

Notes:
- **Single legacy instance per pipeline.** All three adapters wrap the *same* `LocalSTTLlamaElevenLabsHandler` — that's the existing 4a-adapter convention (the LLM and TTS halves share the same llama / ElevenLabs client; the STT half is the mixin baked into the same class).
- **No `tool_dispatcher` plumbed.** `LlamaLLMAdapter.chat` ignores its `tools` arg and dispatches via the legacy `_call_llm` path; the orchestrator's `tool_dispatcher` callback stays unset, exactly as today's adapter contract documents.
- **`copy()` re-runs `_build`.** Every WebRTC peer gets a fresh legacy handler instance and a fresh pipeline (independent history, stop event, adapter state). The closure captures `handler_kwargs` by reference — the dict is built once at the outer call and reused, which matches what FastRTC expects (configuration is the same across peers, only state diverges).

### 3. `ComposableConversationHandler` barge-in fixes

Two small surgical changes inside the wrapper. Both are scoped to the composable path, so the legacy path is untouched.

**Fix A — propagate `_clear_queue` to the underlying TTS handler.** The legacy `LocalSTTInputMixin` listener calls `self._clear_queue()` on whatever instance it's mixed into. In the composable path that instance is the `LocalSTTLlamaElevenLabsHandler` held as `self._tts_handler`. So a `_clear_queue` setter on the wrapper has to forward to the TTS handler too:

```python
# composable_conversation_handler.py — replace the plain attribute init

self.output_queue = pipeline.output_queue
self._clear_queue: Callable[[], None] | None = None

@property
def _clear_queue(self) -> Callable[[], None] | None:  # type: ignore[no-redef]
    return self.__clear_queue

@_clear_queue.setter
def _clear_queue(self, callback: Callable[[], None] | None) -> None:
    """Mirror the queue-flush callback onto the underlying TTS handler so the
    Moonshine listener's barge-in fires through console.clear_audio_queue."""
    self.__clear_queue = callback
    # The underlying legacy handler is what the LocalSTTInputMixin listener
    # closes over. Propagate the callback unless it already has its own.
    if self._tts_handler is not None:
        self._tts_handler._clear_queue = callback
```

(Implementation detail: storing in `self.__clear_queue` with the property dance avoids the recursion that `self._clear_queue = …` inside the setter would cause. The ABC has `_clear_queue` as a class-level annotation, not a descriptor, so descriptor override is well-defined.)

**Fix B — clearing `output_queue` must clear the pipeline's queue.** The simplest fix is to delete the wrapper's `output_queue` alias attribute and replace it with a property that reads and writes through to `pipeline.output_queue`:

```python
@property
def output_queue(self) -> asyncio.Queue[Any]:  # type: ignore[override]
    return self.pipeline.output_queue

@output_queue.setter
def output_queue(self, queue: asyncio.Queue[Any]) -> None:
    """Mirror the clear-queue path: rebinding the wrapper's queue replaces
    the pipeline's queue so emit() sees a fresh empty queue."""
    self.pipeline.output_queue = queue
```

After these two fixes:
- `clear_audio_queue` from `console.py:1309` rebinds `wrapper.output_queue` → property setter → `pipeline.output_queue` swap → next `emit()` reads from the fresh empty queue.
- `LocalSTTInputMixin.listener` on the wrapped legacy handler calls `self._clear_queue()` → the callback we propagated through Fix A → `clear_audio_queue()` (which then calls the wrapper's `output_queue` setter, cycle closes).

Both fixes are observable through unit tests that mock `console.clear_audio_queue` and a fake legacy handler — no fastrtc / hardware needed.

### 4. Resolution of the open question raised in the 4a session

| Question | Resolution in 4b |
|---------|------------------|
| Does `clear_audio_queue` need a TODO or a real fix? | Real fix in 4b. The bug is unique to the wrapper (legacy handlers don't have a pipeline; their `output_queue` *is* the read queue) and ships the moment 4b is operator-reachable. |
| Should `_clear_queue` propagation live in the wrapper or the factory? | Wrapper. The factory passes `tts_handler` into the wrapper at construction; the wrapper is the only place that knows the TTS handler is the listener's host. |

## Files Changed

| File | Change |
|------|--------|
| `src/robot_comic/config.py` | Add `FACTORY_PATH_ENV` constants, `_normalize_factory_path`, `Config.FACTORY_PATH`, `refresh_runtime_config_from_env` update. ~25 LOC. |
| `src/robot_comic/handler_factory.py` | Add `_build_composable_llama_elevenlabs` helper + one new branch in the existing llama+elevenlabs path. ~50 LOC. |
| `src/robot_comic/composable_conversation_handler.py` | Replace plain attributes with `output_queue` and `_clear_queue` property/setter pairs. ~15 LOC delta. |
| `tests/test_handler_factory_factory_path.py` | NEW — assertions for legacy vs composable dispatch. |
| `tests/test_composable_conversation_handler.py` | Add tests for the two property/setter fixes. ~30 LOC delta. |
| `tests/test_config.py` *(or new `test_config_factory_path.py`)* | NEW or appended — `_normalize_factory_path` happy/invalid paths and env-var read. |
| `.env.example` | One-line addition documenting `REACHY_MINI_FACTORY_PATH=legacy`. |

No changes to `composable_pipeline.py`, the adapters, the ABC, or `main.py`. The factory is the only call site that sees `FACTORY_PATH`.

## Success Criteria

- `config.FACTORY_PATH` reads `REACHY_MINI_FACTORY_PATH`, normalises legacy/composable, defaults to `legacy`, warns on invalid input.
- `HandlerFactory.build(moonshine, elevenlabs, deps, pipeline_mode=composable)` with `FACTORY_PATH=legacy` (default) returns a `LocalSTTLlamaElevenLabsHandler` — bit-for-bit current behaviour.
- Same call with `FACTORY_PATH=composable` returns a `ComposableConversationHandler` wrapping a `ComposablePipeline` whose adapters all wrap one `LocalSTTLlamaElevenLabsHandler` instance.
- `wrapper.copy()` returns a *different* `ComposableConversationHandler` whose adapters wrap a *different* `LocalSTTLlamaElevenLabsHandler` instance (no aliasing).
- `wrapper._clear_queue = some_callable` also assigns `wrapper._tts_handler._clear_queue = some_callable`.
- `wrapper.output_queue = fresh_queue` replaces `wrapper.pipeline.output_queue`; subsequent `wrapper.emit()` reads from the new queue.
- No other triple's routing changes: `(moonshine, chatterbox)`, `(moonshine, gemini_tts)`, `(moonshine, openai_realtime_output)`, `(moonshine, hf_output)`, all three bundled-realtime modes, and Gemini-LLM variants all return the same classes as before, regardless of `FACTORY_PATH`.
- New ruff check / format / mypy / pytest all green from repo root.
- Existing 1669 tests still pass (modulo the one container-environment failure in `test_cleanup_worktrees_script.py` which is a git-signing quirk unrelated to this work).

## Test Plan

### Unit tests for the config dial

| Test | What it asserts |
|------|-----------------|
| `test_factory_path_default_is_legacy` | `_normalize_factory_path(None) == FACTORY_PATH_LEGACY`; ditto for empty/whitespace. |
| `test_factory_path_normalises_known_values` | `_normalize_factory_path("composable")`, `"COMPOSABLE"`, `"  legacy  "` all normalise correctly. |
| `test_factory_path_invalid_falls_back_to_legacy_with_warning` | Returns legacy, emits a `logger.warning` mentioning the choices. |
| `test_factory_path_read_from_env` | With `REACHY_MINI_FACTORY_PATH=composable` set, `config.FACTORY_PATH == "composable"` after `refresh_runtime_config_from_env`. |

### Factory dispatch tests (`test_handler_factory_factory_path.py`)

| Test | What it asserts |
|------|-----------------|
| `test_legacy_path_returns_legacy_handler_for_llama_elevenlabs` | `FACTORY_PATH=legacy` + `(moonshine, elevenlabs, llama)` → `LocalSTTLlamaElevenLabsHandler`. Patches the class with a `_fake_cls` like the existing factory tests do. |
| `test_composable_path_returns_wrapper_for_llama_elevenlabs` | `FACTORY_PATH=composable` + same triple → `isinstance(result, ComposableConversationHandler)`; `result.pipeline` is a `ComposablePipeline`; `result._tts_handler` is a `LocalSTTLlamaElevenLabsHandler` (fake-patched). |
| `test_composable_path_wires_three_adapters` | Inspect `result.pipeline.stt / .llm / .tts` — types are `MoonshineSTTAdapter`, `LlamaLLMAdapter`, `ElevenLabsTTSAdapter` and they all wrap the same legacy handler instance. |
| `test_composable_path_seeds_system_prompt_from_get_session_instructions` | Monkeypatch `prompts.get_session_instructions` → assert `pipeline._conversation_history[0]["content"]` matches the patched value. |
| `test_copy_constructs_fresh_legacy_handler` | `result.copy()` returns a different wrapper; its `_tts_handler` is a different `LocalSTTLlamaElevenLabsHandler` instance from the original. |
| `test_composable_path_only_affects_llama_elevenlabs_triple` | `FACTORY_PATH=composable` + `(moonshine, chatterbox)` still returns `LocalSTTChatterboxHandler` (parametrised over the four other moonshine triples). |
| `test_composable_path_ignored_in_bundled_realtime_modes` | `FACTORY_PATH=composable` + `pipeline_mode=hf_realtime|openai_realtime|gemini_live` still returns the respective bundled handler. |
| `test_composable_path_with_gemini_llm_unchanged` | `FACTORY_PATH=composable` + `(moonshine, elevenlabs, gemini)` returns `GeminiTextElevenLabsHandler` (4c will migrate this; 4b leaves it alone). |

### Wrapper barge-in fixes (`test_composable_conversation_handler.py` additions)

| Test | What it asserts |
|------|-----------------|
| `test_clear_queue_assignment_propagates_to_tts_handler` | After `wrapper._clear_queue = cb`, both `wrapper._clear_queue is cb` and `wrapper._tts_handler._clear_queue is cb`. |
| `test_clear_queue_assignment_handles_none` | `wrapper._clear_queue = None` clears both attributes. |
| `test_output_queue_setter_replaces_pipeline_queue` | `fresh = asyncio.Queue(); wrapper.output_queue = fresh; assert wrapper.pipeline.output_queue is fresh`. |
| `test_output_queue_getter_returns_pipeline_queue` | After `wrapper.pipeline.output_queue = q2`, `wrapper.output_queue is q2` (the wrapper does not cache). |
| `test_emit_reads_from_replaced_queue_after_clear` | Put a sentinel on the original queue; rebind `wrapper.output_queue = fresh`; `emit()` blocks until something is on the fresh queue — confirms the read path follows the rebind. |

All wrapper tests use the existing `_make_wrapper()` helper pattern, with a fake `_tts_handler` MagicMock.

### What we don't add tests for

- Hardware barge-in latency / audio path. The 4b PR can't validate that without a robot; the wrapper-side fixes are the necessary precondition, and the operator will validate end-to-end on the robot after merge (Option C's "operator validates each triple on hardware" milestone).
- `apply_personality` re-seeding interaction with the new wiring. Already covered by 4a tests; unchanged in 4b.
- Lifecycle-hook gaps (telemetry, supporting events, joke history, history trim, echo guard). Each gets its own follow-up PR per the plan.

## Migration notes

- Default behaviour is unchanged. Existing `.env` files keep working. Operators opt into the new path with `export REACHY_MINI_FACTORY_PATH=composable`.
- Reverting is one env-var flip — no code change, no rebuild.
- 4c onward will keep `FACTORY_PATH` as the single gate; expanding to other triples means adding more `if FACTORY_PATH == composable:` branches inside the existing per-triple blocks of `handler_factory.py`. The dial itself doesn't change shape.
- 4d (flip default) will change `DEFAULT_FACTORY_PATH = FACTORY_PATH_COMPOSABLE` in one line; 4e/4f then delete the dial when all triples are migrated and `BACKEND_PROVIDER` is retired.
