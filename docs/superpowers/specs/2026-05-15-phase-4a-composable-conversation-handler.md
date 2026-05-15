# Phase 4a — `ComposableConversationHandler` Wrapper

**Date:** 2026-05-15
**Scope:** New file `src/robot_comic/composable_conversation_handler.py`; tests under `tests/`
**Epic:** #337 — Pipeline refactor (Option C, incremental retirement)
**Predecessors:** Phase 0 (#341 PIPELINE_MODE dial), Phase 1 (#342 STT/LLM/TTS Protocols), Phase 2 (#343 `ComposablePipeline`), Phase 3a/b/c (#344/#350/#346 adapters)
**Successor:** Phase 4b — factory dual path behind `REACHY_MINI_FACTORY_PATH` flag for `(moonshine, llama, elevenlabs)`

## Background

`ComposablePipeline` (Phase 2) and the three adapter classes (Phase 3) exist on `main` but no factory path routes to them. The factory still returns legacy concrete handlers (`LocalSTTLlamaElevenLabsHandler` et al.) that inherit from `ConversationHandler` and implement the ABC surface FastRTC + admin routes depend on. `ComposablePipeline` does not implement `ConversationHandler` — it has its own `start_up/shutdown/feed_audio/reset_history` surface and is missing seven ABC methods (`copy`, `receive`, `emit`, `apply_personality`, `change_voice`, `get_available_voices`, `get_current_voice`).

This sub-phase closes that gap with a single wrapper class. 4b then teaches the factory to return the wrapper behind a flag.

## Goal

Introduce `ComposableConversationHandler(ConversationHandler)` that wraps one `ComposablePipeline` instance and forwards voice/personality calls to the underlying legacy TTS handler that the TTS adapter wraps. After 4a, the factory could route to it — but does not yet (that's 4b).

## Out of scope (deferred to later sub-phases)

| Item | Sub-phase |
|------|-----------|
| Factory routing changes / dual-path flag | 4b |
| `ChatterboxTTSAdapter`, `GeminiLLMAdapter`, `GeminiTTSAdapter` | 4c |
| Sibling `HybridRealtimePipeline` class for `LocalSTT*RealtimeHandler` pair | 4c-tris |
| Flip default to composable | 4d |
| Delete legacy handlers + orphan `LocalSTTLlamaGeminiTTSHandler` + test rewrites | 4e |
| Retire `BACKEND_PROVIDER` / `LOCAL_STT_RESPONSE_BACKEND` config dials | 4f |
| Plumbing the missing lifecycle hooks (telemetry, supporting events, joke history, history trim, echo guard) through the wrapper | Per-hook follow-ups between 4b and 4d |

The lifecycle-hook gaps are real — `ComposablePipeline` doesn't fire `telemetry.record_llm_duration`, the four supporting events from boot-timeline (#321), `record_joke_history` (`llama_base.py:553-568`), `history_trim.trim_history_in_place`, or the `_speaking_until` echo-guard timestamp updates. 4a does NOT close these; each is a small follow-up PR with hardware validation. 4a inserts `# TODO(phase4-lifecycle): …` markers at the integration points so they're easy to find.

## Design

### Public class surface

```python
class ComposableConversationHandler(ConversationHandler):
    def __init__(
        self,
        pipeline: ComposablePipeline,
        *,
        tts_handler: ConversationHandler,
        deps: ToolDependencies,
        build: Callable[[], "ComposableConversationHandler"],
    ) -> None: ...

    # ConversationHandler ABC methods
    def copy(self) -> "ComposableConversationHandler": ...
    async def start_up(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def receive(self, frame: AudioFrame) -> None: ...
    async def emit(self) -> HandlerOutput: ...
    async def apply_personality(self, profile: str | None) -> str: ...
    async def get_available_voices(self) -> list[str]: ...
    def get_current_voice(self) -> str: ...
    async def change_voice(self, voice: str) -> str: ...
```

**Wrapper holds:**
- `self.pipeline: ComposablePipeline` — the orchestrator. Source of truth for conversation state and audio I/O.
- `self._tts_handler` — the underlying legacy TTS handler (accessed via the TTS adapter's `.handler` attribute, but the wrapper takes it directly to keep the dependency one-way and avoid leaking the adapter type). Holds the voice/personality methods we forward to.
- `self.deps: ToolDependencies` — required by the `ConversationHandler` ABC's typed attribute; the wrapper does not use them directly but exposes them so tool wiring in 4b can plug in via the pipeline's `tool_dispatcher`.
- `self._build: Callable[[], ComposableConversationHandler]` — closure for `copy()` to reconstruct a fresh wrapper + fresh pipeline. The factory will provide this closure in 4b.

**Wrapper does NOT hold:** the adapter instances or the LLM/STT legacy handlers. Those live inside `pipeline.stt/llm/tts`. The wrapper exposes only the voice surface.

### Method semantics

| Method | Implementation |
|--------|----------------|
| `start_up()` | `await self.pipeline.start_up()` — blocks until shutdown, matching ABC contract. |
| `shutdown()` | `await self.pipeline.shutdown()` |
| `receive(frame)` | `await self.pipeline.feed_audio(frame)` |
| `emit()` | `await wait_for_item(self.pipeline.output_queue)` — same `fastrtc.wait_for_item` pattern as `ElevenLabsTTSResponseHandler.emit()`. Imported lazily to avoid pulling gradio at boot. |
| `copy()` | `return self._build()` — fresh wrapper, fresh pipeline state, no aliasing. |
| `apply_personality(profile)` | Calls `set_custom_profile(profile)` (matches legacy behavior in `elevenlabs_tts.py:486`), then `self.pipeline.reset_history(keep_system=False)`, then re-seeds: `self.pipeline._conversation_history.append({"role": "system", "content": get_session_instructions()})`. Returns `f"Applied personality {profile!r}. Conversation history reset."` on success, `f"Failed to apply personality: {exc}"` on exception. |
| `get_available_voices()` | `return await self._tts_handler.get_available_voices()` |
| `get_current_voice()` | `return self._tts_handler.get_current_voice()` |
| `change_voice(voice)` | `return await self._tts_handler.change_voice(voice)` |

### Resolution of memo §6 open questions for 4a

| Question | Resolution in 4a |
|---------|------------------|
| §6-#1 history persistence on persona switch | `apply_personality()` calls `pipeline.reset_history(keep_system=False)` and re-seeds system message. Matches existing handlers. |
| §6-#3 system prompt staleness | Re-seeded in `apply_personality()`. Per-turn refresh deferred: leave a TODO referencing the open question. Operator can revisit if profile-mid-conversation refresh becomes a real need. |
| §6-#4 `change_voice` design | Wrapper keeps typed reference to legacy TTS handler and forwards. No Protocol churn. Memo's recommended (b). |
| §6-#5 lifecycle hooks | Flagged with TODO markers, deferred to follow-up sub-phases. Each hook gets its own small PR. |
| §6-#8 `copy()` semantics | Constructor takes a `build` closure; `copy()` invokes it for a fresh wrapper + pipeline. |

### Why a `build` closure for `copy()`

FastRTC clones the handler per WebRTC peer; cloning must produce a fully independent pipeline (separate `_conversation_history`, separate `_stop_event`, separate adapter instances if those carry per-session state). Sharing any of `pipeline.stt/llm/tts` across copies would couple sessions.

Reproducing the construction in `copy()` requires knowing how to build the adapter chain — which is factory-level knowledge. Rather than duplicate that knowledge inside the wrapper (would tightly couple wrapper to `LlamaLLMAdapter`/`ElevenLabsTTSAdapter`/`MoonshineSTTAdapter` concrete classes), we accept a `build` callable from the constructor. The factory (4b) will provide:

```python
def _build_wrapper() -> ComposableConversationHandler:
    stt_handler = LocalSTTInputMixinHostHandler(deps=deps)  # whatever the factory does today
    stt = MoonshineSTTAdapter(stt_handler)
    tts_handler = LlamaElevenLabsTTSResponseHandler(deps=deps, ...)
    tts = ElevenLabsTTSAdapter(tts_handler)
    llm = LlamaLLMAdapter(tts_handler)  # llama uses the same handler
    pipeline = ComposablePipeline(stt, llm, tts, ...)
    return ComposableConversationHandler(
        pipeline=pipeline,
        tts_handler=tts_handler,
        deps=deps,
        build=_build_wrapper,
    )
```

4a's tests provide a tiny in-test `build` closure for `copy()` assertions; the factory side is 4b.

## Files Changed

| File | Change |
|------|--------|
| `src/robot_comic/composable_conversation_handler.py` | NEW. One class `ComposableConversationHandler`. ~120 LOC including the TODO markers. |
| `tests/test_composable_conversation_handler.py` | NEW. Unit + small integration tests, all using mocked adapters. |

No changes to `handler_factory.py`, `composable_pipeline.py`, adapters, or legacy handlers. 4b handles factory wiring.

## Success Criteria

- `ComposableConversationHandler` instantiates with a `ComposablePipeline` and a legacy TTS handler stub.
- All nine ABC methods implemented and individually unit-tested.
- `copy()` produces an independent instance (different object id; mutations to one's pipeline history don't affect the other).
- `apply_personality(profile)` resets history and re-seeds with `get_session_instructions()`.
- Voice methods forward to the underlying TTS handler with no in-wrapper logic.
- `receive` → `feed_audio` → STT-completed callback → LLM mock → TTS mock → `emit` round-trip works in an integration test using stubbed adapters.
- New `ruff check`, `ruff format`, `mypy`, `pytest` all green.
- No changes to existing tests or production code.

## Test Plan

| Test | What it asserts |
|------|-----------------|
| `test_wrapper_implements_conversation_handler_abc` | `isinstance(wrapper, ConversationHandler)`; class is not abstract. |
| `test_start_up_delegates_to_pipeline` | `start_up()` awaits `pipeline.start_up()`. |
| `test_shutdown_delegates_to_pipeline` | `shutdown()` awaits `pipeline.shutdown()`. |
| `test_receive_forwards_to_feed_audio` | `await wrapper.receive(frame)` calls `pipeline.feed_audio(frame)`. |
| `test_emit_pulls_from_output_queue` | Put item on `pipeline.output_queue`; `emit()` returns it. |
| `test_get_current_voice_delegates` | Returns whatever the stub TTS handler returns. |
| `test_get_available_voices_delegates` | Async forward to stub. |
| `test_change_voice_delegates` | Async forward to stub; preserves return value. |
| `test_apply_personality_resets_history_and_reseeds` | After call, `pipeline._conversation_history == [{"role": "system", "content": <new instructions>}]`. |
| `test_apply_personality_returns_success_message` | On success, returns the documented success string. |
| `test_apply_personality_returns_failure_message_on_error` | Stub `set_custom_profile` to raise; return string mentions the exception. |
| `test_copy_returns_new_instance_from_build_closure` | `id(wrapper.copy()) != id(wrapper)`; the build closure was called once. |
| `test_copy_does_not_share_pipeline_state` | Mutate original's history; copy's history is independent. |
| `test_integration_transcript_to_audio_frame` | End-to-end with mocked `STTBackend`/`LLMBackend`/`TTSBackend` exercising the full transcript→LLM→TTS→emit path through the wrapper. |

All tests use `pytest-asyncio` and `unittest.mock` patterns matching the existing `tests/adapters/test_*.py` files.
