# Phase 4c.1 — ChatterboxTTSAdapter + factory routing for `(moonshine, chatterbox, llama)`

**Date:** 2026-05-15
**Scope:** New `ChatterboxTTSAdapter` (~120 LOC); new `_build_composable_llama_chatterbox` helper in `handler_factory.py`; one new branch in the existing `(moonshine, chatterbox)` block; tests under `tests/`.
**Epic:** #337 — Pipeline refactor (Option C, incremental retirement)
**Predecessors:** Phase 4b (#359 / 8f94691) — `FACTORY_PATH` dial landed; `(moonshine, llama, elevenlabs)` is routed through `ComposableConversationHandler` under `FACTORY_PATH=composable`.
**Successors:** Phase 4c.2 — `GeminiLLMAdapter` + `(moonshine, chatterbox, gemini)`. Phase 4c.3–5 cover the remaining triples.

## Background

`FACTORY_PATH=composable` is operator-reachable, but only for one triple. Every other moonshine triple still flows through legacy concrete handlers. The next-cheapest triple to migrate is `(moonshine, chatterbox, llama)` — it is the second-most-common production pipeline (chatterbox is the primary local TTS) and shares two of three adapters with 4b (`MoonshineSTTAdapter`, `LlamaLLMAdapter`). The only new piece is a `ChatterboxTTSAdapter`.

The legacy handler `LocalSTTChatterboxHandler` is composed of:

- `LocalSTTInputMixin` — moonshine listener (already adapted by `MoonshineSTTAdapter`).
- `ChatterboxTTSResponseHandler(BaseLlamaResponseHandler)` — llama-server LLM (already adapted by `LlamaLLMAdapter`) + chatterbox-server TTS.

The TTS half lives in `ChatterboxTTSResponseHandler._synthesize_and_enqueue(response_text, tts_start, target_queue)`. Even though the base-class signature names a `target_queue` parameter, the concrete chatterbox method ignores it and pushes frames to `self.output_queue` directly (see `chatterbox_tts.py:352, 371, 374`). That makes the adapter pattern identical to `ElevenLabsTTSAdapter`: substitute `handler.output_queue` for a temp queue, await `_synthesize_and_enqueue`, yield items from the temp queue, restore on `finally`.

## Goal

Add `ChatterboxTTSAdapter` and route `(moonshine, chatterbox, llama)` through `ComposableConversationHandler` when `REACHY_MINI_FACTORY_PATH=composable`. Default `legacy` keeps `LocalSTTChatterboxHandler` — zero behaviour change for unset / unchanged operators.

## Out of scope (deferred)

| Item | Sub-phase |
|------|-----------|
| `(moonshine, chatterbox, gemini)` — `GeminiTextChatterboxHandler` | 4c.2 (needs `GeminiLLMAdapter`) |
| `(moonshine, elevenlabs, gemini)` | 4c.3 |
| `(moonshine, elevenlabs, gemini-fallback)` — `LocalSTTGeminiElevenLabsHandler` | 4c.4 |
| `(moonshine, gemini_tts)` — `LocalSTTGeminiTTSHandler` | 4c.5 (needs `GeminiTTSAdapter`) |
| Sibling `HybridRealtimePipeline` for `LocalSTT*RealtimeHandler` | 4c-tris |
| Flip default to `composable` | 4d |
| Delete legacy handlers | 4e |
| Retire `BACKEND_PROVIDER` | 4f |
| Lifecycle hooks: telemetry, boot-timeline events (#321), joke history, history trim, echo-guard timestamps | Per-hook follow-up PRs |
| First-audio marker / tag plumbing through `LLMResponse.metadata` | Tracked in 4b spec as a known TODO; same TODO carries over here |

## Design

### 1. `ChatterboxTTSAdapter`

New file `src/robot_comic/adapters/chatterbox_tts_adapter.py`. Mirrors `ElevenLabsTTSAdapter` exactly:

```python
class ChatterboxTTSAdapter:
    """Adapter exposing ``ChatterboxTTSResponseHandler`` as ``TTSBackend``."""

    def __init__(self, handler: "ChatterboxTTSResponseHandler") -> None:
        self._handler = handler

    async def prepare(self) -> None:
        await self._handler._prepare_startup_credentials()

    async def synthesize(
        self,
        text: str,
        tags: tuple[str, ...] = (),
    ) -> AsyncIterator[AudioFrame]:
        # tags ignored — chatterbox handler does not consume them today
        # (it uses persona-driven tag translation via translate()).
        temp_queue: asyncio.Queue[Any] = asyncio.Queue()
        original_queue = self._handler.output_queue
        self._handler.output_queue = temp_queue

        async def _stream_and_signal() -> None:
            try:
                await self._handler._synthesize_and_enqueue(text)
            finally:
                try:
                    await temp_queue.put(_STREAM_DONE)
                except Exception:
                    pass

        task = asyncio.create_task(_stream_and_signal(), name="chatterbox-tts-adapter")
        try:
            while True:
                item = await temp_queue.get()
                if item is _STREAM_DONE:
                    break
                # Chatterbox emits (sample_rate, frame) PCM tuples just like
                # ElevenLabs; AdditionalOutputs items go through the same path.
                if isinstance(item, tuple) and len(item) == 2:
                    sample_rate, frame = item
                    yield AudioFrame(samples=frame, sample_rate=sample_rate)
                # Non-tuple items (e.g. AdditionalOutputs error sentinels)
                # are dropped on the floor — the legacy handler emits them
                # for error reporting; the adapter's caller has no protocol
                # channel for them yet.
            await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            self._handler.output_queue = original_queue

    async def shutdown(self) -> None:
        http = getattr(self._handler, "_http", None)
        if http is not None:
            try:
                await http.aclose()
            except Exception as exc:
                logger.warning("ChatterboxTTSAdapter shutdown: aclose() raised: %s", exc)
            self._handler._http = None
```

Key invariants:

- **Same legacy handler instance is wrapped by all three adapters** (`MoonshineSTTAdapter`, `LlamaLLMAdapter`, `ChatterboxTTSAdapter`), exactly like the 4b factory does.
- **Auto-gain / target-dBFS knobs continue to work** because the adapter delegates straight to `_synthesize_and_enqueue` → `_call_chatterbox_tts` → `_wav_to_pcm(..., auto_gain=self._auto_gain_enabled, target_dbfs=self._target_dbfs)`. Those properties read from `config.py` on the legacy handler instance. The adapter never intercepts that chain.
- **`tags` argument is accepted for Protocol compliance and dropped.** Chatterbox doesn't honour external delivery tags; it derives per-segment exaggeration/cfg-weight from `chatterbox_tag_translator.translate()` driven by the persona. The adapter logs no warning (parity with legacy behaviour where the orchestrator never passed tags either).
- **`AdditionalOutputs` items are dropped.** The legacy chatterbox handler pushes an `AdditionalOutputs({"role": "assistant", "content": "[TTS error]"})` sentinel when no audio is produced. The Protocol has no metadata channel for these yet; dropping is the same behaviour as letting `wait_for_item` skip the frame on the legacy path (the wrapper's `emit()` would surface it, but the orchestrator-level pipeline doesn't have an analogous path until lifecycle-hook PRs land). This matches the `ElevenLabsTTSAdapter` TODO documented in its module docstring.
- **No `tags_list` translation.** Unlike ElevenLabs whose `_stream_tts_to_queue(text, first_audio_marker, tags)` takes tags by-position, `_synthesize_and_enqueue(response_text, tts_start, target_queue)` doesn't accept tags at all. The adapter therefore drops them silently rather than forwarding `None`.

### 2. `_build_composable_llama_chatterbox` in `handler_factory.py`

Mirror the existing `_build_composable_llama_elevenlabs` helper. The factory branch lives inside the existing `LLM_BACKEND=llama` + `AUDIO_OUTPUT_CHATTERBOX` fallthrough — but today there is no explicit branch (the llama+chatterbox triple falls through to the lower `AUDIO_OUTPUT_CHATTERBOX` block at line 252 because `ChatterboxTTSResponseHandler` is already `BaseLlamaResponseHandler`-derived, so the factory's llama-LLM gate at line 188 doesn't have a chatterbox arm).

Two options:

A. **Add an explicit llama+chatterbox gate at the top** (alongside the llama+elevenlabs gate in lines 188–207), so the composable check happens *before* the fallthrough.
B. **Add the composable check to the existing line 252 chatterbox branch.**

Option A is the right pattern — it mirrors 4b's structure exactly and keeps each `_llm_backend == LLM_BACKEND_LLAMA` arm side-by-side. Option B would split the composable gate across two locations and make 4c.2 (the gemini-chatterbox triple) harder to add cleanly.

The new factory branch:

```python
if _llm_backend == LLM_BACKEND_LLAMA:
    if output_backend == AUDIO_OUTPUT_ELEVENLABS:
        ...  # existing 4b branch
    if output_backend == AUDIO_OUTPUT_CHATTERBOX:
        if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
            logger.info(
                "HandlerFactory: selecting ComposableConversationHandler "
                "(%s → %s, llm=%s, factory_path=composable)",
                input_backend, output_backend, LLM_BACKEND_LLAMA,
            )
            return _build_composable_llama_chatterbox(**handler_kwargs)
        # else: fall through to legacy LocalSTTChatterboxHandler branch below
```

The helper:

```python
def _build_composable_llama_chatterbox(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, chatterbox, llama) pipeline."""
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

### 3. Resolution of open design questions from the briefing

| Question | Resolution |
|----------|-----------|
| How does `ChatterboxTTSAdapter` surface the auto-gain / target-dBFS knobs? | It does not surface them explicitly. The legacy handler reads them from `config.py` via properties (`_auto_gain_enabled`, `_target_dbfs`, `_gain`); the adapter delegates straight into `_synthesize_and_enqueue` which calls `_call_chatterbox_tts` which uses those properties. No change to the adapter is needed. |
| Does `ChatterboxTTSResponseHandler` expose `change_voice` / `get_available_voices` / `get_current_voice`? | Yes — see `chatterbox_tts.py:308–326`. The wrapper's voice methods (`composable_conversation_handler.py:141–151`) forward to `_tts_handler` which is the legacy handler instance, so voice switching keeps working without adapter changes. |
| First-audio marker and tags TODO | Parity TODO with `ElevenLabsTTSAdapter`. Document the gap in the module docstring; defer to a future PR that broadens `LLMResponse.metadata` plumbing. |

## Files Changed

| File | Change |
|------|--------|
| `src/robot_comic/adapters/chatterbox_tts_adapter.py` | NEW — ~120 LOC, mirrors `ElevenLabsTTSAdapter`. |
| `src/robot_comic/adapters/__init__.py` | EDIT — export `ChatterboxTTSAdapter`. |
| `src/robot_comic/handler_factory.py` | EDIT — new branch in the llama-LLM gate + `_build_composable_llama_chatterbox` helper. ~45 LOC delta. |
| `tests/adapters/test_chatterbox_tts_adapter.py` | NEW — mirrors `test_elevenlabs_tts_adapter.py`: stub handler + prepare / synthesize / queue isolation / shutdown / protocol tests. |
| `tests/test_handler_factory_factory_path.py` | EDIT — add composable-path tests for `(moonshine, chatterbox, llama)`; loosen the existing parametrised "other triples unchanged" test to drop chatterbox from the migrated list. |

No changes to `composable_pipeline.py`, `composable_conversation_handler.py`, the ABC, or `main.py`.

## Success Criteria

- `FACTORY_PATH=legacy` (default) + `(moonshine, chatterbox, llama)` → `LocalSTTChatterboxHandler` (bit-for-bit current behaviour).
- `FACTORY_PATH=composable` + same triple → `ComposableConversationHandler` whose pipeline holds `MoonshineSTTAdapter`, `LlamaLLMAdapter`, `ChatterboxTTSAdapter` — all wrapping a single `LocalSTTChatterboxHandler` instance.
- `wrapper.copy()` returns a different wrapper whose `_tts_handler` is a different `LocalSTTChatterboxHandler` instance.
- Voice switching (`change_voice`, `get_available_voices`, `get_current_voice`) works through the wrapper.
- `(moonshine, chatterbox, gemini)` and every other triple still returns its legacy concrete class.
- `(moonshine, llama, elevenlabs)` from 4b still works.
- Bundled-realtime modes ignore `FACTORY_PATH`.
- `ruff check` / `ruff format --check` / `mypy --pretty` / `pytest tests/ -q` all green from the repo root.

## Test Plan

### Unit tests for `ChatterboxTTSAdapter` (`tests/adapters/test_chatterbox_tts_adapter.py`)

Mirrors `test_elevenlabs_tts_adapter.py` task-for-task:

| Test | Asserts |
|------|---------|
| `test_prepare_calls_handler_prepare` | `adapter.prepare()` calls `_prepare_startup_credentials`. |
| `test_synthesize_yields_one_audio_frame_per_pushed_item` | Each `(sample_rate, ndarray)` tuple → one `AudioFrame`. |
| `test_synthesize_forwards_text_to_handler` | `_synthesize_and_enqueue` receives the text. |
| `test_synthesize_does_not_leak_into_handlers_original_output_queue` | After the generator drains, `handler.output_queue` is the original queue and is empty. |
| `test_synthesize_restores_original_queue_after_exception` | Exception in `_synthesize_and_enqueue` still restores the queue. |
| `test_synthesize_with_no_frames_yields_nothing` | Empty handler push → empty generator. |
| `test_synthesize_propagates_handler_exception` | Exception in handler propagates to consumer. |
| `test_synthesize_cleans_up_when_consumer_breaks_early` | `gen.aclose()` cancels the streaming task. |
| `test_synthesize_drops_non_tuple_items` | `AdditionalOutputs`-shaped items don't yield (current behaviour; future PR may surface them). |
| `test_shutdown_closes_handler_http` | `adapter.shutdown()` calls `_http.aclose()` and nulls `_http`. |
| `test_shutdown_with_no_open_http_is_safe` | No-op when `_http` is `None`. |
| `test_adapter_satisfies_tts_backend_protocol` | `isinstance(adapter, TTSBackend)` is `True`. |

### Factory dispatch tests (additions to `tests/test_handler_factory_factory_path.py`)

| Test | Asserts |
|------|---------|
| `test_legacy_path_returns_legacy_handler_for_llama_chatterbox` | `FACTORY_PATH=legacy` + chatterbox triple → `LocalSTTChatterboxHandler`. |
| `test_composable_path_returns_wrapper_for_llama_chatterbox` | `FACTORY_PATH=composable` + chatterbox triple → `ComposableConversationHandler` with `LocalSTTChatterboxHandler` as `_tts_handler`. |
| `test_composable_path_wires_three_adapters_for_llama_chatterbox` | `pipeline.stt/llm/tts` are `MoonshineSTTAdapter`, `LlamaLLMAdapter`, `ChatterboxTTSAdapter`; all wrap the same legacy instance. |
| `test_composable_path_seeds_system_prompt_for_llama_chatterbox` | `pipeline._conversation_history[0]` is the patched `get_session_instructions` value. |
| `test_composable_path_copy_constructs_fresh_legacy_for_chatterbox` | `copy()` produces a new wrapper + a new `LocalSTTChatterboxHandler`. |

Update the existing parametrised `test_composable_path_only_affects_llama_elevenlabs` to remove `AUDIO_OUTPUT_CHATTERBOX` from the "still goes through legacy" matrix (chatterbox is now migrated). Rename the test or keep the name and let the parameter list shrink — the simpler rename keeps git blame clean.

### What we don't add tests for

- Hardware barge-in (operator validates on robot after merge).
- End-to-end transcript → audio path with real chatterbox/llama servers (covered by the legacy handler's existing test suite; this PR doesn't change those flows).
- Voice switching propagation — already covered by Phase 4a wrapper tests against a `MagicMock` TTS handler; the chatterbox handler exposes the same surface so the existing tests still pass.

## Migration Notes

- Default behaviour is unchanged — operators with `REACHY_MINI_FACTORY_PATH` unset (or `legacy`) keep `LocalSTTChatterboxHandler`.
- Operators who already set `FACTORY_PATH=composable` from 4b's rollout will, after this PR lands, also get the composable path for the chatterbox triple. There is currently no operator using composable on chatterbox (the dial only activates llama+elevenlabs as of 4b on main), so the expanded scope is opt-in.
- Reverting is one env-var flip.
- 4c.2 onwards: the same shape — one new adapter (`GeminiLLMAdapter` for 4c.2) and one new factory branch.

## Risks

- **`AdditionalOutputs` error-sentinel drop.** When chatterbox can't synthesise (all retries failed), the legacy handler pushes an `AdditionalOutputs` to surface a chat-message error in the UI. The adapter currently drops it — the operator sees silence instead of an error bubble. Acceptable risk for 4c.1; future PR can add a metadata channel.
- **Voice-clone reference path.** `_prepare_startup_credentials` resolves `_voice_clone_ref_path` from the active profile. The adapter calls this in `prepare()`. If the profile changes via the wrapper's `apply_personality`, the legacy handler's `_voice_clone_ref_path` is not re-resolved — same bug as today (operators have to restart for clone-ref switches). Out of scope.
- **`_warmup_tts` runs at `prepare()`.** The legacy handler does a one-shot synthesis to warm the voice model. Under the adapter, `prepare()` triggers `_prepare_startup_credentials` which triggers `_warmup_tts` which calls `_call_chatterbox_tts` directly (no queue interaction). That's fine — `_warmup_tts` doesn't use `self.output_queue`.

## After-merge follow-ups (out of scope for 4c.1)

- 4c.2: `GeminiLLMAdapter` + `(moonshine, chatterbox, gemini)`.
- Lifecycle hooks: per-PR rollout per the operating manual.
- First-audio marker + tag plumbing through `LLMResponse.metadata` — joint PR covering both ElevenLabs and Chatterbox adapters.
