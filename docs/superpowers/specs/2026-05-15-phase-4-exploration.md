# Phase 4 Exploration Memo — Pipeline Refactor (Epic #337)

**Date:** 2026-05-15
**Status:** Planning memo, NOT a final spec — for operator + Claude review.

## 1. Current state survey

### `src/robot_comic/handler_factory.py` (origin/main, 304 lines)

A pure routing layer. `HandlerFactory.build(input_backend, output_backend, deps, *, pipeline_mode=None, sim_mode, instance_path, startup_voice)`:

1. Resolves `pipeline_mode` via `derive_pipeline_mode(input, output)` when not passed.
2. **Bundled-realtime fast path:** if mode is `hf_realtime` / `openai_realtime` / `gemini_live`, instantiates `HuggingFaceRealtimeHandler` / `OpenaiRealtimeHandler` / `GeminiLiveHandler` and returns. STT/LLM/TTS dials are ignored in these modes.
3. **Composable fast path:** asserts `pipeline_mode == composable`, then reads `config.LLM_BACKEND` and branches on `(input_backend, output_backend, LLM_BACKEND)`:
   - `(moonshine, elevenlabs, llama)` → `LocalSTTLlamaElevenLabsHandler`
   - `(moonshine, chatterbox, gemini)` → `GeminiTextChatterboxHandler`
   - `(moonshine, elevenlabs, gemini)` → `GeminiTextElevenLabsHandler`
   - `(moonshine, chatterbox, *)` → `LocalSTTChatterboxHandler` (llama by default)
   - `(moonshine, gemini_tts, *)` → `LocalSTTGeminiTTSHandler`
   - `(moonshine, elevenlabs, gemini-fallback)` → `LocalSTTGeminiElevenLabsHandler`
   - `(moonshine, openai_realtime_output)` → `LocalSTTOpenAIRealtimeHandler` *(still composable mode because input != openai_realtime_input)*
   - `(moonshine, hf_output)` → `LocalSTTHuggingFaceRealtimeHandler`

Unmatched combinations raise `NotImplementedError` with a docs pointer.

### `src/robot_comic/composable_pipeline.py` (origin/main)

`ComposablePipeline(stt, llm, tts, *, output_queue, tool_dispatcher, tools_spec, max_tool_rounds=8, system_prompt)`. Notable surface:

- `_conversation_history` lives on the pipeline (seeded with the optional `system_prompt`).
- `start_up()` calls `llm.prepare()`, `tts.prepare()`, `stt.start(on_completed=...)`, then blocks on `_stop_event` for the lifetime of the session. Mirrors the existing `ConversationHandler` ABC's contract.
- `feed_audio(frame)` forwards into `stt.feed_audio(frame)`.
- `_on_transcript_completed(transcript)` appends `{role: user}` and runs `_run_llm_loop_and_speak()` — up to `max_tool_rounds` LLM calls, dispatches each tool call via `tool_dispatcher`, appends `{role: tool, tool_call_id, name, content}`, finally speaks the assistant text via `tts.synthesize(text)`.
- `reset_history(keep_system=True)`. No `apply_personality`, `change_voice`, `get_available_voices`, or `get_current_voice` — those don't yet exist on the orchestrator.

### Adapters (`src/robot_comic/adapters/`, origin/main)

All three are "reach into the legacy handler's internals" thin wrappers:

- **`LlamaLLMAdapter(handler: BaseLlamaResponseHandler)`** — `prepare()` calls `handler._prepare_startup_credentials()`. `chat(messages, tools)` saves and clears `handler._conversation_history`, calls `handler._call_llm(extra_messages=messages)`, restores history in `finally`, and converts the raw tool-call dicts to `ToolCall`. The `tools` parameter is **ignored** (the legacy handler sources tools from `deps` via `get_active_tool_specs`).
- **`ElevenLabsTTSAdapter(handler: ElevenLabsTTSResponseHandler)`** — `synthesize()` substitutes `handler.output_queue` with a temp queue, runs `_stream_tts_to_queue(text, tags)` in a task, consumes frames as `AudioFrame(samples, sample_rate)` until a `_STREAM_DONE` sentinel. Cancels the task and restores the queue in `finally` so the handler stays reusable. `first_audio_marker` is not surfaced (TODO).
- **`MoonshineSTTAdapter(handler)`** — monkey-patches `handler._dispatch_completed_transcript` to call the registered `TranscriptCallback`. Restores in `stop()`. The "host handler" must mix in `LocalSTTInputMixin`.

### `src/robot_comic/config.py` cluster

Active dials consumed by today's `handler_factory.py`:
- `PIPELINE_MODE` (`composable` | `openai_realtime` | `gemini_live` | `hf_realtime`) — added by Phase 0. Has env override and `derive_pipeline_mode()` fallback.
- `LLM_BACKEND` (`llama` | `gemini`) — branches the composable path inside the factory.
- `AUDIO_INPUT_BACKEND` (`moonshine` | `*_realtime_input`) and `AUDIO_OUTPUT_BACKEND` (`chatterbox` | `gemini_tts` | `elevenlabs` | `*_realtime_output`).
- `BACKEND_PROVIDER` (`huggingface` | `openai` | `gemini` | `local_stt`) — *legacy*; the modular AUDIO_INPUT/OUTPUT dials now override it but the values still derive from it when those vars are unset. `main.py:240-251` still reads `config.BACKEND_PROVIDER` directly for HF-specific logging paths.
- `LOCAL_STT_RESPONSE_BACKEND` (`openai` | `huggingface` | `chatterbox` | `gemini_tts` | `elevenlabs`) — still defined in config (lines 414–431) but **no longer read by `handler_factory.py`**. It feeds `derive_audio_backends()` to produce AUDIO_INPUT/OUTPUT.
- `LOCAL_STT_PROVIDER`, `LOCAL_STT_CACHE_DIR`, `LOCAL_STT_LANGUAGE`, `LOCAL_STT_MODEL`, `LOCAL_STT_UPDATE_INTERVAL` — Moonshine model loader knobs (lines 412–434). Independent of the dispatch refactor.

---

## 2. Inventory of legacy classes

**Composable-mode handlers (in scope for Phase 4 retirement):**

| Class | File | Bases | Notes |
| --- | --- | --- | --- |
| `BaseLlamaResponseHandler` | `llama_base.py:64` | `AsyncStreamHandler, ConversationHandler` | LLM phase + tool loop + history. Foundation of all llama- and gemini-text handlers. |
| `LlamaElevenLabsTTSResponseHandler` | `llama_elevenlabs_tts.py:92` | `BaseLlamaResponseHandler` | Adds ElevenLabs TTS streaming. |
| `LocalSTTLlamaElevenLabsHandler` | `llama_elevenlabs_tts.py:359` | `LocalSTTInputMixin, LlamaElevenLabsTTSResponseHandler` | The default composable pipeline today. |
| `LlamaGeminiTTSResponseHandler` | `llama_gemini_tts.py:59` | `BaseLlamaResponseHandler` | Llama LLM + Gemini TTS. |
| `LocalSTTLlamaGeminiTTSHandler` | `llama_gemini_tts.py:234` | `LocalSTTInputMixin, LlamaGeminiTTSResponseHandler` | Not reached by current factory routing — orphan? |
| `ChatterboxTTSResponseHandler` | `chatterbox_tts.py:49` | `BaseLlamaResponseHandler` | Llama LLM + Chatterbox TTS. |
| `LocalSTTChatterboxHandler` | `chatterbox_tts.py:467` | `LocalSTTInputMixin, ChatterboxTTSResponseHandler` | |
| `ElevenLabsTTSResponseHandler` | `elevenlabs_tts.py:232` | `AsyncStreamHandler, ConversationHandler` | Hardcodes Gemini for LLM via `genai.Client` in `_prepare_startup_credentials`. |
| `LocalSTTGeminiElevenLabsHandler` | `elevenlabs_tts.py:1053` | `LocalSTTInputMixin, ElevenLabsTTSResponseHandler` | Aliased as legacy `LocalSTTElevenLabsHandler`. |
| `GeminiTTSResponseHandler` | `gemini_tts.py:195` | `AsyncStreamHandler, ConversationHandler` | Native Gemini LLM+TTS. |
| `LocalSTTGeminiTTSHandler` | `gemini_tts.py:583` | `LocalSTTInputMixin, GeminiTTSResponseHandler` | |
| `GeminiTextResponseHandler` | `gemini_text_base.py:30` | `BaseLlamaResponseHandler` | Replaces `_call_llm` with Gemini API. |
| `GeminiTextChatterboxResponseHandler` | `gemini_text_handlers.py:34` | `GeminiTextResponseHandler, ChatterboxTTSResponseHandler` | MRO diamond. |
| `GeminiTextChatterboxHandler` | `gemini_text_handlers.py:84` | `LocalSTTInputMixin, GeminiTextChatterboxResponseHandler` | |
| `GeminiTextElevenLabsResponseHandler` | `gemini_text_handlers.py:97` | `GeminiTextResponseHandler, ElevenLabsTTSResponseHandler` | MRO diamond. |
| `GeminiTextElevenLabsHandler` | `gemini_text_handlers.py:180` | `LocalSTTInputMixin, GeminiTextElevenLabsResponseHandler` | |
| `LocalSTTInputMixin` | `local_stt_realtime.py:230` | (none) | The Moonshine STT loop. Must survive in some form — `MoonshineSTTAdapter` depends on it. |

**Hybrid: Moonshine STT + bundled-realtime output** (don't decompose cleanly — still need the bundled session for their output half):

- `LocalSTTOpenAIRealtimeHandler` (`local_stt_realtime.py:869`) — `LocalSTTInputMixin, OpenaiRealtimeHandler`
- `LocalSTTHuggingFaceRealtimeHandler` (`local_stt_realtime.py:910`) — `LocalSTTInputMixin, HuggingFaceRealtimeHandler`

**Bundled-realtime (preserved by Phase 4):**

- `BaseRealtimeHandler` (`base_realtime.py:90`)
- `OpenaiRealtimeHandler` (`openai_realtime.py:27`)
- `HuggingFaceRealtimeHandler` (`huggingface_realtime.py:62`)
- `GeminiLiveHandler` (`gemini_live.py:315`)

The bundled-realtime path is explicitly out of scope. The hybrid `LocalSTT*RealtimeHandler` pair is awkward: their LLM+TTS half lives inside the bundled websocket session, so they don't compose into the `STT/LLM/TTS` Protocol triple. Treat them as out-of-scope for now.

---

## 3. Inventory of legacy config dials

| Dial | Today's role | Phase 4 disposition (proposed) |
| --- | --- | --- |
| `PIPELINE_MODE` | Top-level mode selector | **Keep.** Becomes the sole dispatch axis in the factory. |
| `AUDIO_INPUT_BACKEND` | Resolves to `moonshine` for composable mode | **Keep (composable only).** ComposablePipeline needs to know which STT to plug in (today: only Moonshine). Future-proofs Whisper / Deepgram. |
| `AUDIO_OUTPUT_BACKEND` | Selects TTS for composable mode | **Keep.** ComposablePipeline needs to know which TTS adapter to instantiate. |
| `LLM_BACKEND` | Selects LLM provider for composable mode | **Keep.** Selects which LLM adapter to instantiate (`llama` → `LlamaLLMAdapter`; `gemini` → new `GeminiLLMAdapter` *(does not yet exist)*). |
| `BACKEND_PROVIDER` | Pre-modular legacy, drives derivation of AUDIO_INPUT/OUTPUT when those are unset | **Defer.** Still consumed by `main.py:240` for HF-specific logging and by `derive_audio_backends()`. Retiring this is its own mini-refactor (touches main.py, profiles, env templates, deploy scripts). Recommend leaving in place and addressing in a Phase 5. |
| `LOCAL_STT_RESPONSE_BACKEND` | Used by `derive_audio_backends()` only | **Defer with BACKEND_PROVIDER** — they're a unit. |
| `LOCAL_STT_PROVIDER` / `_CACHE_DIR` / `_LANGUAGE` / `_MODEL` / `_UPDATE_INTERVAL` | Moonshine model knobs | **Keep.** These tune the STT backend itself; Phase 4 doesn't change them. |
| `DEFAULT_BACKEND_PROVIDER`, `HF_BACKEND`, `OPENAI_BACKEND`, `GEMINI_BACKEND`, `LOCAL_STT_BACKEND` constants | Symbolic names used across config / profile logic | **Keep** — still referenced by profile validation, voice catalog, prompt selection. |

**Honest read:** Phase 4 retiring `BACKEND_PROVIDER` is harder than the prompt suggests. It's woven into profile defaults, model-name derivation, and operator-facing env files. Cutting it should be its own PR.

---

## 4. Proposed approach (three options)

### Option A — Big-bang
One PR: gut `handler_factory.py`'s composable branch, replace with `ComposablePipeline` instantiation. Delete `LocalSTTChatterboxHandler`, `LocalSTTGeminiElevenLabsHandler`, `LocalSTTLlamaElevenLabsHandler`, `LocalSTTLlamaGeminiTTSHandler`, `LocalSTTGeminiTTSHandler`, `GeminiTextChatterboxHandler`, `GeminiTextElevenLabsHandler`, and the orchestration code on `BaseLlamaResponseHandler` / `ChatterboxTTSResponseHandler` / `GeminiTextResponseHandler` / `ElevenLabsTTSResponseHandler`. Keep the underlying API-call code paths. Rewrite ~10 test files.

**Pros:** Fastest; one shippable artifact; no dead code left behind.
**Cons:** Single PR will be 2k+ LOC delta. If any composable pipeline regresses on the robot, you can't easily bisect to which retirement broke it. Risky to validate without staging time on hardware.

### Option B — Parallel paths
Add a new config flag (`USE_COMPOSABLE_PIPELINE=0/1`, default 0). When 1, factory builds via `ComposablePipeline` + adapters. When 0, legacy handlers as today. Operator flips the flag per profile, validates, and after ≥1 cycle of production use we delete the legacy path.

**Pros:** Safest. Trivially revertible. Real-world validation before deletion.
**Cons:** Two pipelines in tree for some time; the surface of `ConversationHandler` ABC vs. `ComposablePipeline` diverges (voice / personality methods missing on the orchestrator — see open questions). Operator must remember to flip the flag.

### Option C — Incremental retirement (recommended)
- **4a — Adapter ABC / wrapper.** Wrap `ComposablePipeline` in a `ComposableConversationHandler(ConversationHandler)` shim that exposes `copy()`, `start_up()`, `shutdown()`, `receive()`, `emit()`, `apply_personality()`, `change_voice()`, `get_available_voices()`, `get_current_voice()`. The wrapper holds the underlying legacy handler instances *too* (the adapters need them) and forwards the voice/personality methods to the appropriate one. This is the "Phase 3d" that was deferred.
- **4b — Factory dual path behind config flag.** `handler_factory.py` reads a new `REACHY_MINI_FACTORY_PATH=legacy|composable` dial (default `legacy`). When `composable`, builds via the 4a wrapper for the *one* pipeline triple we've validated end-to-end first (suggest: `moonshine + llama + elevenlabs`, the prod default). All other triples still flow through the legacy branch.
- **4c — Expand coverage.** One sub-PR per remaining composable triple to flip it onto the new path. After each, operator validates on hardware.
- **4d — Flip default.** `REACHY_MINI_FACTORY_PATH=composable` becomes the default.
- **4e — Delete legacy.** Remove `LocalSTT*Handler` and `*ResponseHandler` orchestration code (keep the API-call helpers and `LocalSTTInputMixin`). Rewrite tests. Remove the flag.

**Pros:** Each sub-PR is small and reviewable. Always revertible. Validates each pipeline triple in isolation. Lets the operator catch regressions one triple at a time.
**Cons:** Slowest. Five PRs. Adds a transient flag that we then delete. Risk of the "delete legacy" step being deprioritised forever — needs operator commitment to follow through.

**Recommendation: Option C.** The orchestrator is unproven on hardware. Test coverage of the existing handlers is uneven (`test_chatterbox_auto_gain.py`, `test_elevenlabs_start_up_supporting_events.py` exercise startup events that the new path needs to preserve but doesn't yet). Going incremental lets us spot what `ComposablePipeline` is missing (telemetry spans, supporting events, joke history capture, history trim, first-audio markers, echo guard) before committing to a default flip. Each gap closure is its own small PR.

---

## 5. Test migration sketch

Adapter tests already exist on `origin/main` under `tests/adapters/` (3 files: `test_llama_llm_adapter.py`, `test_elevenlabs_tts_adapter.py`, `test_moonshine_stt_adapter.py`) and `tests/test_composable_pipeline.py`, `tests/test_backends_protocols.py`. Those cover the new surface.

Tests that will need attention in Phase 4:

| Test file | Action | Reason |
| --- | --- | --- |
| `tests/test_handler_factory.py` | **Rewrite** | Today asserts `LocalSTTLlamaElevenLabsHandler` is the concrete class returned for `(moonshine, elevenlabs, llama)`. Under Option C-4d that becomes the wrapper. Update assertions to check `isinstance(handler, ComposableConversationHandler)` plus the wrapped adapter types. |
| `tests/test_handler_factory_llama_llm.py` | **Rewrite** | Same as above for llama-specific routing. |
| `tests/test_handler_factory_gemini_llm.py` | **Rewrite** | Needs a `GeminiLLMAdapter` to exist first; gemini composable path is currently blocked by lack of that adapter. |
| `tests/test_handler_factory_pipeline_mode.py` | **Parametric updates** | Pipeline-mode dispatch is unchanged; mostly stays. Add a case for the flag. |
| `tests/test_llama_base.py` (28.7 KB) | **Keep + reshape** | Tests `_call_llm`, tool dispatch, history trimming. The underlying methods survive; only the orchestrator (`start_up`, `receive`) goes away. Likely a 30–50% trim. |
| `tests/test_llama_elevenlabs_tts.py` (16.3 KB) | **Keep + reshape** | Same story — TTS streaming primitives survive, the dispatcher loop is replaced. |
| `tests/test_llama_gemini_tts.py` | **Keep + reshape** | Same. |
| `tests/test_llama_streaming.py` | **Keep mostly** | Streaming-specific, not orchestration. |
| `tests/test_chatterbox_auto_gain.py` | **Keep** | Audio-processing unit tests, orthogonal. |
| `tests/test_elevenlabs_tts.py` (45.8 KB) | **Keep + reshape** | Biggest test file. Substantial portions are TTS-internals tests that survive. Orchestration tests (start_up, receive flow) get deleted/replaced. |
| `tests/test_elevenlabs_start_up_supporting_events.py` | **Migrate** | These supporting events need to fire from the new path too. Tests should be parametrized over legacy and composable. |
| `tests/test_elevenlabs_profile_configs.py` | **Keep** | Config-level, not handler-level. |
| `tests/test_gemini_tts_handler.py`, `test_gemini_tts_styling.py` | **Keep + reshape** | Same as llama_elevenlabs. |
| `tests/test_gemini_llm.py` | **Keep** | Tests `GeminiLLMClient` primitives needed by a future `GeminiLLMAdapter`. |
| `tests/test_local_stt_realtime.py` (15.4 KB) | **Keep almost entirely** | Listener / rearm / heartbeat tests; the mixin survives. |
| `tests/test_local_stt_dispatch_hook.py` | **Keep** | Already tests the dispatch-hook surface the adapter relies on. |
| `tests/test_local_stt_diag.py` | **Keep** | Diagnostics, orthogonal. |
| `tests/test_handler_factory.py` startup-event assertions | **Migrate** | Boot timeline (`#321`) expects four supporting events; new path must emit them or the welcome flow breaks. |
| `tests/test_admin_pipeline_3column.py` | **Audit** | Admin UI route — might rely on handler class names. |
| `tests/test_prompt_backend_awareness.py` | **Audit** | Backend-aware prompting; verify the new path threads the same context. |
| `tests/test_george_carlin_profile.py`, `test_rodney_dangerfield_profile.py` | **Keep** | Profile-config tests, orthogonal to handler class identity. |

Rough count: ~6 files need substantial rewrites, ~10 need partial reshapes, the rest are orthogonal.

---

## 6. Risks & open questions

1. **History persistence.** Today's `BaseLlamaResponseHandler._conversation_history` lives on the handler instance and is cleared in a few places (e.g. `apply_personality` does `self._conversation_history.clear()` at line 230). `ComposablePipeline` also keeps history per-instance. **Question:** is history persisted across sessions today? If not, no migration is needed — but `reset_history(keep_system=True)` doesn't currently get called when the operator switches personas. The wrapper needs to call it from `apply_personality()`.

2. **Tool dispatcher wiring.** `LlamaLLMAdapter.chat` *ignores* its `tools` arg because the legacy handler sources tools from `deps` via `get_active_tool_specs(deps)`. The orchestrator's `tool_dispatcher` callback is also missing — adapters today do tool dispatch *internally* via `_call_llm` returning the raw tool calls. **Decision needed:** does Phase 4 hand `ComposablePipeline` a real `tool_dispatcher` callback that knows `ToolDependencies`, or does the wrapper short-circuit and let the adapter keep doing what it does today? Probably the former, but it means writing a `ToolDependencies → ToolDispatcher` shim (~40 LOC) that mirrors `_start_tool_calls` + `_await_tool_results` from `llama_base.py:572-617`. Background tools (the `bg_tools` list at `llama_base.py:570`) complicate this — they're not synchronous-return, so the orchestrator's `await tool_dispatcher(call)` model needs an answer for them.

3. **System prompt.** `BaseLlamaResponseHandler._call_llm` reads `system_prompt = get_session_instructions()` at line 731 *every call* (so persona switches reflect immediately). `ComposablePipeline` takes a static `system_prompt` at construction. **Resolution options:** (a) make `ComposablePipeline.system_prompt` a property that re-reads on each turn; (b) have the wrapper call `pipeline.reset_history(); pipeline._conversation_history[0] = {"role": "system", "content": get_session_instructions()}` inside `apply_personality()`; (c) push the system prompt into `messages` per-call from the wrapper. Option (b) is cleanest.

4. **`ConversationHandler` ABC surface gap.** ComposablePipeline does not implement `copy()`, `emit()`, `apply_personality()`, `change_voice()`, `get_available_voices()`, `get_current_voice()`. FastRTC and the admin routes call these. The 4a wrapper has to. `change_voice` is tricky — it operates on the *TTS backend instance*, which is hidden behind `TTSBackend`. Either expose a `change_voice` method on the Protocol (Protocol churn), or the wrapper keeps a typed reference to the underlying handler and calls through.

5. **Lifecycle hooks.** Telemetry (`telemetry.record_llm_duration`), supporting events (`#321` boot timeline), joke history capture (`llama_base.py:553-568`), echo guard (`elevenlabs_tts.py:313`), `history_trim` (`llama_base.py:480`) — none of these are in `ComposablePipeline`. Each needs an answer:
   - Telemetry → adapter wraps + records → straightforward
   - Supporting events → must fire from `start_up()` of the wrapper
   - Joke history → orchestrator-level hook, or do it in `LlamaLLMAdapter.chat` post-call
   - Echo guard → already lives on `ElevenLabsTTSResponseHandler` and the adapter delegates, so this works "for free"
   - History trim → orchestrator-level concern

6. **Bundled-realtime + Moonshine STT hybrids.** `LocalSTTOpenAIRealtimeHandler` / `LocalSTTHuggingFaceRealtimeHandler` exist for "talk to a realtime endpoint but transcribe locally". They are *not* composable in the new sense (the realtime endpoint owns LLM+TTS as one). Are they still supported configurations? If yes, they survive Phase 4 untouched. If no, drop them (saves one source file's worth of complexity).

7. **`GeminiLLMAdapter` doesn't exist yet.** Composable + `LLM_BACKEND=gemini` has no adapter. Phase 4 either needs to build it (~150 LOC mirroring `LlamaLLMAdapter` against `GeminiTextResponseHandler._call_llm`) or explicitly defer the Gemini composable path to Phase 5.

8. **`copy()` semantics for FastRTC.** FastRTC clones the handler instance per WebRTC peer. `ComposablePipeline` has stateful backends — what does cloning look like? Probably: the wrapper's `copy()` reconstructs a fresh pipeline with fresh backends from the factory. Needs verification against FastRTC's expectations.

9. **The "orphan" `LocalSTTLlamaGeminiTTSHandler`.** Defined in `llama_gemini_tts.py:234` but I don't see it referenced from `handler_factory.py` — recommend confirming it's actually reachable today before bothering to migrate it.

---

## 7. Effort estimate

Calibration: a "session" here = one focused work session, ~2–4 hours of Claude + operator collaboration ending in a merged PR.

- **Option A (big-bang):** 3–5 sessions. One big PR + 1–2 hardware-validation revs. High variance — could blow up if regressions cascade.
- **Option B (parallel paths):** 4–6 sessions. One PR to add the flag, ~1 session per pipeline triple to validate, ~1 session of cleanup. Moderate variance.
- **Option C (incremental, recommended):**
  - 4a — wrapper + voice/personality forwarding: **1 session**
  - 4b — factory dual path for one triple: **1 session**
  - 4c — expand to remaining 4–5 triples: **2–3 sessions**
  - 4d — flip default + soak: **0.5 session**
  - 4e — delete legacy + test rewrites: **1–2 sessions**
  - **Total: 5–7 sessions.** Lower variance than A or B; each sub-PR is independently shippable so partial progress isn't wasted.

Add **+1 session** in any option for a `GeminiLLMAdapter` if we want gemini composable parity in Phase 4 rather than deferring.

---

## Things to confirm before turning this into a real spec

- Yes/no on Option C as the chosen path
- Decision on the orphan `LocalSTTLlamaGeminiTTSHandler` and the hybrid `LocalSTT*RealtimeHandler` pair
- Decision on whether `GeminiLLMAdapter` ships in Phase 4 or Phase 5
- Confirmation that `BACKEND_PROVIDER` / `LOCAL_STT_RESPONSE_BACKEND` retirement is explicitly out of Phase 4 scope (I think it should be)
- A target pipeline triple to migrate first under 4b — recommend `moonshine + llama + elevenlabs` since it's the prod default and has the most existing test coverage to lean on
