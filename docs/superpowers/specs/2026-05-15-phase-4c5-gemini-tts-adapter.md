# Phase 4c.5 — GeminiTTSAdapter + composable routing for `(moonshine, gemini_tts)`

**Date:** 2026-05-15
**Scope:** New `GeminiTTSAdapter` (~140 LOC); new `GeminiBundledLLMAdapter` (~110 LOC) that wraps `GeminiTTSResponseHandler._run_llm_with_tools`; new `_build_composable_gemini_tts` helper in `handler_factory.py`; route `(moonshine, gemini_tts)` through `ComposableConversationHandler` when `REACHY_MINI_FACTORY_PATH=composable`. Under default `legacy`, `LocalSTTGeminiTTSHandler` is still returned — zero behaviour change for default users.
**Epic:** #337 — Pipeline refactor (Option C, incremental retirement)
**Predecessors:** Phase 4c.4 (#365 / `ac1232e`) — `(moonshine, elevenlabs, gemini-fallback)` composable routing landed; all four `(moonshine, *, *)` non-bundled triples now routed under `FACTORY_PATH=composable` except the gemini-native bundled-TTS triple addressed here.
**Successors:** Phase 4c-tris (hybrid realtime memo + impl); Phase 4d (default flip).

## Background

`FACTORY_PATH=composable` already covers four triples:

- `(moonshine, llama, elevenlabs)` — Phase 4b.
- `(moonshine, llama, chatterbox)` — Phase 4c.1.
- `(moonshine, chatterbox, gemini)` — Phase 4c.2.
- `(moonshine, elevenlabs, gemini)` — Phase 4c.3.
- `(moonshine, elevenlabs, gemini-fallback)` — Phase 4c.4.

The last remaining non-hybrid composable triple is `(moonshine, gemini_tts)`. The legacy class `LocalSTTGeminiTTSHandler` (`gemini_tts.py:583`) extends `LocalSTTInputMixin + GeminiTTSResponseHandler`. The interesting class is the response-handler base:

```python
class GeminiTTSResponseHandler(AsyncStreamHandler, ConversationHandler):  # gemini_tts.py:195
    def __init__(...) -> None:
        self._client: genai.Client | None = None
        self._conversation_history: list[dict[str, Any]] = []
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()

    async def _prepare_startup_credentials(self) -> None:
        self._client = genai.Client(api_key=...)            # one Client for both LLM + TTS

    async def _run_llm_with_tools(self) -> str:             # LLM half + internal tool dispatch
        # walks _LLM_MAX_TOOL_ROUNDS; dispatches tools INSIDE; returns final text
        # uses self._client.aio.models.generate_content(model=GEMINI_TTS_LLM_MODEL, ...)

    async def _call_tts_with_retry(text, system_instruction=None) -> bytes | None:
        # single TTS call returning raw PCM bytes
        # uses self._client.aio.models.generate_content(model=GEMINI_TTS_MODEL, ...)

    # change_voice / get_available_voices / get_current_voice — inherited from
    # ConversationHandler ABC (defined in this class — voice list is GEMINI_TTS_AVAILABLE_VOICES).
```

## Goal

1. Build `GeminiTTSAdapter` wrapping the **TTS half** of `GeminiTTSResponseHandler` — exposes `TTSBackend` Protocol. The adapter:
   - In `prepare()`, calls `_prepare_startup_credentials()` (idempotent on the wrapped handler).
   - In `synthesize(text, tags)`, splits *text* into sentences (replicating the legacy `split_sentences` + `_DELIVERY_TAG_RE` extraction), calls `_call_tts_with_retry` per sentence, chunks the returned PCM via `_pcm_to_frames`, and yields :class:`AudioFrame` for each chunk. `[short pause]` tags produce a silence frame; remaining delivery tags drive `build_tts_system_instruction`.
   - In `shutdown()`, defensively no-ops (the underlying `genai.Client` has no explicit `aclose` — see Risks).
2. Build `GeminiBundledLLMAdapter` wrapping the **LLM half** of `GeminiTTSResponseHandler` — exposes `LLMBackend` Protocol. The adapter:
   - In `prepare()`, calls `_prepare_startup_credentials()` (idempotent — see Q3).
   - In `chat(messages, tools)`, swaps `_conversation_history` (mirror of `LlamaLLMAdapter`/`GeminiLLMAdapter` pattern, but using Gemini's `[{"role": ..., "parts": [{"text": ...}]}]` shape — the helper converts the orchestrator's `{"role", "content"}` messages on the way in), invokes `_run_llm_with_tools()`, and returns `LLMResponse(text=<returned string>, tool_calls=())`. Tools are dispatched **inside** `_run_llm_with_tools` against `self.deps`; the orchestrator's `tool_dispatcher` is unused for this triple.
   - In `shutdown()`, no-op.
3. New factory helper `_build_composable_gemini_tts` mirrors prior 4c.* helpers. Constructs **one** `LocalSTTGeminiTTSHandler` instance and wraps it with `MoonshineSTTAdapter` + `GeminiBundledLLMAdapter` + `GeminiTTSAdapter`, all three holding the same `legacy` reference. This is the "two adapters on one handler" pattern from 4c.3 — the orchestrator sees separate LLM and TTS Protocols, but a single `genai.Client` instance (on the shared handler) backs both calls.
4. Composable-path gate inside the `(moonshine, gemini_tts)` factory branch.

## Out of scope (deferred)

| Item | Sub-phase |
|------|-----------|
| Hybrid bundled-realtime triples (`LocalSTTOpenAIRealtimeHandler`, `LocalSTTHuggingFaceRealtimeHandler`) | 4c-tris |
| Flip default to `composable` | 4d |
| Delete `LocalSTTGeminiTTSHandler` and friends | 4e |
| Retire `BACKEND_PROVIDER` | 4f |
| Lifecycle hooks: telemetry, boot-timeline events (#321), joke history (already present in legacy `_dispatch_completed_transcript`), history trim, echo-guard timestamps | Per-hook follow-up PRs |
| Plumbing `[short pause]` delivery tags through `LLMResponse.metadata` so the orchestrator forwards them via `tts.synthesize(text, tags=...)` | Future PR; today the adapter parses tags out of the raw text inline |
| Joke-history capture in `_dispatch_completed_transcript` (lines 380-394 in `gemini_tts.py`) is **not** ported to the adapter; lifecycle-hook follow-up will revisit |
| Synthetic-status-marker guard (line 376 in `gemini_tts.py`) is **not** ported; orchestrator history is owned by `ComposablePipeline` and synthetic markers don't traverse it |

## Design

### Design question Q1 — bundled LLM+TTS: where does the seam go?

**Three options were considered:**

- (a) **`GeminiTTSAdapter` covers TTS only; reuse the existing `GeminiLLMAdapter` (4c.2) for the LLM half.**
- (b) **`GeminiTTSAdapter` covers TTS only; build a sibling `GeminiBundledLLMAdapter` that wraps `_run_llm_with_tools` directly.**
- (c) `GeminiTTSAdapter` covers both halves as one adapter. **Rejected** per the briefing — the orchestrator's Protocols are separate, and one bundled adapter can only present one Protocol at a time.

**Resolution: option (b).**

Justification — option (a) is structurally incompatible:

- `GeminiLLMAdapter.chat` calls `self._handler._call_llm(extra_messages=messages)` (`gemini_llm_adapter.py:95`).
- `GeminiTTSResponseHandler` does **not** implement `_call_llm`. It implements `_run_llm_with_tools` which (i) walks `_LLM_MAX_TOOL_ROUNDS=5` Gemini tool round-trips internally, (ii) dispatches each tool inside the loop via `dispatch_tool_call(name, json.dumps(args), self.deps)`, (iii) returns the final assistant **text** as a `str`, and (iv) pushes "🛠️ Used tool X" `AdditionalOutputs` sentinels into `self.output_queue` as side effects.
- The shapes don't match. `_call_llm` returns `(text, raw_tool_calls, raw_msg)` — three values, with `raw_tool_calls` in llama-server-shaped dict form for the orchestrator's tool loop. `_run_llm_with_tools` returns one string, no tool surface, because tools have already been dispatched.
- Reconciling via Protocol broadening would require synthesising fake `raw_tool_calls` from the side-effect markers pushed to `output_queue` — that's not a Protocol broadening, that's a behaviour rewrite. Out of bounds for this sub-phase per "Do not modify any adapter".

Option (b) — a dedicated `GeminiBundledLLMAdapter` — fits the structural reality:

- The `LLMBackend` Protocol is satisfied: `chat(messages, tools)` returns one `LLMResponse` with text and empty `tool_calls`. The orchestrator sees a one-shot LLM that never requests tools; its tool-round loop exits after the first call. Inside the adapter, `_run_llm_with_tools` runs its own multi-round loop and dispatches tools. **The orchestrator's `tool_dispatcher` is never invoked for this triple** — this is documented and tested.
- This preserves the legacy behaviour exactly. Tool dispatch wiring in `dispatch_tool_call(name, args, deps)` reads tools from `self.deps`, which is what the legacy class already does.
- The orchestrator's tool-call round-trip telemetry, max-round cap, etc., simply don't fire on this triple. That's a known regression from a "pure orchestrator owns tools" perspective, but it's the same observable behaviour the legacy handler has today. **The composable path for `(moonshine, gemini_tts)` therefore behaves identically to legacy — by design.**
- Re-implementing `_run_llm_with_tools` in terms of `ComposablePipeline._run_llm_loop_and_speak`'s tool-round loop would require: (i) splitting the per-round Gemini call into a `chat()` adapter that returns `tool_calls`, (ii) translating Gemini's `function_call`/`FunctionResponse` shape to the orchestrator's `ToolCall` dataclass, (iii) translating the orchestrator's tool-result-as-string back to Gemini's `Part(function_response=...)` shape on the next round. That's *exactly* what `GeminiLLMClient.call_completion` does today and what `GeminiLLMAdapter` consumes — but `GeminiTTSResponseHandler` doesn't use `GeminiLLMClient`. Migrating it to do so is its own PR (a "Gemini-native bundled migration" change), out of 4c.5 scope.

Constraint compliance: the orchestrator (`ComposablePipeline`) still treats LLM and TTS as separate Protocols. The two adapters (`GeminiBundledLLMAdapter` + `GeminiTTSAdapter`) wrap the **same underlying handler instance** — the existing "shared legacy handler" pattern proven in 4c.3 / 4c.4. One `genai.Client` instance, two adapter wrappers, two Protocol satisfactions.

### Design question Q2 — `change_voice` / `get_available_voices` / `get_current_voice`

`GeminiTTSResponseHandler` defines all three (`gemini_tts.py:336–350`) on the handler class itself, not on a separate response-handler base. The wrapper's voice-forwarding shims (`composable_conversation_handler.py:141–151`) call them on `_tts_handler`, which is the `LocalSTTGeminiTTSHandler` instance — and that class inherits the methods from `GeminiTTSResponseHandler`. **No additional plumbing is needed; the wrapper's existing forward path covers Gemini TTS voices.** Pinned by a new test.

### Design question Q3 — `prepare()` double-init

Both `GeminiBundledLLMAdapter.prepare()` and `GeminiTTSAdapter.prepare()` call `_prepare_startup_credentials()` on the shared handler. The legacy method is idempotent in spirit — it reassigns `self._client = genai.Client(...)` unconditionally, so the second call constructs a fresh client and drops the first one. `genai.Client` opens HTTP connections lazily, so no leak.

Same regression as flagged in the 4c.3 spec for `GeminiLLMAdapter + ElevenLabsTTSAdapter` sharing a `GeminiTextElevenLabsHandler` — the double-init is on the legacy side, not the adapters', and it's been there since 4b. **Out of scope for this PR**, flagged in the PR body consistent with 4c.3.

### 1. `GeminiTTSAdapter`

New file `src/robot_comic/adapters/gemini_tts_adapter.py`. The TTS half does **not** follow the temp-queue + sentinel pattern — `GeminiTTSResponseHandler` exposes `_call_tts_with_retry(text, system_instruction=None) -> bytes | None`, a single-shot call returning raw 16-bit PCM bytes. The adapter directly chunks and yields, mirroring the legacy sentence-loop in `_dispatch_completed_transcript`:

```python
class GeminiTTSAdapter:
    def __init__(self, handler: "_GeminiTTSCompatibleHandler") -> None:
        self._handler = handler

    async def prepare(self) -> None:
        await self._handler._prepare_startup_credentials()

    async def synthesize(self, text: str, tags: tuple[str, ...] = ()) -> AsyncIterator[AudioFrame]:
        # Replicate legacy _dispatch_completed_transcript's sentence loop:
        #   - split_sentences(text)
        #   - per sentence: strip_gemini_tags, extract_delivery_tags,
        #                   [short pause] → silence frame, then
        #                   build_tts_system_instruction + _call_tts_with_retry
        #   - _pcm_to_frames each PCM blob → AudioFrame per chunk
        base = load_profile_tts_instruction()
        sentences = split_sentences(text) or [text]
        for sentence in sentences:
            spoken = strip_gemini_tags(sentence)
            if not spoken:
                continue
            sentence_tags = extract_delivery_tags(sentence)
            if SHORT_PAUSE_TAG in sentence_tags:
                for frame in self._handler._pcm_to_frames(_silence_pcm(SHORT_PAUSE_MS)):
                    yield AudioFrame(samples=frame, sample_rate=GEMINI_TTS_OUTPUT_SAMPLE_RATE)
            instruction = build_tts_system_instruction(base, sentence_tags)
            pcm_bytes = await self._handler._call_tts_with_retry(spoken, system_instruction=instruction)
            if pcm_bytes is None:
                continue
            for frame in self._handler._pcm_to_frames(pcm_bytes):
                yield AudioFrame(samples=frame, sample_rate=GEMINI_TTS_OUTPUT_SAMPLE_RATE)

    async def shutdown(self) -> None:
        # genai.Client has no aclose(); the handler doesn't own httpx for TTS.
        # No-op — Phase 4e cleanup may revisit.
        return None
```

Key invariants:

- **Tag handling.** The Protocol's `tags` argument is **accepted and ignored** because today's `ComposablePipeline._speak_assistant_text` does not yet plumb delivery tags from `LLMResponse.metadata`. The adapter parses tags **out of the LLM text itself** (where the persona prompt embeds them — see `profiles/don_rickles/instructions.txt`'s "Physical Beats" / delivery cues) via `extract_delivery_tags`, matching exactly what the legacy `_dispatch_completed_transcript` does. Once tag-from-LLM-metadata plumbing lands (out of scope), the adapter can switch to using the Protocol's `tags` parameter as the source of truth.
- **No temp-queue swap.** Unlike ElevenLabs/Chatterbox adapters whose underlying handler pushes to `self.output_queue`, `_call_tts_with_retry` *returns* PCM bytes — the adapter chunks them inline without queue manipulation. Cleaner code path, no streaming-task lifecycle to track.
- **`shutdown()` is a no-op.** `genai.Client` has no explicit close method in the SDK as currently used by this handler. The legacy `LocalSTTGeminiTTSHandler.shutdown` only drains `output_queue` — not relevant on the adapter side because we don't own a queue. Phase 4e cleanup can revisit when the bundled legacy is deleted.
- **`extract_delivery_tags` / `SHORT_PAUSE_TAG` / `build_tts_system_instruction` / `load_profile_tts_instruction` / `_silence_pcm` / `GEMINI_TTS_OUTPUT_SAMPLE_RATE` / `SHORT_PAUSE_MS`** are all imported from `robot_comic.gemini_tts`. `split_sentences` is imported from `robot_comic.llama_base` (already a shared utility, used by the legacy class too). `strip_gemini_tags` from `robot_comic.chatterbox_tag_translator`. These are all stable existing-module exports.

### 2. `GeminiBundledLLMAdapter`

New file `src/robot_comic/adapters/gemini_bundled_llm_adapter.py`. Wraps `_run_llm_with_tools` and adapts the orchestrator's stateless-history contract to the handler's stateful one.

```python
class GeminiBundledLLMAdapter:
    def __init__(self, handler: "_GeminiTTSCompatibleHandler") -> None:
        self._handler = handler

    async def prepare(self) -> None:
        # Idempotent on the wrapped handler. Shared with GeminiTTSAdapter.prepare —
        # same double-init flagged on 4c.3, out of scope to fix here.
        await self._handler._prepare_startup_credentials()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if tools is not None:
            logger.debug(
                "GeminiBundledLLMAdapter.chat: ignoring %d tools arg (legacy handler "
                "reads tools from deps); see module docstring",
                len(tools),
            )
        # Swap the handler's _conversation_history with the orchestrator's
        # messages, converted to Gemini's [{"role", "parts": [{"text"}]}] shape.
        # _run_llm_with_tools reads from self._conversation_history.
        saved_history = self._handler._conversation_history
        self._handler._conversation_history = _orchestrator_messages_to_gemini(messages)
        try:
            text = await self._handler._run_llm_with_tools()
        finally:
            self._handler._conversation_history = saved_history
        return LLMResponse(text=text, tool_calls=())

    async def shutdown(self) -> None:
        return None
```

Helper:

```python
def _orchestrator_messages_to_gemini(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert orchestrator-flavoured messages to Gemini's parts-shape history.

    Orchestrator messages use ``{"role": "user|assistant|system|tool", "content": str}``.
    Gemini's API uses ``{"role": "user|model", "parts": [{"text": ...}]}``.

    Translations:
        user      → role="user"
        assistant → role="model"
        system    → role="user" (Gemini doesn't accept "system" in history; the
                                  system prompt is passed as system_instruction
                                  to GenerateContentConfig — see Q4 in module
                                  docstring).
        tool      → skipped — there are no tool messages because
                    GeminiBundledLLMAdapter.chat returns no tool_calls so the
                    orchestrator never appends a tool turn to history.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "user":
            out.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            out.append({"role": "model", "parts": [{"text": content}]})
        elif role == "system":
            # Gemini doesn't accept role="system" in history; prepend as a
            # user-role primer. The legacy _run_llm_with_tools also sets
            # GenerateContentConfig.system_instruction=get_session_instructions()
            # so the model already has the system prompt; this primer is
            # belt-and-braces equivalent to today's first user turn that
            # carries any non-system_instruction system text.
            out.append({"role": "user", "parts": [{"text": content}]})
        # role == "tool" or unknown: skip — no tool history in this triple.
    return out
```

Notes:

- **Why a duck-typed handler Protocol?** Following the 4c.3 pattern (where `_ElevenLabsCompatibleHandler` captured the duck-typed surface), `_GeminiTTSCompatibleHandler` is a Protocol private to whichever module needs it. Since both new adapters touch the same handler members, we define it **once** in `gemini_tts_adapter.py` and re-import from the LLM adapter. The Protocol exposes:
  - `output_queue: asyncio.Queue[Any]` — defensively included even though `GeminiTTSAdapter` doesn't need it (the handler still uses it for "🛠️ Used tool X" side effects during `_run_llm_with_tools`; the orchestrator-level pipeline doesn't drain those today but the legacy handler attaches them to its own queue, which still works because the handler instance owns the queue).
  - `_conversation_history: list[dict[str, Any]]` — swapped by the LLM adapter.
  - `_client: Any` — required for `_run_llm_with_tools`'s `assert self._client is not None`. Typed `Any` to avoid importing `google.genai` at type level.
  - `_prepare_startup_credentials() -> None`.
  - `_run_llm_with_tools() -> str`.
  - `_call_tts_with_retry(text, system_instruction=None) -> bytes | None`.
  - `_pcm_to_frames(pcm_bytes) -> list` (static-method on the legacy handler; the Protocol can't easily express staticmethod, so the Protocol declares it as a regular method — Python's structural matching doesn't care). Alternative: import the static directly from `robot_comic.gemini_tts` and bypass the handler attribute. We choose the latter to keep the Protocol surface lean: the adapter calls `GeminiTTSResponseHandler._pcm_to_frames(pcm_bytes)` as a static.

### 3. `_build_composable_gemini_tts` in `handler_factory.py`

Mirrors `_build_composable_gemini_elevenlabs` (4c.3) with substitutions: `GeminiLLMAdapter` → `GeminiBundledLLMAdapter`, `ElevenLabsTTSAdapter` → `GeminiTTSAdapter`, `GeminiTextElevenLabsHandler` → `LocalSTTGeminiTTSHandler`:

```python
def _build_composable_gemini_tts(**handler_kwargs: Any) -> Any:
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        GeminiBundledLLMAdapter,
        GeminiTTSAdapter,
        MoonshineSTTAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.gemini_tts import LocalSTTGeminiTTSHandler
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        legacy = LocalSTTGeminiTTSHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = GeminiBundledLLMAdapter(legacy)
        tts = GeminiTTSAdapter(legacy)
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

### 4. Factory branch wiring

Inside the existing `(moonshine, gemini_tts)` block (currently `handler_factory.py:304-312`), prepend the composable check:

```python
if output_backend == AUDIO_OUTPUT_GEMINI_TTS:
    if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
        logger.info(
            "HandlerFactory: selecting ComposableConversationHandler "
            "(%s → %s, llm=gemini-bundled, factory_path=composable)",
            input_backend, output_backend,
        )
        return _build_composable_gemini_tts(**handler_kwargs)
    from robot_comic.gemini_tts import LocalSTTGeminiTTSHandler

    logger.info(
        "HandlerFactory: selecting LocalSTTGeminiTTSHandler (%s → %s)",
        input_backend, output_backend,
    )
    return LocalSTTGeminiTTSHandler(**handler_kwargs)
```

### 5. Adapter `__init__.py` export

Add `GeminiBundledLLMAdapter` and `GeminiTTSAdapter` to `src/robot_comic/adapters/__init__.py`.

## Files Changed

| File | Change |
|------|--------|
| `src/robot_comic/adapters/gemini_tts_adapter.py` | NEW — ~140 LOC. Defines `_GeminiTTSCompatibleHandler` Protocol and `GeminiTTSAdapter`. |
| `src/robot_comic/adapters/gemini_bundled_llm_adapter.py` | NEW — ~110 LOC. Defines `GeminiBundledLLMAdapter` + the `_orchestrator_messages_to_gemini` helper. Re-imports the Protocol from `gemini_tts_adapter` to keep one source of truth. |
| `src/robot_comic/adapters/__init__.py` | EDIT — export both new adapters. |
| `src/robot_comic/handler_factory.py` | EDIT — composable gate inside `(moonshine, gemini_tts)` block + new `_build_composable_gemini_tts` helper. ~50 LOC delta. |
| `tests/adapters/test_gemini_tts_adapter.py` | NEW — mirrors `test_chatterbox_tts_adapter.py`: stub handler + prepare / synthesize / sentence-loop / tag handling / empty-text / shutdown / protocol-conformance tests. |
| `tests/adapters/test_gemini_bundled_llm_adapter.py` | NEW — mirrors `test_gemini_llm_adapter.py`: stub handler + prepare / chat / history-swap / message-shape conversion / empty tools / shutdown / protocol-conformance tests. |
| `tests/test_handler_factory_factory_path.py` | EDIT — add a Phase 4c.5 section with legacy + composable dispatch tests. |
| `PIPELINE_REFACTOR.md` | EDIT — mark 4c.5 ✅ Done and the 4c umbrella ✅ Done. |

No changes to `composable_pipeline.py`, `composable_conversation_handler.py`, the ABC, `main.py`, or any existing adapter / legacy handler module.

## Success Criteria

- `FACTORY_PATH=legacy` (default) + `(moonshine, gemini_tts)` → `LocalSTTGeminiTTSHandler` (bit-for-bit current behaviour).
- `FACTORY_PATH=composable` + same triple → `ComposableConversationHandler` whose pipeline holds `MoonshineSTTAdapter`, `GeminiBundledLLMAdapter`, `GeminiTTSAdapter` — all wrapping a single `LocalSTTGeminiTTSHandler` instance, all sharing one `genai.Client` instance.
- `wrapper.copy()` returns a different wrapper whose `_tts_handler` is a different `LocalSTTGeminiTTSHandler` instance.
- Voice methods (`change_voice`, `get_available_voices`, `get_current_voice`) work transparently through the wrapper → underlying handler.
- All other triples (4b, 4c.1, 4c.2, 4c.3, 4c.4) still return their composable wrappers.
- Bundled-realtime modes ignore `FACTORY_PATH`.
- `uvx ruff@0.12.0 check` / `uvx ruff@0.12.0 format --check` / mypy on changed files / pytest all green from the repo root.

## Test Plan

### Unit tests for `GeminiTTSAdapter` (`tests/adapters/test_gemini_tts_adapter.py`)

Stub handler simulates `_call_tts_with_retry` returning canned PCM bytes and `_pcm_to_frames` chunking them (or we call the real static `GeminiTTSResponseHandler._pcm_to_frames` directly — preferred, since it's a pure numpy function).

| Test | Asserts |
|------|---------|
| `test_prepare_calls_handler_prepare` | `adapter.prepare()` invokes `_prepare_startup_credentials`. |
| `test_synthesize_yields_audio_frames_for_one_sentence` | One sentence, one TTS call, N chunks → N `AudioFrame`s at 24 kHz. |
| `test_synthesize_yields_audio_frames_for_multiple_sentences` | Multi-sentence text → multiple `_call_tts_with_retry` calls; total frames concatenate. |
| `test_synthesize_strips_gemini_tags_from_spoken_text` | A sentence with `[annoyance]` markers passes `strip_gemini_tags`'d text to `_call_tts_with_retry`. |
| `test_synthesize_inserts_silence_for_short_pause_tag` | `[short pause]` tag emits an extra silence frame before the spoken-text frames. |
| `test_synthesize_forwards_delivery_tags_to_system_instruction` | A `[fast]` tag injects the cue into the `system_instruction` arg of `_call_tts_with_retry`. |
| `test_synthesize_skips_sentence_when_tts_returns_none` | `_call_tts_with_retry` returning None → no frame emitted for that sentence; loop continues. |
| `test_synthesize_with_empty_text_yields_nothing` | Empty input → empty generator. |
| `test_synthesize_ignores_protocol_tags_arg` | Adapter's `tags=("fast",)` kwarg is accepted-and-ignored (the adapter parses tags from the text). |
| `test_shutdown_is_noop` | `adapter.shutdown()` does not raise and does not touch the handler. |
| `test_adapter_satisfies_tts_backend_protocol` | `isinstance(adapter, TTSBackend)` is `True`. |

### Unit tests for `GeminiBundledLLMAdapter` (`tests/adapters/test_gemini_bundled_llm_adapter.py`)

Stub handler captures the `_conversation_history` that `_run_llm_with_tools` sees and returns a canned string.

| Test | Asserts |
|------|---------|
| `test_prepare_calls_handler_prepare` | `adapter.prepare()` invokes `_prepare_startup_credentials`. |
| `test_chat_returns_llmresponse_with_text_and_no_tool_calls` | `chat(messages)` returns `LLMResponse(text=<handler-return>, tool_calls=())`. |
| `test_chat_swaps_history_for_duration_of_call` | While `_run_llm_with_tools` runs, the handler's `_conversation_history` equals the converted-from-messages list. After the call returns, the handler's original history is restored. |
| `test_chat_restores_history_on_exception` | If `_run_llm_with_tools` raises, the saved history is still restored. |
| `test_chat_converts_orchestrator_messages_to_gemini_shape` | The history seen by `_run_llm_with_tools` has `[{"role": "user", "parts": [{"text": ...}]}]` shape — user→user, assistant→model, system→user, tool→dropped. |
| `test_chat_ignores_tools_arg` | Passing `tools=[...]` does not affect behaviour. |
| `test_chat_empty_history_passes_empty_list` | `chat([])` results in an empty `_conversation_history` during the call. |
| `test_shutdown_is_noop` | `adapter.shutdown()` does not raise and does not touch the handler. |
| `test_adapter_satisfies_llm_backend_protocol` | `isinstance(adapter, LLMBackend)` is `True`. |

### Factory dispatch tests (additions to `tests/test_handler_factory_factory_path.py`)

Mirrors the 4c.4 section, with `(moonshine, gemini_tts)`:

| Test | Asserts |
|------|---------|
| `test_legacy_path_returns_legacy_handler_for_gemini_tts` | `FACTORY_PATH=legacy` + `(moonshine, gemini_tts)` → `LocalSTTGeminiTTSHandler`. |
| `test_composable_path_returns_wrapper_for_gemini_tts` | `FACTORY_PATH=composable` + same triple → `ComposableConversationHandler` with `LocalSTTGeminiTTSHandler` as `_tts_handler`. |
| `test_composable_path_wires_three_adapters_for_gemini_tts` | `pipeline.stt/llm/tts` are `MoonshineSTTAdapter`, `GeminiBundledLLMAdapter`, `GeminiTTSAdapter`; all wrap the same legacy instance. |
| `test_composable_path_seeds_system_prompt_for_gemini_tts` | `pipeline._conversation_history[0]` is the patched `get_session_instructions` value. |
| `test_composable_path_copy_constructs_fresh_legacy_for_gemini_tts` | `copy()` produces a new wrapper + a new `LocalSTTGeminiTTSHandler`. |

### What we don't add tests for

- Hardware audio rendering on the robot (operator validates after merge).
- End-to-end transcript → audio path against real Gemini servers (covered by the legacy class's existing test suite).
- Voice switching propagation through the wrapper — covered by existing Phase 4a wrapper tests; the legacy handler exposes the same voice surface so existing tests pass unmodified. A single new test pinning the wrapper-forward path on a `LocalSTTGeminiTTSHandler`-shaped stub is added if mypy considers the wrapper's `_tts_handler: ConversationHandler` annotation a sticking point — the `LocalSTTGeminiTTSHandler` does inherit from `ConversationHandler` via `GeminiTTSResponseHandler` so this should be fine.
- The `_prepare_startup_credentials` double-init — same as 4c.3, deferred.

## Migration Notes

- Default behaviour unchanged — operators with `REACHY_MINI_FACTORY_PATH` unset (or `legacy`) keep `LocalSTTGeminiTTSHandler`.
- Operators who set `FACTORY_PATH=composable` AND `AUDIO_OUTPUT_BACKEND=gemini_tts` see `ComposableConversationHandler` wrapping `LocalSTTGeminiTTSHandler`. Behaviour is semantically equivalent: same Gemini Client, same LLM tool loop, same TTS path, same delivery-tag parsing.
- Reverting is one env-var flip.
- After this PR, all five `(moonshine, *, *)` composable triples are routed under `FACTORY_PATH=composable`. The umbrella 4c row in `PIPELINE_REFACTOR.md` flips to ✅ Done.

## Risks

- **`GeminiBundledLLMAdapter` bypasses the orchestrator's tool loop.** The orchestrator's `tool_dispatcher` is never invoked for this triple because `_run_llm_with_tools` dispatches internally and returns no `tool_calls`. This is **intentional and documented** — Phase 4e cleanup will refactor `GeminiTTSResponseHandler` to expose `_call_llm`-shaped surface so a single `GeminiLLMAdapter` can drive both this triple and `(moonshine, *, gemini)`. Today the legacy class doesn't use `GeminiLLMClient`; that refactor is out of scope. Operator-observable difference from legacy: **none** — the legacy `LocalSTTGeminiTTSHandler` also dispatches tools inside `_run_llm_with_tools` with no orchestrator involvement.
- **`shutdown()` is a no-op on both new adapters.** `genai.Client` has no explicit close method as used here; the legacy class doesn't own a separate httpx client for this triple. Phase 4e cleanup may revisit.
- **Joke-history capture and synthetic-status-marker filtering are not ported to the adapter.** The legacy `_dispatch_completed_transcript` does both (`gemini_tts.py:376` and `381-394`); the orchestrator's `_speak_assistant_text` does neither. **Regression for this triple under `FACTORY_PATH=composable`**: joke history will not record this triple's punchlines while operators are on the composable path. Acceptable for the staged rollout — joke history is a lifecycle-hook concern explicitly deferred per the operating manual ("Deferred lifecycle hooks" in `PIPELINE_REFACTOR.md`). Same gap exists for every other 4c triple already on `FACTORY_PATH=composable` — this is not a new regression introduced by 4c.5.
- **Voice methods on a `LocalSTTGeminiTTSHandler` instance via the wrapper** — `GeminiTTSResponseHandler.get_current_voice` returns a sync `str`; the wrapper forwards it sync. `change_voice` / `get_available_voices` are async; the wrapper forwards them async. All consistent with the `ConversationHandler` ABC's signatures.
- **`_orchestrator_messages_to_gemini` "system"→"user" mapping** — Gemini doesn't accept `role="system"` in history; the system prompt is meant to be passed as `system_instruction` to `GenerateContentConfig`. The legacy `_run_llm_with_tools` already sets `system_instruction=get_session_instructions()` (`gemini_tts.py:480`), so the system content shows up twice (once in `system_instruction`, once as a user-role primer). Cosmetic duplication, no behaviour break. Lifecycle-hook follow-up can clean this up by either (a) stripping `system` messages in the helper, or (b) passing them through `system_instruction` via Gemini SDK config — the latter requires API changes the adapter doesn't have today.

## After-merge follow-ups (out of scope for 4c.5)

- 4c-tris: hybrid realtime triples (`LocalSTTOpenAIRealtimeHandler`, `LocalSTTHuggingFaceRealtimeHandler`).
- Lifecycle hooks: per-PR rollout per the operating manual. Joke-history and history-trim are the most relevant for the gemini-tts triple specifically.
- Phase 4d: flip default to `composable`.
- Phase 4e: delete `LocalSTTGeminiTTSHandler` + the orphan `LocalSTTLlamaGeminiTTSHandler` confirmed unreachable from the factory; refactor `GeminiTTSResponseHandler` to expose `_call_llm` so the bundled adapter pair collapses to a single `GeminiLLMAdapter` + `GeminiTTSAdapter`.
