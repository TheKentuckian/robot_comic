# Phase 5 Exploration Memo — Post Pipeline-Refactor (Epic #337 closed)

**Date:** 2026-05-16
**Status:** Planning memo, NOT a final spec — for operator review before
any Phase 5 work is scoped.
**Predecessor epic:** #337 — pipeline refactor, closed by commit `1b6bc7b`
on `origin/main`.
**Style template:** mirrors `docs/superpowers/specs/2026-05-15-phase-4-exploration.md`
and `docs/superpowers/specs/2026-05-15-phase-4c-tris-hybrid-realtime-design.md`.

This memo is a decision document. §1 recaps what Phase 4 actually shipped;
§2 expands each "out of scope for Phase 4" item with concrete file-and-line
citations; §3 inventories the surviving TODOs; §4 brainstorms sub-phase
groupings as a DAG; §5 gives the recommendation; §6 is a focused reading
list for the future Phase 5 manager.

No `src/` or `tests/` changes ride along with this PR. Path-filtered CI
(`.github/workflows/{lint,typecheck,pytest,uv-lock-check}.yml`) will not run
for a docs-only diff; `mergeable: MERGEABLE, mergeStateStatus: CLEAN,
statusCheckRollup: []` is the normal state — squash-merge directly.

---

## §1 — Recap of Phase 4 outcome

Phase 4 (`PIPELINE_REFACTOR.md`) executed Option C from the Phase 4
exploration memo (incremental retirement). The full sub-phase table is
locked in `PIPELINE_REFACTOR.md:11-24`; the abbreviated outcome:

- **4a (#355)** — `ComposableConversationHandler(ConversationHandler)`
  wrapper over `ComposablePipeline`.
- **4b (#359)** — factory dual path behind `REACHY_MINI_FACTORY_PATH=legacy|composable`,
  one triple wired (`moonshine + llama + elevenlabs`).
- **4c.1–4c.5 (#361 / #362 / #364 / #365 / #366)** — `ChatterboxTTSAdapter`,
  `GeminiLLMAdapter`, `GeminiTTSAdapter`, `GeminiBundledLLMAdapter` shipped;
  all five non-hybrid composable triples routed through the wrapper.
- **4c-tris (#369)** — **Skipped (Option B per operator).** The two hybrid
  handlers `LocalSTTOpenAIRealtimeHandler` (`local_stt_realtime.py:869`)
  and `LocalSTTHuggingFaceRealtimeHandler` (`local_stt_realtime.py:910`)
  survive forever; the dial no-ops for `(moonshine, openai_realtime_output)`
  and `(moonshine, hf_output)` — they return the same legacy concrete class
  on both `FACTORY_PATH` values. See
  `docs/superpowers/specs/2026-05-15-phase-4c-tris-hybrid-realtime-design.md`
  §6 for the recommendation and §4.3 for the long-term cost.
- **4d (#378)** — `DEFAULT_FACTORY_PATH = FACTORY_PATH_COMPOSABLE`.
- **4e (#379)** — six legacy concrete handler classes deleted; `FACTORY_PATH`
  retired; factory now only exposes the composable path plus the two
  hybrid carve-outs.
- **4f (#381)** — `BACKEND_PROVIDER` and `LOCAL_STT_RESPONSE_BACKEND` config
  dials retired across five surfaces (`prompts.py`, `base_realtime.py`,
  `main.py`, `console.py`, admin UI). `PROVIDER_ID` ClassVar survives on
  the bundled-realtime handlers for telemetry attribute values.
- **Lifecycle hooks #1–#5** (`PIPELINE_REFACTOR.md:308-314`) — all five
  landed before 4d. Doc's "for free via adapter delegation" claim was wrong
  on 5/5 hooks; every fix required real code changes plus regression tests.
- **Lifecycle hook #3b (#382)** — late add: `GeminiTTSAdapter` now emits
  `first_greeting.tts_first_audio` via `emit_first_greeting_audio_once()`
  (`adapters/gemini_tts_adapter.py:197-205`). Flagged during hook #3 and
  closed before this memo.

**Net state on `origin/main` at `1b6bc7b`:**

- Composable path is the only path for five triples; hybrid carve-out for
  two; bundled-realtime fast path for three (`HuggingFaceRealtimeHandler`,
  `OpenaiRealtimeHandler`, `GeminiLiveHandler`).
- Seven adapters in `src/robot_comic/adapters/`:
  `moonshine_stt_adapter.py` (1 STT),
  `llama_llm_adapter.py` + `gemini_llm_adapter.py` + `gemini_bundled_llm_adapter.py`
  (3 LLM), `elevenlabs_tts_adapter.py` + `chatterbox_tts_adapter.py` +
  `gemini_tts_adapter.py` (3 TTS).
- Five factory-private mixin host classes
  (`handler_factory.py:131-163`) compose `LocalSTTInputMixin` over each
  surviving `*ResponseHandler` base. These hosts are what the adapters
  reach into; they're not exported.
- Surviving response-handler bases used as adapter targets:
  `BaseLlamaResponseHandler`, `LlamaElevenLabsTTSResponseHandler`,
  `ChatterboxTTSResponseHandler`, `GeminiTextChatterboxResponseHandler`,
  `GeminiTextElevenLabsResponseHandler`, `GeminiTTSResponseHandler`, plus
  `BaseLlamaResponseHandler._call_llm` / `GeminiTextResponseHandler._call_llm`
  / `_run_llm_with_tools` etc. as internal entry points.

The five-row "Deferred lifecycle hooks" table (`PIPELINE_REFACTOR.md:308-314`)
is now ✅ on every row. Epic #337 is mergeably done.

---

## §2 — Phase 4 out-of-scope carry-forwards

`PIPELINE_REFACTOR.md:384-390` lists Phase 4's explicit out-of-scope items.
Each is expanded below with concrete file-and-line context plus an honest
scope estimate.

### 2.1 New STT backends (Whisper, Distil-Whisper, Deepgram, NVIDIA Parakeet, faster-whisper)

**Today's STT surface** is a two-piece structure: a `LocalSTTInputMixin`
(`local_stt_realtime.py:230-866`) that drives a Moonshine streaming
recogniser via its bundled `_MoonshineListener`, and a thin
`MoonshineSTTAdapter` (`adapters/moonshine_stt_adapter.py`) that exposes
the mixin-hosted handler as the `STTBackend` Protocol from
`backends.py:111-136`.

The adapter is intimately coupled to the mixin's *implementation* shape,
not just the Protocol shape:

- `MoonshineSTTAdapter.__init__` (`moonshine_stt_adapter.py:40-48`) takes a
  *host handler* (not a Moonshine client), which must mix in
  `LocalSTTInputMixin`.
- `start()` (`:50-75`) monkey-patches
  `handler._dispatch_completed_transcript` so the Protocol callback
  intercepts before the legacy method runs. This works because the mixin's
  internal Moonshine listener fires `_dispatch_completed_transcript(text)`
  on the asyncio loop when a transcript completes.
- `feed_audio()` (`:77-82`) calls `handler.receive((sample_rate, samples))`,
  which is the mixin's audio ingestion path (`local_stt_realtime.py:748-834`).
- `stop()` (`:84-93`) restores the original dispatch and shuts the handler.

The adapter is therefore a single-purpose bridge: it doesn't abstract
"any STT backend," it adapts *one specific mixin's implementation surface*
to the Protocol shape. The Protocol is defined; nothing implements it
*independently of Moonshine*.

**What adding a new STT entails today:** the work depends on which class
of backend:

- **Locally-loaded streaming model (faster-whisper, Distil-Whisper)** —
  closest fit. Could either (a) extend `LocalSTTInputMixin` to abstract
  over the underlying transcriber, or (b) write a parallel
  `LocalWhisperInputMixin` that mirrors the Moonshine listener shape and
  ship a `WhisperSTTAdapter` mirroring `MoonshineSTTAdapter`. Option (a)
  is cleaner long-term but the mixin is 636 lines and tightly bound to
  Moonshine semantics (the `_pending_stream_rearm` hack for one-line-per-stream
  is Moonshine-specific, see #279). Option (b) doubles the mixin file with
  parallel structure.
- **Network streaming (Deepgram)** — doesn't fit the mixin model at all
  (no local model loader, no per-frame `add_audio` push; protocol is a
  websocket where you stream raw PCM and consume transcript events). A
  network-streaming STT adapter would implement `STTBackend` directly
  without a host-handler indirection. That's the *first* implementor of
  `STTBackend` that isn't an adapter-over-legacy-mixin; it would
  exercise the Protocol abstraction for the first time in anger.
- **Cloud chunked API (NVIDIA Parakeet, Whisper API)** — neither streaming
  nor mixin-based. Different lifecycle altogether (buffer until silence,
  send chunk, receive transcript). A `ParakeetSTTAdapter` would be a
  buffering wrapper.

The **factory plumbing** is also asymmetric. `handler_factory.py:246-348`
branches on `input_backend == AUDIO_INPUT_MOONSHINE` and assumes the
mixin host pattern. A new STT would either need (a) a new
`AUDIO_INPUT_*` constant + a parallel branch + factory-private host
classes mirroring `_LocalSTT*Host` for each TTS, or (b) a refactor of
the factory to compose the STT adapter independently of the TTS host —
which is most of what Phase 5 should do.

**Honest scope estimate per new STT backend:**

- Local streaming, mirror-the-mixin approach: 2–3 sessions per backend
  (one to write the mixin/adapter; one for factory wiring across all five
  TTS triples; one for tests and #279-class regression coverage).
- Network/cloud, Protocol-direct approach: 1–2 sessions for the adapter +
  tests, plus 1–2 sessions for the factory refactor that decouples the
  STT from the host mixin. **The factory refactor is the long pole.**

The right Phase 5 move is **factory STT decoupling first**, then one
backend at a time. See §4.

### 2.2 Tool-system refactor (orchestrator's `tool_dispatcher` callback wiring)

`ComposablePipeline` defines `tool_dispatcher: ToolDispatcher | None`
(`composable_pipeline.py:75-78`, `:105`, `:115`, `:225-249`) — a callback
that takes a `ToolCall` and returns a `str` result for the LLM's next
turn. The orchestrator's loop in `_run_llm_loop_and_speak`
(`composable_pipeline.py:219-240`) is conditional:

```python
if response.tool_calls:
    if self.tool_dispatcher is None:
        logger.warning("LLM requested tools but no dispatcher is configured; ignoring ...")
        break
    await self._dispatch_tools_and_record(response.tool_calls)
    continue
```

**In production, `tool_dispatcher` is always `None`** — every factory
builder in `handler_factory.py:392-619` constructs `ComposablePipeline(
stt, llm, tts, system_prompt=...)` with no dispatcher argument. The
warning branch never fires because every LLM adapter today *also* returns
`tool_calls=()` from its `chat()` even when tools are dispatched. The
mechanism is double-redirected:

- `LlamaLLMAdapter.chat` (`llama_llm_adapter.py:55-92`) calls
  `handler._call_llm(extra_messages=messages)` which returns
  `(text, raw_tool_calls, raw_msg)`. The adapter converts `raw_tool_calls`
  to the Protocol's `ToolCall` shape and stuffs them in `LLMResponse.tool_calls`.
  But the *legacy* `_call_llm` only fires *once* in the original handler's
  `_run_response_loop`, with tool dispatch handled by the *outer loop*
  there. The composable path's outer loop is `_run_llm_loop_and_speak`,
  which sees `LLMResponse(tool_calls=...)` and would dispatch via
  `tool_dispatcher` — but `tool_dispatcher` is `None`, so the loop
  *breaks*.
- `GeminiLLMAdapter.chat` (`gemini_llm_adapter.py:55-118`) has the same
  shape.
- `GeminiBundledLLMAdapter.chat` (`gemini_bundled_llm_adapter.py`) does
  things *differently* — it calls `handler._run_llm_with_tools()` which is
  a *bundled* method that dispatches tools internally and returns the
  final assistant text only. So this adapter returns `LLMResponse(text=...,
  tool_calls=())` always (see its module docstring lines 12–19: "tools
  are dispatched inside the wrapped handler against its `self.deps`").

**So tools work in production only by accident**: the bundled-Gemini case
dispatches internally; the llama+chatterbox / llama+elevenlabs /
gemini+chatterbox / gemini+elevenlabs cases dispatch *not at all on the
composable path*, because `_call_llm` is only one round-trip and the
multi-round loop in legacy `_run_response_loop` (`llama_base.py:570-617`)
is bypassed.

Wait — that can't be right, since the hooks 1–5 work landed without
regressions. Let me re-read. Yes, it is right: when the LLM requests
tools, the composable path's pipeline gets `LLMResponse(tool_calls=...)`,
falls through the `if self.tool_dispatcher is None` warning branch, and
*breaks the loop without speaking*. The user gets silence on that turn.
The reason this didn't blow up during 4c.1–4c.5 hardware validation is
that the smoke test transcripts didn't trigger tool calls. **This is a
latent bug, not a clean carry-forward.** Worth flagging — see §3 / §5.

A real tool-system refactor would:

1. **Wire `tool_dispatcher` at factory build time.** Every
   `_build_composable_*` helper in `handler_factory.py:392-619` should
   construct a `ToolDispatcher` shim that knows `ToolDependencies` and
   calls `tools.core_tools.dispatch_tool_call(name, args_json, deps)`.
   The shim is ~40 LOC mirroring `BaseLlamaResponseHandler._start_tool_calls`
   + `_await_tool_results` (`llama_base.py:570-617`). The exploration
   memo §6.2 (`2026-05-15-phase-4-exploration.md:179`) flagged this and
   the answer it landed on — "adapter keeps doing what it does today" —
   is partially right (bundled-Gemini does dispatch internally) and
   partially wrong (the other four triples don't dispatch at all on the
   composable path).
2. **Background tools.** `BaseLlamaResponseHandler._start_tool_calls`
   distinguishes synchronous from background tools
   (`llama_base.py:572-617`, references `bg_tools`). The composable
   orchestrator's `await tool_dispatcher(call)` model is synchronous-return
   per call — there's no place for `BackgroundToolManager` to slip in.
   Two answers: (a) `tool_dispatcher` returns immediately with a placeholder
   for background tools and `BackgroundToolManager` continues to fire
   completion notifications back through the legacy notification queue;
   (b) the orchestrator gains a "background results pending" channel and
   awaits them between turns. (a) is cheaper.
3. **Tool-spec source.** `LlamaLLMAdapter.chat`'s `tools` arg is *ignored*
   (`llama_llm_adapter.py:65-70`); the legacy handler reads from
   `get_active_tool_specs(deps)`. A proper refactor would make the
   orchestrator's `tools_spec` the source of truth, populated by the
   factory builder calling `get_active_tool_specs(deps)` once at
   construction.

**Honest scope:** 2–4 sessions. The shim is small; the multiplicative
test coverage (per-triple integration test that tool calls round-trip
correctly) is bigger.

### 2.3 Voice / personality method redesign

`composable_conversation_handler.py:175-185` forwards
`get_available_voices`, `get_current_voice`, and `change_voice` to
`self._tts_handler` (the underlying legacy response handler held by
reference for exactly this reason). `apply_personality` (`:156-173`)
does its own work: `set_custom_profile(profile)` then
`pipeline.reset_history(keep_system=False)` then re-seeds
`pipeline._conversation_history` with a fresh `system` message.

The TODO on `composable_conversation_handler.py:158-166` reads:

> legacy handlers also clear per-session echo-guard state on persona
> switch. Compose that in when the echo-guard hook lands. Joke history
> is a cross-persona file (`~/.robot-comic/joke-history.json`) …
> intentionally persists across personality switches …

Lifecycle Hook #1 (echo-guard, PR #372) shipped — but that hook
addressed the per-turn `_speaking_until` write site, not the per-persona
state reset. So the TODO is partially obsolete and partially live: the
echo-guard write site is wired, but if there's per-session
echo-guard state on the legacy handler that should reset on persona
switch, the composable path doesn't clear it. Worth a half-session of
audit.

**The deeper redesign question** is that voice/personality methods on
`ConversationHandler` are conflated:

- `apply_personality` does conversation-state surgery (history reset +
  system prompt) that's *pipeline-shaped* — it should live on the
  pipeline.
- `change_voice` / `get_available_voices` / `get_current_voice` are
  *TTS-shaped* — they should live on `TTSBackend`.

Today the `TTSBackend` Protocol (`backends.py:172-211`) has neither.
`composable_conversation_handler.py` works around this by holding a
reference to the underlying TTS-handler-instance and forwarding to it.
This works because every TTS-holding response handler implements those
methods directly. But it means:

- The wrapper can't accept an arbitrary `TTSBackend` — it needs the
  underlying handler. A future "fully-decoupled" composable pipeline
  (Phase 5 §4.2 of the exploration memo's roadmap) wouldn't fit.
- New TTS adapters that don't wrap a legacy response handler would have
  to implement the four methods themselves — or the wrapper would have
  to grow `if self._tts_handler is None: …` branches.

The right shape is probably to extend `TTSBackend` with the three
voice methods, default-implemented as `NotImplementedError`, and have
`apply_personality` move onto `ComposablePipeline`. Then the wrapper
becomes a thin pass-through that doesn't hold the legacy handler at
all — that's the "Phase 4a was deferred" thing the original exploration
memo (§6.4) flagged.

**Honest scope:** 1–2 sessions. The Protocol change touches every TTS
adapter; the test surface is small (parametric voice-switch tests
exist).

### 2.4 `ConversationHandler` ABC rewrite

`conversation_handler.py:19-69` defines the ABC. Its concrete shape
today:

```python
class ConversationHandler(ABC):
    deps: ToolDependencies
    output_queue: asyncio.Queue[QueueItem]
    _clear_queue: Callable[[], None] | None
    @abstractmethod def copy(self) -> ConversationHandler: ...
    @abstractmethod async def start_up(self) -> None: ...
    @abstractmethod async def shutdown(self) -> None: ...
    @abstractmethod async def receive(self, frame: AudioFrame) -> None: ...
    @abstractmethod async def emit(self) -> HandlerOutput: ...
    @abstractmethod async def apply_personality(self, profile: str | None) -> str: ...
    @abstractmethod async def get_available_voices(self) -> list[str]: ...
    @abstractmethod def get_current_voice(self) -> str: ...
    @abstractmethod async def change_voice(self, voice: str) -> str: ...
```

What's specifically wrong with it:

1. **It conflates three roles.** It's a FastRTC adapter shim
   (`copy()`, `receive()`, `emit()`, `output_queue`, `_clear_queue`), a
   lifecycle owner (`start_up()`, `shutdown()`), and a conversation
   surface (`apply_personality()`, voice methods). Implementors today
   pay the cost of all three even when one is irrelevant
   (e.g. `BaseRealtimeHandler` has no meaningful `_clear_queue` semantics
   on the live path).
2. **`deps: ToolDependencies` is a misplaced field.** The factory passes
   `deps` to every concrete handler so tools work; but with Phase 4's
   adapter pattern, `deps` is actually owned by the legacy
   `*ResponseHandler` inside the wrapper, not by the wrapper itself. The
   wrapper's `deps` is a duplicate. The factory injects it at
   `composable_conversation_handler.py:50` to satisfy the ABC's class-level
   annotation, then never reads it on the wrapper itself.
3. **`_clear_queue` is a leaky abstraction.** It's a callback that
   `LocalStream.__init__` writes to so barge-in can flush the player. The
   wrapper has to *mirror* it onto the underlying TTS handler
   (`composable_conversation_handler.py:74-94`) because the
   `LocalSTTInputMixin` listener inside the host calls `self._clear_queue`
   on the *legacy host instance*, not on the wrapper. The setter logic is
   five lines of indirection that exists because the ABC put `_clear_queue`
   on the wrong object.
4. **`copy()` semantics are FastRTC-specific.** FastRTC clones the
   handler per-peer. The composable wrapper implements
   `copy()` as `return self._build()` (the factory closure). The ABC
   makes this a top-level concern even for use cases that don't have
   peer-clone semantics (e.g. unit tests, sim mode).
5. **`emit()` returns `HandlerOutput = Any`.** The type system gives no
   protection; every implementor defines its own queue item shape.
6. **There's no abstraction over the conversation loop itself.** The ABC
   is a *handler surface* (one event in, one event out) — it doesn't
   express the conversation flow. `ComposablePipeline` is that
   expression, sitting *inside* a `ConversationHandler` implementor. The
   ABC could be smaller (FastRTC adapter only) and the conversation flow
   could live on a separate `ConversationOrchestrator` type that the
   adapter delegates to.

`PIPELINE_REFACTOR.md` doesn't say *what* needs to change about the ABC
— it just lists it as Phase 5. The exploration memo's §6.4
(`2026-05-15-phase-4-exploration.md:183`) sketched the surface gap
between `ComposablePipeline` and `ConversationHandler` ABC that
`ComposableConversationHandler` was built to bridge. Phase 5 would
either (a) shrink the ABC to the FastRTC-shim role and move
voice/personality off it, or (b) collapse the ABC entirely and have the
factory return `ComposablePipeline` directly with FastRTC adapter
methods on the pipeline. (b) is bolder but the existing bundled-realtime
handlers (`HuggingFaceRealtimeHandler` et al.) still need an ABC to
satisfy because they don't decompose into a pipeline.

**Honest scope:** 3–5 sessions. Big diff (every concrete handler must be
audited), high reviewability per PR (each method moved is a separate
small PR), low risk if done one method at a time.

---

## §3 — Surviving TODOs flagged during Phase 4

A `Grep` of `src/robot_comic/` for `TODO|FIXME|XXX` returns four hits
inside the new code added by epic #337. Most pre-existing TODOs are
outside this scope.

### 3.1 `composable_conversation_handler.py:158-166` — `apply_personality` per-session state

```python
# TODO(phase4-lifecycle): legacy handlers also clear per-session
# echo-guard state on persona switch. Compose that in when the
# echo-guard hook lands. Joke history is a cross-persona file …
```

Lifecycle Hook #1 (echo-guard, PR #372) landed *one specific*
write-site move — into `_enqueue_audio_frame`. The TODO is about
*reset semantics on persona switch*: when the operator switches
persona mid-session, the composable path's wrapper does
`pipeline.reset_history(keep_system=False)` but doesn't touch the
echo-guard state on the underlying TTS handler. If `_speaking_until`
points to a long-running playback that's about to end, the next
persona's STT would still drop frames for the remaining window.

**Category:** (b) Phase 5 candidate. Half-session of work to audit the
legacy handlers' per-session state and wire equivalents.

### 3.2 `composable_pipeline.py:278-283` — delivery tag plumbing

```python
# TODO(phase3): plumb delivery tags from ``response.metadata`` into
# ``tts.synthesize(text, tags=...)``. The TTS Protocol accepts a
# ``tags`` tuple (``fast``/``slow``/``annoyance``/etc.) that the
# existing ElevenLabs handler uses, but it has no channel here yet
# — adapters in Phase 3 will surface the existing handlers' tag
# extraction and we'll thread it through metadata at that point.
```

`GeminiTTSAdapter` and `ChatterboxTTSAdapter` both parse delivery tags
out of the LLM-generated text itself (see
`gemini_tts_adapter.py:166-167` and module docstring "Tag handling —
known gap"). `ElevenLabsTTSAdapter` *drops* the tags entirely; legacy
`_stream_tts_to_queue` accepted them as a separate parameter, and the
adapter passes `tags_list = list(tags) if tags else None`
(`elevenlabs_tts_adapter.py:127`). Tags from `tags_list` work; tags
from `response.metadata` don't exist yet.

**Category:** (b) Phase 5 candidate. One session to thread metadata
through — touches `LLMResponse.metadata`, every LLM adapter (to populate
it), `_speak_assistant_text`, every TTS adapter (to consume from
`tags=` parameter instead of parsing from text).

### 3.3 `chatterbox_tts_adapter.py:47-50` — frame-shape TODO + first-audio marker

```python
# ``isinstance(item, tuple)`` unpack. Mirrors the parallel TODO in
# ...
# - **No first-audio marker.** Same TODO as :class:`ElevenLabsTTSAdapter`.
```

Two micro-TODOs:

- **Tuple unpack** — the chatterbox handler pushes `(sample_rate, ndarray)`
  tuples, which the adapter unpacks via `isinstance(item, tuple)`. Pattern
  parity with the ElevenLabs adapter. Not a bug, just brittle to a future
  legacy-handler refactor.
- **First-audio marker** — same gap `ElevenLabsTTSAdapter.synthesize`
  flags in its module docstring (`elevenlabs_tts_adapter.py:43-49`): the
  legacy `_stream_tts_to_queue` accepts a `first_audio_marker: list[float]
  | None` for echo-guard / first-audio-latency telemetry. The adapter
  doesn't surface it. Lifecycle Hook #3b (PR #382) closed the
  `first_greeting.tts_first_audio` event for the `GeminiTTSAdapter` case
  specifically, but the marker mechanism is broader — it's the
  per-turn first-audio timestamp that feeds the boot-timeline.

**Category:** (b) Phase 5 candidate. Combine with §3.2 — both are
"plumb non-text data through the TTSBackend Protocol."

### 3.4 The OTel `gen_ai.system` "inheritance accident"

Phase 4 noted (`adapters/gemini_llm_adapter.py:103-105`):

> ``gen_ai.system="gemini"`` is semantically correct (mirrors
> ``elevenlabs_tts.py:244``); the legacy ``_run_response_loop`` emits
> ``"llama_cpp"`` by inheritance accident — fixed on the new surface.

Post-4e the legacy `_run_response_loop` is *gone* (deleted with the
concrete handler classes), so the inheritance accident no longer emits
new rows. **But the historical data is still in dashboards** — any
gen-ai-attribution query that filters on
`gen_ai.system="llama_cpp"` will conflate llama-server traffic with
historical gemini-text traffic for queries spanning the pre-4e period.
This is a dashboard artifact, not a bug to fix in code.

**Category:** (c) deferred-forever / live-with. Document if it surfaces
in a dashboard review.

### 3.5 `tests/test_llama_base.py` `coroutine never awaited` RuntimeWarnings

Flagged in `docs/superpowers/specs/2026-05-16-lifecycle-record-joke-history.md:211-214`
during PR #375 review: a previous agent noticed
`coroutine never awaited` RuntimeWarnings in `test_llama_base.py` on
`main`. Confirmed pre-existing, unrelated to the joke-history hook.

`Grep` of the source tree for the warning content turns up no
production-code matches; the issue is test-fixture-only. Most likely
suspect: a test that constructs an async-mock and forgets to `await` or
register a `return_value` of an awaitable.

**Category:** (a) trivial cleanup. Half-session to find and fix.

---

## §4 — Brainstorm sub-phase groupings

Phase 5 broken into sub-phases analogous to Phase 4. Each has a goal,
estimated scope, predecessor DAG, pause points, and risk.

```
                              ┌── 5b: Tool-dispatcher wiring (LATENT-BUG FIX)
                              │
        5a: Adapter cleanup ──┼── 5c: Voice/personality split onto TTSBackend
                              │           │
                              │           └── 5d: ConversationHandler ABC shrink
                              │
                              └── 5e: Factory STT decoupling
                                         │
                                         └── 5f: New STT backend(s) (per-backend PRs)
```

5a is a low-risk prelude that lands the trivial TODOs (§3.1, §3.2, §3.3,
§3.5). It's not a predecessor for anything — runs independently any time.
5b is the latent-bug fix from §2.2 and is *probably the right place to
start* (see §5). 5c → 5d is the ABC redesign. 5e → 5f is the STT
extension work.

### 5a — Adapter & TODO cleanup

**Goal:** close §3's trivial TODOs and the per-persona echo-guard reset.
Pay down the small debts before structural work.

- §3.1 — `apply_personality` echo-guard reset audit (~150 LOC source +
  test).
- §3.2 + §3.3 — delivery-tag plumbing through `LLMResponse.metadata` →
  `TTSBackend.synthesize(tags=...)` → every TTS adapter consumes from
  the parameter rather than parsing the text. Adds first-audio-marker
  plumbing while we're in there.
- §3.5 — fix the `coroutine never awaited` RuntimeWarnings in
  `test_llama_base.py`.

**Estimated scope:** 1–2 sessions, ~3 PRs (one per item).
**Predecessors:** none — independent.
**Pause points:** none — pure cleanup.
**Risk:** low.

### 5b — Tool-dispatcher wiring (LATENT-BUG FIX)

**Goal:** close the §2.2 latent bug. Wire a real `ToolDispatcher` into
every composable factory builder so tool calls round-trip on the
composable path for `(moonshine, llama, *)` and `(moonshine, gemini, *)`
triples — *not* just the bundled-Gemini case.

The dispatcher shim mirrors `BaseLlamaResponseHandler._start_tool_calls`
+ `_await_tool_results` (`llama_base.py:570-617`) but adapted to the
orchestrator's `await tool_dispatcher(call)` model. Background tools
(`bg_tools` list) need an answer — recommend "return immediately with a
placeholder; let `BackgroundToolManager` keep firing completion
notifications via the existing notification queue."

**Estimated scope:** 2–4 sessions, 2–3 PRs (shim + per-triple test + tool
spec source-of-truth migration).
**Predecessors:** none — independent of 5a.
**Pause points:** **mandatory hardware validation** between shim landing
and per-triple flip. The bug is latent today because smoke-test
transcripts don't trigger tool calls; the validation has to use a
transcript that *does* trigger a tool to verify the wiring works.
**Risk:** medium. The latent bug means there's no existing regression
test to compare against; the operator's manual hardware test is the
ground truth.

### 5c — Voice/personality split onto `TTSBackend` Protocol

**Goal:** §2.3. Add `change_voice` / `get_available_voices` /
`get_current_voice` to `TTSBackend` Protocol (default-implemented as
`NotImplementedError`). Move `apply_personality` onto `ComposablePipeline`
(it's pipeline-shaped). Shrink the wrapper's voice/personality methods
to thin pass-throughs that don't rely on holding the legacy handler by
reference.

Touches: `backends.py`, every TTS adapter
(`chatterbox_tts_adapter.py`, `elevenlabs_tts_adapter.py`,
`gemini_tts_adapter.py`), `composable_pipeline.py`, and
`composable_conversation_handler.py`. Plus
`handler_factory.py:392-619` to drop the `tts_handler=host` argument
from the wrapper builder once the wrapper no longer needs it.

**Estimated scope:** 2 sessions, 2 PRs (Protocol extension + adapter
migration; wrapper simplification).
**Predecessors:** none — but ideally runs after 5a's tag plumbing so the
TTS Protocol churn happens once.
**Pause points:** none if the existing parametric voice-switch tests
cover all triples; one-session hardware spot-check otherwise.
**Risk:** low-medium. Protocol churn but the abstraction direction is
sound.

### 5d — `ConversationHandler` ABC shrink

**Goal:** §2.4. Move the ABC toward the FastRTC-shim role. Specifically:
remove `apply_personality` / voice methods from the ABC (now on the
pipeline / TTS Protocol after 5c). Audit `deps` — is the field needed
on the ABC at all post-5c? Probably not; the factory carries it. Audit
`_clear_queue` — can the mirroring shim
(`composable_conversation_handler.py:74-94`) be deleted once it's no
longer needed?

Side-effect: every concrete handler in the bundled-realtime path
(`HuggingFaceRealtimeHandler`, `OpenaiRealtimeHandler`, `GeminiLiveHandler`,
the two `LocalSTT*RealtimeHandler` hybrids) has to be audited because
they inherit voice/personality from `BaseRealtimeHandler`
(`base_realtime.py:300-377`). The shrink doesn't *delete* those
methods; it just removes the ABC's `@abstractmethod` annotation.
`BaseRealtimeHandler`'s implementations stay.

**Estimated scope:** 3 sessions, 3–4 PRs (one per method moved).
**Predecessors:** 5c (5d's "move voice methods off ABC" depends on 5c's
"put voice methods onto TTSBackend").
**Pause points:** hardware validation after the ABC shape settles, before
the wrapper's `_tts_handler` field is deleted (which is the final
visible-change PR).
**Risk:** medium. Bundled-realtime handlers are out of #337 scope; this
PR touches them for the first time since Phase 0.

### 5e — Factory STT decoupling

**Goal:** §2.1 prerequisite. Refactor `handler_factory.py:246-348` so
the STT adapter is composed independently of the TTS host. Today every
`_build_composable_*` helper instantiates a factory-private mixin host
class (`_LocalSTTLlamaElevenLabsHost` etc., `handler_factory.py:131-163`)
that bakes `LocalSTTInputMixin` over the TTS response handler. Adding a
non-Moonshine STT requires either parallel mixin hosts (host count
multiplies by STT count × TTS count) or this refactor.

The decoupling pattern: the factory composes an STT adapter
(`MoonshineSTTAdapter` today, others tomorrow) and a TTS host (the
existing `_LocalSTT*Host` minus the `LocalSTTInputMixin`) separately,
then wires them through `ComposablePipeline`. The `MoonshineSTTAdapter`
becomes a standalone Protocol implementor — no host-handler indirection.

This is the *biggest* of the §2 carry-forwards in raw LOC terms (every
factory builder rewrites) but each builder rewrite is mechanical and the
test surface is parametric. Recommend one PR per triple, mirroring 4c's
sub-PR cadence.

**Estimated scope:** 5 sessions, 5–6 PRs (one per triple + a Moonshine
adapter rewrite to detach from the host pattern).
**Predecessors:** none — but ideally after 5d (the ABC shrink) so the
wrapper's interface is settled.
**Pause points:** **mandatory hardware validation per-triple**, mirroring
Phase 4c's cadence.
**Risk:** medium. The Moonshine listener's `_pending_stream_rearm`
recovery (`local_stt_realtime.py:265, 748-834`, see #279) is
implementation-tangled with the host pattern; pulling it apart is the
sharp edge.

### 5f — New STT backends (per-backend, gated on 5e)

**Goal:** ship one or more of: faster-whisper, Distil-Whisper, Deepgram,
NVIDIA Parakeet. Each is a fresh `STTBackend` implementor written against
the post-5e clean Protocol surface.

**Estimated scope:** 1–3 sessions per backend, depending on the class
(see §2.1 breakdown).
**Predecessors:** 5e (factory decoupling).
**Pause points:** operator chooses which backend(s) to ship and in what
order, after 5e lands.
**Risk:** medium-high per backend — each has its own integration
quirks (network timeouts, model-loading latencies, language detection
fallbacks). Phase 5 should *not* commit to a backend list before 5e is
done; instead, 5e unblocks the menu.

---

## §5 — Recommendations

### 5.1 Start with 5b (tool-dispatcher wiring)

The latent bug in §2.2 is the most consequential carry-forward. Today
the composable path silently breaks the conversation loop when an LLM
on the `(moonshine, llama, *)` or `(moonshine, gemini, *)` triples
requests a tool. The bundled-Gemini triple works by coincidence.
4e's hardware soak didn't catch it because Don Rickles-style transcripts
trigger tools rarely; the moment a profile or operator session exercises
tools, the user hears silence on that turn.

**5b should be Phase 5's first sub-phase, not 5a.** 5a (TODO cleanup)
is cheap and reviewable, but it's not a bug. 5b is.

The operator should also verify on hardware whether the latent-bug
hypothesis is correct before committing to 5b — a single test
conversation that triggers a tool call (e.g. asking the robot to
dance) on the composable `(moonshine, llama, elevenlabs)` path would
either confirm the bug or reveal that something else (a fallback in the
adapter or a leak through `_call_llm`'s internals) is keeping the turn
alive.

### 5.2 Items to defer indefinitely

- **OTel `gen_ai.system` inheritance accident historical data (§3.4).**
  Documented; no fix path. Operators querying multi-month gen-ai data
  should `OR` the filter to include `llama_cpp` rows from the pre-4e
  window.
- **The two hybrid `LocalSTT*RealtimeHandler` classes (§1, also Phase
  4c-tris).** Already declared "legacy forever" by the operator. No
  Phase 5 action needed. If a future contributor proposes wrapping them,
  point them at `docs/superpowers/specs/2026-05-15-phase-4c-tris-hybrid-realtime-design.md`.
- **`_clear_queue` mirroring shim
  (`composable_conversation_handler.py:74-94`).** Will probably disappear
  during 5d but if 5d gets descoped, leaving the mirror in place is
  *fine* — it's five lines, well-commented, and works. Don't refactor
  for elegance alone.
- **Background tool dispatch.** Touched in §2.2. The cheapest path
  ("return placeholder, fire notification later") is what the legacy
  code did. Don't over-design — keep it.

### 5.3 Open questions for the operator

1. **Confirm the §2.2 latent bug exists.** Run one hardware session on
   `(moonshine, llama, elevenlabs)` that triggers a tool call (e.g.
   "robot, do a dance"). If the dance fires, the bug isn't a bug and 5b
   reduces to "wire the dispatcher anyway for clarity, low priority." If
   the dance silently fails, 5b is P0.

2. **Sub-phase ordering preference.** Does the operator prefer to start
   with the latent-bug fix (5b), the trivial cleanup (5a), or the bigger
   structural work (5d / 5e)? Recommendation is 5b first, then 5a, then
   the structural items by reviewer appetite.

3. **STT backend wishlist.** §2.1 mentioned Whisper / Distil-Whisper /
   Deepgram / Parakeet. Which (if any) is the operator actually planning
   to ship? 5e is mostly worthwhile *only* if at least one new backend is
   slated. If the answer is "none in the next two quarters," 5e can be
   deferred to Phase 6 and the factory's current shape is fine.

4. **`ConversationHandler` ABC: shrink or collapse?** §2.4 sketched two
   directions. Collapse (the ABC goes away entirely; `ComposablePipeline`
   grows FastRTC adapter methods) is bolder; shrink (ABC stays as a
   FastRTC-shim only) is safer. The shrink-first path can later choose
   to collapse without rework.

5. **Tag plumbing scope.** §3.2 + §3.3 want `LLMResponse.metadata` to
   carry delivery tags into `TTSBackend.synthesize(tags=...)`. Today
   `GeminiTTSAdapter` and `ChatterboxTTSAdapter` parse tags out of text;
   if 5a switches them to consume from `tags=`, the LLM prompt
   instructions that today embed `[fast]` / `[short pause]` markers in
   the response need to *also* stop doing that *or* the LLM keeps
   embedding markers but the adapter ignores them. Which? (Recommended:
   adapter consumes from `tags=` only, the LLM stops embedding markers
   — but that's an LLM-prompt change that touches profile `instructions.txt`
   files. Bigger than it looks.)

6. **Phase 5 cadence.** Phase 4 ran for a single session window with a
   manager-driven sub-agent loop. Phase 5 has fewer interdependencies
   (5a / 5b / 5c-d / 5e-f are largely independent), so could run with
   parallel sub-agents per memory file
   `feedback_agent_concurrency_cap.md` (cap ~4). The operator's call on
   whether to revive the manager pattern or run one sub-phase at a time
   manually.

---

## §6 — Reading list for a future Phase 5 manager

Bootstrap reading, in order:

1. **`PIPELINE_REFACTOR.md`** — the status table at lines 11–24 plus
   the "Out of scope" section at lines 384–390. The "Deferred lifecycle
   hooks" table at lines 308–314 is the post-mortem you want before
   committing to anything analogous on Phase 5.
2. **This memo.** §2 is the carry-forward expansion; §3 the surviving
   TODOs; §4 the proposed DAG; §5 the recommendation.
3. **`docs/superpowers/specs/2026-05-15-phase-4-exploration.md`** — the
   style template for this memo. The "Things to confirm before turning
   this into a real spec" section at the bottom is the right cadence for
   the operator sign-off step.
4. **`docs/superpowers/specs/2026-05-15-phase-4c-tris-hybrid-realtime-design.md`** —
   how an operator-decision-memo reads when the recommendation is
   "skip." Phase 5's 5d (collapse vs shrink) is a similar shape.
5. **`src/robot_comic/composable_pipeline.py`** — the orchestrator. Read
   `_run_llm_loop_and_speak` carefully (lines 210–240) to understand
   why the §2.2 latent bug exists.
6. **`src/robot_comic/composable_conversation_handler.py`** — the wrapper.
   Especially the `_clear_queue` mirror (lines 74–94) and the
   `apply_personality` TODO (158-166).
7. **`src/robot_comic/backends.py`** — the three Protocols. The
   `TTSBackend` (lines 172–211) is what Phase 5's 5c will extend.
8. **`src/robot_comic/handler_factory.py`** — read top-to-bottom. The
   factory-private mixin hosts (lines 131–163) are 5e's refactor target;
   the per-triple `_build_composable_*` helpers (392-619) are the
   call-sites that change.
9. **`src/robot_comic/adapters/moonshine_stt_adapter.py`** — the only STT
   adapter today. Understand the monkey-patch pattern (lines 50–75)
   before approaching 5e.
10. **`src/robot_comic/adapters/llama_llm_adapter.py` + `gemini_bundled_llm_adapter.py`** —
    contrast these. The llama adapter exposes `tool_calls` to the
    orchestrator (which then breaks because the dispatcher is `None`);
    the bundled-Gemini adapter dispatches internally and never exposes
    them. Both are correct given today's wiring; 5b unifies them.
11. **`src/robot_comic/llama_base.py:570-617`** — the legacy
    `_start_tool_calls` + `_await_tool_results` pattern that 5b's
    dispatcher shim has to mirror.
12. **`src/robot_comic/tools/background_tool_manager.py`** — the
    background-tool queue mechanism that 5b's design has to coexist
    with.

Memory files to honour every Phase 5 session (under
`docs/superpowers/memory/`):

- `feedback_ruff_check_whole_repo_locally.md` — ruff from repo root.
- `feedback_ci_runs_against_pull_request_merge_commit.md` — green push
  + red pull_request means main is broken; fix main first.
- `feedback_stacked_pr_merge_order.md` — retarget downstream PR bases
  to main before merging the bottom of a stack with `--delete-branch`.
- `project_session_2026_05_15_phase4a_landed_plan_for_phase4.md` — the
  manager-loop pattern and container bootstrap recipe (mostly carries
  over).
- `feedback_agent_concurrency_cap.md` — ~4 sub-agents max.

---

## Appendix — Anything weird worth flagging

- **§2.2 latent tool-dispatcher bug.** The composable path's
  `tool_dispatcher` is always `None` in production, so any LLM that
  emits `tool_calls` on the four non-bundled-Gemini composable triples
  causes the turn to break silently. The bundled-Gemini case works by
  internal dispatch. This is the highest-impact finding in this memo
  and the only one that's a *bug* rather than a *design carry-forward*.
  Worth verifying on hardware before scoping Phase 5.

- **`derive_audio_backends` lineage.** Phase 4f retired
  `LOCAL_STT_RESPONSE_BACKEND` and its derivation machinery; the
  exploration memo §3 noted this as Phase 5 territory. Phase 4f did the
  work, so this is *not* a Phase 5 carry-forward — it just isn't yet
  reflected in `PIPELINE_REFACTOR.md`'s "Out of scope" section, which
  still says Phase 5 inherits the dial retirement. The status table
  (`PIPELINE_REFACTOR.md:24`) correctly marks 4f done; the prose at
  `PIPELINE_REFACTOR.md:388` is now stale. Worth a one-line edit during
  whatever first Phase 5 PR touches the file.

- **`PROVIDER_ID` ClassVar (post-4f).** Phase 4f renamed
  `BACKEND_PROVIDER` ClassVar to `PROVIDER_ID` on the bundled-realtime
  handlers without changing the emitted OTel attribute *values*. If
  Phase 5's 5d ABC work touches those handlers, double-check that
  telemetry still emits `"openai"` / `"huggingface"` / `"gemini"` as
  attribute values regardless of the ClassVar name — dashboards depend
  on the values, not the field name.

- **The orphan `LocalSTTLlamaGeminiTTSHandler` is gone.** The Phase 4
  exploration memo §6.9 flagged it as "defined in `llama_gemini_tts.py:234`
  but unreferenced from `handler_factory.py`." Phase 4e deleted the file.
  Not a Phase 5 concern, just confirming the loose end is closed.

- **No Phase 5 epic exists yet.** Phase 4 ran under epic #337. Phase 5
  should get its own epic before any sub-PR opens, so the sub-phase PRs
  can reference it the way 4c-* referenced #337. Recommend opening one
  early with this memo's §4 as the starter table.
