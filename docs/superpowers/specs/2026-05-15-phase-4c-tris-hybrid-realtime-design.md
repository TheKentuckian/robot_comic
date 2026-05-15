# Phase 4c-tris — `HybridRealtimePipeline` Design Memo

**Date:** 2026-05-15
**Status:** Design memo for operator sign-off — NO code in this PR.
**Scope:** Decide whether to ship a new `HybridRealtimePipeline` sibling to
`ComposablePipeline` so the two `LocalSTT*RealtimeHandler` hybrids can be
routed through `ComposableConversationHandler`, or leave them as legacy
forever.
**Epic:** #337 — Pipeline refactor (Option C, incremental retirement).
**Predecessors:**
- Phase 4a (#355) — `ComposableConversationHandler` wrapper.
- Phase 4b (#359) — factory dual path behind `REACHY_MINI_FACTORY_PATH`.
- Phase 4c.1–4c.5 (#361/#362/#364/#365/#366) — all five non-hybrid
  composable triples migrated.
- `docs/superpowers/specs/2026-05-15-phase-4-exploration.md` §6.6 — the
  original framing of the hybrid problem.
**Successor (if shipped):** three PRs per the plan in §7; then Phase 4d.
**Successor (if skipped):** Phase 4d (default flip) and Phase 4e (legacy
deletion) — but 4e's deletion list must explicitly preserve
`LocalSTTOpenAIRealtimeHandler` + `LocalSTTHuggingFaceRealtimeHandler`
forever.

This memo is structured as a decision document. §1–§5 give the operator
the facts. §6 gives a recommendation. §7 sketches the plan if 4c-tris
ships. §8 lists the open questions the operator should answer in
sign-off.

---

## §1 — Problem statement

### 1.1 The two hybrid handlers

`src/robot_comic/local_stt_realtime.py` defines two classes that
combine local Moonshine STT with a bundled realtime endpoint for LLM+TTS:

- `LocalSTTOpenAIRealtimeHandler(LocalSTTInputMixin, OpenaiRealtimeHandler)`
  — `local_stt_realtime.py:869`.
- `LocalSTTHuggingFaceRealtimeHandler(LocalSTTInputMixin, HuggingFaceRealtimeHandler)`
  — `local_stt_realtime.py:910`.

Both are reachable from the factory today
(`handler_factory.py:360-378`):

- `(moonshine, openai_realtime_output)` → `LocalSTTOpenAIRealtimeHandler`
- `(moonshine, hf_output)` → `LocalSTTHuggingFaceRealtimeHandler`

They are the only composable-mode triples not yet routed through
`ComposableConversationHandler`. Sub-phases 4c.1–4c.5 covered the other
five.

### 1.2 Why they don't fit the existing Protocols

The Protocols in `backends.py` model a 3-phase pipeline:

- `STTBackend` — `start(on_completed)` / `feed_audio(frame)` / `stop()`
  (`backends.py:111-135`).
- `LLMBackend` — `prepare()` / `chat(messages, tools) -> LLMResponse` /
  `shutdown()` (`backends.py:138-168`).
- `TTSBackend` — `prepare()` / `synthesize(text, tags) -> AsyncIterator[AudioFrame]`
  / `shutdown()` (`backends.py:171-210`).

`ComposablePipeline._run_llm_loop_and_speak` (`composable_pipeline.py:208-231`)
assumes these three are independent objects: call `llm.chat()`, dispatch
tools, append the assistant text to history, then iterate
`tts.synthesize(text)` and push each `AudioFrame` to the output queue.

The OpenAI Realtime and HuggingFace Realtime endpoints don't decompose
that way. Both classes inherit `BaseRealtimeHandler` (`base_realtime.py:90`),
which runs one bidirectional websocket session
(`base_realtime.py:728-1125`) where:

- The server owns LLM, TTS, and (in non-hybrid mode) STT, all in one
  session.
- The client sends user input via `conversation.item.create(...)`
  (`base_realtime.py:625-631` / `local_stt_realtime.py:625-636`), then
  enqueues `response.create()` via `_safe_response_create`
  (`base_realtime.py:520-525`).
- The server pushes back interleaved events on the same websocket:
  `response.created`, `response.output_audio.delta`,
  `response.function_call_arguments.done`,
  `response.output_audio_transcript.done`, `response.done`, etc.
  (`base_realtime.py:776-1114`).
- Audio frames are decoded from `response.output_audio.delta` events
  inline in `_run_realtime_session` and pushed directly onto
  `self.output_queue` as `(sample_rate, NDArray[int16])` tuples
  (`base_realtime.py:1011-1040`).
- Tool calls fire mid-stream via `response.function_call_arguments.done`
  (`base_realtime.py:1042-1088`) and the tool result is sent back via
  another `conversation.item.create` with `type="function_call_output"`
  (`base_realtime.py:637-645`) followed by a fresh
  `_safe_response_create` (`base_realtime.py:715-721`). The model owns
  the LLM-tool round-trip; the client just brokers the I/O.

There is no place in this flow where the client receives "the assistant
text response" and then feeds it to a separate TTS. There is no
`chat(messages, tools)` round-trip the client can drive. The LLM half
and the TTS half are fused inside the websocket session.

### 1.3 What 4c-tris would add

Per the brief in `PIPELINE_REFACTOR.md:136-171`:

> introduce a sibling pipeline class so `LocalSTTOpenAIRealtimeHandler`
> and `LocalSTTHuggingFaceRealtimeHandler` can be routed through the same
> `ComposableConversationHandler`-shaped wrapper, even though their
> LLM+TTS half lives inside a single websocket session.
>
> **Why a sibling class:** the STT/LLM/TTS Protocol from `backends.py`
> doesn't fit — the realtime endpoint owns LLM+TTS as one unit.
> `HybridRealtimePipeline` exposes `STTBackend` for the Moonshine half
> and a single `RealtimeBackend` Protocol for the bundled half. The
> wrapper's interface stays the same so the factory doesn't need a third
> dispatch path beyond `FACTORY_PATH=composable`.

The exploration memo §6.6 (`2026-05-15-phase-4-exploration.md:192`)
flagged the same question:

> **Bundled-realtime + Moonshine STT hybrids.**
> `LocalSTTOpenAIRealtimeHandler` / `LocalSTTHuggingFaceRealtimeHandler`
> exist for "talk to a realtime endpoint but transcribe locally". They
> are *not* composable in the new sense (the realtime endpoint owns
> LLM+TTS as one). Are they still supported configurations? If yes, they
> survive Phase 4 untouched. If no, drop them (saves one source file's
> worth of complexity).

The decision: ship `HybridRealtimePipeline` (Option A below), or leave
them legacy forever (Option B). The two hybrids are explicitly preserved
in 4e's keep-list — `PIPELINE_REFACTOR.md:211-216` deletes seven legacy
handler classes but does **not** list the two hybrids — so Option B is
already the de facto plan unless we decide otherwise here.

---

## §2 — Today's behaviour, mapped concretely

### 2.1 Class topology

```
LocalSTTOpenAIRealtimeHandler  (local_stt_realtime.py:869)
   │ inherits from →
   ├── LocalSTTInputMixin       (local_stt_realtime.py:230)  — Moonshine STT
   └── OpenaiRealtimeHandler    (openai_realtime.py:27)
          │ inherits from →
          └── BaseRealtimeHandler   (base_realtime.py:90)    — websocket session
                 │ inherits from →
                 ├── AsyncStreamHandler  (fastrtc)
                 └── ConversationHandler (robot_comic.conversation_handler)
```

`LocalSTTHuggingFaceRealtimeHandler` (`local_stt_realtime.py:910`) is
structurally identical, swapping `OpenaiRealtimeHandler` for
`HuggingFaceRealtimeHandler` (`huggingface_realtime.py:62`).

`LocalSTTOpenAIRealtimeHandler` adds two overrides over the bare diamond
(`local_stt_realtime.py:889-907`):

- `_get_session_config(tool_specs)` — strips the realtime `input` audio
  config from the OpenAI session payload, because input is now driven by
  Moonshine over `conversation.item.create` rather than by the realtime
  server's VAD on the input audio stream. Only `output` and `tools` stay.
- `get_current_voice()` — same as the parent except it normalises against
  `get_default_voice_for_backend(OPENAI_BACKEND)` explicitly.

`LocalSTTHuggingFaceRealtimeHandler` adds no overrides — just the diamond
and its cost-class constants (`local_stt_realtime.py:912-920`).

### 2.2 Lifecycle, end to end

#### 2.2.1 `__init__`

- `LocalSTTInputMixin.__init__` (`local_stt_realtime.py:238-293`) calls
  `super().__init__()` (which threads up to `BaseRealtimeHandler.__init__`
  at `base_realtime.py:120-205`) and then sets up Moonshine state
  (`_local_stt_stream`, `_local_stt_transcriber`, `_local_stt_listener`,
  `_heartbeat`, `_welcome_gate`, …). All Moonshine state lives on
  `self`, mixed in over the BaseRealtime state.

#### 2.2.2 `start_up()`

`BaseRealtimeHandler.start_up()` (`base_realtime.py:452-483`):

1. `await self._prepare_startup_credentials()` — `LocalSTTInputMixin`
   overrides this (`local_stt_realtime.py:317-321`) to chain to the
   parent (e.g. `OpenaiRealtimeHandler._prepare_startup_credentials` at
   `openai_realtime.py:39-57`, which waits for `OPENAI_API_KEY`) and
   then build the Moonshine stream in a thread.
2. `self.client = await self._build_realtime_client()` — provider-specific.
3. Loop: `await self._run_realtime_session()` — opens the websocket,
   sends `session.update(session=self._get_session_config(tool_specs))`,
   then `async for event in self.connection` processes server events
   until the session ends (`base_realtime.py:728-1125`).
4. On `ConnectionClosedError`: rebuild the client (if
   `REFRESH_CLIENT_ON_RECONNECT`, true for HF, false for OpenAI), back
   off, retry — up to three attempts (`base_realtime.py:457-476`).

The point: there is no place to insert a `HybridRealtimePipeline` "loop"
without re-shaping `_run_realtime_session`. The websocket session **is**
the loop.

#### 2.2.3 `receive(frame)`

`LocalSTTInputMixin.receive` (`local_stt_realtime.py:748-834`) is the
input path. It:

- Honours `self._pending_stream_rearm` to recover from Moonshine's
  one-line-per-stream limit (issue #279).
- Drops audio while `self._speaking_until` is in the future (echo guard
  for the robot's own TTS).
- Resamples to 16 kHz and feeds the Moonshine stream via
  `self._local_stt_stream.add_audio(audio_payload, self.local_stt_sample_rate)`.

Note this **shadows** `BaseRealtimeHandler.receive`
(`base_realtime.py:1128-1168`), which would otherwise send the raw mic
audio bytes to `self.connection.input_audio_buffer.append(...)`. In hybrid
mode the mic audio never goes over the websocket.

#### 2.2.4 `_dispatch_completed_transcript(transcript)`

`LocalSTTInputMixin._dispatch_completed_transcript` (`local_stt_realtime.py:616-636`):

```python
async def _dispatch_completed_transcript(self, transcript: str) -> None:
    if not self.connection:
        logger.debug("Local STT transcript ready but realtime connection is not connected")
        return
    await self.connection.conversation.item.create(
        item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": transcript}],
        },
    )
    await self._safe_response_create(
        response=RealtimeResponseCreateParamsParam(
            instructions="Answer the user's transcribed speech naturally and concisely in audio.",
        ),
    )
```

This is the seam between the two halves. The Moonshine listener fires
`_handle_local_stt_event(kind="completed", text=...)` on the asyncio
loop (`local_stt_realtime.py:491-614`), which after a series of
guardrails (echo guard, welcome gate, pause controller, duplicate
suppression) calls `await self._dispatch_completed_transcript(transcript)`
at line 614. That call writes the user text into the websocket session
as a user-role message and enqueues a `response.create()` to ask the
realtime server to answer.

From this point on, the websocket-event loop in
`BaseRealtimeHandler._run_realtime_session` owns the rest: it receives
`response.created`, decodes `response.output_audio.delta` events into
PCM frames and puts them on `self.output_queue` (`base_realtime.py:1011-1040`),
handles `response.function_call_arguments.done` by spawning a background
tool (`base_realtime.py:1042-1088`), and processes `response.done` to
close out turn telemetry (`base_realtime.py:863-904`).

#### 2.2.5 `emit()`

`BaseRealtimeHandler.emit` (`base_realtime.py:1170-1186`) drains
`self.output_queue` via `_wait_for_output_item`, with an idle-detection
path that fires `send_idle_signal` after 180 s of inactivity. FastRTC
calls this on its read schedule.

#### 2.2.6 `shutdown()`

`LocalSTTInputMixin.shutdown` (`local_stt_realtime.py:836-866`) cancels
the Moonshine heartbeat / diag tasks, chains to `super().shutdown()` —
which is `BaseRealtimeHandler.shutdown` (`base_realtime.py:1188-1219`),
closing the websocket and draining the queue — then closes the Moonshine
stream and transcriber in a thread.

### 2.3 Lifecycle hooks the hybrids fire today

These are the surfaces a future `ComposableConversationHandler`-shaped
wrapping needs to either preserve or accept the loss of:

| Hook | Where it fires | Notes |
|------|----------------|-------|
| Telemetry — turn span | `base_realtime.py:789-797` (start) / `:421-444` (close) | Already fully instrumented in the realtime loop. Survives any wrapping. |
| Telemetry — `stt.infer` span | `local_stt_realtime.py:533-537` (start) / `:565-574` (end) | Fired from the Moonshine event handler. Survives. |
| Telemetry — `llm.request` span + `record_llm_duration` | `base_realtime.py:850-861` (start) / `:881-900` (end) | Inside the realtime loop. Survives. |
| Telemetry — `tts.synthesize` span + `record_tts` + `record_tts_first_audio` | `base_realtime.py:811-825` and `:1028-1030` | Inside the realtime loop. Survives. |
| Telemetry — `record_ttft` | `base_realtime.py:1024-1027` | Inside the realtime loop. Survives. |
| Welcome gate | `local_stt_realtime.py:295-315` (build) / `:607-612` (consume) | In the Moonshine event handler. Survives. |
| Echo guard (`_speaking_until`) | Set by realtime audio handling — currently **NOT** set on the realtime path. The legacy ElevenLabs handlers set it; the bundled-realtime path doesn't. See §5 cell `_speaking_until`. | Pre-existing gap, not caused by this design choice. |
| Pause controller | `local_stt_realtime.py:588-596` | In the Moonshine event handler. Survives. |
| Engagement / soften-note guardrail | `base_realtime.py:983-1000` | Inside the realtime loop, on user-transcript completion (note: that event fires server-side in non-hybrid mode; in hybrid mode it doesn't fire because input audio never reaches the server). Currently **dead code on the hybrid path** — pre-existing bug, not caused by 4c-tris. |
| Background tool dispatch | `base_realtime.py:1068-1088` (start) / `:601-726` (result handling) | Inside the realtime loop. Survives. |
| Idle signal | `base_realtime.py:1176-1184` / `:1236-1256` | Inside `emit`. Survives. |
| Boot-timeline supporting events (#321) | Not implemented yet on any path; cross-cutting follow-up. | Same as for all triples. |
| `record_joke_history` | Not implemented on the realtime path (it's an LLM-message-shape concern, and the realtime API doesn't surface assistant text as a discrete `chat()` round-trip). | Pre-existing gap on bundled-realtime. |
| `history_trim` | Not applicable — realtime API owns history server-side, not bound by `REACHY_MINI_MAX_HISTORY_TURNS` (see `CLAUDE.md` "Conversation history bound" note). | N/A by design. |
| Voice / personality methods | `BaseRealtimeHandler.change_voice` (`base_realtime.py:300-310`), `BaseRealtimeHandler.get_current_voice` (`:312-315`), `BaseRealtimeHandler.get_available_voices` (`:1228-1230`), `BaseRealtimeHandler.apply_personality` (`:317-377`) | All present and working. `apply_personality` even sends a live `session.update` over the websocket. Survives. |

**Net read:** the hybrid handlers already fire essentially every
lifecycle hook the operator cares about, and they do it without help
from the orchestrator because the bundled-realtime parent class already
implements them inline. The composable triples are the ones starving for
plumbing, not these.

---

## §3 — Option A: Ship `HybridRealtimePipeline`

### 3.1 What to ship

**A new Protocol in `backends.py`:**

```python
@runtime_checkable
class RealtimeBackend(Protocol):
    """Bundled LLM+TTS realtime backend driven by a single websocket session."""

    async def prepare(self) -> None: ...
    async def run_session(
        self,
        on_user_text: Callable[[str], Awaitable[None]],
        output_queue: asyncio.Queue[Any],
        tool_dispatcher: ToolDispatcher | None = None,
        tools_spec: list[dict[str, Any]] | None = None,
    ) -> None:
        """Block until the session ends.

        Implementations call ``on_user_text`` registration once and from
        then on accept user-text input via ``send_user_text(text)``
        (separate method, omitted here for brevity). Output frames push
        directly into ``output_queue`` as the websocket events arrive.
        Tool calls dispatch via ``tool_dispatcher`` if provided, otherwise
        through the implementation's own background tool manager.
        """
        ...

    async def send_user_text(self, text: str) -> None:
        """Inject a user-role message into the active session and request a response."""
        ...

    async def shutdown(self) -> None: ...
```

The Protocol shape is the messy part. There is no clean "round-trip"
abstraction the way `LLMBackend.chat` is — the session is long-lived,
events stream in both directions, and tool calls are mid-stream. A real
spec has to either:

- Expose a coarse `run_session()` that blocks for the whole session (like
  the proposal above), and a separate `send_user_text()` to push input
  in mid-session. The pipeline wires `on_user_text` to the
  `MoonshineSTTAdapter` callback. This is closest to today's flow.
- Or expose a fine-grained event-stream Protocol where the pipeline pulls
  events from the backend. That's a substantial rewrite of
  `_run_realtime_session` and ~600 lines of test surface, and we don't
  have a second `RealtimeBackend` implementor (Gemini Live is a *third*
  shape, see §3.4), so the abstraction would be over-fit.

Either way the Protocol is awkward — it's modelling a thing that doesn't
generalise yet.

**A new pipeline class:**

```python
class HybridRealtimePipeline:
    def __init__(
        self,
        stt: STTBackend,
        realtime: RealtimeBackend,
        *,
        output_queue: asyncio.Queue[Any] | None = None,
        tool_dispatcher: ToolDispatcher | None = None,
        tools_spec: list[dict[str, Any]] | None = None,
    ) -> None: ...

    async def start_up(self) -> None:
        await self.realtime.prepare()
        # Bind STT's callback to push transcripts into the realtime session.
        await self.stt.start(on_completed=self._on_transcript_completed)
        # Run the realtime session for the lifetime of the pipeline.
        await self.realtime.run_session(
            on_user_text=self._noop,  # input comes via _on_transcript_completed
            output_queue=self.output_queue,
            tool_dispatcher=self.tool_dispatcher,
            tools_spec=self.tools_spec,
        )

    async def _on_transcript_completed(self, transcript: str) -> None:
        await self.realtime.send_user_text(transcript)

    async def feed_audio(self, frame: AudioFrame) -> None:
        await self.stt.feed_audio(frame)

    async def shutdown(self) -> None: ...
```

The class is short. The complexity is hidden in the Protocol contract
and in the adapters.

**Two adapters:**

- `OpenAIRealtimeBackend(handler: LocalSTTOpenAIRealtimeHandler)` —
  thin wrapper that delegates `run_session` to
  `handler._run_realtime_session()` (after `start_up()`'s prepare +
  client-build steps) and `send_user_text` to
  `handler._dispatch_completed_transcript`. Same monkey-patch pattern
  the existing adapters use: reach into the legacy handler's internals.
- `HuggingFaceRealtimeBackend(handler: LocalSTTHuggingFaceRealtimeHandler)`
  — identical shape, different host class.

Estimated adapter LOC: ~80 each, mostly docstrings and lifecycle
forwarding.

**Factory routing:**

Two new helpers in `handler_factory.py`, mirroring the existing
`_build_composable_<output>_<llm>` pattern:

```python
def _build_hybrid_openai_realtime(**kwargs: Any) -> Any:
    legacy = LocalSTTOpenAIRealtimeHandler(**kwargs)
    stt = MoonshineSTTAdapter(legacy)
    realtime = OpenAIRealtimeBackend(legacy)
    pipeline = HybridRealtimePipeline(stt, realtime)
    return ComposableConversationHandler(
        pipeline=pipeline,
        tts_handler=legacy,
        deps=kwargs["deps"],
        build=lambda: _build_hybrid_openai_realtime(**kwargs),
    )

def _build_hybrid_hf_realtime(**kwargs: Any) -> Any:
    # identical, swap the host class
```

…and gating them inside the existing per-triple branches at
`handler_factory.py:360-378` on `FACTORY_PATH=composable`, mirroring the
pattern at lines 190-198 etc.

**`ComposableConversationHandler` compatibility:**

The wrapper at `composable_conversation_handler.py:34-151` takes a
`pipeline` typed as `ComposablePipeline`. To accept
`HybridRealtimePipeline` we'd need to:

- Either generalise the `pipeline` attribute to a `Protocol` covering
  `start_up` / `shutdown` / `feed_audio` / `output_queue` /
  `reset_history` / `_conversation_history`.
- Or accept `pipeline: ComposablePipeline | HybridRealtimePipeline` and
  add a `Union`-aware code path for the no-history case in
  `apply_personality` (the realtime server owns history; the wrapper
  shouldn't try to mutate `_conversation_history`).

The latter is simpler and matches the "concrete-types-not-Protocols-for-internal-glue"
pattern the rest of the codebase already uses.

### 3.2 Estimated LOC and PR count

`PIPELINE_REFACTOR.md:161` says "one PR per hybrid handler migration" if
the memo green-lights implementation. The §7 plan in this memo expands
that to three PRs because the Protocol + pipeline class is itself a
shippable, testable unit before any factory routing:

- PR 1: `RealtimeBackend` Protocol in `backends.py` + `HybridRealtimePipeline`
  in `hybrid_realtime_pipeline.py` + unit tests against stubs. No factory
  wiring. ~250 LOC source + ~400 LOC tests.
- PR 2: `OpenAIRealtimeBackend` adapter + factory route for
  `(moonshine, openai_realtime_output)`. ~150 LOC source + ~250 LOC tests.
- PR 3: `HuggingFaceRealtimeBackend` adapter + factory route for
  `(moonshine, hf_output)`. ~150 LOC source + ~250 LOC tests.

Plus tweaks to `composable_conversation_handler.py` to accept the new
pipeline type (probably folded into PR 1).

**Total: ~3 PRs, ~1450 LOC including tests. Roughly 2–3 sessions.**

That's about the same as 4c.5 (`GeminiTTSAdapter` + bundled-LLM routing,
PR #366) — a comparable design problem (an adapter that doesn't fit the
"chat → tts" two-step), so the calibration is sound.

### 3.3 Risks

**3.3.1 Websocket lifecycle vs. `start_up()` semantics.**
`ComposablePipeline.start_up` (`composable_pipeline.py:127-149`) prepares
all backends and then blocks on `self._stop_event.wait()`. The
non-hybrid backends' `prepare()` is short-lived. A `RealtimeBackend`'s
"prepare and then start the websocket session" is a long-lived operation
— the websocket lives for the entire pipeline lifetime. Mapping that
onto a `prepare()` + `run_session()` Protocol works but it means the
pipeline's `start_up` has to await `run_session` to keep the session
alive (as sketched in §3.1). If `run_session` re-connects internally
(it does — the existing `start_up` retries up to 3 times on
`ConnectionClosedError`, `base_realtime.py:457-476`), the pipeline can't
observe individual session lifetimes. That's probably fine, but the
contract needs to spell it out.

**3.3.2 Transcript routing.**
The current `_dispatch_completed_transcript` lives on the host handler
(`local_stt_realtime.py:616-636`) and is overridden by
`MoonshineSTTAdapter`'s monkey-patch
(`adapters/moonshine_stt_adapter.py:55-63`). Under 4c-tris the patched
`_dispatch_completed_transcript` would call `pipeline._on_transcript_completed`,
which calls `realtime.send_user_text`, which calls
`handler._dispatch_completed_transcript` — except the original one was
just monkey-patched away. Solution: the adapter has to capture the
original `_dispatch_completed_transcript` (the
`MoonshineSTTAdapter._original_dispatch` field at
`adapters/moonshine_stt_adapter.py:46`) and `OpenAIRealtimeBackend.send_user_text`
has to invoke the captured original rather than going through
`handler._dispatch_completed_transcript` (which is now the bridge). This
is workable but the call-graph is twisty enough to introduce a class of
bugs the non-hybrid path doesn't have.

**3.3.3 Tool-call flow.**
`base_realtime._run_realtime_session` dispatches tool calls via
`self.tool_manager.start_tool(...)` (`base_realtime.py:1068-1076`) — the
bundled `BackgroundToolManager` already wires tool results back into the
session via `_handle_tool_result` (`base_realtime.py:601-726`). The
`ComposablePipeline.tool_dispatcher` is a different shape: a
synchronous-return callback per tool call (`composable_pipeline.py:76`,
`:233-249`). The two models don't compose. The honest answer: the
`tool_dispatcher` arg on `HybridRealtimePipeline` is ignored, and tools
keep going through `tool_manager` inside the bundled session. We pay
the cost of putting it in the Protocol signature for shape parity,
nothing more.

**3.3.4 `apply_personality` divergence.**
`composable_conversation_handler.apply_personality`
(`composable_conversation_handler.py:127-139`) calls
`pipeline.reset_history(keep_system=False)` and re-seeds the system
prompt into `pipeline._conversation_history`. For the hybrid case the
realtime server owns history, so this code path is wrong. The right
thing is the legacy behaviour:
`BaseRealtimeHandler.apply_personality` (`base_realtime.py:317-377`)
sends `session.update(session=RealtimeSessionCreateRequestParam(instructions=...))`
over the live websocket, then force-restarts the session. Under 4c-tris
the wrapper would need a code path that delegates `apply_personality` to
the underlying realtime handler instead of mutating pipeline state — yet
another union/`isinstance` branch in the wrapper.

**3.3.5 `output_queue` ownership.**
`ComposablePipeline.output_queue` is owned by the pipeline; the wrapper
exposes it via property forwarding (`composable_conversation_handler.py:55-70`).
`BaseRealtimeHandler.output_queue` is owned by the handler
(`base_realtime.py:150`), and the realtime event loop pushes frames
into it directly (`base_realtime.py:1035-1040`). Sharing one queue
between the two requires either (a) the adapter forwards the handler's
queue to the pipeline at construction time, or (b) the pipeline
overwrites `handler.output_queue = self.output_queue` before starting
the session. Both work, both have failure modes if the wrapper does the
barge-in `output_queue = asyncio.Queue()` swap at
`composable_conversation_handler.py:60-70` and the handler still holds
the old reference. Solvable but it's a class of bugs we don't otherwise
have.

**3.3.6 Test surface.**
The non-hybrid composable triples were covered by ~250 LOC of TDD-style
tests each (`tests/adapters/test_*.py`,
`tests/test_handler_factory_*.py`). The hybrid case adds: websocket-event
stubbing (the existing `tests/test_local_stt_realtime.py` already has
this scaffolding, 15.4 KB worth), `RealtimeBackend` Protocol conformance
tests, `HybridRealtimePipeline` lifecycle tests, and per-adapter routing
tests. Realistic floor: ~900 LOC tests across the three PRs.

### 3.4 Why this doesn't generalise to `GeminiLiveHandler`

`GeminiLiveHandler` (`gemini_live.py:315`) is a **third** wire shape —
it uses the `google.genai` live-API SDK rather than the OpenAI realtime
SDK that `BaseRealtimeHandler` is built around. It doesn't inherit from
`BaseRealtimeHandler`, doesn't speak the same event types, and isn't
reached through the hybrid pattern (the factory routes
`PIPELINE_MODE_GEMINI_LIVE` directly to `GeminiLiveHandler` —
`handler_factory.py:165-169`).

If we ever wanted Gemini-Live-with-Moonshine-STT we'd ship a third
`RealtimeBackend` implementor (`GeminiLiveBackend`) that maps the same
`prepare`/`run_session`/`send_user_text`/`shutdown` Protocol onto the
`google.genai` API. The Protocol is supposed to abstract over both
SDKs but the abstraction is paying for two implementors in the future,
not the two we have today.

This is a real argument for shipping `HybridRealtimePipeline` — it's the
plumbing that future Gemini-Live-hybrid work would land on top of. It's
also speculative: there's no operator demand for that pipeline today,
and adding it would be its own design memo.

---

## §4 — Option B: Leave the hybrids legacy forever

### 4.1 What NOT shipping means

The two `LocalSTT*RealtimeHandler` classes stay in
`src/robot_comic/local_stt_realtime.py` past 4e. The factory keeps the
two branches at `handler_factory.py:360-378` that route directly to
those concrete classes. `FACTORY_PATH=composable` returns
`ComposableConversationHandler` for the five "real" composable triples
and the two legacy hybrid classes for the other two. The dispatch
matrix becomes:

| Triple | FACTORY_PATH=legacy | FACTORY_PATH=composable |
|--------|---------------------|-------------------------|
| (moonshine, llama, elevenlabs) | `LocalSTTLlamaElevenLabsHandler` | `ComposableConversationHandler` |
| (moonshine, chatterbox, llama) | `LocalSTTChatterboxHandler` | `ComposableConversationHandler` |
| (moonshine, chatterbox, gemini) | `GeminiTextChatterboxHandler` | `ComposableConversationHandler` |
| (moonshine, elevenlabs, gemini) | `GeminiTextElevenLabsHandler` | `ComposableConversationHandler` |
| (moonshine, elevenlabs, gemini-fallback) | `LocalSTTGeminiElevenLabsHandler` | `ComposableConversationHandler` |
| (moonshine, gemini_tts) | `LocalSTTGeminiTTSHandler` | `ComposableConversationHandler` |
| **(moonshine, openai_realtime_output)** | `LocalSTTOpenAIRealtimeHandler` | `LocalSTTOpenAIRealtimeHandler` |
| **(moonshine, hf_output)** | `LocalSTTHuggingFaceRealtimeHandler` | `LocalSTTHuggingFaceRealtimeHandler` |

The bottom two rows are the giveaway: under Option B the dial does
nothing for the hybrid triples.

### 4.2 Confirming 4e's keep-list

`PIPELINE_REFACTOR.md:209-216` lists what 4e deletes:

> - `src/robot_comic/llama_elevenlabs_tts.py::LocalSTTLlamaElevenLabsHandler` (lines 359–366).
> - `src/robot_comic/chatterbox_tts.py::LocalSTTChatterboxHandler`.
> - `src/robot_comic/elevenlabs_tts.py::LocalSTTGeminiElevenLabsHandler` + the legacy alias `LocalSTTElevenLabsHandler`.
> - `src/robot_comic/gemini_tts.py::LocalSTTGeminiTTSHandler`.
> - `src/robot_comic/gemini_text_handlers.py::GeminiTextChatterboxHandler` and `GeminiTextElevenLabsHandler` …
> - `src/robot_comic/llama_gemini_tts.py::LocalSTTLlamaGeminiTTSHandler` …

The two hybrids are **not** in that list. The keep-list at
`PIPELINE_REFACTOR.md:218-226` doesn't name them either, but
`LocalSTTInputMixin` is preserved ("`MoonshineSTTAdapter` depends on
it") and both hybrid classes mix it in, so they survive by
construction. Confirmed by `Grep` for the class names in `src/`:
`LocalSTTOpenAIRealtimeHandler` and `LocalSTTHuggingFaceRealtimeHandler`
are defined only in `local_stt_realtime.py` and referenced only by
`handler_factory.py:361-378` plus tests. Nothing in the 4c.1–4c.5
adapters needs them.

### 4.3 Long-term cost of Option B

- **Two handler shapes forever.** Even after 4f finishes retiring
  `BACKEND_PROVIDER`, the codebase has two parallel composable handler
  shapes: `ComposableConversationHandler(ComposablePipeline)` and these
  two `LocalSTTInputMixin + BaseRealtimeHandler` legacy classes. New
  contributors have to learn both.
- **Lifecycle hooks need plumbing twice.** The deferred lifecycle hooks
  (`telemetry.record_llm_duration`, boot-timeline events, history trim,
  echo guard, joke history) all need to be threaded through either the
  composable wrapper *or* the legacy hybrid classes. Today the bundled
  realtime path already fires telemetry, has a no-op
  history-trim contract, and is missing echo-guard / joke-history — so
  the cost is asymmetric: composable triples need more new plumbing, but
  any future lifecycle work that *also* applies to the hybrid path has
  to land twice (once in the wrapper, once in `BaseRealtimeHandler` or
  `LocalSTTInputMixin`).
- **Test coverage stays split.** `tests/test_local_stt_realtime.py`
  (15.4 KB, per the exploration memo §5) stays as-is — it tests the
  hybrid classes directly. `tests/test_handler_factory_factory_path.py`
  has to special-case the two hybrid triples (the dial returns the same
  class regardless of value).
- **Phase 5 work is more awkward.** If we ever want to unify
  `ConversationHandler`'s ABC across composable and bundled-realtime
  paths (Phase 5 territory per the exploration memo), the hybrid classes
  are an extra integration target.

### 4.4 What it saves

- **One design memo** (this one, after sign-off).
- **Three PRs** (~1450 LOC source + tests).
- **2–3 sessions** of engineering time.
- **Maintenance of a `RealtimeBackend` Protocol** that doesn't yet have
  a second implementor and may never get one (Gemini Live is a separate
  branch in the factory; an operator decision to fold it in is a future
  call).
- **The risks in §3.3** — they're real, all five of them are solvable,
  but Option B sidesteps them by not introducing the abstraction at all.

---

## §5 — Decision criteria

Read this as a yes/no checklist the operator can run through. Each row
nudges toward Option A (ship) or Option B (skip).

| Question | If yes | If no | Operator's answer |
|----------|--------|-------|-------------------|
| Is the operator actively running `(moonshine, openai_realtime_output)` or `(moonshine, hf_output)` in production? | A — worth uniformity for a maintained path | B — uniformity for a dead path is waste | ? |
| Are the deferred lifecycle hooks (telemetry duration, boot-timeline events, joke history, history trim, echo guard) going to land on the composable path between now and 4d? | A — would simplify the parallel-plumbing problem | B — same plumbing burden either way; defer | ? |
| Is there a roadmap item that needs Gemini Live + Moonshine STT? | A — the Protocol pays for the future implementor | B — speculative | ? |
| Are we likely to add new realtime providers (e.g. a third vendor's bundled-realtime API)? | A — Protocol is the seam | B — YAGNI | ? |
| Is the cost of "two handler shapes forever" higher than the cost of three more PRs + ~1450 LOC? | A | B | ? |
| Does 4d's default flip need the hybrid path to also be on `composable` for the dial to be coherent? | A | B (dial just no-ops for hybrid triples) | ? |
| Is the operator comfortable with the wrapper having a `isinstance(pipeline, HybridRealtimePipeline)` branch for `apply_personality` (per §3.3.4)? | A | B (no branch needed if hybrid isn't wrapped) | ? |
| Has the operator ever asked for a unified mental model of the dispatch matrix, vs. asked for less code? | A | B | ? |

**Boot-timeline / lifecycle context:** the deferred lifecycle hooks in
`PIPELINE_REFACTOR.md:303-320` are themselves a separate work stream.
They will mostly land between 4c.5 (done) and 4d (default flip) as small
follow-up PRs. If those land *before* 4c-tris, they'd need to be
re-implemented inside `HybridRealtimePipeline` too — that's more work
under Option A. If 4c-tris lands first, the same hooks land in *both*
pipeline classes simultaneously — also more work under Option A. Under
Option B the hooks only need to land in `ComposablePipeline` + the
wrapper, and the hybrid path already has most of them via
`BaseRealtimeHandler` (see §2.3 table).

---

## §6 — Recommendation

**Skip 4c-tris (Option B).**

The reasoning, in order of weight:

1. **The bundled realtime classes already implement the lifecycle hooks
   we care about.** §2.3's table shows that telemetry spans (`turn`,
   `stt.infer`, `llm.request`, `tts.synthesize`), TTFT recording,
   tool-call dispatch, the welcome gate, pause controller, voice
   switching, and live `apply_personality` all fire from
   `BaseRealtimeHandler` / `LocalSTTInputMixin` today. The composable
   triples are the ones starving for plumbing. Wrapping the hybrid
   classes in a new pipeline class doesn't *unlock* hooks — they're
   already there. It just changes who fires them.

2. **The Protocol is over-fit to one shape.** `RealtimeBackend` would
   have one implementor (well, two if you count both vendors — but the
   `OpenAIRealtimeBackend` and `HuggingFaceRealtimeBackend` differ only
   in which subclass of `BaseRealtimeHandler` they wrap; they're not
   substantively distinct shapes). There's no third implementor on the
   roadmap, and `GeminiLiveHandler` would need a third Protocol shape
   anyway (§3.4). Abstractions earn their keep when they have ≥3
   implementors. This one has 1.5.

3. **The wrapper-side `apply_personality` divergence is real.** §3.3.4
   spells it out: under Option A the wrapper has to do
   `isinstance(pipeline, HybridRealtimePipeline)` to skip the
   history-reset path, because the realtime server owns history. That
   `isinstance` is exactly the kind of leak the abstraction is supposed
   to prevent. Going the other direction — duplicating
   `_conversation_history`-style state inside `HybridRealtimePipeline`
   for symmetry — is even worse (it would invent a fake history that
   the server doesn't see).

4. **The dial just no-ops for two triples — that's fine.** Under
   Option B, `FACTORY_PATH=composable` returns the same class as
   `FACTORY_PATH=legacy` for the two hybrid triples. That's a
   documentation problem (which `handler_factory.py`'s docstring at
   lines 7-18 already partly handles), not a correctness problem. The
   operator's dial works for the triples that matter and is a no-op
   for the two that don't fit.

5. **4e's deletion list already greenlights this.**
   `PIPELINE_REFACTOR.md:209-216` lists six legacy concrete handlers
   for deletion and explicitly does not list the two hybrids. The
   keep-list relies on `LocalSTTInputMixin` surviving for
   `MoonshineSTTAdapter` to depend on, which transitively keeps the
   hybrid classes alive. The author of `PIPELINE_REFACTOR.md` (this
   manager session's predecessor, May 14–15) wrote it that way
   deliberately. The cheapest path is to honour that intent.

6. **Cost-benefit on the engineering time.** 2–3 sessions of focused
   work and ~1450 LOC of source+tests, against a uniformity benefit
   that nobody has asked for. Other Phase 4 / Phase 5 work (the
   deferred lifecycle hooks, `BACKEND_PROVIDER` retirement, the
   tool-dispatcher refactor flagged in the exploration memo §6.2)
   has clearer payoff.

**Caveat — when to revisit:** if the operator decides to add a Gemini
Live + Moonshine STT hybrid (or a third vendor's bundled-realtime
endpoint), or if the deferred lifecycle hooks turn out to require
plumbing that's only natural to add at the orchestrator level (e.g. a
unified joke-history capture across all triples), the calculus
changes. Both feel like Phase 5+ concerns today.

---

## §7 — If we ship (Option A): rough plan

If the operator overrides §6 and green-lights shipping, here is the
shape — at outline depth only. Each bullet is one PR; the full TDD
plans would be written separately, following the 4c.1–4c.5 pattern.

**PR 1 — `RealtimeBackend` Protocol + `HybridRealtimePipeline` class.**
Adds `RealtimeBackend` to `backends.py` alongside the existing three
Protocols. Adds `hybrid_realtime_pipeline.py` with the class sketched
in §3.1. Updates `composable_conversation_handler.py` to accept
`pipeline: ComposablePipeline | HybridRealtimePipeline` and adds the
`isinstance` branch in `apply_personality` (delegates to the underlying
realtime handler's `apply_personality` instead of mutating pipeline
history). Adds unit tests against in-memory stubs for both classes,
matching the depth of `tests/test_composable_pipeline.py`. No factory
wiring. ~250 LOC source + ~400 LOC tests.

**PR 2 — `OpenAIRealtimeBackend` adapter + factory route.**
Adds `src/robot_comic/adapters/openai_realtime_backend.py` wrapping
`LocalSTTOpenAIRealtimeHandler`. Adds
`_build_hybrid_openai_realtime` to `handler_factory.py` and gates the
existing `(moonshine, openai_realtime_output)` branch at
`handler_factory.py:360-368` on `FACTORY_PATH=composable`. Adds adapter
unit tests (websocket-event stubbing, mirroring the existing
`tests/test_local_stt_realtime.py` patterns) and a factory routing test
(mirroring `tests/test_handler_factory_factory_path.py`). ~150 LOC
source + ~250 LOC tests.

**PR 3 — `HuggingFaceRealtimeBackend` adapter + factory route.**
Same shape as PR 2 but for the HF triple. ~150 LOC source + ~250 LOC
tests.

**Ordering and hardware validation:** PR 1 is purely additive (no
behaviour change). PR 2 and PR 3 each flip one triple on the
composable path. Operator should validate each on the robot before
merging the next, per the manager-loop pattern in
`PIPELINE_REFACTOR.md:31-49`. Estimated total time: 2–3 sessions.

**What this does NOT do:** it doesn't touch `GeminiLiveHandler` (a
third wire shape — see §3.4). It doesn't change the bundled-realtime
fast path at `handler_factory.py:153-169`. It doesn't add the deferred
lifecycle hooks; those follow-up PRs would have to land in both
`ComposablePipeline` and `HybridRealtimePipeline` if relevant.

---

## §8 — Open questions for the operator

To sign off on this memo the operator needs to answer:

1. **Option A vs Option B?** The recommendation is B. State your
   preference; if A, override the recommendation with reasoning so a
   future manager session can re-read the decision context.

2. **If Option A: PR ordering preference?** The §7 plan is PR 1
   (Protocol+class) → PR 2 (OpenAI) → PR 3 (HF). Acceptable variations:
   merge PR 1 + PR 2 as one PR (route + scaffold together) for less
   ceremony at the cost of a bigger diff; or land PR 3 first if the
   operator has more confidence in the HF endpoint than the OpenAI one.

3. **If Option A: any lifecycle hooks strictly required for the hybrid
   triples?** §2.3 inventories what fires today. The
   `_speaking_until` echo guard is a pre-existing gap on the
   bundled-realtime path (the legacy ElevenLabs handlers set it; the
   realtime handlers don't). Should 4c-tris close that gap on the way
   through, or is that a separate follow-up?

4. **If Option B: does 4f's `BACKEND_PROVIDER` retirement need any
   special handling for the hybrid path?** The two hybrid classes
   declare `BACKEND_PROVIDER = OPENAI_BACKEND` and
   `BACKEND_PROVIDER = HF_BACKEND` as class attributes
   (`local_stt_realtime.py:872, 913`). 4f's scope per
   `PIPELINE_REFACTOR.md:280-292` includes removing the
   `BACKEND_PROVIDER` config dial, but the *class attribute* is used by
   `_compute_response_cost` and telemetry (`base_realtime.py:794, 819,
   823, 857, 879, 895, 1026`). The class attribute may need to survive
   even after the config dial dies. Worth a memo-of-its-own when 4f
   starts, regardless of Option A/B today.

5. **Confirm the docstring update for `handler_factory.py`.** Under
   Option B the file's lead docstring at lines 7-18 already correctly
   describes the two hybrid triples as routing to the legacy classes.
   No change needed. Under Option A the docstring would need to add
   "(composable: `ComposableConversationHandler` wrapping
   `HybridRealtimePipeline`)" for the two rows. Confirm we're not
   missing a docstring update either way.

---

## Appendix — Files touched by this memo

This PR ships docs only:

- `docs/superpowers/specs/2026-05-15-phase-4c-tris-hybrid-realtime-design.md` (new — this file).
- `PIPELINE_REFACTOR.md` — one-line status update on the 4c-tris row
  pointing readers to this memo. No structural change.

No `src/` or `tests/` changes. Per
`docs/superpowers/memory/feedback_pr_354_merged_red_branch_protection_check.md`
and the path-filtered CI workflows (`lint.yml`, `typecheck.yml`,
`pytest.yml`, `uv-lock-check.yml` all filter on
`src/**`/`tests/**`/`pyproject.toml`), a doc-only PR is expected to
report `mergeable: MERGEABLE, mergeStateStatus: CLEAN,
statusCheckRollup: []`. That's the normal state; do not wait for CI
that won't run.
