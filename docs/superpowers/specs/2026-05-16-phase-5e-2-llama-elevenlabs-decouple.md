# Phase 5e.2 — Migrate `(moonshine, llama, elevenlabs)` off `LocalSTTInputMixin`

**Date:** 2026-05-16
**Status:** Spec — implementation in progress on
`claude/phase-5e-2-llama-elevenlabs-decouple`.
**Tracks:** epic #391; `docs/superpowers/specs/2026-05-16-phase-5-exploration.md`
§2.1.
**Predecessor:** Phase 5e.1 (`#405`, standalone `MoonshineSTTAdapter()` shape
+ `MoonshineListener`). All Phase 5a/5b/5c/5d/5e.1 work is done on `main`
at `847e6ac`.

---

## §1 — Problem

5e.1 landed the standalone `MoonshineSTTAdapter()` shape and
`MoonshineListener`, but did NOT migrate any factory call site. Every
composable triple still constructs a factory-private mixin host
(`_LocalSTT*Host`, `handler_factory.py:138-170`) and feeds it into
`MoonshineSTTAdapter(host)` (host-coupled mode). The mixin
(`LocalSTTInputMixin`, `local_stt_realtime.py:234-890`) still owns five
concerns the standalone listener deliberately punted on:

| Mixin concern | Today's site | Where it goes in 5e.2 |
|---|---|---|
| Turn-span open/close (root `turn` + `stt.infer`) | `local_stt_realtime.py:522-555` (on `started`), `:584-591` (on `completed`) | `ComposablePipeline` (`_on_event`) |
| `output_queue` `user_partial` publishing | `local_stt_realtime.py:564-568` | `ComposablePipeline` (`_on_event`) |
| `output_queue` `user` publishing | `local_stt_realtime.py:605` | `ComposablePipeline` (`_on_transcript_completed`) |
| `deps.movement_manager.set_listening(True/False)` | `local_stt_realtime.py:561` (True), `:581` (False) | `ComposablePipeline` (`_on_event`) |
| Pause-controller integration (`handle_transcript` → drop on `HANDLED`) | `local_stt_realtime.py:607-615` | `ComposablePipeline._on_transcript_completed` |
| Welcome gate (`WelcomeGate.consider` in WAITING state) | `local_stt_realtime.py:626-631` | `ComposablePipeline._on_transcript_completed` |
| Echo-guard skip in audio ingestion (drop frames during TTS playback) | `local_stt_realtime.py:796-798` | `MoonshineSTTAdapter` via `should_drop_frame` callback |
| Name-validation `record_user_transcript` | `local_stt_realtime.py:603` | `ComposablePipeline._on_transcript_completed` |
| Duplicate-completion suppression (0.75s window) | `local_stt_realtime.py:574-578` | `ComposablePipeline._on_event` |
| `head_wobbler.reset()` on speech start | `local_stt_realtime.py:559-560` | `ComposablePipeline._on_event` |
| `_clear_queue()` (barge-in flush) on speech start | `local_stt_realtime.py:557-558` | `ComposablePipeline._on_event` |
| `_mark_activity(...)` calls | `local_stt_realtime.py:518, :566, :580` | Drop — `_mark_activity` is a debug `logger.debug`; not load-bearing |
| Heartbeat / `MOONSHINE_DIAG` instrumentation | `local_stt_realtime.py:721-770` | Drop — punted with 5e.1 (out of scope per its §3 / §8) |

This PR migrates the `(moonshine, llama, elevenlabs)` triple — and ONLY
that triple — off the mixin host. The other four triples
(`llama_chatterbox`, `gemini_chatterbox`, `gemini_elevenlabs`,
`gemini_tts`) continue to use their existing `_LocalSTT*Host` hosts
unchanged. 5e.3-5e.6 migrate them one at a time using the pattern this
PR establishes.

## §2 — Scope (this PR)

1. **Extend the `STTBackend` Protocol** with an `on_partial` callback
   parameter on `start()` (default no-op via Protocol method body, same
   trick 5c.2 used for `reset_per_session_state`). The pipeline subscribes
   via this callback to publish `user_partial` rows.

2. **Add `MoonshineEventCallback` to `MoonshineSTTAdapter`'s standalone
   path** so the orchestrator can subscribe to `started` / `partial` /
   `completed` events (not just `completed`). Reshape the standalone
   `start(on_completed, on_partial)` filter so:
   - `completed` → fires `on_completed(text)` (existing behaviour).
   - `partial` (and `started` with non-empty text) → fires
     `on_partial(text)` (new).
   - `started`/`error` with no text → trigger orchestrator-side
     turn-span boundary via a separate `on_speech_started` callback OR
     via the `on_partial` path with empty text. Decision: a third
     callback `on_speech_started()` makes the turn-span open site
     explicit and avoids overloading `on_partial`.

3. **Add `should_drop_frame: Callable[[], bool] | None` parameter to
   `MoonshineSTTAdapter.__init__`** (standalone-mode only — ignored when
   `_handler is not None`). When provided, `feed_audio` consults the
   callable before forwarding to the listener; truthy means drop the
   frame. The factory builder closes over the host instance and supplies
   `lambda: time.perf_counter() < getattr(host, "_speaking_until", 0.0)`.

4. **Move host concerns onto `ComposablePipeline`** as described in §1's
   table, gated on injected dependencies. Specifically:
   - New `__init__` kwarg `deps: ToolDependencies | None = None` (default
     None → behaviour matches today, no movement/pause/welcome wiring).
   - New `__init__` kwarg `welcome_gate: WelcomeGate | None = None`.
   - New STT subscription wires `on_partial` and `on_speech_started`.
   - New `_on_speech_started` opens the turn span (mirrors mixin lines
     522-555), fires `set_listening(True)`, `head_wobbler.reset()`,
     `_clear_queue()` (when set), publishes via `output_queue`.
   - `_on_transcript_completed` (existing) gains the welcome-gate +
     pause-controller + echo-guard checks + duplicate-suppression +
     name-validation + `user`-publish + `set_listening(False)` + close
     stt-infer span — mirrors mixin lines 564-633.

5. **Wire the `_clear_queue` callback into `ComposablePipeline`** via a
   new attribute (`pipeline._clear_queue: Callable[[], None] | None`).
   `ComposableConversationHandler._clear_queue` setter mirrors onto the
   pipeline instead of the host (for the migrated triple) — but the
   mirror still falls through to `_tts_handler` for triples that haven't
   migrated yet. See §3.3 for the precise wrapper change.

6. **Rewrite `_build_composable_llama_elevenlabs`** to construct a plain
   `LlamaElevenLabsTTSResponseHandler` (no host shell), a standalone
   `MoonshineSTTAdapter(should_drop_frame=...)`, and pass `deps` +
   `welcome_gate` into `ComposablePipeline`. Delete
   `_LocalSTTLlamaElevenLabsHost` (it has no other call sites).

7. **Add an idempotency guard to
   `LlamaElevenLabsTTSResponseHandler._prepare_startup_credentials`** —
   see §3.5 for why this is required.

## §3 — Design details

### §3.1 — `STTBackend` Protocol extension

```python
@runtime_checkable
class STTBackend(Protocol):
    async def start(
        self,
        on_completed: TranscriptCallback,
        on_partial: TranscriptCallback | None = None,
        on_speech_started: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Bind transcript callbacks and prepare to receive audio.

        ``on_completed`` (required): fires once per completed user line.
        ``on_partial`` (optional): fires repeatedly for in-progress
            partial transcripts. ``None`` opts out (legacy
            host-coupled behaviour). Standalone backends ignore the
            callback when ``None``.
        ``on_speech_started`` (optional): fires once per turn at the
            start of speech. Used by the orchestrator to open the
            turn-span and reset speech-reactive secondary state. ``None``
            opts out.
        """
        ...
```

The two optional callbacks default to None so existing call sites
(host-coupled `MoonshineSTTAdapter(host)` plus pipeline integrations
that haven't migrated to 5c.2-style subscribers) keep working unchanged.

`TranscriptCallback` already typed as `Callable[[str], Awaitable[None]]`
(`backends.py:116`). The new `on_speech_started` doesn't take a text
argument because the started event's text is rarely useful (Moonshine
fires the event at first VAD activity, with the partial text often
empty).

### §3.2 — `MoonshineSTTAdapter` standalone-mode extension

```python
class MoonshineSTTAdapter:
    def __init__(
        self,
        handler: Any = None,
        *,
        should_drop_frame: Callable[[], bool] | None = None,
    ) -> None:
        self._handler = handler
        self._should_drop_frame = should_drop_frame
        # ... existing state
```

`feed_audio` consults `_should_drop_frame` in standalone mode only
(host-coupled mode already drops via the mixin's `receive`):

```python
async def feed_audio(self, frame: AudioFrame) -> None:
    ...
    if self._standalone:
        if self._should_drop_frame is not None and self._should_drop_frame():
            return
        assert self._listener is not None
        await self._listener.feed_audio(frame.sample_rate, samples)
    else:
        await self._handler.receive((frame.sample_rate, samples))
```

`start` accepts the new callbacks and forwards them in standalone mode:

```python
async def _start_standalone(
    self,
    on_completed: TranscriptCallback,
    on_partial: TranscriptCallback | None,
    on_speech_started: Callable[[], Awaitable[None]] | None,
) -> None:
    listener = MoonshineListener()
    async def _on_event(kind: str, text: str) -> None:
        if kind == EVENT_STARTED:
            if on_speech_started is not None:
                await on_speech_started()
            return
        if kind == EVENT_PARTIAL:
            if on_partial is not None and text:
                await on_partial(text)
            return
        if kind == EVENT_COMPLETED:
            await on_completed(text)
            return
        # error: dropped (orchestrator tracks via telemetry separately)
    ...
```

Host-coupled mode ignores `on_partial` / `on_speech_started` (the mixin
handles those itself). Documented in the docstring; no error raised
when they're supplied to a host-coupled adapter (forward-compatible).

### §3.3 — `ComposablePipeline` host-concern landing

`ComposablePipeline.__init__` gains three optional kwargs:

```python
def __init__(
    self,
    stt: STTBackend,
    llm: LLMBackend,
    tts: TTSBackend,
    *,
    deps: ToolDependencies | None = None,
    welcome_gate: WelcomeGate | None = None,
    # ... existing kwargs
) -> None:
    ...
    self.deps = deps
    self.welcome_gate = welcome_gate
    self._clear_queue: Callable[[], None] | None = None
    self._last_completed_transcript: str = ""
    self._last_completed_at: float = 0.0
    self._turn_span: Any = None
    self._turn_ctx_token: Any = None
    self._stt_infer_span: Any = None
    self._stt_infer_start: float = 0.0
```

`start_up` now subscribes to all three callbacks:

```python
await self.stt.start(
    on_completed=self._on_transcript_completed,
    on_partial=self._on_partial_transcript,
    on_speech_started=self._on_speech_started,
)
```

`_on_speech_started` (new):

- Open root `turn` span + `stt.infer` child, mirrors mixin lines
  522-555. Attach span to OTel context.
- Call `self._clear_queue()` if set (barge-in flush).
- If `deps` provided and `deps.head_wobbler` is not None:
  `deps.head_wobbler.reset()`.
- If `deps` provided: `deps.movement_manager.set_listening(True)`.

`_on_partial_transcript` (new):

- If transcript is non-empty: publish `AdditionalOutputs({"role":
  "user_partial", "content": transcript})` to `output_queue`.

`_on_transcript_completed` (existing — extend, do not replace):

- Duplicate-suppression (0.75s window) — mirrors mixin lines 574-578.
- If `deps`: `deps.movement_manager.set_listening(False)`, close
  `stt.infer` span with `turn.excerpt`, tag root `turn` span.
- If `deps`: `record_user_transcript(deps.recent_user_transcripts,
  transcript)`.
- Publish `AdditionalOutputs({"role": "user", "content": transcript})`
  to `output_queue`.
- Pause-controller integration: if `deps.pause_controller` is not
  None and `handle_transcript(transcript)` returns
  `TranscriptDisposition.HANDLED`, drop.
- Welcome-gate: if `self.welcome_gate is not None` and `gate.state is
  WAITING`: only dispatch on `gate.consider(transcript)` returning
  True.
- Existing body: append to history + `_run_llm_loop_and_speak`.

Echo-guard skip in `_on_transcript_completed` (mixin lines 619-622)
moves to *audio ingestion* — the STT adapter's `should_drop_frame`
callback. That's the more correct site (drop the frames, not the
completed transcript that resulted from them); the mixin's
text-level check was belt-and-braces. Confirmed by the mixin's own
comment at line 793-795 ("Without this, Moonshine's streaming VAD
treats the robot's own TTS as one continuous utterance and never
emits completed").

### §3.4 — `ComposableConversationHandler._clear_queue` mirror update

Today's setter (`composable_conversation_handler.py:89-104`) mirrors
onto `self._tts_handler`. Post-5e.2, the migrated triple's pipeline
needs the callback too — but the four un-migrated triples still rely on
the legacy host mirror. The setter becomes:

```python
@_clear_queue.setter
def _clear_queue(self, callback: Callable[[], None] | None) -> None:
    self.__clear_queue = callback
    # Mirror onto the pipeline for migrated triples (Phase 5e.2+).
    if self.pipeline is not None:
        self.pipeline._clear_queue = callback
    # Mirror onto the legacy host for triples that still use
    # LocalSTTInputMixin (Phase 5e.3-5e.6 retire each one).
    if getattr(self, "_tts_handler", None) is not None:
        self._tts_handler._clear_queue = callback
```

The double-mirror is intentional during the 5e.* transition; 5e.6
deletes the `_tts_handler` branch.

### §3.5 — `_prepare_startup_credentials` idempotency

Pre-5e.2, `LocalSTTInputMixin._prepare_startup_credentials`
(`local_stt_realtime.py:321-340`) gated the three-adapter call chain
behind `_startup_credentials_ready`. Without the mixin in the migrated
triple, each adapter (`LlamaLLMAdapter.prepare`, `ElevenLabsTTSAdapter.prepare`)
calls the underlying handler's `_prepare_startup_credentials` independently,
leaking an extra `httpx.AsyncClient` per duplicate call.

5e.2 adds a per-handler idempotency guard to
`LlamaElevenLabsTTSResponseHandler._prepare_startup_credentials`:

```python
async def _prepare_startup_credentials(self) -> None:
    if getattr(self, "_startup_credentials_ready", False):
        return
    await super()._prepare_startup_credentials()
    self._http = httpx.AsyncClient(timeout=30.0)
    # ... existing body
    self._startup_credentials_ready = True
```

5e.3-5e.6 add analogous guards on their respective handlers
(`ChatterboxTTSResponseHandler`, `GeminiTextChatterboxResponseHandler`,
`GeminiTextElevenLabsResponseHandler`, `GeminiTTSResponseHandler`).

### §3.6 — Factory rewrite

```python
def _build_composable_llama_elevenlabs(**handler_kwargs: Any) -> Any:
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        LlamaLLMAdapter,
        MoonshineSTTAdapter,
        ElevenLabsTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.composable_conversation_handler import ComposableConversationHandler
    from robot_comic.welcome_gate import make_gate_for_profile  # lazy
    from pathlib import Path

    def _build() -> ComposableConversationHandler:
        # Plain handler — no LocalSTTInputMixin shell.
        host = LlamaElevenLabsTTSResponseHandler(**handler_kwargs)

        # Echo-guard closure: standalone STT consults TTS playback deadline.
        def _should_drop_frame() -> bool:
            import time as _time
            return _time.perf_counter() < getattr(host, "_speaking_until", 0.0)

        stt = MoonshineSTTAdapter(should_drop_frame=_should_drop_frame)
        llm = LlamaLLMAdapter(host)
        tts = ElevenLabsTTSAdapter(host)

        # Welcome gate (operator-config-driven, may be None).
        welcome_gate = _maybe_build_welcome_gate()

        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
            deps=handler_kwargs["deps"],
            welcome_gate=welcome_gate,
            tool_dispatcher=_make_tool_dispatcher(host),
            system_prompt=get_session_instructions(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=host,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()
```

`_maybe_build_welcome_gate` is a new factory helper that ports
`LocalSTTInputMixin._build_welcome_gate` (`local_stt_realtime.py:299-319`).
It lives in `handler_factory.py` as a module-private function so all five
triples can reuse it during 5e.3-5e.6.

`_LocalSTTLlamaElevenLabsHost` is deleted; the other four `_LocalSTT*Host`
classes survive verbatim.

## §4 — Pattern for 5e.3-5e.6

Subsequent triples follow the same pattern. Each PR is roughly:

1. Add idempotency guard to the triple's leaf `_prepare_startup_credentials`.
2. Rewrite `_build_composable_*` to construct the plain handler, a
   standalone `MoonshineSTTAdapter(should_drop_frame=...)`, and pass
   `deps` + `welcome_gate` into `ComposablePipeline`.
3. Delete the corresponding `_LocalSTT*Host` class.
4. Update factory tests (the per-triple `_tts_handler` isinstance
   assertion will reflect the new plain handler type — already true today
   because the mixin host inherits from the handler, so `isinstance` still
   passes).

**No `ComposablePipeline` changes** are needed in 5e.3-5e.6: the host-concern
landing in §3.3 is general-purpose (works for any TTS adapter / handler
combination that exposes `_speaking_until`, `_clear_queue`,
`movement_manager.set_listening`).

**No `STTBackend` Protocol changes** are needed in 5e.3-5e.6: the
`on_partial` / `on_speech_started` callbacks are wired once in
`ComposablePipeline.start_up` and reused by every migrated triple.

5e.6 ALSO deletes:

- `LocalSTTInputMixin` — no production call sites remain. (The two
  `LocalSTT*RealtimeHandler` hybrids import the mixin via class
  inheritance; per the Phase 4c-tris decision they're legacy-forever,
  so 5e.6 either keeps the mixin file alive for them or moves the
  mixin into the hybrid file. Operator decision in 5e.6.)
- The `_tts_handler` branch from
  `ComposableConversationHandler._clear_queue` setter.

## §5 — Tests

### New tests

- `tests/adapters/test_moonshine_stt_adapter.py`:
  - `test_standalone_should_drop_frame_when_callback_returns_true` —
    `feed_audio` does NOT forward when callback is truthy.
  - `test_standalone_should_drop_frame_when_callback_returns_false` —
    `feed_audio` forwards when callback is falsy.
  - `test_standalone_should_drop_frame_callback_not_called_when_handler_provided` —
    host-coupled mode ignores the callback.
  - `test_standalone_start_invokes_on_partial_for_partial_event`.
  - `test_standalone_start_invokes_on_speech_started_for_started_event`.
  - `test_standalone_start_does_not_invoke_partial_for_completed`.
  - `test_standalone_start_does_not_invoke_speech_started_for_partial`.
  - `test_standalone_start_with_no_partial_callback_does_not_crash_on_partial_event` —
    backwards compat.

- `tests/test_composable_pipeline.py` extensions:
  - `test_speech_started_callback_opens_turn_span` — assertion on
    `_turn_span` / `_stt_infer_span` set.
  - `test_speech_started_callback_calls_set_listening_true_when_deps_provided`.
  - `test_speech_started_callback_calls_head_wobbler_reset_when_deps_provided`.
  - `test_speech_started_callback_calls_clear_queue_when_set`.
  - `test_speech_started_no_deps_no_op` — no exception when `deps=None`.
  - `test_partial_callback_publishes_user_partial_to_output_queue`.
  - `test_partial_callback_does_not_publish_empty_string`.
  - `test_completed_callback_publishes_user_to_output_queue_when_deps_provided`.
  - `test_completed_callback_records_user_transcript_when_deps_provided`.
  - `test_completed_callback_calls_set_listening_false_when_deps_provided`.
  - `test_completed_callback_suppresses_duplicate_within_window`.
  - `test_completed_callback_pause_controller_handled_drops_transcript`.
  - `test_completed_callback_pause_controller_dispatch_proceeds`.
  - `test_completed_callback_welcome_gate_waiting_drops_on_no_match`.
  - `test_completed_callback_welcome_gate_waiting_opens_on_match_and_dispatches`.
  - `test_completed_callback_welcome_gate_gated_dispatches_immediately`.

- `tests/test_handler_factory.py`:
  - Existing `test_moonshine_elevenlabs_default_llama_routes_to_composable`
    assertion on `result._tts_handler` still passes (the plain
    `LlamaElevenLabsTTSResponseHandler` is now what's wrapped instead
    of `_LocalSTTLlamaElevenLabsHost`; `isinstance` still satisfied
    since the host subclassed the handler).
  - New: `test_moonshine_llama_elevenlabs_uses_standalone_moonshine_adapter` —
    assert `result.pipeline.stt._handler is None`.
  - New: `test_moonshine_llama_elevenlabs_wires_should_drop_frame` —
    assert `result.pipeline.stt._should_drop_frame is not None`.

- `tests/test_llama_elevenlabs_tts.py` (or wherever the handler is
  tested) — new:
  - `test_prepare_startup_credentials_is_idempotent` — second call
    does not re-build `_http`.

### Migrated tests

The mixin-specific tests in `tests/test_local_stt_realtime.py`
(`test_local_stt_completion_sends_text_turn_and_queues_response`,
`test_local_stt_receive_resamples_and_feeds_stream`, the heartbeat /
stall / rearm cluster) stay unchanged — they pin the surviving
mixin behaviour for the four un-migrated triples + the two hybrids.
**Do NOT delete them.**

The host-concern coverage that the mixin previously held is now
duplicated at the pipeline level for the migrated path; the mixin
tests stay until 5e.6 closes the mixin out.

## §6 — Out of scope

- Migrating any other triple. **Strict: only `(moonshine, llama,
  elevenlabs)`.**
- Deleting `LocalSTTInputMixin`. Survives until 5e.6 (and even then
  may stay for the two hybrids).
- Adding faster-whisper or any new STT backend (5f).
- Touching `BaseRealtimeHandler` / hybrid `LocalSTT*RealtimeHandler`.
- Porting heartbeat / `MOONSHINE_DIAG` instrumentation. 5e.1 explicitly
  punted; 5e.2 holds the line. A future PR can lift the heartbeat into
  `MoonshineListener` if operators report on-robot regression.

## §7 — Risk

**Medium-high** — the largest 5e.* PR by behaviour-coverage. The
host-concern port to `ComposablePipeline` introduces five new code
paths (speech-started, partial, completed-extended) that previously
lived inside one cohesive mixin. Unit-test coverage is the safety
net; **mandatory hardware validation** between this PR landing and
5e.3 starting, per the 5e exploration memo §4.

Specific risk vectors:

- **Echo-guard timing.** Moving the skip from
  text-level (`_dispatch_completed_transcript`) to audio-level
  (`feed_audio`) is a *better* place to drop, but the timing window
  shifts: today the check fires at completion (after VAD closes the
  line); post-5e.2 it fires per-frame. Net: fewer transcripts reach
  Moonshine during TTS playback, which is the goal — but if
  `_speaking_until` is sticky-on after playback ends due to a
  cooldown miscalculation, the first turn after a long response
  could be partially dropped. Verified by the per-frame test +
  hardware validation.

- **Turn-span attribution.** The `_on_speech_started` callback opens
  the root `turn` span from `ComposablePipeline` instead of from the
  handler. If a parallel `LocalStream` code path expects to find the
  turn span at the handler instance, attribute lookup fails silently.
  Mitigation: span context attaches to OTel context, not the handler
  field — span attributes flow to OTel collectors regardless.

- **`_clear_queue` double-mirror.** During 5e.* the wrapper writes the
  callback to both the pipeline AND the `_tts_handler` host. For the
  migrated triple, the host write is dead code (the host has no
  listener calling it). Cost is one no-op `setattr`; risk is zero.

- **Welcome-gate move.** The mixin built the gate in
  `_build_welcome_gate` (`local_stt_realtime.py:299-319`). The factory
  helper `_maybe_build_welcome_gate` ports the same logic verbatim and
  is shared across triples. Risk: if the helper reads config at
  factory-build time but the operator re-configures profile mid-session,
  the gate stays stale. Same risk exists today (the mixin reads config
  at `__init__`); not a regression.

## §8 — Verification

```bash
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/composable_pipeline.py \
    src/robot_comic/adapters/moonshine_stt_adapter.py \
    src/robot_comic/backends.py \
    src/robot_comic/handler_factory.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

Known flakes per manager brief (re-run, do not fix):

- `test_huggingface_realtime::test_run_realtime_session_passes_allocated_session_query`
- `test_openai_realtime::test_openai_excludes_head_tracking_when_no_head_tracker`
- `test_handler_factory::test_moonshine_openai_realtime_output`,
  `test_moonshine_hf_output` (env-var leakage).

## §9 — Diff budget

Target: ≤1500 LOC across all touched files. If actuals exceed the
budget, STOP and report — the operator can split into 5e.2a (Protocol
+ pipeline host-concerns) and 5e.2b (factory triple swap).

Rough budget:

- `backends.py`: +15 LOC (Protocol extension).
- `adapters/moonshine_stt_adapter.py`: +40 LOC (`should_drop_frame`,
  `on_partial` / `on_speech_started` standalone wiring).
- `composable_pipeline.py`: +180 LOC (deps/welcome/`_clear_queue`
  fields, three callback handlers, history of mixin-line copies).
- `handler_factory.py`: -25 LOC (delete `_LocalSTTLlamaElevenLabsHost`),
  +60 LOC (rewrite builder + `_maybe_build_welcome_gate` helper).
- `composable_conversation_handler.py`: +5 LOC (`_clear_queue`
  pipeline mirror).
- `llama_elevenlabs_tts.py`: +4 LOC (idempotency guard).
- Tests: +400 LOC.

Total: ~680 LOC. Well under budget; no split needed unless surprises
land.
