# Phase 5e.1 — Moonshine STT Adapter Decoupling (Standalone Listener)

**Date:** 2026-05-16
**Status:** Spec — implementation in progress on
`claude/phase-5e-1-moonshine-adapter-decouple`.
**Tracks:** epic #391; `docs/superpowers/specs/2026-05-16-phase-5-exploration.md`
§2.1 / §4 (5e DAG).
**Predecessor:** Phase 5d (`#403`, `ConversationHandler` ABC shrink). All
Phase 5a/5c/5d work is done on `main` at `c8fef7d`.

---

## §1 — Problem

The composable pipeline's only STT adapter, `MoonshineSTTAdapter`
(`src/robot_comic/adapters/moonshine_stt_adapter.py`), is *intimately coupled
to a host handler's implementation surface* — not to a clean Protocol target.

Concretely, it requires a host instance that mixes in `LocalSTTInputMixin`:

- `__init__(handler)` — takes the host handler (not a Moonshine client).
- `start(on_completed)` — monkey-patches
  `handler._dispatch_completed_transcript` to intercept transcripts.
- `feed_audio(frame)` — forwards `(sample_rate, samples)` to
  `handler.receive(...)`, the mixin's audio ingestion path.
- `stop()` — calls `handler.shutdown()` and restores the dispatch hook.

Every composable factory builder
(`handler_factory._build_composable_*`, `:514-745`) constructs one of five
factory-private mixin host classes (`_LocalSTTLlamaElevenLabsHost` etc.,
`handler_factory.py:138-170`) that bake `LocalSTTInputMixin` on top of the
TTS response handler. The STT adapter then reaches into that host instance.

This **prevents** adding any STT backend that doesn't ship as a host-mixin
implementation (faster-whisper, Distil-Whisper, Deepgram, Parakeet, …). The
Protocol exists; nothing implements it without going through the host.

## §2 — Scope of 5e.1 (this PR)

5e.1 is the **foundational PR**. It introduces the standalone
implementation so that the per-triple factory rewrites (5e.2–5e.6) have
something to wire up. **It does NOT migrate any factory call site.**

Specifically:

1. Add `MoonshineListener` — a standalone class that owns the Moonshine
   transcriber + stream + per-event listener + audio ingestion + #279
   re-arm recovery. Exposes a callback-driven interface that takes no
   reference to any host handler.

2. Extend `MoonshineSTTAdapter` with a second construction shape:
   `MoonshineSTTAdapter(handler=None)` (standalone mode) that creates its
   own `MoonshineListener` internally. The existing
   `MoonshineSTTAdapter(host_handler)` shape continues to work unchanged.

3. **Do NOT touch** `LocalSTTInputMixin`, any `_LocalSTT*Host` class, any
   `_build_composable_*` factory helper, the bundled-realtime
   `LocalSTT*RealtimeHandler` hybrids, or any production call site. The
   mixin and the existing host-mode adapter survive untouched; subsequent
   PRs (5e.2-5e.6) migrate each triple.

4. Tests cover the new standalone path: listener event dispatch, audio
   ingestion, rearm recovery (#279 parity), shutdown idempotency, Protocol
   conformance, and backwards-compat for the existing host-mode shape.

## §3 — `MoonshineListener` shape (standalone)

New module: `src/robot_comic/adapters/moonshine_listener.py`.

```python
class MoonshineEventKind:
    STARTED   = "started"
    PARTIAL   = "partial"
    COMPLETED = "completed"
    ERROR     = "error"

EventCallback = Callable[[str, str], Awaitable[None]]
#                          ^kind ^text


class MoonshineListener:
    """Standalone Moonshine streaming-STT listener.

    Owns the transcriber, stream, listener bridge, and #279 rearm flag.
    Has zero references to any conversation-handler host: no `deps`, no
    `output_queue`, no `_speaking_until`, no `_dispatch_*` method.
    Surfaces every stream event (started / partial / completed / error)
    to the registered callback; the adapter (or future consumers) filter.

    Lifecycle:
        listener = MoonshineListener(language=..., model_name=...)
        await listener.start(on_event)
        # audio loop:
        await listener.feed_audio(sample_rate=24000, samples=ndarray_int16)
        ...
        await listener.stop()

    Threading model is the same as the mixin: the Moonshine listener
    callbacks fire from the transcriber's worker thread; the listener
    schedules the asyncio callback via the loop captured in start().
    """

    def __init__(
        self,
        *,
        language: str | None = None,
        model_name: str | None = None,
        update_interval: float | None = None,
        cache_root: Path | None = None,
    ) -> None: ...

    async def start(self, on_event: EventCallback) -> None: ...
    async def feed_audio(self, sample_rate: int, samples: np.ndarray) -> None: ...
    async def stop(self) -> None: ...
```

Internals mirror the mixin's STT-only pieces (the listener bridge, model
load, stream open/rearm, audio ingestion). Two simplifications vs. the
mixin:

- **No echo-guard `_speaking_until` skip in `feed_audio`.** That's a
  host concern; the adapter handles it (or, post-5e.6, the composable
  pipeline does). For 5e.1 the standalone listener feeds every frame.
- **No heartbeat / MOONSHINE_DIAG instrumentation.** Those are also
  mixin-only for 5e.1 to keep the diff small. Subsequent PRs may port
  them; their absence does not break #279 or any STT correctness path.

The `_pending_stream_rearm` recovery (#279) **is preserved**: when the
listener callback fires `on_line_completed`, the rearm flag is set, and
the next `feed_audio()` call rebuilds the stream before pushing audio.
Same semantics as the mixin.

## §4 — `MoonshineSTTAdapter` shape (post-5e.1)

```python
class MoonshineSTTAdapter:
    """STTBackend implementation backed by Moonshine streaming STT.

    Two construction shapes:

    1. Standalone (post-5e.1, target shape):
           adapter = MoonshineSTTAdapter()
       Owns its own MoonshineListener; no host handler required. Future
       backends (faster-whisper, Deepgram) follow this shape.

    2. Host-coupled (pre-5e.1 legacy, surviving until 5e.6):
           adapter = MoonshineSTTAdapter(host_handler)
       Wraps an existing LocalSTTInputMixin host and monkey-patches its
       _dispatch_completed_transcript. All five composable factory
       builders still use this shape today.
    """

    def __init__(self, handler: Any = None) -> None:
        self._handler = handler  # None in standalone mode
        self._listener: MoonshineListener | None = None  # set in standalone start()
        self._on_completed: TranscriptCallback | None = None
        self._original_dispatch: Any = None  # used only in host-coupled mode
```

`start()`, `feed_audio()`, `stop()` switch on `self._handler is None`:

- **Standalone mode** — instantiate `MoonshineListener()`, call its
  `start(on_event)`, and translate `on_event(kind, text)` to the
  `STTBackend` `on_completed(text)` callback (firing only when
  `kind == "completed"`, dropping started/partial/error). `feed_audio`
  forwards to the listener; `stop` shuts it down.

- **Host-coupled mode** — exactly the current behaviour. Monkey-patch
  `handler._dispatch_completed_transcript`, forward audio to
  `handler.receive((sample_rate, samples))`, restore on stop.

## §5 — Backwards-compatibility

After this PR:

- `MoonshineSTTAdapter(host_handler)` — works unchanged (host-coupled
  mode). All factory `_build_composable_*` helpers continue using this
  shape.
- `MoonshineSTTAdapter()` — new standalone mode; not yet used by any
  factory call site (5e.2–5e.6 migrate them).
- `LocalSTTInputMixin` — untouched. Continues to own all STT *plus* host
  concerns for the host-coupled path. Will be retired only after all
  five triples migrate off it (post-5e.6).
- `_LocalSTT*Host` classes — untouched.
- Existing tests
  (`tests/adapters/test_moonshine_stt_adapter.py`,
  `tests/test_local_stt_realtime.py`, every triple-factory test) — pass
  unchanged.

## §6 — Migration plan for 5e.2 through 5e.6

Each subsequent PR is one triple, ~50-100 LOC change per:

### 5e.2 — `(moonshine, llama, elevenlabs)`

Rewrite `_build_composable_llama_elevenlabs` and delete
`_LocalSTTLlamaElevenLabsHost`. The host class collapses to a direct
`LlamaElevenLabsTTSResponseHandler` instance (no `LocalSTTInputMixin`
mix-in). STT becomes `MoonshineSTTAdapter()` (standalone). The
host-handler is still passed to the LLM/TTS adapters and as
`tts_handler=` to `ComposableConversationHandler`. *Critical:* port any
host-side concerns the mixin previously handled (echo-guard skip in
audio ingestion, output_queue partial publishing, set_listening calls,
turn span open/close, pause controller integration, welcome gate,
echo guard, name-validation `record_user_transcript`) into the
`ComposablePipeline.on_transcript` path or onto the adapter — TBD per
operator review of the diff. **This is the largest 5e.* PR by LOC** and
the one with the highest correctness risk; recommend hardware
validation between 5e.2 landing and 5e.3 starting.

### 5e.3 — `(moonshine, llama, chatterbox)`

Mirror of 5e.2 against `_LocalSTTLlamaChatterboxHost` /
`_build_composable_llama_chatterbox`. The transcript-handling-side work
established in 5e.2 should be reusable verbatim.

### 5e.4 — `(moonshine, gemini, chatterbox)`

Mirror against `_LocalSTTGeminiChatterboxHost` /
`_build_composable_gemini_chatterbox`. Identical pattern.

### 5e.5 — `(moonshine, gemini, elevenlabs)`

Mirror against `_LocalSTTGeminiElevenLabsHost` /
`_build_composable_gemini_elevenlabs`. Identical pattern.

### 5e.6 — `(moonshine, gemini-bundled, gemini_tts)`

Mirror against `_LocalSTTGeminiTTSHost` /
`_build_composable_gemini_tts`. After this PR all five host classes are
gone and `LocalSTTInputMixin` has no production call sites — delete it
(or stage that into a 5e.7 cleanup PR depending on review appetite).
The two hybrid `LocalSTT*RealtimeHandler` classes
(`local_stt_realtime.py:893+, :934+`) still import the mixin via class
inheritance — confirm 5e.6 doesn't kill them. (Reminder: those two
hybrids are declared "legacy forever" per the Phase 4c-tris decision.)

Per-triple PRs MUST include hardware validation per memo §4 (5e
"mandatory hardware validation per-triple").

## §7 — Tests added in 5e.1

New file: `tests/adapters/test_moonshine_listener.py`

- `test_listener_emits_started_partial_completed_to_callback` —
  in-process simulation of stream callbacks.
- `test_listener_sets_pending_rearm_on_completed` — #279 parity.
- `test_listener_sets_pending_rearm_on_error` — #279 parity.
- `test_listener_feed_audio_rebuilds_stream_when_rearm_flag_set` —
  next-frame rebuild path.
- `test_listener_feed_audio_resamples_when_sample_rate_mismatch` —
  resample parity with mixin's `receive`.
- `test_listener_stop_closes_stream_and_transcriber` — cleanup.
- `test_listener_stop_is_idempotent` — safe re-call.

Extend `tests/adapters/test_moonshine_stt_adapter.py`:

- `test_standalone_mode_constructor_accepts_no_handler` — new shape.
- `test_standalone_mode_routes_completed_to_callback` — end-to-end with
  a stubbed listener.
- `test_standalone_mode_drops_non_completed_events` — started/partial
  are not surfaced to the `STTBackend.on_completed` callback.
- `test_standalone_mode_feed_audio_forwards_to_listener` — Protocol
  shape works.
- `test_standalone_mode_stop_calls_listener_stop` — cleanup.
- `test_standalone_mode_satisfies_stt_backend_protocol` — isinstance.

All existing tests (host-coupled mode) MUST continue to pass.

## §8 — Out of scope (explicit)

- Factory rewrites (5e.2-5e.6).
- New STT backends (5f).
- Deleting `LocalSTTInputMixin`.
- Touching `BaseRealtimeHandler` / the hybrid `LocalSTT*RealtimeHandler`
  classes.
- Porting host concerns (turn-span, output_queue, pause controller,
  welcome gate, deps.movement_manager.set_listening, echo guard,
  name-validation) into the standalone listener or adapter. Those live
  inside the mixin today and stay there for this PR. The per-triple PRs
  decide whether to push them into `ComposablePipeline`,
  `MoonshineSTTAdapter`, or a new orchestrator hook.
- Porting the heartbeat / MOONSHINE_DIAG instrumentation.
- Telemetry attribute changes.

## §9 — Risk assessment

**Low.** Net-new code; existing production paths and tests are
untouched. The only file that gets a substantive edit is
`adapters/moonshine_stt_adapter.py` (constructor + start/stop/feed_audio
branch on `self._handler is None`); the host-coupled branch is the
existing body verbatim.

The chief risk for *later* sub-phases (5e.2-5e.6) is the host-concern
port mentioned in §6. 5e.1 deliberately leaves that decision to those
PRs so each can be reviewed against an isolated diff against a single
triple.

## §10 — Verification

```bash
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
/venvs/apps_venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/adapters/moonshine_stt_adapter.py \
    src/robot_comic/adapters/moonshine_listener.py \
    src/robot_comic/local_stt_realtime.py
/venvs/apps_venv/bin/python -m pytest tests/ -q \
    --ignore=tests/vision/test_local_vision.py
```

Known flakes per the manager's brief (re-run, do not fix):

- `test_huggingface_realtime::test_run_realtime_session_passes_allocated_session_query`
- `test_openai_realtime::test_openai_excludes_head_tracking_when_no_head_tracker`
- `test_handler_factory::test_moonshine_openai_realtime_output`,
  `test_moonshine_hf_output` (env-var leakage).
