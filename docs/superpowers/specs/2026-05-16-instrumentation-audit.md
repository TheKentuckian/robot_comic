# App-Instrumentation Audit — Monitor-Effectiveness Gap Memo

**Date:** 2026-05-16
**Base commit:** `1b6bc7b` (Phase 4c.5 + 3b landed)
**Scope:** Research-only audit of the robot_comic telemetry / observability surface
and the gaps that would limit the operator monitor's diagnostic effectiveness.
**Non-goals:** No code changes. Recommended emissions are described, not written.

---

## TL;DR

The OTel-flavoured tracing surface is in good shape for the **post-startup
turn loop**: every hot path (STT → LLM → TTS → frame-out → turn-close)
emits both spans and histograms, and the Phase 4 lifecycle hooks now keep
the composable orchestrator parity with the legacy class hierarchy. The
boot-timeline lane (PR #321 + Hook #3 + Hook #3b) closes the
process-launch-to-first-audio window.

The most operator-impactful gaps are:

1. **Tool execution outcome is not on the metric surface.** `tool.execute`
   spans exist on legacy realtime path only (`base_realtime.py`/`elevenlabs_tts.py`)
   — the composable orchestrator's tool dispatcher emits nothing, no
   duration histogram, no error counter. A stuck or failing tool is invisible
   to dashboards until the next turn fails.
2. **Moonshine cold-load — the dominant boot delay (~20 s of ~42 s
   cold-start) — has no supporting event.** Operators can read
   `app.startup` and `first_greeting.tts_first_audio` but cannot attribute
   the gap between them.
3. **ALSA capture (Stream A) and head-wobbler playback have no underrun /
   drop counters wired** — `playback_underruns` is defined but never
   incremented; `frame_drops` only fires for the head wobbler, not for any
   STT input pump.
4. **Camera worker / head-tracker has no FPS or detection-rate metric**,
   even though it owns the head-tracking control surface and has known
   IK / collision warning paths.
5. **`stt.duration` is recorded once per completed transcript, but
   Moonshine state transitions (`started` / `partial` / `completed`,
   plus the documented `state=idle` stall) only live in `logger.info` /
   `logger.warning`** — the monitor would have to scrape logs to detect
   the very stall the heartbeat loop exists to surface.

The single most-impactful remediation is **wiring tool-dispatch duration +
outcome to the metric surface** (§5 Rec 1). Everything else is either
boot-time diagnostics (one supporting event each) or counter increments
behind paths that already emit logs.

---

## §1 — Inventory: what's instrumented today

### `src/robot_comic/telemetry.py`

#### Gate
- `ROBOT_INSTRUMENTATION` env: `unset|""` → no-op, `trace` → console-line
  exporter, `remote` → OTLP traces + metrics + console.
- `ROBOT_METRICS_CONSOLE` env: opt-in console metric exporter (off by default to
  prevent journald spam).
- `_init_otel()` builds `TracerProvider` with `CompactLineExporter` (writes
  one-line `RCSPAN <json>` to stdout — the monitor's primary ingest format).

#### Span attribute allowlist (`_SPAN_ATTRS_TO_KEEP`)
The exporter drops any attribute not in this set, so it doubles as the
operator-visible schema:

`turn.id`, `session.id`, `turn.outcome`, `turn.excerpt`, `robot.mode`,
`robot.persona`, `gen_ai.system`, `gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`, `tool.name`, `tool.id`, `vad.duration_ms`,
`stt.type`, `event.kind`, `event.dur_ms`, `from_persona`, `to_persona`,
`outcome`, `aplay.exit_code`, `aplay.command`.

#### Histograms (`robot_comic` meter)
| Instrument | Name | Unit | Recorded by |
|---|---|---|---|
| `turn_duration` | `robot.turn.duration` | s | `base_realtime._close_turn_span`, `gemini_live`, `elevenlabs_tts._dispatch_completed_transcript_impl`, `llama_base` |
| `llm_operation_duration` | `gen_ai.client.operation.duration` | s | `base_realtime` (response.done), `elevenlabs_tts._call_llm`, `llama_base`, `LlamaLLMAdapter.chat`, `GeminiLLMAdapter.chat`, `GeminiBundledLLMAdapter.chat` |
| `ttft` | `gen_ai.server.time_to_first_token` | s | `base_realtime` (first audio delta), `gemini_live`, `llama_base` |
| `stt_duration` | `robot.stt.duration` | s | `base_realtime` (server VAD), `local_stt_realtime` (Moonshine completed) |
| `tts_duration` | `robot.tts.duration` | s | `base_realtime` (response.output_audio.done), `elevenlabs_tts._stream_tts_to_queue` finally, `llama_base` |
| `tts_first_audio` | `robot.tts.time_to_first_audio` | s | `base_realtime`, `chatterbox_tts`, `elevenlabs_tts._stream_tts_to_queue`, `llama_elevenlabs_tts`, `llama_gemini_tts` |

#### Counters
| Instrument | Name | Unit | Incremented by |
|---|---|---|---|
| `frame_drops` | `robot.audio.capture.frame_drops` | frames | `audio/head_wobbler.py:146` (head wobbler audio lag) |
| `playback_underruns` | `robot.audio.playback.underruns` | events | **never** (defined, no callers) |
| `errors` | `robot.errors` | events | **never** (defined, no callers) |

#### Supporting-event surface (`emit_supporting_event` — PR #321)
Pre-init queue (`_pending_supporting`) buffers calls until `init()` runs
and then flushes — necessary because the early-welcome path (#317/#324)
fires before `telemetry.init()`.

Each emit opens-and-closes a zero-duration span with `event.kind=supporting`,
optional `event.dur_ms`, and `extra_attrs` (allowlisted).

#### First-greeting once-guard
`emit_first_greeting_audio_once()` flips a module-level boolean and emits
`first_greeting.tts_first_audio` with `event.dur_ms = since_startup()*1000`.
Idempotent — every TTS frame-enqueue site may call it.

### Boot-timeline event surface (PR #321 / Hook #3 / Hook #3b)

Five named events; emission sites in source order:

| Event | Emitted from | Trigger |
|---|---|---|
| `app.startup` | `main.py:231` | `telemetry.init()` returned |
| `welcome.wav.played` | `main.py:82` (early), `warmup_audio.py:480` (in-process) | Popen of `aplay` dispatched |
| `welcome.wav.completed` | `warmup_audio.py:342` (thread), `warmup_audio.py:382` (sync) | `aplay` Popen exited; carries `aplay.exit_code` + `aplay.command` |
| `handler.start_up.complete` | `elevenlabs_tts.py:362` (legacy), `composable_conversation_handler.py:125` (Hook #3) | Handler ready to accept audio |
| `first_greeting.tts_first_audio` | `base_realtime.py:1039`, `chatterbox_tts.py:368`, `elevenlabs_tts.py:999`, `gemini_live.py:1036`, `gemini_tts.py:415`, `llama_gemini_tts.py:171`, `adapters/gemini_tts_adapter.py:204` (Hook #3b) | First PCM frame of process-life enqueued |

Process-wide ordering (cold-boot, on-robot):
```
app.startup → welcome.wav.played → welcome.wav.completed
            → handler.start_up.complete → first_greeting.tts_first_audio
```

### OTel spans (live trace surface)

Span names found in `src/`:

- `turn` — outer per-utterance root span (`base_realtime`, `elevenlabs_tts`,
  `gemini_live`, `gemini_tts`, `llama_base`, `local_stt_realtime`).
- `stt.infer` — Moonshine streaming inference window (`local_stt_realtime`).
- `vad.endpoint` — server-VAD endpoint duration (`base_realtime`); carries
  `vad.duration_ms`.
- `llm.request` — realtime LLM round-trip (`base_realtime`); carries
  `gen_ai.usage.{input,output}_tokens` when available.
- `tts.synthesize` — TTS streaming wrap (`base_realtime`, `chatterbox_tts`,
  `elevenlabs_tts`, `llama_base`).
- `tool.execute` — tool dispatch (`background_tool_manager.py:212`, attrs
  `tool.name` + `tool.id`); also `elevenlabs_tts.py:867` for the in-line
  llama-style dispatch path.

### Logging surface (hot-path signals the monitor would scrape)

- **Moonshine heartbeat** — `local_stt_realtime._log_heartbeat` (every 1 s
  while stream active). `INFO` line `[Moonshine] state=… last_event=…
  age=… frames=… text=…` with dedupe + 30 s repeat-INFO floor. Idle-stall
  WARNING after 10 s of `state=idle` past the 10 s startup grace.
- **MOONSHINE_DIAG=1** opt-in verbose tracing of listener wiring,
  add_audio frames, periodic state dumps (issue #314 disambiguation).
  Operator-flipped at runtime; no metric surface.
- **Turn-latency logs** — `INFO Turn latency: response.created %.0f ms`
  and `first audio delta %.0f ms` in `base_realtime` (#1024). Useful for
  monitor regression detection but not on the metric surface.
- **Tool-call info** — `INFO Tool '%s' (id=%s) executed successfully` /
  `INFO Tool call received` (`base_realtime`). No structured emission.
- **Backoff / rate-limit** — `WARNING ElevenLabs TTS 429`,
  `WARNING TTS attempt %d/%d failed`, `WARNING LLM call failed`,
  `WARNING Gemini ... rate-limited` (per CLAUDE.md Gemini retry path).
  None increment a counter.
- **Camera worker** — `logger.exception("Camera worker error: %s")` with
  60 s same-error suppression; no metric.
- **Startup checkpoints** — `startup_timer.log_checkpoint` fires ~15
  `Startup: +Xs <label>` INFO lines from `main.py` and `warmup_audio.py`.
  These ride the boot-timeline plot for humans reading journald, but
  only four of them are mirrored as supporting events.

### Performance / liveness counters

- `_heartbeat["audio_frames"]` — internal Moonshine counter, surfaced
  only in the heartbeat log line.
- `cumulative_cost` — per-backend dollar accumulator
  (`base_realtime._compute_response_cost`, `elevenlabs_tts` char-cost).
  Logged but not exported as a metric or attribute.
- Per-turn API-call counter (`api_call_counter` in `elevenlabs_tts`) is
  surfaced as `gen_ai.usage.api_call_count` on the turn span — orphan
  attribute, not in `_SPAN_ATTRS_TO_KEEP` so it never reaches the monitor.

---

## §2 — Coverage map (conversation hot path)

Walking the live turn, segment by segment. Symbols: ✅ instrumented,
⚠️ partial, ❌ absent.

| Segment | Status | Signal source |
|---|---|---|
| Process import → `telemetry.init()` | ⚠️ | `app.startup` event (one number). No checkpoint events for the 15 `log_checkpoint(...)` sites between import and handler init. |
| Welcome WAV dispatch & exit | ✅ | `welcome.wav.played` + `welcome.wav.completed` (with `aplay.exit_code`). |
| Handler init / credentials | ⚠️ | `handler.start_up.complete` event (one number). No sub-span for Gemini client init, ElevenLabs voice catalog fetch, llama-server warmup. |
| **Moonshine model load** | ❌ | No supporting event. `_build_local_stt_stream` runs serially inside `_prepare_startup_credentials` (~20 s on-robot). Only audible signal is the gap between `handler.start_up.complete` and `first_greeting.tts_first_audio`. |
| Camera/vision init | ❌ | Log only (`log_checkpoint("camera/vision init")`). |
| Mic audio frames → `LocalSTTInputMixin.receive` | ⚠️ | `audio_frames` counter is internal; `MOONSHINE_DIAG=1` per-frame logs exist; no metric exported. |
| **ALSA RW capture (Stream A)** | ❌ | `AlsaRwCapture` emits two `logger.warning` lines on shutdown; no per-second frame counter, no drop / underrun signal. |
| Moonshine streaming inference | ✅ | `stt.infer` span + `robot.stt.duration` histogram with `stt.type=moonshine`. |
| **Moonshine idle stall** | ⚠️ | `WARNING "idle for %.1fs … possible thread-lock or model stall"` is **log-only**. The monitor cannot detect this without parsing journald. |
| LLM call (realtime path) | ✅ | `llm.request` span + `gen_ai.client.operation.duration` histogram + `gen_ai.usage.{input,output}_tokens` + `ttft` histogram. |
| LLM call (composable path — Hook #2) | ✅ | `record_llm_duration` in three adapters (Llama, Gemini-text, Gemini-bundled). |
| **LLM retries (Gemini 503, ElevenLabs 429, llama timeout)** | ⚠️ | Logged at WARNING with attempt counts; `gen_ai.usage.api_call_count` is set on turn span but dropped by exporter allowlist. No counter. |
| **Tool dispatch (composable orchestrator)** | ❌ | `ComposablePipeline._dispatch_tools_and_record` emits no span, no counter, no duration. Tool errors only land in `logger.exception`. |
| Tool dispatch (legacy realtime / bundled-Gemini) | ⚠️ | `tool.execute` span exists via `BackgroundToolManager`; carries `tool.name` + `tool.id` only — no duration histogram, no outcome attr, no error counter. |
| TTS first-audio | ✅ | `tts.synthesize` span + `robot.tts.time_to_first_audio` + `first_greeting.tts_first_audio` boot event. |
| TTS streaming / completion | ✅ | `tts.synthesize` span end + `robot.tts.duration` histogram. |
| **TTS retries** | ⚠️ | Logged with attempt counter; no counter metric. Rate-limit state surfaced via the `[ElevenLabs TTS rate-limited; …]` synthetic status marker → monitor's status-marker row, but no histogram of *how often* this happens. |
| Audio frames → output queue → speakers | ⚠️ | `_enqueue_audio_frame` (ElevenLabs path) maintains `_response_audio_bytes` for echo guard — internal only. No `frames_sent` / playback-throughput metric. |
| **Speaker underruns** | ❌ | `playback_underruns` counter is defined but never incremented. The ALSA sink owner is the daemon, not our process, so we'd need to scrape `dmesg` / `/proc` to detect xruns. Out of band for now, but the gap should be acknowledged. |
| Move execution (gesture/dance/emotion) | ❌ | No spans, no counters. The movement manager runs at 60 Hz and a stuck primary-move would only surface via the user noticing the robot is frozen. |
| Echo guard (`_speaking_until`) | ⚠️ | Lifecycle Hook #1 wired in `_enqueue_audio_frame`; no metric. `INFO "Echo guard: discarding transcript during TTS playback"` log line only. |
| Joke history capture (Hook #4) | ⚠️ | `logger.debug("joke_history capture failed")` on error; no success/fail counter, no extraction-latency histogram (the LLM call is bounded at 500 ms — would be useful to know how often it times out). |
| History trim (Hook #5) | ⚠️ | `logger.info("Trimmed %d entries …")` when trimming occurs; no counter. |
| Camera worker / head tracker | ❌ | No FPS, no detection-rate, no tracker-output magnitude. `clamp_tracker_rotation_offsets` is invoked silently. |
| Admin UI / FastAPI routes | ❌ | No request count, no latency, no error count for `init_admin_ui` routes. |
| WS endpoint (Pi↔laptop) | ⚠️ | Send/recv logged at INFO; `pi_status` heartbeat sent per turn; no span / metric. |

---

## §3 — Gaps the monitor would be blind to

Ranked by **operator pain × diagnostic value / cost**. Each gap is named
by the failure mode it lets through.

### Rank 1 — Tool dispatch is opaque on the composable path

`ComposablePipeline._dispatch_tools_and_record` (composable_pipeline.py:242)
runs each tool sequentially via the operator-supplied callback, catches
exceptions, and appends to history. There is **no span around the
callback**, **no duration histogram**, and **no error counter**. The only
signal is `logger.exception("Tool %s raised", call.name)`.

Concretely: if `dance` or `camera` blocks for 30 s (gstreamer pipeline
stall, motor IK collision retry, vision model thrash), the operator sees:

- The turn span stays open (no `turn.outcome=success` until LLM next
  responds).
- No `tool.execute` span — that surface only fires from
  `BackgroundToolManager` on the legacy path.
- The next turn's STT activity halts because the LLM is mid-tool-round.

**Diagnostic value:** High. Tool stalls are the #1 source of "the robot
is frozen but not crashed" reports. **Cost:** Low — one span wrap + one
histogram emit in the two existing dispatch sites
(`ComposablePipeline._dispatch_tools_and_record`,
`BackgroundToolManager._run_tool_routine` if not already wrapped).

### Rank 2 — Moonshine cold-load swallows ~20 s with no supporting event

`prewarm_model_file` + `Transcriber(...)` + `_open_local_stt_stream` run
inside `_prepare_startup_credentials` (`local_stt_realtime.py:317-391`).
The transcriber constructor blocks 15-20 s on cold-boot (per
MEMORY.md `project_moonshine_cold_load`).

There's no `moonshine.model.loaded` supporting event. Operators can only
infer the gap from `handler.start_up.complete - app.startup` ≈ Moonshine
load.

**Diagnostic value:** High — this is the dominant cold-start cost and
the operator needs to know when it's the model vs other startup paths.
**Cost:** One `emit_supporting_event("moonshine.model.loaded",
dur_ms=…)` call.

### Rank 3 — Moonshine idle-stall and rearm only surface in logs

`_log_heartbeat` warns at WARNING when state=idle for >10 s with
`audio_frames > 0` (`local_stt_realtime.py:691`). `_rearm_local_stt_stream`
is called silently after every `on_line_completed` (#279).

The monitor has no way to detect either condition without log scraping.
A stuck transcriber (the symptom that motivated MOONSHINE_DIAG) is
invisible to dashboards.

**Diagnostic value:** Medium-high — this *is* the bug class the diag
infrastructure exists for. **Cost:** Low — one `inc_errors({"component":
"moonshine", "kind": "idle_stall"})` next to the warning + a
`inc_errors({"component": "moonshine", "kind": "rearm"})` next to the
rearm log. (The current `errors` counter has zero callers, so this also
exercises a dead instrument.)

### Rank 4 — `playback_underruns` is defined but never incremented

`telemetry.inc_playback_underruns` exists; no source file calls it.
ALSA xruns on the daemon-owned sink are not directly observable from our
process, but `aplay` exit codes on the welcome-WAV path (carried on
`welcome.wav.completed`) and TTS frame-enqueue lag (we know byte-counts
in `_enqueue_audio_frame`) are signals we control.

**Diagnostic value:** Medium — audio glitches are user-visible.
**Cost:** Either retire the unused counter or wire it once
(non-zero aplay exit code → `inc_playback_underruns`).

### Rank 5 — Camera / head-tracker has no telemetry whatsoever

`CameraWorker.working_loop` runs at ~25 FPS (`time.sleep(0.04)`).
No FPS metric, no detection-rate (face found / not found), no
`clamp_tracker_rotation_offsets` invocation counter. The 60 s
same-error suppression on `_last_error_msg` is the only signal, and it
only fires on full exceptions — a tracker that returns `None` forever
is silent.

**Diagnostic value:** Medium — operator already kill-switched
head-tracker pending #264/#272/#308 (per MEMORY.md), so this is post-fix
hygiene. **Cost:** Low — three counters (`frames_captured`,
`faces_detected`, `tracker_clamp_engaged`) at the existing
hot-loop sites.

### Rank 6 — Tool span attribute set is too thin to diagnose

`tool.execute` carries `tool.name` + `tool.id`. It carries no:

- Tool result outcome (success / error / cancelled / timeout)
- Argument size (LLM hallucinated 20 k chars of JSON is a real failure
  mode)
- Result kind (was this an idle tool call?)

**Diagnostic value:** Medium — once Rank 1 lands, this is the natural
next step to enrich it. **Cost:** Low — `set_attribute` calls inside the
existing span.

### Rank 7 — LLM retry / rate-limit count is dropped

`gen_ai.usage.api_call_count` is set on the turn span in
`elevenlabs_tts.py:901` but is **not in `_SPAN_ATTRS_TO_KEEP`**, so the
exporter drops it. Similarly `_last_tts_rate_limited` exists as an
instance attribute but only surfaces via the synthetic status marker.

**Diagnostic value:** Medium — frequent retries are an early warning for
quota/billing burn (Gemini free tier is ~10 req/day per CLAUDE.md
"Gemini 429 handling"). **Cost:** Trivial — add the key to the
allowlist (one-line); harder fix is a proper counter.

### Rank 8 — Composable orchestrator has no turn-level duration

`ComposablePipeline._run_llm_loop_and_speak` does not call
`telemetry.record_turn`. The legacy realtime + ElevenLabs +
gemini_live + llama_base sites all do. After Phase 4d's default flip,
the composable path's turns will silently disappear from the
`robot.turn.duration` histogram.

**Diagnostic value:** Medium-high (regression-after-Phase-4d). **Cost:**
Low — wrap `_run_llm_loop_and_speak` with `perf_counter` + record at
end with `{robot.mode, turn.outcome}` matching the legacy shape.

### Rank 9 — `record_joke_history` extraction-latency is invisible

The Hook #4 extraction call has a 500 ms timeout (`joke_history.py:51`).
We have no signal on how often it times out or how the latency
distribution looks. Free-tier llama-server quirks (cold-load, model
swap) could make this bite without anyone noticing.

**Diagnostic value:** Low-medium. **Cost:** Low — one histogram emit
inside `extract_punchline_via_llm`.

### Rank 10 — `history_trim` invocations only logged

`trim_history_in_place` logs `INFO "Trimmed %d entries …"` only when
`removed > 0`. We have no counter for how often we hit the cap, which is
the signal we'd want for tuning `REACHY_MINI_MAX_HISTORY_TURNS`.

**Diagnostic value:** Low. **Cost:** Trivial.

---

## §4 — Monitor compatibility

The monitor (`src/robot_comic/monitor.py`) understands:

- **Turn rows** built from `turn` spans + children
  (`stt.infer`, `vad.endpoint`, `llm.request`, `tts.synthesize`,
  `tool.execute`).
- **Supporting-event rows** (`event.kind=supporting`) — dimmer cyan,
  STT/LLM/TTS columns blank, displays `event.dur_ms` in the Total
  column.

Confirmed from `monitor.py:137` (tool-count derived from `tool.execute`
children) and `monitor.py:209-244` (supporting-event routing + child
naming).

### Orphan emissions (emitted but no monitor consumer the audit could verify)

- `gen_ai.usage.api_call_count` (set on turn span, dropped by exporter
  allowlist — never reaches the monitor).
- `tts.voice_id`, `tts.char_count` (set on `tts.synthesize` in
  `elevenlabs_tts.py:973-975`, dropped by the same allowlist).
- `gen_ai.server.time_to_first_token` as an attribute on `llm.request`
  span (`base_realtime.py:1028`) — also not in allowlist; only the
  histogram side reaches anything downstream.

These three are zero-cost wins: either add to `_SPAN_ATTRS_TO_KEEP` (and
the monitor table) or stop setting them.

### Missing emissions the monitor likely expects

The monitor's display logic in `monitor.py:244` (`elif name ==
"tool.execute"`) suggests it counts tool spans per turn. With **no**
`tool.execute` spans emitted on the composable path, the tools column
will read 0 for every composable turn that called tools — a UI lie. Rank
1 fixes this.

### Semconv alignment

The OTel attribute set tracks GenAI Semantic Conventions
(`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
`gen_ai.client.operation.duration`, `gen_ai.server.time_to_first_token`).
A downstream OTLP consumer (SigNoz / Grafana / a future external
monitor) that follows semconv will recognise these without configuration.

`robot.*` attributes (`robot.mode`, `robot.persona`, `robot.turn.*`,
`robot.audio.*`, `robot.tts.*`, `robot.stt.*`) are project-local but
namespaced; safe.

---

## §5 — Recommendations (rank-ordered, smallest blast first)

### Rec 1 — Wrap tool dispatch on the composable orchestrator (Rank 1 gap)

**Where:** `composable_pipeline.py:_dispatch_tools_and_record`.

**Add:**
```
with tracer.start_as_current_span(
    "tool.execute",
    attributes={"tool.name": call.name, "tool.id": call.id},
) as span:
    _t0 = time.perf_counter()
    try:
        result = await self.tool_dispatcher(call)
        span.set_attribute("outcome", "success")
    except Exception as exc:
        span.set_attribute("outcome", "error")
        telemetry.inc_errors({"component": "tool", "tool.name": call.name})
        raise
    finally:
        telemetry.record_tts(...)  # or a new robot.tool.duration histogram
```

(`outcome` is already in `_SPAN_ATTRS_TO_KEEP`.)

**Why first:** Closes the largest live-path observability hole; no
architectural prereq; same shape as `BackgroundToolManager`'s existing
`tool.execute` span so the monitor needs no change.

**Note:** Adding a `robot.tool.duration` histogram is the natural
companion; it can ship in the same PR.

### Rec 2 — Emit `moonshine.model.loaded` supporting event

**Where:** end of `_build_local_stt_stream`
(`local_stt_realtime.py:391`).

**Add:** `telemetry.emit_supporting_event("moonshine.model.loaded",
dur_ms=load_elapsed_ms)` — measure from method entry. Closes Rank 2.

**Cost:** ~3 lines. Operator visibility: massive (resolves "is it
Moonshine or is it something else").

### Rec 3 — Wire `inc_errors` at the existing Moonshine warning sites

**Where:** `local_stt_realtime._log_heartbeat` idle-stall block
(line 691), `_rearm_local_stt_stream` (line 482),
`_MoonshineListener.on_error` (line 218).

**Add:** `telemetry.inc_errors({"component": "moonshine", "kind": …})`
in each.

**Why:** Exercises an unused counter and gives the monitor a numeric
signal for the very stall class that has its own diagnostic infrastructure
(MOONSHINE_DIAG).

### Rec 4 — Record `robot.turn.duration` from `ComposablePipeline`

**Where:** `_run_llm_loop_and_speak` — wrap with `perf_counter` and
record at end.

**Why:** Phase 4d will flip the default to composable; without this the
turn-duration histogram drops a lane.

### Rec 5 — Add the existing dropped-attribute keys to the exporter allowlist

**Where:** `telemetry.py:_SPAN_ATTRS_TO_KEEP`.

**Add:** `gen_ai.usage.api_call_count`, `gen_ai.server.time_to_first_token`,
`tts.voice_id`, `tts.char_count`. Document that any new attr must be
added here too.

**Cost:** Literally one tuple edit. Zero new emission code; we already
set these attributes.

### Rec 6 — Add `outcome` attribute to existing `tool.execute` spans

**Where:** `background_tool_manager.py:212` and `elevenlabs_tts.py:867`.

**Add:** Set `outcome=success|error|cancelled` on the span before
`end()`. Already in allowlist.

### Rec 7 — Wire the welcome-WAV `aplay.exit_code != 0` path to `inc_playback_underruns`

**Where:** `warmup_audio._emit_completion_now` and `_wait_and_emit_completion`.

**Add:** When `exit_code != 0`, `inc_playback_underruns({"path":
"welcome.wav"})`. (Optional: retire the unused counter if we're not
going to use it.)

### Rec 8 — Emit `moonshine.transcriber.rearmed` supporting event

**Where:** after `_open_local_stt_stream` is called from
`_rearm_local_stt_stream`.

**Why:** Operator can see how often the stream is being rebuilt across a
session — the #279 work-around frequency is a useful signal.

### Rec 9 — Add a histogram for `joke_history` extraction latency

**Where:** `joke_history.extract_punchline_via_llm`.

**Add:** Optional new `robot.joke_history.extract.duration` histogram
with `{outcome=success|timeout|http_error|parse_error}`.

### Rec 10 — Camera-worker FPS + detection counter

**Where:** `camera_worker.working_loop`.

**Add:** Two counters — `robot.camera.frames` (every loop with non-None
frame) + `robot.camera.faces_detected` (when `eye_center is not None`).

**Defer:** Until the head-tracker kill-switch is lifted (#264/#272/#308
per MEMORY.md) — the data is meaningless while tracking is off.

---

## §6 — Anti-recommendations (do NOT instrument)

### Don't emit a counter per audio frame in / out

`_enqueue_audio_frame` runs hundreds of times per turn. The existing
`_response_audio_bytes` accumulator is private to the echo guard and
should stay private. A broader per-frame counter would balloon the
metric cardinality with no diagnostic gain — the `tts.synthesize` span +
duration histogram already encodes the same information once per turn.

### Don't promote `logger.debug` to spans

The Moonshine path has `[MOONSHINE_DIAG]` opt-in verbose tracing. The
right path for the operator's worst-day debugging is `MOONSHINE_DIAG=1`
+ journalctl, not making the monitor swim in per-`add_audio` spans.

### Don't add per-move telemetry to MovementManager

A 60 Hz control loop will saturate any exporter. The right move signal
is "primary move *queue* depth" + "is a move executing right now",
exposed once per turn — not a 60 Hz span stream. And even that should
wait until the field-test backlog (#264/#272/#308) lands.

### Don't instrument `pause_controller.handle_transcript`

It's a string-match against a few-dozen-entry list and runs on every
completed transcript. The decision (`HANDLED|DISPATCH`) is already
logged on the rare `HANDLED` branch (`INFO`). Span overhead would
swamp the actual work.

### Don't instrument the admin UI FastAPI routes

The admin UI is operator-driven, not user-facing; route latency or
error count is a feature we'd add when there's a known UI failure
mode, not pre-emptively.

### Don't add structured emission for every `logger.warning`

The Gemini-retry, ElevenLabs-429, llama-timeout WARNINGs are noisy in
the journal but they already speak. Wiring `inc_errors` on every WARNING
turns the error counter into an aggregate-noise channel; pick the high-pain
sites (Moonshine, tool dispatch) and leave the rest as logs.

---

## Appendix A — Where the boot-timeline events fire (exact lines)

| Event | File:line | Trigger |
|---|---|---|
| `app.startup` | `main.py:231` | `telemetry.init()` returned (deferred buffer drained) |
| `welcome.wav.played` (early) | `main.py:82` | aplay Popen succeeded in pre-init early path |
| `welcome.wav.played` (in-proc) | `warmup_audio.py:480` | aplay/winsound/afplay/pw-play dispatch returned |
| `welcome.wav.completed` (threaded) | `warmup_audio.py:342` | `aplay` exited (daemon thread wait()) |
| `welcome.wav.completed` (sync) | `warmup_audio.py:382` | aplay had already exited at helper-call time |
| `handler.start_up.complete` (legacy) | `elevenlabs_tts.py:362` | credentials prepared, startup-trigger task scheduled |
| `handler.start_up.complete` (composable) | `composable_conversation_handler.py:125` | Before pipeline.start_up await |
| `first_greeting.tts_first_audio` | 7 sites (see §1) | First PCM frame of process-life |

## Appendix B — Files referenced (absolute paths)

- `D:\Projects\robot_comic\.claude\worktrees\agent-ac4cb18d805fb51e5\src\robot_comic\telemetry.py`
- `...\src\robot_comic\main.py`
- `...\src\robot_comic\warmup_audio.py`
- `...\src\robot_comic\composable_conversation_handler.py`
- `...\src\robot_comic\composable_pipeline.py`
- `...\src\robot_comic\adapters\gemini_tts_adapter.py`
- `...\src\robot_comic\adapters\gemini_llm_adapter.py`
- `...\src\robot_comic\adapters\llama_llm_adapter.py`
- `...\src\robot_comic\adapters\gemini_bundled_llm_adapter.py`
- `...\src\robot_comic\adapters\moonshine_stt_adapter.py`
- `...\src\robot_comic\base_realtime.py`
- `...\src\robot_comic\local_stt_realtime.py`
- `...\src\robot_comic\elevenlabs_tts.py`
- `...\src\robot_comic\llama_base.py`
- `...\src\robot_comic\camera_worker.py`
- `...\src\robot_comic\audio\head_wobbler.py`
- `...\src\robot_comic\audio_input\alsa_rw_capture.py`
- `...\src\robot_comic\tools\background_tool_manager.py`
- `...\src\robot_comic\monitor.py`
- `...\src\robot_comic\joke_history.py`
- `...\src\robot_comic\history_trim.py`
- `...\src\robot_comic\startup_timer.py`
