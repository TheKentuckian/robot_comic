# Reachy Mini boot-time optimization — survey & next attack vectors

**Date:** 2026-05-16
**Status:** Decision memo (research only, no code changes)
**Author:** boot-time research agent
**Base commit:** `1b6bc7b` (post-Phase-4 epic close; Hook 3b shipped)

---

## §1 — Current state recap

The most recent measured baseline is captured in
`memory/project_moonshine_cold_load.md` (verified on `ricci` cold boot
2026-05-15 07:44Z at HEAD `746693d`, after all overnight perf PRs landed):

| Milestone | Wall-clock from systemd start | From Python `exec` |
|---|---|---|
| Welcome WAV dispatched (#317 early-play) | ~1 s | — |
| Python imports done (`+5.55s tool deps ready`) | ~6 s | ~5.5 s |
| Handler init done (Moonshine model loaded) | ~26 s | ~25 s |
| Moonshine `stream_opened` (listener bound) | ~41 s | ~36 s |
| First synthesized speech (canned opener) | **~42 s** | ~37 s |

**Key established facts:**

- **Welcome WAV early-play (PR #317)** moved first-audio-out from ~13 s
  to ~1 s — the perceived "robot is alive" win.
- **Pre-handler import budget** is now ~5.5 s, the result of PR #327
  (fastrtc lazy), #332 (camera worker defer), #336 (mediapipe lazy +
  play_emotion catalog cache). Further import-side perf does not move
  the needle.
- **Handler init** (~20 s) is dominated by Moonshine ONNX/ORT model
  load and is unchanged by any current PR. This is the dominant cost.
- **Total cold-start to first synthesized speech** is roughly flat
  versus the 2026-05-14 40.3 s baseline because Moonshine eats the
  ~8 s of import savings.
- Once warm (post-cold-boot), perceived perf is fine.
- The split `welcome_intro.wav` / `welcome_picker.wav` assets (#311)
  chain on a daemon thread so audio output stays continuous through
  the 20 s Moonshine gap.

**Tracking issues:** #323 (cold-start tracking), #270 (first-STT-cold-start),
#314 (cold-boot stall — fixed by Stream A / PR #347 for the ALSA MMAP arm).

**Known unknowns going into this memo:**

- What `reachy_mini_daemon` is doing during the 0-1 s window before our
  Python `exec`. We don't ship that daemon — only its install scripts —
  so reasoning is via systemd dependencies and the `wait-for-reachy-daemon.sh`
  poll behaviour.
- The breakdown of the Moonshine 20 s itself: how much is `import
  moonshine_voice`, how much is `Transcriber(...)` construction, how
  much is `prewarm_model_file` page-cache read, how much is ONNX
  Runtime session construction. The current boot-timeline collapses
  this into one "handler init" span.
- Why `stream_opened → first synthesized speech` is +5 s (start-up has
  finished but the canned opener takes that long to actually reach the
  speakers). This points at the TTS warm-up path (ElevenLabs HTTP
  round-trip and/or model load) rather than STT.

---

## §2 — Segment-by-segment boot path

### Segment A — OS kernel handoff → systemd ready

We don't have Raspberry Pi OS configs in this repo. The known boot order from
the autostart unit (`deploy/systemd/reachy-app-autostart.service`):

```
[Unit]
After=network-online.target reachy-daemon.service
Wants=network-online.target
```

- `network-online.target` typically waits for the carrier link, not for
  Internet reachability. On a static-IP Pi this is fast; on Wi-Fi with
  DHCP this can be 1-5 s.
- `reachy-daemon.service` is the third-party daemon ship from Pollen.
  We don't see its unit file, but `wait-for-reachy-daemon.sh` polls
  `http://127.0.0.1:8000/api/motors/get_mode` with 50 × 0.2 s tries
  (10 s cap), so the daemon is typically ready within a few seconds.
- The unit then issues two synchronous `curl` calls: `motors/set_mode/enabled`
  and `move/play/wake_up`. Both are blocking `ExecStartPre` lines —
  they run *before* `python -m robot_comic.main` starts.

**Cost:** unknown to us, but likely 2-5 s from systemd "start" to our Python
`exec`. The memory snapshot shows welcome WAV dispatched ~1 s from `systemctl
start`, but that's our app's first instruction, *after* the daemon poll, motors
enable, and wake-up POST have already completed. So either the snapshot's "1 s"
clock starts from `python exec` (most likely), or the daemon was already up
from a previous boot. **Status:** suspected cost, not directly measured.

### Segment B — `reachy_mini_daemon` startup

We control nothing here. From our app's POV the daemon is a black box exposing:

- HTTP API at `127.0.0.1:8000` (motors, wake_up, status, simulation flags).
- A persistent ALSA dmix sink (`plug:reachymini_audio_sink`) and source
  (`reachymini_audio_src`) that survive across our app's lifetime
  (see `memory/project_daemon_vs_app_lifecycle.md`).
- The XMOS USB chip — note from `reference_xmos_chip_config.md` that
  the daemon doesn't actually reach the chip on our hardware (wrong
  vid/pid), so the daemon's chip-init time is probably near-zero for
  us.

**Cost:** unknown. The boot timeline has no event for "daemon ready" —
the only relevant signal is whether `wait-for-reachy-daemon.sh` exits
in well under 10 s.

### Segment C — `reachy-app-autostart` → `main.py` import

`ExecStart=/venvs/apps_venv/bin/python -u -m robot_comic.main`

The top of `src/robot_comic/main.py` (lines 1-105) is carefully ordered:

1. **stdlib only** (os, sys, subprocess, pathlib) — fires
   `_play_welcome_early()` (lines 21-105).
2. `_play_welcome_early` spawns `aplay -D plug:reachymini_audio_sink`
   via `subprocess.Popen` and returns. Sets
   `REACHY_MINI_EARLY_WELCOME_PLAYED=1`. Fires the
   `welcome.wav.played` supporting event (queued — telemetry not yet
   initialised — see issue #337).
3. `from robot_comic import startup_timer` — captures `STARTUP_T0`
   (line 112). **This is our wall-clock origin for all "+X.XXs"
   logs.** Note: STARTUP_T0 is captured *after* welcome WAV dispatch,
   so the "+1 s welcome" number in the snapshot under-counts the true
   systemd→audio delta by a few hundred ms of Python startup +
   `_play_welcome_early` body time.
4. `import time, signal, asyncio, argparse, threading, ...` — pure
   stdlib, fast.
5. `from reachy_mini import ReachyMini, ReachyMiniApp` — this is the
   first non-stdlib import. The Pollen SDK pulls grpc / urllib3 /
   numpy.
6. `from robot_comic.utils import ...` — pulls the rest of our
   transitive import graph indirectly: anything `utils` touches
   (logger setup, head-tracker resolver, vision init helper).

**Cost:** the gap between welcome WAV dispatch and `STARTUP_T0` is
sub-100ms (3 lines of code). The gap between `STARTUP_T0` and the
first `log_checkpoint` call inside `run()` is dominated by the import
chain — measured at +5.55 s in the snapshot. **Status:** measured.

### Segment D — `main.py::run()` startup sequence

Inside `run()` (lines 161-703), the blocking order is:

| Step | What it does | Telemetry | Cost |
|---|---|---|---|
| 1. `setup_logger` | Console + journald sink wiring | — | <50 ms |
| 2. `StartupSettings()` + `.env` load | JSON read + dotenv | — | <100 ms |
| 3. `telemetry.init()` | OTel TracerProvider + drains pending supporting events | emits `app.startup` (since_startup ms) | <100 ms typical, 1-2 s if `ROBOT_INSTRUMENTATION=remote` (OTLP exporter init) |
| 4. `from robot_comic.pause import PauseController` (and 4 more `log_checkpoint`-tagged imports: console, pause_settings, core_tools, head_wobbler) | Deferred to here so the dashboard load is fast | "+Xs import pause/console/..." | These are the bulk of the 5.5 s import budget; #327/#332/#336 made them lazy |
| 5. `ReachyMini()` | SDK auto-detect backend; talks to daemon | — | <500 ms once daemon is up |
| 6. `play_warmup_wav` (no-op — early path already played) | Logs skip, emits checkpoint | "+Xs warmup wav skipped" | <10 ms |
| 7. `robot.client.get_status()` | HTTP RPC to daemon | — | <100 ms |
| 8. `initialize_camera_and_vision(args, robot)` | Constructs `CameraWorker` lazily — does NOT start the gstreamer thread (#332) | "+Xs camera/vision init" | <100 ms; full ~5 s cost is deferred to first `get_latest_frame()` |
| 9. `MovementManager(...)` | Move queue + 60 Hz thread (not started yet) | "+Xs movement manager" | <50 ms |
| 10. `HeadWobbler(...)` | Construct, no thread yet | — | <50 ms |
| 11. `read_pause_settings + PauseController` | JSON + wake-word setup | "+Xs pause controller ready" | <100 ms |
| 12. Optional face-recognition init | Only if `FACE_RECOGNITION_ENABLED` | "+Xs face recognition init" | 0 (default) |
| 13. `ToolDependencies(...)` | Dataclass pack | "+Xs tool deps ready" | trivial — but **this checkpoint at +5.55 s is the import-budget anchor**, marking end of Segment D |
| 14. `HandlerFactory.build(...)` | Instantiates `_LocalSTT*Host` + 3 adapters + `ComposablePipeline` + wrapper. Does NOT run `start_up` yet. | "+Xs handler init" | This logs at ~25 s in the snapshot, but the **construction** is sub-second; the ~20 s lives in step 16 below. The "handler init" label is misleading. |
| 15. Optional WS endpoint thread (gated by `WS_ENABLED`) | New asyncio loop on a daemon thread | — | <50 ms |
| 16. Optional `run_startup_screen` (`STARTUP_SCREEN_ENABLED` + non-sim) | Chatterbox pre-warm probe; awaits an async helper on the main loop | — | unknown; gated, but **synchronous on the main event loop** |
| 17. `LocalStream(...)` construction | FastAPI/Uvicorn setup, admin UI mount | "+Xs LocalStream constructed" | <500 ms |
| 18. `movement_manager.start()` + `head_wobbler.start()` | Two daemon threads | — | trivial |
| 19. `stream_manager.launch()` → FastRTC → `handler.start_up()` | Enters the asyncio event loop; first thing FastRTC awaits is `handler.start_up()`. **This is where the Moonshine 20 s lives.** | "+Xs handler.start_up.complete" (Hook #3) | ~20 s |

Note that the "handler init" `log_checkpoint` in step 14 sits at ~25 s in the
snapshot. That's misleading — the factory `build()` call itself is fast; the
delay is between `main.py` returning from `run_startup_screen` (if enabled) or
`LocalStream constructed` and the next blocking step. Worth verifying.
**Status:** import phase measured, model-load phase measured, but the
`handler init` label hides where the time actually goes.

### Segment E — Composable handler `start_up()`

`ComposableConversationHandler.start_up()` (lines 104-134) does:

1. Emit `handler.start_up.complete` supporting event (Hook #3).
   **Note: this fires *before* the delegate, so the row records "I'm
   about to start" — not "I'm done". The name is a misnomer; renaming
   to `handler.start_up.entered` would be clearer.**
2. `await self.pipeline.start_up()` — blocks until shutdown.

`ComposablePipeline.start_up()` (lines 129-151) is the critical path:

```python
await self.llm.prepare()      # Step P1
await self.tts.prepare()      # Step P2
await self.stt.start(...)     # Step P3  ← Moonshine model load lives here
self._started = True
await self._stop_event.wait() # Block forever
```

For the production triple `(moonshine, llama, elevenlabs)`:

- **P1** `LlamaLLMAdapter.prepare()` → `handler._prepare_startup_credentials()`
  — httpx client construction + tool-list build. Maybe an environment probe
  for the llama-server URL. Likely <1 s, possibly <100 ms.
- **P2** `ElevenLabsTTSAdapter.prepare()` → also calls
  `handler._prepare_startup_credentials()`. **On the diamond-MRO host**
  (`_LocalSTTLlamaElevenLabsHost`), this is the SAME `_prepare_startup_credentials`
  method as P1 — so **it runs twice** on the same handler. The adapter
  docstrings (4c.3, 4c.5) explicitly flag this as a known but
  out-of-scope double-init.
- **P3** `MoonshineSTTAdapter.start()` → re-binds `_dispatch_completed_transcript`
  to the pipeline bridge, then awaits `handler._prepare_startup_credentials()` —
  **a THIRD invocation of the same method** — and (via `LocalSTTInputMixin._prepare_startup_credentials`)
  calls `super()._prepare_startup_credentials()` then `asyncio.to_thread(self._build_local_stt_stream)`.

  `_build_local_stt_stream` (lines 323-391 of `local_stt_realtime.py`):
  - `import moonshine_voice` — first-time import. Pulls ONNX Runtime
    + tokenizers + numpy bindings.
  - `get_model_for_language(...)` — file/cache resolution.
  - `resolve_ort_model_path(...)` — prefer `.ort` over `.onnx`.
  - `prewarm_model_file(model_path)` — sequential read into the page
    cache. No-op on Windows. **This is one read of the model file
    from disk; on a cold SD card this is multi-second.**
  - `Transcriber(model_path=..., model_arch=...)` — **constructs the ONNX
    Runtime session.** Loads weights into RAM and builds the inference
    graph. *This is almost certainly the single biggest item in the 20 s.*
  - `self._open_local_stt_stream()` — `transcriber.create_stream()`,
    add listener, `stream.start()`. Should be sub-second once the
    session exists.

**Cost:** the 20 s in the snapshot is essentially all P3, and inside P3 most of
it is `Transcriber(...)` + `import moonshine_voice`.

**Status:** the *outer* P3 boundary is measured (handler.start_up.complete →
stream_opened in the snapshot), but the *internal* breakdown (`import` vs
`prewarm` vs `Transcriber.__init__` vs `create_stream`) is **not instrumented**.

### Segment F — First audio frame

Once `start_up` returns from the model load:

1. The canned-opener path in the host handler (`_speak_canned_opener` —
   see PR #319 × #327 fix) is what generates the synthetic startup
   turn. It runs TTS via the (already prepared) ElevenLabs adapter,
   pushes frames into the pipeline's output queue, and FastRTC
   drains them.
2. First synthesized frame fires `first_greeting.tts_first_audio`
   (telemetry.py:347).

**Cost:** ~5 s from `handler.start_up.complete` to `first_greeting.tts_first_audio`
in the snapshot. This is the ElevenLabs HTTP synthesis round-trip + first-chunk
arrival. **Status:** measured (Hook 3b just shipped the Gemini TTS coverage in
PR #382 too).

---

## §3 — Dominant costs (ranked)

### Rank 1 — Moonshine `Transcriber(...)` construction
**Cost:** ~15-18 s of the ~20 s handler-init window.
**Where:** `src/robot_comic/local_stt_realtime.py:371` (the `Transcriber(...)`
call inside `_build_local_stt_stream`).
**Telemetry:** **NOT individually instrumented.** Collapsed into
`handler.start_up.complete` minus the prior event. We have no
event between "ComposablePipeline.start_up enters" and "Moonshine stream_opened".
**Parallelism candidate:** Yes — runs strictly *after* `llm.prepare` and
`tts.prepare` today, but nothing in the LLM/TTS prepare paths depends on STT
state. Also could be moved out of the app process entirely (pre-fork daemon).
**Sequential block:** today, yes.

### Rank 2 — Pre-handler Python import budget
**Cost:** ~5.5 s.
**Where:** the chain of `from robot_comic.* import ...` lines in `run()`
(main.py:170-273) plus what they transitively pull. Specifically:
- `from reachy_mini import ReachyMini` (line 128) — Pollen SDK + grpc + numpy.
- `from robot_comic.console import LocalStream` — FastAPI + uvicorn + jinja.
- `from robot_comic.tools.core_tools import ToolDependencies` — pulls tool
  registry (mediapipe is now lazy thanks to #336).
**Telemetry:** measured via the `log_checkpoint` cadence ("+X.XXs import …").
Lands as plain INFO log lines, not OTel spans.
**Parallelism candidate:** mostly no — Python imports are inherently serial.
But moving some imports into the asyncio prelude (Segment E) is viable.
**Sequential block:** yes.

### Rank 3 — ElevenLabs first-frame latency
**Cost:** ~5 s (from `handler.start_up.complete` to `first_greeting.tts_first_audio`).
**Where:** `_speak_canned_opener` path + ElevenLabs streaming HTTP. The
adapter `prepare()` already runs at P2, so the network warmth should be
there — but the first synthesis still costs a full request round-trip.
**Telemetry:** captured (Hook 3b just added the missing GeminiTTSAdapter
emission in #382). Adapter `prepare` does not have its own span.
**Parallelism candidate:** Yes — the canned opener could start synthesizing
*during* Moonshine load (the audio backend is already prepared by step 12 of
the snapshot timeline). The current sequential `await stt.start()` blocks the
loop, but `asyncio.gather` could run the canned-opener synthesis in parallel
with `stt.start()`.
**Sequential block:** today, yes; structurally, no.

### Rank 4 — Systemd-level pre-app cost
**Cost:** unknown — somewhere between 1 s and 5 s, depending on Wi-Fi and
daemon cold-start.
**Where:** before our Python process exists. `wait-for-reachy-daemon.sh`,
two `curl` calls, daemon's own init.
**Telemetry:** none from our side. Could add via the `systemd-analyze
blame` route or pre-app shell logs.
**Parallelism candidate:** Yes — `motors/set_mode/enabled` and
`move/play/wake_up` could overlap with our app's import phase if they
were demoted from `ExecStartPre` to background fire-and-forget.
**Sequential block:** today, yes.

### Rank 5 — `prewarm_model_file` page-cache read
**Cost:** uncertain, probably 0-3 s depending on SD card / cold cache.
**Where:** `local_stt_realtime.py:99-127`.
**Telemetry:** none.
**Parallelism candidate:** Yes — this is just a sequential file read; it
could fire from a thread the moment Python starts (in parallel with imports).
**Sequential block:** today, yes — but trivially defeatable.

---

## §4 — Attack vectors

### V1 — Run STT model load in parallel with LLM/TTS prepare *and* canned-opener TTS
**What it changes:** in `ComposablePipeline.start_up`, use `asyncio.gather`
to run `llm.prepare`, `tts.prepare`, and `stt.start` concurrently. Even
better, *also* kick off the canned-opener synthesis as soon as the TTS
backend's `prepare` resolves — it doesn't need Moonshine ready.
**Estimated win:** 3-5 s if `llm.prepare + tts.prepare` total ~1-2 s
and run in parallel with the 20 s STT load (savings = the LLM/TTS prepare
time). 5-10 s if we can start the canned-opener TTS during Moonshine load
(audio is generated and queued by the time the listener attaches).
**Risk:** the `_prepare_startup_credentials` triple-call (P1/P2/P3 all
invoke it on the same handler) creates ordering hazards if it becomes
parallel. Need an idempotency guard on `_prepare_startup_credentials`
first — small refactor.
**Telemetry to prove:** add per-adapter `prepare` spans + a `pipeline.start_up.complete`
span; today none of these exist.

### V2 — Pre-fork Moonshine into a sibling daemon
**What it changes:** ship a tiny `reachy-moonshine-warmer.service` systemd
unit that runs in parallel with `reachy-app-autostart`. It imports
`moonshine_voice`, constructs the `Transcriber`, and exposes the loaded
session over a Unix socket (or just keeps the model file mmap'd to
prime the page cache for the real app).
**Estimated win:** 10-18 s — the full Moonshine load moves to the
background while the OS is still booting userspace.
**Risk:** medium. Two processes sharing an ORT session via shared memory
is non-trivial; the simpler "prime the page cache" variant gives a smaller
but easier win (probably 1-3 s). A `socket.send_fd` handoff of the loaded
model is feasible but moves us off Moonshine's documented API.
**Telemetry to prove:** existing `handler.start_up.complete` event suffices
to read the delta off the boot timeline.

### V3 — Demote `motors/set_mode/enabled` + `move/play/wake_up` from `ExecStartPre` to background
**What it changes:** instead of blocking the unit's start sequence on
two synchronous `curl` calls, fire them from inside our Python app
(or as a `&`-detached shell command) so motors enable in parallel
with import.
**Estimated win:** 0.5-2 s, but only if the curl calls are slow today.
The wake_up animation itself is asynchronous on the daemon side — what
blocks is the HTTP round-trip.
**Risk:** low. The motors enable is idempotent, and the wake_up animation
just queues onto the daemon's move queue.
**Telemetry to prove:** would need a span for "motors enabled" — currently
none. Easy to add.

### V4 — Pre-cache Moonshine model file at boot
**What it changes:** a tiny systemd unit (or an `ExecStartPre` in the
existing unit, run with `&` so it doesn't block) that does
`cat /path/to/moonshine.ort > /dev/null` during the daemon-wait window.
**Estimated win:** 0-3 s (only helpful on a cold SD card; a warm boot
already has it in cache).
**Risk:** trivial.
**Telemetry to prove:** need a "moonshine file cached" supporting event
to know whether the cache was warm or cold at app start.

### V5 — Parallelise the canned-opener TTS with STT bring-up
**What it changes:** today the canned opener fires *after* `start_up`
returns (i.e. after STT is fully up). If we kick the opener as soon as
TTS `prepare` resolves and the LLM is ready to be skipped (canned text,
no LLM round-trip), we can have the first audio frame out before
Moonshine finishes loading.
**Estimated win:** 5-10 s on perceived "robot speaks for real" latency.
The robot would talk during Moonshine load and only start *listening*
afterwards — UX win identical to the welcome WAV story but for the
real greeting.
**Risk:** medium. Need to confirm that ElevenLabs can synthesize without
the host handler being fully started, and that the audio sink isn't
contended with anything else during boot. The diamond-MRO host means
the handler is "real" — only the STT side is laggy.
**Telemetry to prove:** `first_greeting.tts_first_audio` already exists
(#301, plus #382 Gemini coverage). Would be a direct read.

### V6 — Move `STARTUP_T0` to the very first line of `main.py`
**What it changes:** today `STARTUP_T0` is captured by importing
`startup_timer` at line 112, which means everything above it (welcome
WAV dispatch, three stdlib imports) is uncounted. Move the `time.perf_counter()`
capture to a literal first line so the boot timeline reflects the true
Python wall-clock window.
**Estimated win:** 0 s (instrumentation only) but it would expose maybe
50-200 ms of "missing" boot time and remove a foot-gun for future
analysis.
**Risk:** trivial.
**Telemetry to prove:** the change *is* the telemetry fix.

### V7 — Idempotency guard on `_prepare_startup_credentials`
**What it changes:** the three adapter `prepare()` paths all currently
call the host handler's `_prepare_startup_credentials` — this triggers
a triple-init when LLM and TTS share a handler (every composable
triple does). Add a `self._prepared = True` guard so the second and
third calls are cheap no-ops.
**Estimated win:** 0.5-2 s depending on how heavy the redundant work is
(Gemini client init, httpx client construction).
**Risk:** low — the adapter docstrings already flag this is desired but
out-of-scope when 4c.5 landed.
**Telemetry to prove:** add adapter `prepare` spans to see double-init
collapse to single.

### V8 — Lazy-import `moonshine_voice` from the warmup audio thread
**What it changes:** move the `import moonshine_voice` line out of
`_build_local_stt_stream` into a thread that fires the moment
`startup_timer` is imported. By the time `start_up()` actually runs,
the import is already done.
**Estimated win:** 1-3 s (moonshine_voice import alone is significant;
hard to estimate without profiling).
**Risk:** low — the import is a side-effect-free Python module load.
Just ensure thread safety with no further `import moonshine_voice`
elsewhere on the hot path.
**Telemetry to prove:** add a span around the import in
`_build_local_stt_stream`; we'd see it drop to <100 ms.

### V9 — Warm Welcome WAV blip + drop the silent gap
**What it changes:** the optional `REACHY_MINI_WARMUP_BLIP_ENABLED`
already exists in `warmup_audio.py`. Enable it by default on-robot so
there's an instant in-process sine tone before the WAV file is even
read.
**Estimated win:** improves *perceived* boot by 200-500 ms.
**Risk:** trivial — fully gated, easy to revert via env.
**Telemetry to prove:** none needed; aural.

### V10 — Profile the inside of `Transcriber.__init__`
**What it changes:** wrap the line at `local_stt_realtime.py:371` with a
span + log how much is ORT session construction vs file IO vs tokenizer
load. *This is not an optimisation by itself, but it's a prerequisite
for picking between V2/V4/V8.*
**Estimated win:** 0 s (instrumentation). Unblocks the others.
**Risk:** trivial.
**Telemetry to prove:** the span *is* the telemetry.

---

## §5 — Recommended next 2-3 PRs

**Ordering rationale:** ship the low-risk instrumentation-fix first so the
follow-up wins are measurable, then take the biggest single win (parallelise
prepare), then ship the perceived-UX win (canned-opener-during-load).

### PR 1 — Instrument the Moonshine load segment
- Add `pipeline.start_up.complete` span, per-adapter
  `prepare` spans, and a span around `_build_local_stt_stream` with
  sub-spans for `import moonshine_voice`, `prewarm_model_file`,
  `Transcriber(...)`, and `create_stream + start`.
- Also relocate `STARTUP_T0` to the first line of `main.py` (V6).
- **No perf change. Pure measurement.** Without this, PR 2 lands blind.
- ROI: enables the next two PRs. ~30-line change.

### PR 2 — Parallelise STT load with LLM/TTS prepare + add idempotency guard
- V1 + V7 together: idempotency guard on `_prepare_startup_credentials`,
  then `asyncio.gather` the three `prepare/start` calls in
  `ComposablePipeline.start_up`.
- Estimated 2-5 s saved.
- ROI: solid mid-size PR (~50 lines of code, careful tests).
- Risk gated by PR 1's spans — if the guard breaks something, the
  per-adapter `prepare` spans surface it immediately.

### PR 3 — Speak the canned opener during STT load
- V5: kick off `_speak_canned_opener` (or whatever the equivalent
  post-#319 surface is) the moment TTS `prepare` resolves, regardless
  of STT readiness. The pipeline's input side (STT listener) wires
  up afterwards.
- Estimated 5-10 s of *perceived* first-real-greeting latency removed.
- ROI: biggest UX win. Mid-risk because it changes turn ordering.

**Deferred:** V2 (pre-fork daemon) is structurally the biggest win but is a
significant architectural commitment. Recommend it after PR 1 measures the
remaining gap. V4 (file-cache prime) and V9 (warmup blip) are tiny follow-ups
that can ride any release.

---

## §6 — Boot-specific telemetry gaps

The instrumentation-audit agent will see the full picture; the boot-specific
gaps that would block any of the above PRs:

1. **No per-adapter `prepare` span.** Today `handler.start_up.complete` is
   the only event between welcome WAV and stream_opened. To pick between
   V1, V7, and V8 we need to know how much of the 20 s is LLM prepare vs
   TTS prepare vs STT load vs the Moonshine import itself.
2. **No sub-span inside `_build_local_stt_stream`.** Without this we
   can't confirm whether the win from V8 (lazy-import moonshine_voice
   in a background thread) is real or noise.
3. **`handler.start_up.complete` fires on *entry*, not on completion.**
   Either rename it to `handler.start_up.entered` or fire a second
   event on actual completion. As-is, the name is misleading and the
   monitor TUI presents it as the wrong concept (#321/#301 lineage).
4. **No span around `wait-for-reachy-daemon.sh`.** Pre-Python is opaque.
   A `systemd-notify --status` line from inside the wait script would
   be visible in `systemctl status` but not on the OTel timeline. A
   tiny pre-app script that emits a supporting event over the local
   OTLP endpoint (or just a touch-file with a timestamp our app reads)
   would close this.
5. **`STARTUP_T0` is captured too late.** Moving it to literally line 1 of
   `main.py` (V6) is a 1-line change that closes a 50-200 ms gap.

---

## Appendix — Cited files & line numbers

- `deploy/systemd/reachy-app-autostart.service` — unit file with
  `After=network-online.target reachy-daemon.service`, the
  `wait-for-reachy-daemon.sh` `ExecStartPre`, and two `curl`
  `ExecStartPre` lines.
- `scripts/wait-for-reachy-daemon.sh:14-29` — 50 × 0.2 s poll loop.
- `src/robot_comic/main.py:21-105` — `_play_welcome_early` (welcome WAV
  dispatch before any non-stdlib import).
- `src/robot_comic/main.py:112` — `from robot_comic import startup_timer`
  (STARTUP_T0 capture point).
- `src/robot_comic/main.py:170-273` — the deferred-import block that
  defines the 5.5 s import budget.
- `src/robot_comic/main.py:402-430` — handler factory + the misleading
  "handler init" checkpoint.
- `src/robot_comic/main.py:578-585` — camera worker NOT started (deferred
  to first tool call; #323/#332).
- `src/robot_comic/composable_pipeline.py:129-151` —
  `start_up`: serial `llm.prepare → tts.prepare → stt.start`.
- `src/robot_comic/composable_conversation_handler.py:104-134` —
  `handler.start_up.complete` emission point (fires *before* delegate).
- `src/robot_comic/adapters/moonshine_stt_adapter.py:50-75` —
  `MoonshineSTTAdapter.start` calls `_prepare_startup_credentials` (the
  third invocation on the shared host).
- `src/robot_comic/local_stt_realtime.py:317-321` — `LocalSTTInputMixin._prepare_startup_credentials`
  → `asyncio.to_thread(self._build_local_stt_stream)`.
- `src/robot_comic/local_stt_realtime.py:323-391` — `_build_local_stt_stream`:
  `import moonshine_voice` + `prewarm_model_file` + `Transcriber(...)` +
  `_open_local_stt_stream()`.
- `src/robot_comic/telemetry.py:347-368` — `emit_first_greeting_audio_once`.
- `src/robot_comic/telemetry.py:382-428` — `emit_supporting_event` + the
  pre-init pending-events queue (#337).
- `src/robot_comic/warmup_audio.py:441-509` — `play_warmup_wav` (skips
  cleanly when `REACHY_MINI_EARLY_WELCOME_PLAYED=1`).
- `src/robot_comic/warmup_audio.py:493-501` — the optional `WARMUP_BLIP_ENABLED`
  fast-path.
