# Moonshine reliability, alternate STT models, sibling-daemon architecture

**Status**: Decision memo (research only — no code changes proposed in this PR).
**Author**: Research agent, dispatched by operator.
**Context**: Post-Stream-A (PR #347 merged). Audio-pipeline correctness verified.
Moonshine `state=idle` stall persists independently of capture path. Operator
wants a confidence read on three options.
**Main HEAD at writing**: `1b6bc7b` (Phase 4f / GeminiTTSAdapter lifecycle hook 3b).

---

## TL;DR

| Question | Verdict | One-line reasoning |
|---|---|---|
| (a) Can Moonshine be made permanently reliable? | **Low confidence — partial-fix only.** | The remaining failure mode (rearm-N-then-die, per #314 update 2026-05-15) is internal C-library state we can't see into; one promising untested lever exists (rebuild the Transcriber, not just the Stream). |
| (b) Best alternate STT to evaluate next | **faster-whisper `tiny.en` (int8 CTranslate2)**, with **Distil-Whisper `small.en`** as the accuracy fallback. | Mature, single-file CTranslate2 model, ~75 MB, ~1-2s cold-load on Pi 5, well-trodden streaming wrappers exist, permissive license. Distil-Whisper if WER from tiny is unacceptable. |
| (c) Moonshine-as-sibling-daemon | **Defer.** | Real cold-start win (~20s) but the wins are smaller than they look (perceived cold-start already mitigated by `welcome.wav` early-play), and the residency + ALSA contention + lifecycle complexity arrive in one bundle. Revisit only if (a) lands a permanent fix or (b) shows the new STT also has a long cold-load. |

---

## (a) Can we permanently fix Moonshine reliability?

### Inventory of failure modes

Read of `src/robot_comic/local_stt_realtime.py` (`LocalSTTInputMixin`), `src/robot_comic/adapters/moonshine_stt_adapter.py`, issue #314 + its 2026-05-15 update comment, and the recent `git log` for `local_stt_realtime.py`:

#### Mode 1 — Rearm-N-then-die (active blocker)

**Symptom (from #314 comment 2026-05-15)**: After ~2-4 successful transcripts, the next stream rebuilt by `_rearm_local_stt_stream()` silently stops emitting any line events. `add_audio()` still accepts frames; `window_peak=0.12` speech-level audio reaches the C library; no `on_line_started` / `on_line_updated` / `on_line_completed` ever fires. PR #370's RMS-in-diag instrumentation made this provable.

Today's rearm flow (`local_stt_realtime.py:444-482`):
```python
old_stream.stop(); old_stream.close()
self._open_local_stt_stream()           # transcriber.create_stream(...)
```
The `Transcriber` is preserved across rearms — only the `Stream` is recreated. From the line-by-line trace in the issue, streams 1-4 emit cleanly with the same listener wiring; stream 5 (different `stream_id` per the diag log) silently goes dark.

- **Root cause confidence**: **Low-to-medium.** Hypotheses 1-4 (listener wiring, VAD threshold, sample-rate/shape, CPU starvation) are ruled out by the diag data in the issue comment. The remaining explanation is **C-side accumulator state inside `moonshine_voice.Transcriber` that survives `Stream` recreation**. We can't see into the C library to confirm.
- **Permanent-fix feasibility**: **Partial.** The operator-named untested lever — rebuild the `Transcriber` itself, not just the `Stream`, after every line (or every N lines) — is the obvious next experiment. Cost: the 20s ONNX model load (per [[project_moonshine_cold_load]]) would re-fire on every rebuild. **That cost is unacceptable as the default behaviour.** Mitigation paths:
  1. **Rebuild on N-th completion** — pick a safe N (say, 3) below the observed dying point. Burns one ~20s pause per N turns. Awful UX, but technically a workaround.
  2. **Rebuild lazily on stall detection** — current watchdog (line 691, "[Moonshine] idle for %.1fs with %d audio frames received") already fires; instead of just logging, trigger a Transcriber rebuild from the watchdog. Same ~20s pause but only when the failure mode actually fires. Better UX, but conversation is dead during the rebuild.
  3. **Background Transcriber pool** — keep a second pre-warmed `Transcriber` instance hot in a worker thread. Swap on stall. Doubles the resident model memory (~200-400 MB extra for Moonshine `small_streaming`). Eliminates the user-visible pause but is operationally heavy.
- **Mitigations available today** (zero code change): set up a systemd-level watchdog on the journal pattern `"idle for [0-9]+s with [0-9]+ audio frames"` and `systemctl restart reachy-app-autostart` on hit. Costs the full ~42s cold-start every wedge but recovers without operator intervention.

#### Mode 2 — Cold-load latency (known, not a bug)

20s on Pi 5 cold boot for the ONNX `small_streaming` model load ([[project_moonshine_cold_load]]). Dominates total wall-clock-to-first-greeting (~42s). Already mitigated perceptually by `welcome.wav` early-play (#317): user hears "robot is alive" at ~1s, real greeting at ~42s.

- **Root cause confidence**: **High.** Profiled and traced in `project_moonshine_cold_load`.
- **Permanent-fix feasibility**: **No (within Moonshine).** This is the model's inherent setup cost. Levers: smaller model (already on `small_streaming`), `.ort` fast-load (already shipped #194), page-cache prewarm (already shipped, `prewarm_model_file` line 99). All extracted.
- **Mitigations**: pre-forking the load process (the sibling-daemon proposal in (c)) is the only remaining lever for the wall-clock 20s.

#### Mode 3 — First-utterance latency (`stt.infer dur_ms=5381`, issue #270)

First inference after model load takes ~5s; subsequent inferences are 1-3s. Caused by ONNX runtime + XNNPACK warmup on first real-audio batch. Not a stall — the first transcript does eventually fire.

- **Root cause confidence**: **High.** Issue #270 captures the pattern; consistent with cold-XNNPACK behaviour.
- **Permanent-fix feasibility**: **Yes, partial.** Issue #270's recommendation — feed a short silent buffer at startup, before `Application startup complete` — is the standard warmup trick. This is a one-day fix that should land independently of any of the larger questions in this memo. Worth pulling forward.
- **Mitigations**: none beyond the warmup pass.

#### Mode 4 — Self-echo VAD lock (already mitigated)

If the robot's own TTS audio reaches the mic during playback, Moonshine's streaming VAD treats it as one continuous utterance and never emits `completed`. The `receive()` method (line 772-774) drops mic frames while `_speaking_until` is in the future. Echo guard at line 600 also discards transcripts that arrive during TTS playback.

- **Root cause confidence**: **High.** Code is correct.
- **Permanent-fix feasibility**: **Done.** No further work expected.
- **Mitigations**: already in production.

#### Mode 5 — Per-completion rearm (the #279 fix — already mitigated)

After `LineCompleted`, Moonshine's Stream stops emitting further events; only a recreated Stream will fire callbacks for the next utterance. `_pending_stream_rearm` flag + `_rearm_local_stt_stream()` handle this for streams 1-N. **This is the mechanism that, after N rebuilds, hits Mode 1.**

- **Root cause confidence**: **High** (mitigated).
- **Permanent-fix feasibility**: **Done** for the per-completion case; **failed at N>=3** per Mode 1.
- **Mitigations**: see Mode 1.

#### Mode 6 — Silent-input drift / VAD-threshold misfires

Read of `_log_heartbeat` (line 646) shows we only warn if `state == "idle"` for >10s with `audio_frames > 0`. Below that threshold, low-amplitude input passes silently. With AGC target now at 0.1 ([[reference_xmos_chip_config]]) + Stream A direct ALSA RW capture providing peak=1.0 baseline, this should not be a live issue. **No code-level evidence of an active bug here**, listed for completeness.

- **Root cause confidence**: N/A (no symptom).
- **Permanent-fix feasibility**: N/A.

### Overall confidence

**Low.** Modes 2, 4, 5 are solved or solvable on our side. Mode 3 has a known fix path (#270 warmup) but hasn't shipped. Mode 1 — the active blocker — points to internal C-state inside `moonshine_voice.Transcriber` that survives Stream recreation. The operator's hypothesis (rebuild the Transcriber, not the Stream) is the most promising next experiment, but it costs ~20s per rebuild. Even a successful rebuild-on-N workaround leaves the operator paying a 20s pause every 3-5 conversational turns.

**Operator's likely best path if staying on Moonshine**:
1. Ship #270 warmup pass (cheap; orthogonal win).
2. Prototype the Transcriber-rebuild experiment behind a feature flag — gather repro data on whether the rebuild actually fixes Mode 1.
3. If yes: implement the **lazy stall-driven rebuild** (Mode 1 mitigation 2). Pair it with the **background pool** (Mode 1 mitigation 3) to hide the rebuild latency.
4. If no: the C library has unrecoverable state at the process level → only systemd-watchdog restart or a different STT engine fixes this.

This is a lot of speculative work on a model whose own design doc doesn't appear to commit to multi-utterance robustness. **Evaluate at least one alternate in parallel** — see (b).

### One-line recommendation embedded in code (not implemented here)

If you want a single concrete next experiment that's testable in one chassis session: add a `LOCAL_STT_REBUILD_AFTER_N` env, default `0` (off). When set to N, `_rearm_local_stt_stream()` increments a completion counter and, when it hits N, calls `_build_local_stt_stream()` instead of `_open_local_stt_stream()` — i.e., reload the Transcriber. Set `N=2` for a quick repro; if Mode 1 stops firing, the rebuild lever works. Estimated effort: ~30 lines + one test.

---

## (b) Alternate STT model recommendations

Two candidates, ranked.

### Candidate 1 (recommended): faster-whisper `tiny.en` (CTranslate2 int8)

| Property | Value |
|---|---|
| Variant | `tiny.en`, int8 quantised, CTranslate2 backend |
| Cold-load on Pi 5 | ~1-2s (CTranslate2 is mmap-friendly, no graph-parse step like ONNX) |
| Per-utterance inference | ~0.3-0.8s for ~5s audio (faster-whisper benchmarks; conservative for ARM) |
| Accuracy (WER) | ~10-14% on conversational English (vs Moonshine `small_streaming` ~8-11%) |
| Model size | ~75 MB int8 |
| RAM residency | ~150-200 MB resident |
| License | MIT (faster-whisper) over MIT (Whisper itself) — fully permissive, on-device commercial OK |
| Implementation cost | Medium. Whisper is **batch-only**, not native streaming. Need a fixed-length window (e.g., 2-3s rolling buffer) with overlap, or a VAD-front-ended chunker (silero-vad is the standard companion). Estimated: ~150 LoC new adapter + one test file + one config flag. Could ship behind a feature flag in 1-2 days. |
| Streaming support | **Not native.** Two common patterns: (1) VAD-segmented chunks (silero-vad emits utterance boundaries → batch transcribe the segment) — simpler, less CPU, no partials; (2) sliding window with overlap — gives partials at the cost of more CPU. For this app, pattern 1 is enough (we don't use partials for anything user-visible today). |
| Known reliability | Wide community adoption (Home Assistant, Mycroft fork projects, willow.ai). No equivalent of the rearm-N-then-die pattern reported in the faster-whisper issues tracker. Process-state model is clean — `WhisperModel(...).transcribe(audio_array)` is a pure-ish call returning segments. |
| Trade-off vs Moonshine | + Mature C++ runtime, more eyes on bugs<br/>+ No internal Stream state to wedge<br/>+ Cold-load is 10x faster (~2s vs ~20s)<br/>− Higher per-utterance latency (~0.5-0.8s vs Moonshine ~0.3s in the streaming case)<br/>− Slightly worse WER on hard utterances |

**Why this one first**: the CTranslate2 backend was specifically designed to fix the slow-load / heavy-runtime problems that ONNX-streaming has. The cold-load alone is the biggest single architectural advantage. Streaming via VAD-chunking is well-trodden — we already have an analogue (`_speaking_until` echo guard) and the welcome-gate state machine; introducing utterance boundaries is in-house pattern.

### Candidate 2 (fallback for accuracy): Distil-Whisper `small.en`

| Property | Value |
|---|---|
| Variant | Distil-Whisper `distil-small.en` (HF: `distil-whisper/distil-small.en`) |
| Cold-load on Pi 5 | ~3-5s (still much faster than Moonshine; bigger weights than tiny but distilled architecture removes some layers) |
| Per-utterance inference | ~0.6-1.2s for ~5s audio |
| Accuracy (WER) | ~7-9% on conversational English — competitive with Moonshine `small`, better than Whisper `tiny` |
| Model size | ~166 MB int8 (CTranslate2 conversion exists) |
| RAM residency | ~350-450 MB resident |
| License | MIT |
| Implementation cost | Same adapter shape as Candidate 1 — VAD-segment + batch transcribe. Same ~150 LoC. |
| Streaming support | Same as Candidate 1 (batch + VAD chunker). |
| Known reliability | Newer than faster-whisper but distilled from a well-understood teacher; lower variance in the wild than Moonshine streaming. |
| Trade-off vs Candidate 1 | + Better WER<br/>− 2-3x more RAM, 2-3x slower inference |

**Why not first**: if the operator's actual gating constraint is WER, jump straight here. But for the conversational app where transcripts are 3-8 words on average and the LLM is forgiving of light noise, `tiny.en` is more than adequate, and the latency/footprint gap matters when running alongside MediaPipe + ElevenLabs streaming + the LLM on a Pi 5.

### Models considered and dropped

- **NVIDIA Parakeet TDT 0.6B** — top WER on Open ASR Leaderboard, but the official runtime is NeMo/PyTorch, which is heavy on the Pi 5. CTranslate2/ONNX ports exist as community efforts but are not first-class. Skip until/unless an official lightweight runtime ships.
- **whisper.cpp (`tiny.en` ggml)** — viable; similar territory to faster-whisper but with a different runtime. Less Python-friendly. Choose faster-whisper for ecosystem.
- **Vosk** — accuracy materially worse than Whisper/Moonshine. Skip.
- **Cloud STT (OpenAI Realtime / Gemini Live)** — defeats the on-robot principle and is already available via the bundled-pipeline path (`PIPELINE_MODE=openai_realtime`/`gemini_live`). Not an "alternate to Moonshine" — it's a different deployment posture.

### Implementation shape if we adopt Candidate 1

The Phase 1 `STTBackend` Protocol in `backends.py` is already exactly the right contract — it takes an `on_completed` callback, a `feed_audio(AudioFrame)` per chunk, and a `stop()`. A new `FasterWhisperSTTAdapter` slots in next to `MoonshineSTTAdapter` with no change to `ComposablePipeline`. The factory wiring (Phase 4-style) gains one new branch keyed on a new `LOCAL_STT_ENGINE=faster_whisper` enum. No new Protocol needed.

Adapter sketch (not for this PR):
- A `silero-vad` instance maintained in the adapter, fed `feed_audio` chunks.
- On `silero-vad` signalling utterance end → submit the accumulated buffer to `WhisperModel.transcribe` on a `ThreadPoolExecutor` to avoid blocking the asyncio loop.
- Result text → `on_completed(text)`.

---

## (c) Moonshine as a sibling daemon to `reachy_mini_daemon`

### The proposal

Run Moonshine in a separate process — its own systemd unit — booted alongside (or before) `reachy_mini_daemon`. Pre-warm the model there. The app reads transcripts via IPC. Opt-in. Silently bypassed when `PIPELINE_MODE` is one of the bundled streaming s2s modes (`openai_realtime`, `gemini_live`, `hf_realtime`).

### Boot-time win

**Estimated**: the 20s Moonshine cold-load moves off the app's critical path entirely. Wall-clock from `systemctl start reachy-app-autostart` to "first STT-ready" drops from ~26s (per [[project_moonshine_cold_load]]) to roughly ~6s (the Python imports + handler init excluding model load).

**However**: per the same memory file, the `welcome.wav` early-play (#317) already moved "perceived alive" from 13s to 1s. The wall-clock to first **real** greeting is still gated by Moonshine being ready, but the operator already accepted the 42s baseline because the perceived UX is good. **The boot-time win exists but the operator has paid less attention to it since #317 shipped.**

### Memory residency cost

The Moonshine model (`small_streaming`) is ~200-400 MB resident. As a sibling daemon, that footprint is permanent — it doesn't free when the app picks a bundled pipeline (OpenAI Realtime / Gemini Live / HF Realtime) that handles STT server-side. Pi 5 4GB SKU has limited slack; Pi 5 8GB SKU is fine.

**Risk**: if the operator's usage is mostly bundled streaming pipelines, the sibling daemon is wasted residency the entire session. If usage is mixed, the optimisation is real. We don't currently have telemetry on which pipeline the operator actually selects most often.

### ALSA contention

Today, the comic app holds the microphone (post-Stream-A: it spawns its own `arecord` against `reachymini_audio_src`). The daemon also holds a reader (its GStreamer pipeline, used for TTS playback's echo-cancellation reference, etc). dsnoop allows multiple readers, so adding a third (the sibling Moonshine daemon) is mechanically possible.

But: who **owns** the speech-to-text reading? Two viable shapes:

- **Shape A — sibling owns mic, pushes transcripts**: Sibling daemon spawns its own `arecord` reader against `reachymini_audio_src`. App stops spawning its own. App receives transcripts over IPC. Clean, but means the sibling daemon is mandatory when `LOCAL_STT_ENGINE=moonshine` — there's no in-process fallback if the sibling crashes.
- **Shape B — sibling owns model, app pushes audio**: App keeps the `arecord` reader (Stream A as-is). App streams audio frames to sibling daemon over IPC. Sibling daemon does STT and pushes transcripts back. Less mechanical risk but doubles the IPC traffic: 16kHz × 16-bit × 1ch = 32 KB/s, trivial over UNIX socket.

Shape B is lower-risk. The capture path is unchanged; only the STT inference is offloaded. Recommend Shape B if this happens.

### Lifecycle complexity

systemd unit graph if the sibling daemon ships:

```
reachy-daemon.service              # exists, owns motors + GStreamer audio out
  └── reachy-moonshine.service     # new — opt-in via WantedBy/Wants
        └── reachy-app-autostart.service
```

Failure modes:
- **App starts before Moonshine sibling is ready** → already-existing `wait-for-reachy-daemon.sh` pattern works (poll the sibling's UNIX socket). Trivial.
- **Sibling crashes mid-session** → app currently has no story for "STT goes down mid-conversation." Either (1) the app crashes too (worst), (2) the app falls back to in-process Moonshine load (rebuilds 20s state), (3) the app degrades to "no STT available, robot is mute to speech" until restart. Each option is a new failure surface. Today's in-process Moonshine has the same problem but with a smaller blast radius — if it dies, the app dies, systemd restarts the unit.
- **Sibling memory leak or stall** → operator now has *two* services to babysit instead of one. Watchdog story doubles.

### IPC shape

Read of `src/robot_comic/` for daemon IPC patterns: the app talks to `reachy_mini_daemon` via the daemon's HTTP API at `http://127.0.0.1:8000` (see `deploy/systemd/reachy-app-autostart.service` lines 36-39) and via the Python `ReachyMini` class wrapper (consumed in `console.py:21`, `main.py:128`, etc) which is itself an HTTP/WebSocket client to the same daemon. There is **no UNIX-socket or shared-memory IPC** in current use.

For the sibling daemon, three options:
- **HTTP REST + Server-Sent Events** — consistent with the existing daemon pattern. Adapter posts audio chunks to `/feed`, subscribes to `/events` for transcripts. Latency ~10ms overhead per chunk. Adequate.
- **UNIX socket + length-prefixed JSON** — lighter. ~1ms overhead. Standard pattern, no new Python deps beyond stdlib `asyncio.open_unix_connection`.
- **gRPC** — overkill. Don't add a new build/codegen dep for this.

UNIX socket is the right pick — light, fast, no HTTP framing tax on the 32 KB/s audio stream. Recommend UNIX socket if this happens.

### Code shape

A `SiblingDaemonSTTAdapter` slots in next to `MoonshineSTTAdapter` and the future `FasterWhisperSTTAdapter`. Same `STTBackend` Protocol. Internally it opens a UNIX socket, streams `feed_audio` frames in, and dispatches a Protocol-shaped `on_completed` callback whenever the sibling pushes a transcript event back.

`MoonshineSTTAdapter` (in-process) does NOT need to be refactored — it stays as the fallback / non-sibling-daemon path. The two adapters coexist behind an `LOCAL_STT_TRANSPORT=inprocess|sibling_daemon` config dial.

The sibling daemon itself: a new `src/robot_comic_moonshine_daemon/` package (or external repo, but in-tree is simpler), entry point `python -m robot_comic_moonshine_daemon`. Owns the Moonshine model load + Stream lifecycle. Listens on a UNIX socket. Same `_rearm_local_stt_stream` logic, just running in a different process. **Importantly**: this does NOT fix Mode 1 (rearm-N-then-die) — that bug is internal to `moonshine_voice` and will fire equally inside the sibling daemon. The sibling daemon only fixes the cold-start wall-clock problem (Mode 2).

### Operator opt-in mechanism

A new env: `REACHY_MINI_LOCAL_STT_TRANSPORT=inprocess|sibling_daemon`, default `inprocess`. We just retired two dials in Phase 4f ([[project_session_2026_05_15_audio_root_cause]] lineage). Adding another is fine if the value is clear, but the operator should weigh it against the principle of "fewer dials, not more."

Deploy-time choice (operator chooses sibling-daemon install in their setup script, no env at all) is cleaner but loses runtime switchability. **Recommend env if shipped** — matches existing project conventions.

### Risk of partial benefit

If the operator's usage is dominated by bundled streaming pipelines (OpenAI Realtime / Gemini Live / HF Realtime — which manage their own STT server-side), the sibling Moonshine daemon eats ~300 MB of RAM permanently for no benefit. The hard rule needs to be: **the sibling daemon must skip its model load when the active pipeline doesn't need it**. That means the sibling daemon needs to know which pipeline is active — i.e. read the same env var the app reads, OR be told via the same UNIX socket on connect. Doable, but it's another sync point.

### Recommendation: **defer**

Reasoning:
1. **The biggest win — cold-start time — has been partially neutralised by `welcome.wav` early-play.** Perceived UX is good; wall-clock-to-real-greeting is ugly but accepted. The marginal value of moving the 20s off-critical-path is real but smaller than it was three weeks ago.
2. **The sibling daemon does NOT fix Mode 1 (rearm-N-then-die).** It moves the same buggy `moonshine_voice` code into a different process. The current top blocker stays unfixed.
3. **It adds permanent residency cost during bundled-pipeline use.** Without telemetry on actual pipeline-selection mix, we'd be paying ~300 MB and a second systemd unit for an unknown share of sessions.
4. **It adds a new failure surface** (sibling crash mid-session) without resolving an existing one.

**Order of operations the operator should consider**:
1. **First**: ship #270 warmup (cheap, orthogonal).
2. **Second**: prototype the Transcriber-rebuild experiment from (a) — it's the actual blocker test.
3. **Third**: spike `FasterWhisperSTTAdapter` from (b) — gets us off `moonshine_voice` reliability questions entirely, and the faster-whisper cold-load (~2s) is itself most of the sibling-daemon win.
4. **Only then revisit (c)**: if after the three above we still have a real cold-start problem AND telemetry shows local STT is the dominant pipeline, the sibling daemon is worth its cost. Otherwise skip.

In short: **the sibling-daemon proposal optimises the wrong axis given current bug priorities and welcome-gate UX**. Defer until those higher-priority items have settled the architecture.

---

## Appendix: cross-references

- [[project_session_2026_05_15_stream_a]] — Stream A direct ALSA RW capture (PR #347) shipped; field-test confirmed audio level is no longer the bug.
- [[project_session_2026_05_15_audio_root_cause]] — predecessor session that root-caused the MMAP attenuation.
- [[reference_alsa_mmap_attenuation]] — the hardware quirk that Stream A bypasses; retest condition documented.
- [[reference_xmos_chip_config]] — AGC parameter table; we've set the desired-level to 0.1 (improvement vs factory 0.0045).
- [[project_moonshine_cold_load]] — 20s cold-load profiling; `welcome.wav` early-play (#317) mitigated perceived UX.
- GitHub issue [#314](https://github.com/TheKentuckian/robot_comic/issues/314) — Moonshine streaming stall; comment 2026-05-15 isolates the rearm-N-then-die failure mode. **Read the comment in full before any Moonshine work.**
- GitHub issue [#270](https://github.com/TheKentuckian/robot_comic/issues/270) — first-utterance ~5s inference latency; warmup pass recommended.
- GitHub issue [#279](https://github.com/TheKentuckian/robot_comic/issues/279) — original "wedges in state=completed" bug; fixed by per-completion rearm (PR #293). The fix's edge case is Mode 1 of this memo.
- GitHub PR [#370](https://github.com/TheKentuckian/robot_comic/pull/370) — `MOONSHINE_DIAG` rolling peak/RMS; the instrumentation that proved Mode 1 is rearm-state, not silence.
