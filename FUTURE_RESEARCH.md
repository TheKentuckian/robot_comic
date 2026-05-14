# Future Research / Manual Follow-ups

This document collects work items that were closed during the 2026-05-13/14
sweep because they require **on-hardware testing, manual audio/video
recording, or open-ended research** that a remote coding agent on a
Windows dev machine cannot deliver. Each item is structured so it can be
re-opened as a fresh GitHub issue when the corresponding manual session is
scheduled.

The reference to each original issue is included in case any context was
lost in the close comments.

---

## On-robot motion & hardware

### Wake-up animation gentling (was #155)
**What:** The robot's `wake_up` animation on service start is physically
jarring — fast head + neck moves arriving at max joint velocity. Especially
bad on a double-restart sequence.

**Why it's blocked here:** The animation lives in the `reachy_mini` daemon
(upstream `pollen-robotics/reachy-mini`), not this fork. Cannot edit
upstream from this session.

**Recommended next session:**
- Open a fork of `pollen-robotics/reachy-mini` and adjust the wake-up
  trajectory: add cubic ease-in/out, cap angular velocity (~1.0 rad/s),
  consider a debounce so a quick restart-cancel-restart doesn't double-fire.
- Once the upstream PR lands, bump the SDK version pin in `pyproject.toml`.

### Servo calibration + body-collision avoidance (was #19)
**What:** Hand-calibrate each servo's safe range so head/torso moves
cannot drive the head into the cowling or neck base.

**Why it's blocked here:** Requires the physical robot and a manual
sweep-and-measure procedure.

**Recommended next session:**
- Use the on-robot serial console to nudge each axis to its physical limit.
- Capture the safe `(min, max)` per axis and feed into
  `src/robot_comic/motion_safety.py` constants (or the new env vars added
  in PR #176).
- The current envelope was estimated from `MoveHead` docs — replace with
  measured values for precision.

### Basic L-R movement testing (was #20)
**What:** Run the head through its full L-R range with a real persona
session to confirm sweeps look smooth.

**Why it's blocked here:** Visual + audible inspection only.

**Recommended next session:**
- Use the new `scan` gesture (PR #209) and the gemini_live presence-probe
  (PR #189) as exercise paths.
- Record a short video clip for the repo's `docs/` if motion looks clean.

### EEPROM / cmdline.txt / config.txt boot tweaks (was #22)
**What:** Tune Pi boot to drop ~5-15s of boot time via:
- `eeprom_bootloader` config (skip USB enumeration, disable HDMI probing)
- `cmdline.txt`: `quiet`, `loglevel=3`, `nosplash`, disable unused console output
- `config.txt`: disable Bluetooth/WiFi if unused, reduce GPU memory split,
  `boot_delay=0`

**Why it's blocked here:** Requires editing the Pi's boot partition over
SD card or SSH.

**Recommended next session:**
- Document current boot timing (use `systemd-analyze`).
- Apply tweaks one at a time, measuring each delta.
- Land the changes in `deploy/pi/` (new directory) as templated
  config snippets the user can install via a wrapper script.

### Phase 4 movement training from video (was #33)
**What:** Train a movement-from-video model so the robot can mimic comic
delivery gestures observed in stand-up footage.

**Why it's blocked here:** Open-ended ML research; requires a training
pipeline + curated footage + significant compute.

**Recommended next session:** Spin off as a separate research
repo/project. The current `src/robot_comic/gestures/` (PR #209) is the
target API — any trained model should produce gesture sequences that
plug into `GestureRegistry`.

---

## Voice / audio (manual listening + recording)

### Welcome-gate `welcome_name.wav` assets (was #108)
**What:** Record the per-persona `welcome_name.wav` files for 5 comedians
+ 6 character voices. These are short clips the welcome-gate plays as
"the robot heard you say <name>".

**Why it's blocked here:** Requires Gemini TTS quota + listening to pick
the best take per persona.

**Recommended next session:**
- Run `scripts/build_welcome_assets.py` (already in repo) for each
  persona name. The shared narrator WAVs are already committed to
  `assets/welcome/`.
- The per-persona `welcome_name.wav` files are gitignored (per
  `.gitignore`); the user can commit them with `git add -f` after a
  satisfactory listen.

### Manual A/B audio verification of per-sentence delivery cues (was #103)
**What:** PR #149 dropped `system_instruction` for the preview Gemini TTS
model and prepends the delivery cue to the text instead. Confirm by ear
that the new path still produces the correct delivery (fast / slow /
whispered / etc).

**Why it's blocked here:** Requires running real Gemini TTS calls and
listening.

**Recommended next session:**
- A 10-minute listening session with a script that runs Gemini TTS with
  each delivery tag (`[fast]`, `[slow]`, `[whispered]`, `[annoyance]`, …)
  and verifies the spoken output matches expectation.
- Capture the comparison clips in `docs/audio-samples/` if useful for
  regression watching.

### Chatterbox voice-clone calibration per persona (was part of #44)
**What:** Items 3 + 4 of #44 — calibrate `exaggeration` and `cfg_weight`
per persona against actual Chatterbox output with each cloned voice; and
decide whether Chatterbox Turbo's latency win outweighs its zero-shot
cloning quality drop.

**Why it's blocked here:** Listening-and-tuning loop with a real Chatterbox
server.

**Recommended next session:**
- The `PERSONA_BASELINES` in `src/robot_comic/chatterbox_tag_translator.py`
  has the current starting estimates. Walk each persona through 5-10 lines
  and tune the values until the voice feels right.
- For Turbo: run the same set with both standard and Turbo and pick.
- Document the final values in each persona's `chatterbox.txt`.

### Fine-tune Chatterbox on Don Rickles archival audio (was #37)
**What:** Run a Chatterbox fine-tune on the ~30 minutes of cleaned Rickles
clips from `profiles/don_rickles/voice_prep/` to push identity fidelity
past what IVC alone can achieve.

**Why it's blocked here:** Training compute + Chatterbox training stack.

**Recommended next session:** Spin off as a separate research session.
Output is a `.pt` checkpoint that replaces the per-persona generic Chatterbox
model. Add a config slot for `CHATTERBOX_MODEL_OVERRIDE_PATH` in
`config.py` if not already present.

### Benchmark Qwen3 14B dense vs 35B MoE (was #78)
**What:** Run the same persona-eval test set against Qwen3 14B dense and
35B MoE and decide which gives the best quality/latency trade-off for
production.

**Why it's blocked here:** Needs both models loaded on a GPU box.

**Recommended next session:**
- Use `bench_llm.py` (already in repo) as the harness — extend it to
  alternate between two models on a fixed prompt set.
- Capture median latency + a subjective rubric (humor, in-character,
  refusal rate) and report.
- Update `LLAMA_CPP_URL` model pin in deployment scripts based on the
  winner.

### Better Gemini Live voice research (was #15)
**What:** Explore alternative Gemini Live voice configurations beyond the
defaults baked in.

**Why it's blocked here:** Listening-and-comparing across many voice IDs.

**Recommended next session:**
- Cycle through the Gemini Live `prebuilt_voice_config` voices for each
  persona, record short samples.
- Update `profiles/*/voice.txt` with the best per-persona pick.

---

## Tests / UI (require running app + UI interaction)

### Admin Restart end-to-end click-test (was #138)
**What:** The admin web UI exposes a "Restart app" button that triggers
`exit(75)` so systemd relaunches the process. Write an automated e2e
test that drives the click and verifies the systemd unit comes back.

**Why it's blocked here:** Requires headless browser automation on a
running app, plus systemd present (Linux + user systemd or
robot deployment).

**Recommended next session:**
- Spin up the app in `--sim` mode.
- Use Playwright (already added in `tests/vision/`-adjacent files? check)
  to click the Restart button.
- Mock `os.kill(getpid(), 75)` and verify the JSON acknowledgment instead
  of actually exiting — the e2e shouldn't kill the test runner.

### Record a video of the working robot (was #35)
**What:** A short demo video for the repo README / Hugging Face Space
showing a full persona session end-to-end.

**Why it's blocked here:** Physical recording + edit.

**Recommended next session:**
- Use a phone or webcam to capture a 1-2 minute session.
- Add to `docs/demo/` and link from README.

---

## Observability + dev-env

### Win11 laptop setup research (was #27)
**What:** Document the dev-machine setup (Windows 11 + dependencies)
for new contributors.

**Why it's blocked here:** Capture-as-you-go session by the actual user
on their actual machine.

**Recommended next session:** Walk through a fresh checkout on a clean
Win11 box and capture every step into `docs/DEV_SETUP_WIN11.md`. The
existing `DEVELOPMENT.md` covers the cross-platform commands; this would
be Win11-specific.

### Camera + lightweight VLM for closed-loop refinement (was #25)
**What:** Investigate whether a small on-device VLM (vs. backend Gemini
vision) can power closed-loop scene awareness (e.g. notice a hat the user
removed mid-set and reference it).

**Why it's blocked here:** Open-ended research; model selection + latency
profiling on the Pi.

**Recommended next session:**
- Candidates: SmolVLM2 (already in repo as the local-vision option),
  Moondream, or a quantized LLaVA-Phi.
- Measure inference latency on Pi 5 8GB.
- Decide based on time-to-first-token + scene-description fidelity.

---

## Notes on what's already in the tree

- **Generic narrator WAVs** (`assets/welcome/welcome.wav`,
  `list_intro.wav`, `not_understood.wav`, `switching.wav`) ARE committed
  via PR #154. The per-persona name WAVs (#108) are the remaining gap.
- **Motion safety clamps + velocity caps** (PR #176) are in place. The
  manual calibration (#19) tunes the constants more accurately but the
  framework is live.
- **OpenTelemetry + SigNoz deploy** (PRs #2/#3/#4/#5/#13/#207) is done.
  Manual install on the actual Pi is the remaining step.
- **Repo organization** (NOW.md, CONTRIBUTING.md) is in place via PR #206.

---

*Compiled 2026-05-14 during session-end review. Each item above was an
issue closed with "needs manual/hardware/research work" — re-open the
specific item when starting that work.*
