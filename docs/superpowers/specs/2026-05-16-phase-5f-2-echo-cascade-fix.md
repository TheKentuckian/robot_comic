# Phase 5f.2 — fix self-echo barge-in cascade with faster-whisper STT

**Date:** 2026-05-16
**Status:** Spec — implementation on `claude/phase-5f-2-echo-cascade-fix`.
**Tracks:** epic #391; follow-up to Phase 5f (#416) + 5f.1 (#420).
**Predecessor:** Phase 5f.1 (`#420`) — swapped silero-vad for webrtcvad so the
faster-whisper STT adapter fits the chassis eMMC. With that landed, we ran
the first end-to-end hardware validation under
`REACHY_MINI_AUDIO_INPUT_BACKEND=faster_whisper` + ElevenLabs TTS today and
hit a new failure mode that Moonshine had been masking.

---

## §1 — Hardware-observed cascade (2026-05-16)

Running with the faster-whisper STT adapter + ElevenLabs TTS on the chassis,
every assistant turn was being interrupted mid-utterance and replaced with a
new (echo-driven) turn. The journal shows the pattern repeating without
external user input:

```
role=assistant content="Yeah, see, this is what passes for entertainment..."
turn.outcome=interrupted   # speech_started fired while assistant was still speaking
role=user      content="Thank you"
turn.outcome=interrupted
role=user      content="You"
turn.outcome=interrupted
role=user      content="I think he's very young."
turn.outcome=interrupted
(play_emotion called repeatedly — motor jitter observed)
```

### Root cause

1. Don Rickles speaks via ElevenLabs through the chassis speaker.
2. The chassis mic picks up the speaker output (room-acoustic echo).
3. `faster-whisper` transcribes that echo as short fragments — frequently
   single-word ("You") or short stock phrases ("Thank you", "I think he's
   very young.") that whisper-family models are known to hallucinate on
   non-speech audio.
4. `ComposablePipeline._on_speech_started` fires
   (webrtcvad detects energy in the echoed audio), which:
   - Closes the current `turn` span with `turn.outcome=interrupted`.
   - Calls `self._clear_queue()` — barge-in flushes the in-flight TTS.
5. `ComposablePipeline._on_transcript_completed` dispatches the (echo) fragment
   as a new user turn.
6. Gemini sees the "user" said `"Thank you"` and often responds with a
   `play_emotion` tool call — motors jitter.
7. The fresh assistant audio plays through the speaker, the mic picks it
   up, and the cascade continues.

Moonshine doesn't hit this *exact* failure because its streaming listener is
more conservative about emitting `completed` events for short echo, and its
host-coupled mode already has its own echo guard. faster-whisper's
webrtcvad-based segment dispatch is more aggressive.

### Why the existing echo-guard misses

The factory already passes
`lambda: time.perf_counter() < getattr(host, "_speaking_until", 0.0)` to
`FasterWhisperSTTAdapter` (Phase 5e.2 wiring), and
`elevenlabs_tts.py::_enqueue_audio_frame` derives `_speaking_until` from the
cumulative byte count + `config.ECHO_COOLDOWN_MS` (default 300 ms).

Two reasons this isn't sufficient on hardware:

1. **Cooldown is too tight for room-acoustic reverb.** 300 ms covers
   device-buffer + scheduling jitter but not the actual physical decay of
   speaker output in the chassis enclosure. Reverb tail bleeds past the
   deadline; webrtcvad flags it; faster-whisper hallucinates a fragment.
2. **There is no second line of defence.** Once the echo frame slips past
   the input-side guard, every downstream check (duplicate suppression,
   welcome gate) treats the echo as a real user turn.

## §2 — Decision

Defence in depth. Implement **both** fixes:

### Fix A — Orchestrator-side minimum-utterance filter (PRIMARY)

In `ComposablePipeline._on_transcript_completed`, drop transcripts that look
like echo/hallucination before they reach the duplicate-suppression window
or the LLM dispatch:

- Drop if `< MIN_TRANSCRIPT_WORDS` words.
- Drop if `< MIN_TRANSCRIPT_CHARS` characters (after strip).

Thresholds (module-level constants for trivial on-device tuning):

- `MIN_TRANSCRIPT_WORDS = 2`
- `MIN_TRANSCRIPT_CHARS = 8`

**Rationale.** The chosen pair catches the observed echo fragments
(`"You"` = 1 word / 3 chars; `"Thank you"` = 2 words / 9 chars — fails the
char check; `"I think he's very young."` = 5 words / 24 chars — passes,
which is acceptable: that one is rare enough and grammatical enough to be
indistinguishable from real input on text alone). Legitimate short user
input ("yes please" = 2 words / 10 chars; "go on" = 2 words / 5 chars
— fails char check, but the persona doesn't actually need 5-char user
input to be reactive: the user can elaborate). The pair is intentionally
heuristic; the constants are at module top so on-device tuning is a
one-line change.

Drop happens **AFTER** the existing `transcript.strip()` + non-empty check
but **BEFORE** duplicate-suppression. Logs at `DEBUG` like the existing
empty-string drop; no telemetry event (these are pre-dispatch drops,
identical category to the empty-string case at lines 397-399).

This helps **both** backends — Moonshine has occasionally hallucinated
short partials too (see #19), and the orchestrator-side filter is
backend-agnostic.

### Fix B — `_speaking_until` cooldown audit + raise the default

Audit findings:

- `_speaking_until` is set inside
  `elevenlabs_tts.py::_enqueue_audio_frame` (line 486) as
  `playback_end + cooldown_s`, where `cooldown_s = config.ECHO_COOLDOWN_MS / 1000`.
- `playback_end` is derived from cumulative response bytes / sample rate
  — accurate for the moment audio leaves our queue.
- `config.ECHO_COOLDOWN_MS` is an existing env-driven knob
  (`REACHY_MINI_ECHO_COOLDOWN_MS`, `DEFAULT_ECHO_COOLDOWN_MS = 300`) — no
  source surgery needed, just a default bump.
- The same cooldown is consumed by `llama_base.py::_enqueue_audio_frame`
  (line 239); Moonshine standalone reads the *same* `_speaking_until`
  from the host via the factory-injected `should_drop_frame` closure.
- The legacy `LocalSTTInputMixin` echo-guard path also reads
  `host._speaking_until` — so the cooldown is unified across both backends
  and both pipeline modes. Lifting `DEFAULT_ECHO_COOLDOWN_MS` raises the
  bar for every consumer.

Bump `DEFAULT_ECHO_COOLDOWN_MS` from **300** → **800**.

Rationale: 300 ms was chosen for "device-buffer and scheduling jitter"
(per the existing comment at `config.py:432-433`). Hardware data today
proves the real budget includes room-acoustic reverb tail in the chassis
enclosure, which is on the order of several hundred milliseconds. 800 ms
is conservative enough to swallow a typical enclosure reverb tail while
still allowing legitimate user barge-in within a second of the assistant
finishing — well inside the round-trip "what did you say?" expectation.

Trade-off: longer cooldown means a legitimate user interruption launched
within ~800 ms of the assistant's last frame is dropped. Acceptable —
the alternative is an open echo loop that requires power-cycling the
robot to escape. Operators who need tighter barge-in can set
`REACHY_MINI_ECHO_COOLDOWN_MS=300` to restore the old behaviour.

Moonshine is already shipping at the 300 ms cooldown today and behaves
fine; the longer cooldown only ever drops *more* echo frames, never adds
false positives downstream of the guard. Moonshine's reset cycle was
described in prior memory as "quicker" but that observation was about
the *listener's* dispatch behaviour, not the cooldown consumption — both
backends consume the same scalar deadline.

## §3 — What 5f.2 does NOT do

- No STT-backend swap or adapter-shape changes (faster_whisper adapter
  itself is not touched — 5f.1 just landed; no churn).
- No acoustic echo cancellation (AEC) — that's a hardware/DSP concern,
  not a Python fix. We may revisit if Fix A + Fix B still leak.
- No webrtcvad parameter tuning (aggressiveness, frame thresholds) — 5f.1
  just shipped; defer.
- No persona / Gemini / Don Rickles behaviour changes.

## §4 — Files touched

- `src/robot_comic/composable_pipeline.py` — add two module-level constants
  (`MIN_TRANSCRIPT_WORDS`, `MIN_TRANSCRIPT_CHARS`) + filter in
  `_on_transcript_completed`.
- `src/robot_comic/config.py` — bump `DEFAULT_ECHO_COOLDOWN_MS` 300 → 800;
  comment updated to mention room-acoustic reverb tail.
- `tests/test_composable_pipeline.py` — new tests covering both filter
  arms (too-short-word-count drop, too-short-char-count drop, accepted
  longer transcript, ordering relative to duplicate-suppression).

Expected diff: ~80-120 LOC (constants + filter + tests + 1-line config
bump + comment).

## §5 — Test plan

New tests in `tests/test_composable_pipeline.py`:

1. `test_single_word_transcript_dropped_as_likely_echo` — `"You"`
   doesn't reach the LLM; conversation history stays empty.
2. `test_short_char_transcript_dropped_as_likely_echo` — `"hi"` (2 chars,
   1 word) dropped; also exercises the AND-of-conditions edge.
3. `test_short_two_word_transcript_below_char_floor_dropped` — `"a b"`
   (2 words / 3 chars) dropped via the char floor.
4. `test_normal_transcript_passes_filter` — `"hi robot can you dance"`
   reaches the LLM as before (regression guard for the happy path).
5. `test_short_drop_logged_at_debug` — log capture confirms the
   `"dropping short transcript as likely echo"` debug line fires; nothing
   on the output queue.
6. `test_short_drop_runs_before_duplicate_suppression` — sending the
   same short fragment twice in a row exits via the short-drop path
   both times (proves ordering — the duplicate-suppression cache is
   never populated with a dropped value).

No new tests for the cooldown bump — it's a one-line default change to an
existing well-tested config knob. The existing `tests/test_config.py` and
echo-guard tests continue to assert the env-var override path works.

Whole-repo invocation per `docs/superpowers/memory/feedback_ruff_check_whole_repo_locally.md`:

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes src/robot_comic/composable_pipeline.py src/robot_comic/elevenlabs_tts.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

## §6 — Rollout

- PR title: `fix(phase-5f-2): drop short transcripts + raise echo cooldown to break self-echo cascade`
- Reference hardware finding (2026-05-16 validation under faster_whisper
  backend) in the PR body.
- No env-var changes required; operators on the old 300 ms cooldown can
  opt back in via `REACHY_MINI_ECHO_COOLDOWN_MS=300`.
- Next iteration if this isn't enough: webrtcvad aggressiveness bump,
  AEC investigation, or a smarter hallucination-detector
  (whisper confidence threshold).
