# Robot Comic Fixes — Design Spec

**Date:** 2026-05-06
**Status:** Approved

## Overview

Six targeted fixes for the robot_comic project covering Gemini Live delivery-tag
leakage, servo speed control, session stability, TTS pacing, Moonshine STT
diagnostics, and a protobuf deprecation warning.

---

## Fix 1 — Gemini Live: strip TTS delivery tags, add natural prosody guidance

### Problem

`get_session_instructions()` returns identical text for both Gemini Live and
Gemini TTS backends. The Rickles profile's `## GEMINI TTS DELIVERY TAGS`
section contains inline tags like `[fast]`, `[amusement]`, `[short pause]` that
Gemini TTS interprets as vocal delivery cues. Gemini Live reads them aloud
literally.

### Solution — two parts

**Part A — Profile edit (`profiles/don_rickles/instructions.txt`):**

Add a `## GEMINI LIVE DELIVERY GUIDANCE` section alongside the existing TTS
section. Uses natural language that Gemini Live actually responds to: rapid
Brooklyn pacing, strategic pauses after punchlines, escalating contempt on
dismissals, etc. This section is ignored by the TTS path because TTS strips it
(Part B applies in reverse by convention — Live strips TTS tags, TTS ignores the
Live section since the TTS model focuses on the inline tags anyway).

**Part B — Code (`src/robot_comic/gemini_live.py`, `_build_live_config`):**

Post-process the instructions string before passing to Gemini Live:
1. Strip the entire `## GEMINI TTS DELIVERY TAGS` section — from the header
   line through to (but not including) the next `##` section header. Use a
   regex: `r'## GEMINI TTS DELIVERY TAGS\n.*?(?=\n##|\Z)'` with `re.DOTALL`.
2. Strip any residual `[tag]` patterns matching the delivery tag vocabulary
   using: `r'\[(?:fast|slow|short pause|long pause|amusement|annoyance|aggression|enthusiasm)\]'`.
   Apply after step 1 as a safety net for tags that appear outside the section.

No changes to `prompts.py` or the instructions loading pipeline.

### Files changed
- `profiles/don_rickles/instructions.txt` — add `## GEMINI LIVE DELIVERY GUIDANCE` section
- `src/robot_comic/gemini_live.py` — add `_strip_tts_tags(instructions: str) -> str` helper; call it in `_build_live_config()`

---

## Fix 2 — Servo speed: live slider + 0.6 default

### Problem

Emotion and dance servo movements are played at their recorded speed. At full
speed, servo noise drowns out speaker audio.

### Solution

**Config (`src/robot_comic/config.py`):**
- Add `MOVEMENT_SPEED_FACTOR: float = 0.6` to the config dataclass.
- Read from `MOVEMENT_SPEED_FACTOR` env var (float, clamped to 0.1–2.0).

**Move evaluation (`src/robot_comic/dance_emotion_moves.py`):**
- `EmotionQueueMove` and `DanceQueueMove` each gain a `speed_factor: float = 1.0` constructor argument.
- `evaluate(t)` internally calls `self._move.evaluate(t * self.speed_factor)`.
- The `duration` property returns `original_duration / self.speed_factor` so the
  queue correctly waits for the scaled duration before starting the next move.

**MovementManager (`src/robot_comic/moves.py`):**
- Add `self.speed_factor: float` initialized from `config.MOVEMENT_SPEED_FACTOR`.
- Add thread-safe `set_speed_factor(value: float)` that clamps to 0.1–2.0 and
  updates `self.speed_factor` (a float write is atomic under CPython's GIL).
- When dequeuing a move in the worker loop, pass `speed_factor=self.speed_factor`
  to `EmotionQueueMove` / `DanceQueueMove` constructors. Other move types
  (BreathingMove, etc.) are unaffected.

**Admin UI (`src/robot_comic/headless_personality_ui.py` or equivalent):**
- Add a `gr.Slider(minimum=0.1, maximum=2.0, step=0.05, value=0.6, label="Movement Speed")` to the settings panel.
- On change, call `movement_manager.set_speed_factor(value)`.
- Takes effect on the next queued move with no restart needed.

### Files changed
- `src/robot_comic/config.py`
- `src/robot_comic/dance_emotion_moves.py`
- `src/robot_comic/moves.py`
- `src/robot_comic/gradio_personality.py` (Gradio settings panel)

---

## Fix 3 — Gemini Live 1007: disable background video streaming

### Problem

`_run_live_session()` unconditionally starts a `_video_sender_loop()` task
whenever `camera_worker` is not None. This loop sends JPEG frames every second
via `session.send_realtime_input(video=...)`. With model `gemini-3.1-flash-live-preview`,
this triggers WebSocket 1007 "invalid argument" errors ~20 seconds in, crashing
the session and cutting audio output.

The on-demand camera tool (which sends a single frame inline after a tool call)
works correctly and is unaffected.

### Solution

**Config (`src/robot_comic/config.py`):**
- Add `GEMINI_LIVE_VIDEO_STREAMING: bool = False`.
- Read from `GEMINI_LIVE_VIDEO_STREAMING=true` env var.

**Code (`src/robot_comic/gemini_live.py`, `_run_live_session`):**

```python
if self.deps.camera_worker is not None and config.GEMINI_LIVE_VIDEO_STREAMING:
    video_task = asyncio.create_task(self._video_sender_loop(), name="gemini-video-sender")
```

### Files changed
- `src/robot_comic/config.py`
- `src/robot_comic/gemini_live.py`

---

## Fix 4 — Gemini TTS: "brooklyn fast" default delivery

### Problem

Gemini TTS audio playback is too slow by default. The Rickles persona requires
rapid-fire delivery as the baseline; slowdowns should only happen for intentional
comedic effect.

### Solution

Add a `system_instruction` to the `GenerateContentConfig` in
`GeminiTTSResponseHandler._call_tts_with_retry()`:

```python
tts_config = types.GenerateContentConfig(
    system_instruction=(
        "Deliver this text at a fast, clipped Brooklyn pace — "
        "rapid-fire on the insults, short crisp pauses only where marked. "
        "Never drawl or over-enunciate. Keep the energy sharp."
    ),
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name=self.get_current_voice()
            )
        )
    ),
)
```

The inline `[fast]` / `[short pause]` tags in the response text still apply on
top of this baseline. This system instruction is hardcoded in the handler rather
than in the profile so it applies universally to the TTS backend regardless of
profile.

### Files changed
- `src/robot_comic/gemini_tts.py`

---

## Fix 5 — Moonshine STT heartbeat (log-only, env-gated)

### Problem

When using Moonshine STT, there is no visibility into whether the transcriber is
actively receiving audio, stuck waiting, or thread-locked. Failures are silent.

### Solution

**Config (`src/robot_comic/config.py`):**
- Add `MOONSHINE_HEARTBEAT: bool = False`.
- Read from `MOONSHINE_HEARTBEAT=true` env var.

**State tracking (`src/robot_comic/local_stt_realtime.py`, `LocalSTTInputMixin`):**

The `_MoonshineListener` callbacks update a shared dict `_heartbeat` on the
handler:
```python
self._heartbeat = {
    "state": "idle",       # idle | speech_started | partial | completed
    "last_event": None,    # name of last callback fired
    "last_text": "",       # latest partial/completed text
    "last_event_at": 0.0,  # time.monotonic() of last callback
    "audio_frames": 0,     # count of add_audio calls (incremented in receive())
}
```

Callbacks (`on_line_started`, `on_line_updated`, `on_line_completed`) update
`_heartbeat` directly. Under CPython, simple dict key assignment is GIL-protected;
no explicit lock needed for this read/write pattern.

**Heartbeat task:** After `stream.start()` in `_build_local_stt_stream()`, if
`config.MOONSHINE_HEARTBEAT` is True, schedule a 1-second repeating coroutine on
`self._local_loop` via `loop.call_soon_threadsafe`. The coroutine logs:

```
[Moonshine] state=idle  last_event=started  age=3.2s  frames=1024
[Moonshine] state=partial  text="where are you"  age=0.1s  frames=2048
```

Additionally logs a `WARNING` if `state=idle` and `age > 10.0s` and
`audio_frames` has been incrementing — indicating audio is arriving but no
speech is being detected (potential thread-lock or model stall).

The heartbeat task checks `self._local_stt_stream is None` to self-terminate on
shutdown.

### Files changed
- `src/robot_comic/config.py`
- `src/robot_comic/local_stt_realtime.py`

---

## Fix 6 — Suppress SymbolDatabase protobuf deprecation warning

### Problem

`google.protobuf.symbol_database.SymbolDatabase.GetPrototype()` emits a
`UserWarning` on every startup. This comes from Google's own SDK internals.

### Solution

Add a warning filter in the app entry point (`src/robot_comic/main.py`), before
any Google SDK imports:

```python
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="google.protobuf.symbol_database",
)
```

### Files changed
- `src/robot_comic/main.py`

---

## Implementation order

1. Fix 3 (video streaming flag) — stops session crashes, unblocks testing of all other fixes
2. Fix 6 (warning suppression) — one line, zero risk
3. Fix 1 (Gemini Live tag stripping + profile update) — profile edit + small helper
4. Fix 4 (TTS speed) — one parameter addition
5. Fix 2 (servo speed + slider) — most files touched, save for last
6. Fix 5 (Moonshine heartbeat) — self-contained, can be done any time

---

## Out of scope

- Changing TTS speed per-profile (one global baseline is sufficient for now)
- Admin UI display for Moonshine heartbeat status (log-only per decision)
- Re-enabling background video streaming for other models (opt-in via env var)
