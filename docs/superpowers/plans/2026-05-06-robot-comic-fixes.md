# Robot Comic Six Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply six targeted fixes: disable Gemini Live background video streaming, suppress a protobuf warning, strip TTS delivery tags from Gemini Live instructions, add a natural-language prosody section to the Rickles profile, make Gemini TTS speak at a fast Brooklyn default, add servo speed control with a live Gradio slider, and add an env-gated Moonshine STT heartbeat log.

**Architecture:** Each fix is self-contained. Tasks 1–2 lay groundwork (shared config additions + warning suppression). Tasks 3–5 can be dispatched in parallel. Tasks 6–7 follow sequentially. All changes are tested before committing.

**Tech Stack:** Python 3.12, google-genai SDK, Gradio, FastRTC, Moonshine STT, pytest-asyncio

---

## File Map

| File | Changes |
|---|---|
| `src/robot_comic/config.py` | Add `GEMINI_LIVE_VIDEO_STREAMING`, `MOVEMENT_SPEED_FACTOR`, `MOONSHINE_HEARTBEAT` flags |
| `src/robot_comic/main.py` | Add protobuf warning filter before imports |
| `src/robot_comic/gemini_live.py` | Add `_strip_tts_delivery_tags()` helper; guard video task behind config flag |
| `profiles/don_rickles/instructions.txt` | Add `## GEMINI LIVE DELIVERY GUIDANCE` section |
| `src/robot_comic/gemini_tts.py` | Add `system_instruction` to `_call_tts_with_retry` |
| `src/robot_comic/dance_emotion_moves.py` | Add `speed_factor` param to `EmotionQueueMove` and `DanceQueueMove` |
| `src/robot_comic/moves.py` | Add `speed_factor` to `MovementManager`, thread-safe setter, pass to move constructors |
| `src/robot_comic/gradio_personality.py` | Add `gr.Slider` for movement speed, wire event to `movement_manager.set_speed_factor` |
| `src/robot_comic/local_stt_realtime.py` | Add `_heartbeat` dict, update callbacks, add heartbeat async loop |
| `tests/test_gemini_live.py` | Tests for tag stripping and video task guard |
| `tests/test_gemini_tts_handler.py` | Test for system_instruction in TTS config |
| `tests/test_config.py` (new) | Tests for new config flags |

---

## Task 1: Config additions + protobuf warning suppression

**Files:**
- Modify: `src/robot_comic/config.py` (Config class, ~line 497)
- Modify: `src/robot_comic/main.py` (top of file, before imports)
- Create: `tests/test_config_new_flags.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config_new_flags.py
import importlib
import os
import pytest


def _reload_config(monkeypatch, env: dict):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import robot_comic.config as cfg_mod
    importlib.reload(cfg_mod)
    return cfg_mod.config


def test_gemini_live_video_streaming_defaults_false(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.GEMINI_LIVE_VIDEO_STREAMING is False


def test_gemini_live_video_streaming_env_true(monkeypatch):
    cfg = _reload_config(monkeypatch, {"GEMINI_LIVE_VIDEO_STREAMING": "true"})
    assert cfg.GEMINI_LIVE_VIDEO_STREAMING is True


def test_movement_speed_factor_defaults_0_6(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.MOVEMENT_SPEED_FACTOR == pytest.approx(0.6)


def test_movement_speed_factor_clamped_high(monkeypatch):
    cfg = _reload_config(monkeypatch, {"MOVEMENT_SPEED_FACTOR": "5.0"})
    assert cfg.MOVEMENT_SPEED_FACTOR == pytest.approx(2.0)


def test_movement_speed_factor_clamped_low(monkeypatch):
    cfg = _reload_config(monkeypatch, {"MOVEMENT_SPEED_FACTOR": "0.0"})
    assert cfg.MOVEMENT_SPEED_FACTOR == pytest.approx(0.1)


def test_moonshine_heartbeat_defaults_false(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.MOONSHINE_HEARTBEAT is False


def test_moonshine_heartbeat_env_true(monkeypatch):
    cfg = _reload_config(monkeypatch, {"MOONSHINE_HEARTBEAT": "true"})
    assert cfg.MOONSHINE_HEARTBEAT is True
```

- [ ] **Step 2: Run to confirm they fail**

```
pytest tests/test_config_new_flags.py -v
```
Expected: AttributeError on `cfg.GEMINI_LIVE_VIDEO_STREAMING` etc.

- [ ] **Step 3: Add the three config attributes to `Config` class**

In `src/robot_comic/config.py`, add a helper for float with clamping near the `_env_flag` helper (around line 213):

```python
def _env_float_clamped(name: str, default: float, lo: float, hi: float) -> float:
    """Parse a float env var, clamping to [lo, hi]."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning("Invalid float for %s=%r, using default=%.2f", name, raw, default)
        return default
    clamped = max(lo, min(hi, value))
    if clamped != value:
        logger.warning("%s=%.2f clamped to %.2f", name, value, clamped)
    return clamped
```

Then in the `Config` class body (after `LOCAL_STT_UPDATE_INTERVAL`, around line 474), add:

```python
    GEMINI_LIVE_VIDEO_STREAMING = _env_flag("GEMINI_LIVE_VIDEO_STREAMING", default=False)
    MOVEMENT_SPEED_FACTOR = _env_float_clamped("MOVEMENT_SPEED_FACTOR", default=0.6, lo=0.1, hi=2.0)
    MOONSHINE_HEARTBEAT = _env_flag("MOONSHINE_HEARTBEAT", default=False)
```

- [ ] **Step 4: Add protobuf warning suppression to `main.py`**

Add these lines at the very top of `src/robot_comic/main.py`, before any other imports (after the module docstring if there is one):

```python
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="google.protobuf.symbol_database",
)
```

- [ ] **Step 5: Run tests to confirm they pass**

```
pytest tests/test_config_new_flags.py -v
```
Expected: all 7 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/robot_comic/config.py src/robot_comic/main.py tests/test_config_new_flags.py
git commit -m "feat: add GEMINI_LIVE_VIDEO_STREAMING, MOVEMENT_SPEED_FACTOR, MOONSHINE_HEARTBEAT config flags; suppress protobuf warning"
```

---

## Task 2: Disable Gemini Live background video streaming

**Files:**
- Modify: `src/robot_comic/gemini_live.py` (`_run_live_session`, around line 563)
- Modify: `tests/test_gemini_live.py`

- [ ] **Step 1: Write a failing test**

Add to `tests/test_gemini_live.py`:

```python
@pytest.mark.asyncio
async def test_video_task_not_started_when_streaming_disabled(monkeypatch):
    """Video sender task must not start when GEMINI_LIVE_VIDEO_STREAMING=False."""
    monkeypatch.setattr(gemini_mod, "get_session_instructions", lambda: "test")
    monkeypatch.setattr(gemini_mod, "get_session_voice", lambda: "Kore")
    monkeypatch.setattr(gemini_mod, "get_active_tool_specs", lambda _: [])

    camera = MagicMock()
    camera.get_latest_frame.return_value = None
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera,
    )

    handler = GeminiLiveHandler(deps)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    stop_event = asyncio.Event()
    session = _FakeSession(batches=[], stop_event=stop_event)
    fake_client = _FakeLiveClient(session)
    handler.client = fake_client

    # Flag off (default)
    import robot_comic.config as cfg_mod
    monkeypatch.setattr(cfg_mod.config, "GEMINI_LIVE_VIDEO_STREAMING", False)

    video_sender_calls = []
    original = handler._video_sender_loop

    async def spy_video_loop():
        video_sender_calls.append(True)
        await original()

    monkeypatch.setattr(handler, "_video_sender_loop", spy_video_loop)

    stop_event.set()
    await handler._run_live_session()

    assert video_sender_calls == [], "Video sender must not start when flag is False"
```

- [ ] **Step 2: Run to confirm it fails**

```
pytest tests/test_gemini_live.py::test_video_task_not_started_when_streaming_disabled -v
```
Expected: FAIL (video sender loop gets started regardless of flag).

- [ ] **Step 3: Apply the guard in `_run_live_session`**

In `src/robot_comic/gemini_live.py`, find the block (around line 562–564):

```python
                # Start video sender if camera is available
                if self.deps.camera_worker is not None:
                    video_task = asyncio.create_task(self._video_sender_loop(), name="gemini-video-sender")
```

Replace with:

```python
                # Start video sender only when explicitly enabled — continuous
                # streaming triggers 1007 errors on some Gemini Live models.
                if self.deps.camera_worker is not None and config.GEMINI_LIVE_VIDEO_STREAMING:
                    video_task = asyncio.create_task(self._video_sender_loop(), name="gemini-video-sender")
```

- [ ] **Step 4: Run test to confirm it passes**

```
pytest tests/test_gemini_live.py::test_video_task_not_started_when_streaming_disabled -v
```
Expected: PASS.

- [ ] **Step 5: Run full existing test suite to check for regressions**

```
pytest tests/test_gemini_live.py -v
```
Expected: all existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/robot_comic/gemini_live.py tests/test_gemini_live.py
git commit -m "fix: gate Gemini Live background video streaming behind GEMINI_LIVE_VIDEO_STREAMING flag (default off)"
```

---

## Task 3: Strip TTS delivery tags from Gemini Live instructions

**Files:**
- Modify: `src/robot_comic/gemini_live.py` (add helper, call in `_build_live_config`)
- Modify: `profiles/don_rickles/instructions.txt` (add GEMINI LIVE DELIVERY GUIDANCE section)
- Modify: `tests/test_gemini_live.py`

- [ ] **Step 1: Write failing tests for the helper**

Add to `tests/test_gemini_live.py`:

```python
from robot_comic.gemini_live import _strip_tts_delivery_tags


def test_strip_tts_delivery_tags_removes_section():
    instructions = """\
## IDENTITY
You are a robot.

## GEMINI TTS DELIVERY TAGS
Use [fast] for speed.
- [slow] — drag it out
- [amusement] — love your own jokes

## GUARDRAILS
Be safe.
"""
    result = _strip_tts_delivery_tags(instructions)
    assert "GEMINI TTS DELIVERY TAGS" not in result
    assert "[fast]" not in result
    assert "[amusement]" not in result
    assert "IDENTITY" in result
    assert "GUARDRAILS" in result


def test_strip_tts_delivery_tags_removes_stray_tags():
    instructions = "Say [fast] this [amusement] line [slow] clearly."
    result = _strip_tts_delivery_tags(instructions)
    assert "[fast]" not in result
    assert "[amusement]" not in result
    assert "[slow]" not in result
    assert "Say" in result
    assert "this" in result


def test_strip_tts_delivery_tags_leaves_unrelated_brackets():
    instructions = "See section [PHYSICAL BEATS] for moves."
    result = _strip_tts_delivery_tags(instructions)
    # PHYSICAL BEATS is not a delivery tag so it should survive
    assert "[PHYSICAL BEATS]" in result


def test_strip_tts_delivery_tags_no_section_is_noop():
    instructions = "## IDENTITY\nYou are a robot.\n"
    result = _strip_tts_delivery_tags(instructions)
    assert result == instructions
```

- [ ] **Step 2: Run to confirm they fail**

```
pytest tests/test_gemini_live.py::test_strip_tts_delivery_tags_removes_section tests/test_gemini_live.py::test_strip_tts_delivery_tags_removes_stray_tags tests/test_gemini_live.py::test_strip_tts_delivery_tags_leaves_unrelated_brackets tests/test_gemini_live.py::test_strip_tts_delivery_tags_no_section_is_noop -v
```
Expected: ImportError (`_strip_tts_delivery_tags` not defined).

- [ ] **Step 3: Add `_strip_tts_delivery_tags` to `gemini_live.py`**

Add this function near the top of the module (after the existing `_convert_schema_types` helper, around line 113):

```python
import re as _re

_TTS_DELIVERY_TAG_NAMES = (
    "fast", "slow", "short pause", "long pause",
    "amusement", "annoyance", "aggression", "enthusiasm",
)
_TTS_SECTION_RE = _re.compile(
    r"## GEMINI TTS DELIVERY TAGS\b.*?(?=\n##|\Z)",
    _re.DOTALL,
)
_TTS_TAG_RE = _re.compile(
    r"\[(?:" + "|".join(_re.escape(t) for t in _TTS_DELIVERY_TAG_NAMES) + r")\]"
)


def _strip_tts_delivery_tags(instructions: str) -> str:
    """Remove Gemini-TTS-specific delivery tags from system instructions.

    Strips the entire GEMINI TTS DELIVERY TAGS section and any residual
    inline [tag] patterns so Gemini Live does not read them aloud.
    """
    result = _TTS_SECTION_RE.sub("", instructions)
    result = _TTS_TAG_RE.sub("", result)
    # Collapse runs of blank lines left behind by section removal
    result = _re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
```

Then in `_build_live_config`, find the line:
```python
        instructions = get_session_instructions()
```
Replace with:
```python
        instructions = _strip_tts_delivery_tags(get_session_instructions())
```

- [ ] **Step 4: Add `## GEMINI LIVE DELIVERY GUIDANCE` to Rickles profile**

Append the following block to the end of `profiles/don_rickles/instructions.txt`
(after the existing `## GEMINI TTS DELIVERY TAGS` section — this section is for Gemini Live only; the TTS section above is stripped before Live receives it):

```
## GEMINI LIVE DELIVERY GUIDANCE

When speaking, you are performing live stand-up — not reading a script.

**Pace:** Default to a rapid, clipped Brooklyn delivery. Short sentences. Hit the consonants hard. Never drawl.

**Punchlines:** Land the punch word, then stop dead. Let the silence sit for a beat — then pivot.

**The dismissal:** After a particularly savage line, shift tone down slightly, almost matter-of-fact: "And that's all I have to say about that." Then snap back up.

**Escalation:** Start measured. Let each exchange ratchet the energy up one notch. By the fourth or fifth exchange you should be at full speed, full volume, rapid-fire.

**The warm closer:** When you wrap, drop into a genuine, slower register — "But I love ya. I really do." Let it breathe. It earns the previous destruction.

**Never:** Pause before starting a sentence (no "Um," "Well," or "So"). Never explain a joke after it lands.
```

- [ ] **Step 5: Run tests to confirm they pass**

```
pytest tests/test_gemini_live.py -v
```
Expected: all PASS including the four new tests.

- [ ] **Step 6: Commit**

```bash
git add src/robot_comic/gemini_live.py profiles/don_rickles/instructions.txt tests/test_gemini_live.py
git commit -m "feat: strip TTS delivery tags from Gemini Live instructions; add natural prosody guidance to Rickles profile"
```

---

## Task 4: Gemini TTS brooklyn-fast default delivery

**Files:**
- Modify: `src/robot_comic/gemini_tts.py` (`_call_tts_with_retry`, around line 283)
- Modify: `tests/test_gemini_tts_handler.py`

- [ ] **Step 1: Write a failing test**

Add to `tests/test_gemini_tts_handler.py`:

```python
@pytest.mark.asyncio
async def test_tts_call_includes_speed_system_instruction() -> None:
    """_call_tts_with_retry must pass a fast-delivery system_instruction to TTS."""
    handler = _make_handler()

    captured_configs = []

    async def fake_generate(model, contents, config):
        captured_configs.append(config)
        # Return a minimal valid response
        fake_data = b"\x00" * 4800
        import base64
        encoded = base64.b64encode(fake_data).decode()
        part = MagicMock()
        part.inline_data.data = encoded
        candidate = MagicMock()
        candidate.content.parts = [part]
        response = MagicMock()
        response.candidates = [candidate]
        return response

    handler._client.aio.models.generate_content = fake_generate

    result = await handler._call_tts_with_retry("You hockey puck!")

    assert result is not None
    assert len(captured_configs) == 1
    cfg = captured_configs[0]
    assert cfg.system_instruction is not None
    instruction_text = cfg.system_instruction.lower()
    assert "fast" in instruction_text or "brooklyn" in instruction_text or "pace" in instruction_text
```

- [ ] **Step 2: Run to confirm it fails**

```
pytest tests/test_gemini_tts_handler.py::test_tts_call_includes_speed_system_instruction -v
```
Expected: AssertionError (`system_instruction` is None).

- [ ] **Step 3: Add `system_instruction` to `_call_tts_with_retry`**

In `src/robot_comic/gemini_tts.py`, find `_call_tts_with_retry` (around line 279). Replace:

```python
        tts_config = types.GenerateContentConfig(
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

With:

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

- [ ] **Step 4: Run test to confirm it passes**

```
pytest tests/test_gemini_tts_handler.py -v
```
Expected: all PASS including new test.

- [ ] **Step 5: Commit**

```bash
git add src/robot_comic/gemini_tts.py tests/test_gemini_tts_handler.py
git commit -m "feat: add fast Brooklyn delivery system_instruction to Gemini TTS"
```

---

## Task 5: Servo speed factor in move classes and MovementManager

**Files:**
- Modify: `src/robot_comic/dance_emotion_moves.py`
- Modify: `src/robot_comic/moves.py`
- Create: `tests/test_dance_emotion_moves.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dance_emotion_moves.py
from unittest.mock import MagicMock
import numpy as np
import pytest


def _make_recorded_moves(duration: float = 2.0):
    """Build a mock RecordedMoves that returns a fake emotion with the given duration."""
    fake_move = MagicMock()
    fake_move.duration = duration
    eye4 = np.eye(4, dtype=np.float64)
    fake_move.evaluate.side_effect = lambda t: (eye4, (0.0, 0.0), 0.0)

    recorded = MagicMock()
    recorded.get.return_value = fake_move
    return recorded, fake_move


def test_emotion_queue_move_duration_scaled():
    from robot_comic.dance_emotion_moves import EmotionQueueMove

    recorded, _ = _make_recorded_moves(duration=2.0)
    move = EmotionQueueMove("laughing1", recorded, speed_factor=0.5)
    assert move.duration == pytest.approx(4.0)  # 2.0 / 0.5


def test_emotion_queue_move_evaluate_scales_t():
    from robot_comic.dance_emotion_moves import EmotionQueueMove

    recorded, fake_move = _make_recorded_moves(duration=2.0)
    move = EmotionQueueMove("laughing1", recorded, speed_factor=2.0)
    move.evaluate(1.0)
    fake_move.evaluate.assert_called_once_with(2.0)  # t * speed_factor = 1.0 * 2.0


def test_emotion_queue_move_default_speed_factor_1():
    from robot_comic.dance_emotion_moves import EmotionQueueMove

    recorded, fake_move = _make_recorded_moves(duration=2.0)
    move = EmotionQueueMove("laughing1", recorded)
    assert move.duration == pytest.approx(2.0)
    move.evaluate(0.5)
    fake_move.evaluate.assert_called_once_with(0.5)


def test_dance_queue_move_duration_scaled():
    from robot_comic.dance_emotion_moves import DanceQueueMove

    fake_dance = MagicMock()
    fake_dance.duration = 3.0
    eye4 = np.eye(4, dtype=np.float64)
    fake_dance.evaluate.side_effect = lambda t: (eye4, (0.0, 0.0), 0.0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "robot_comic.dance_emotion_moves.DanceMove",
            lambda name: fake_dance,
        )
        move = DanceQueueMove("robot_groove", speed_factor=0.6)

    assert move.duration == pytest.approx(3.0 / 0.6, rel=1e-3)


def test_movement_manager_set_speed_factor_clamps():
    from robot_comic.moves import MovementManager

    mm = MovementManager.__new__(MovementManager)
    mm.speed_factor = 1.0

    mm.set_speed_factor(5.0)
    assert mm.speed_factor == pytest.approx(2.0)

    mm.set_speed_factor(0.0)
    assert mm.speed_factor == pytest.approx(0.1)

    mm.set_speed_factor(0.6)
    assert mm.speed_factor == pytest.approx(0.6)
```

- [ ] **Step 2: Run to confirm they fail**

```
pytest tests/test_dance_emotion_moves.py -v
```
Expected: errors about missing `speed_factor` argument and missing `set_speed_factor`.

- [ ] **Step 3: Update `EmotionQueueMove` in `dance_emotion_moves.py`**

Replace the `EmotionQueueMove` class (lines 56–87) with:

```python
class EmotionQueueMove(Move):  # type: ignore
    """Wrapper for emotion moves to work with the movement queue system."""

    def __init__(self, emotion_name: str, recorded_moves: RecordedMoves, speed_factor: float = 1.0):
        """Initialize an EmotionQueueMove."""
        self.emotion_move = recorded_moves.get(emotion_name)
        self.emotion_name = emotion_name
        self.speed_factor = max(0.1, speed_factor)

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float(self.emotion_move.duration) / self.speed_factor

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate emotion move at time t."""
        try:
            head_pose, antennas, body_yaw = self.emotion_move.evaluate(t * self.speed_factor)

            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating emotion '{self.emotion_name}' at t={t}: {e}")
            from reachy_mini.utils import create_head_pose

            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, np.array([0.0, 0.0], dtype=np.float64), 0.0)
```

- [ ] **Step 4: Update `DanceQueueMove` in `dance_emotion_moves.py`**

Replace the `DanceQueueMove` class `__init__` and `duration` property (lines 22–33):

```python
class DanceQueueMove(Move):  # type: ignore
    """Wrapper for dance moves to work with the movement queue system."""

    def __init__(self, move_name: str, speed_factor: float = 1.0):
        """Initialize a DanceQueueMove."""
        self.dance_move = DanceMove(move_name)
        self.move_name = move_name
        self.speed_factor = max(0.1, speed_factor)

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float(self.dance_move.duration) / self.speed_factor

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate dance move at time t."""
        try:
            head_pose, antennas, body_yaw = self.dance_move.evaluate(t * self.speed_factor)

            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating dance move '{self.move_name}' at t={t}: {e}")
            from reachy_mini.utils import create_head_pose

            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, np.array([0.0, 0.0], dtype=np.float64), 0.0)
```

- [ ] **Step 5: Add `speed_factor` to `MovementManager` in `moves.py`**

In `MovementManager.__init__` (after the `_command_queue` and lock setup, around line 290), add:

```python
        self.speed_factor: float = getattr(config, "MOVEMENT_SPEED_FACTOR", 0.6)
```

And add this method to `MovementManager` (after `set_moving_state`, around line 352):

```python
    def set_speed_factor(self, value: float) -> None:
        """Set the global move playback speed multiplier (0.1–2.0).

        Thread-safe: float assignment is atomic under CPython's GIL.
        Effect applies to the next queued move.
        """
        self.speed_factor = max(0.1, min(2.0, value))
```

- [ ] **Step 6: Pass `speed_factor` when constructing `EmotionQueueMove` in `play_emotion.py`**

In `src/robot_comic/tools/play_emotion.py`, find:

```python
            emotion_move = EmotionQueueMove(emotion_name, RECORDED_MOVES)
```

Replace with:

```python
            speed = getattr(deps.movement_manager, "speed_factor", 1.0)
            emotion_move = EmotionQueueMove(emotion_name, RECORDED_MOVES, speed_factor=speed)
```

- [ ] **Step 7: Pass `speed_factor` when constructing `DanceQueueMove` in `tools/dance.py`**

Open `src/robot_comic/tools/dance.py` and find the line that constructs `DanceQueueMove`. Wrap it the same way:

```python
            speed = getattr(deps.movement_manager, "speed_factor", 1.0)
            dance_move = DanceQueueMove(move_name, speed_factor=speed)
```

(The exact variable name for the move_name may differ — use whatever name the existing code uses.)

- [ ] **Step 8: Run tests to confirm they pass**

```
pytest tests/test_dance_emotion_moves.py -v
```
Expected: all PASS.

- [ ] **Step 9: Run the full test suite to catch regressions**

```
pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add src/robot_comic/dance_emotion_moves.py src/robot_comic/moves.py src/robot_comic/tools/play_emotion.py src/robot_comic/tools/dance.py tests/test_dance_emotion_moves.py
git commit -m "feat: add speed_factor to EmotionQueueMove and DanceQueueMove; MovementManager.set_speed_factor(); default 0.6"
```

---

## Task 6: Gradio movement speed slider

**Files:**
- Modify: `src/robot_comic/gradio_personality.py`

No new tests — Gradio component wiring is verified manually. The underlying `set_speed_factor` is already tested in Task 5.

- [ ] **Step 1: Add the slider component to `PersonalityUI.create_components`**

In `src/robot_comic/gradio_personality.py`, inside `create_components` (after the `save_btn` line, around line 177), add:

```python
        self.speed_slider = gr.Slider(
            minimum=0.1,
            maximum=2.0,
            step=0.05,
            value=0.6,
            label="Movement Speed (0.1 = slow/quiet, 1.0 = normal, 2.0 = fast)",
            interactive=True,
        )
```

- [ ] **Step 2: Expose it in `additional_inputs_ordered`**

In `additional_inputs_ordered` (around line 179), add `self.speed_slider` to the returned list:

```python
    def additional_inputs_ordered(self) -> list[Any]:
        return [
            self.personalities_dropdown,
            self.apply_btn,
            self.new_personality_btn,
            self.status_md,
            self.preview_md,
            self.person_name_tb,
            self.person_instr_ta,
            self.tools_txt_ta,
            self.voice_dropdown,
            self.available_tools_cg,
            self.save_btn,
            self.speed_slider,
        ]
```

- [ ] **Step 3: Wire the slider event in `wire_events`**

In `wire_events` (around line 196), add after the other event bindings (find where `self.save_btn.click` or similar is wired and add below it):

```python
        def _on_speed_change(value: float) -> None:
            mm = getattr(handler, "deps", None)
            if mm is not None:
                movement_manager = getattr(mm, "movement_manager", None)
                if movement_manager is not None and hasattr(movement_manager, "set_speed_factor"):
                    movement_manager.set_speed_factor(value)

        self.speed_slider.change(
            fn=_on_speed_change,
            inputs=[self.speed_slider],
            outputs=[],
        )
```

- [ ] **Step 4: Verify the app starts without error**

```
python -m robot_comic.main --help
```
Expected: help text displays without ImportError or AttributeError.

- [ ] **Step 5: Commit**

```bash
git add src/robot_comic/gradio_personality.py
git commit -m "feat: add movement speed slider to Gradio admin UI"
```

---

## Task 7: Moonshine STT heartbeat log

**Files:**
- Modify: `src/robot_comic/local_stt_realtime.py`
- Modify: `tests/test_local_stt_realtime.py`

- [ ] **Step 1: Write a failing test**

Add to `tests/test_local_stt_realtime.py`:

```python
import asyncio
import time
from unittest.mock import MagicMock, patch
import pytest


def _make_heartbeat_mixin():
    """Build a LocalSTTInputMixin instance with heartbeat attributes initialized."""
    from robot_comic.local_stt_realtime import LocalSTTInputMixin

    class _Concrete(LocalSTTInputMixin):
        # Minimal concrete implementation so the mixin can be instantiated
        BACKEND_PROVIDER = "test"
        SAMPLE_RATE = 24000

        def __init__(self):
            self._heartbeat = {
                "state": "idle",
                "last_event": None,
                "last_text": "",
                "last_event_at": time.monotonic(),
                "audio_frames": 0,
            }
            self._local_loop = None
            self._local_stt_stream = MagicMock()

    return _Concrete()


def test_heartbeat_dict_initialized():
    """LocalSTTInputMixin must expose _heartbeat dict with expected keys."""
    from robot_comic.local_stt_realtime import LocalSTTInputMixin

    # Access via the mixin's __init__ by constructing a concrete subclass
    obj = _make_heartbeat_mixin()
    assert "state" in obj._heartbeat
    assert "last_event_at" in obj._heartbeat
    assert "audio_frames" in obj._heartbeat
    assert obj._heartbeat["state"] == "idle"


@pytest.mark.asyncio
async def test_heartbeat_loop_logs_state(caplog):
    """When MOONSHINE_HEARTBEAT is True, the loop emits INFO logs each tick."""
    import logging
    from robot_comic.local_stt_realtime import LocalSTTInputMixin

    obj = _make_heartbeat_mixin()
    obj._local_loop = asyncio.get_running_loop()

    # Run one tick of the heartbeat coroutine
    with patch("robot_comic.config.config") as mock_cfg:
        mock_cfg.MOONSHINE_HEARTBEAT = True
        with caplog.at_level(logging.INFO, logger="robot_comic.local_stt_realtime"):
            # Simulate one iteration by calling the internal helper
            obj._heartbeat["state"] = "idle"
            obj._heartbeat["last_event_at"] = time.monotonic()
            obj._heartbeat["audio_frames"] = 42
            obj._log_heartbeat()

    assert any("Moonshine" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run to confirm they fail**

```
pytest tests/test_local_stt_realtime.py::test_heartbeat_dict_initialized tests/test_local_stt_realtime.py::test_heartbeat_loop_logs_state -v
```
Expected: AttributeError (`_heartbeat` not present) or similar.

- [ ] **Step 3: Add `_heartbeat` dict to `LocalSTTInputMixin.__init__`**

In `src/robot_comic/local_stt_realtime.py`, inside `LocalSTTInputMixin.__init__` (after the existing attribute assignments, around line 100), add:

```python
        import time as _time
        self._heartbeat: dict = {
            "state": "idle",        # idle | speech_started | partial | completed
            "last_event": None,
            "last_text": "",
            "last_event_at": _time.monotonic(),
            "audio_frames": 0,
        }
        self._heartbeat_future: "asyncio.Future | None" = None
```

- [ ] **Step 4: Update `_MoonshineListener` callbacks to write to `_heartbeat`**

In `_MoonshineListener.on_line_started`:

```python
    def on_line_started(self, event: Any) -> None:
        text = self._text_from_event(event)
        import time as _t
        self.handler._heartbeat.update({"state": "speech_started", "last_event": "started", "last_text": text, "last_event_at": _t.monotonic()})
        self.handler._schedule_local_stt_event("started", text)
```

In `on_line_updated` / `on_line_text_changed`:

```python
    def on_line_updated(self, event: Any) -> None:
        text = self._text_from_event(event)
        import time as _t
        self.handler._heartbeat.update({"state": "partial", "last_event": "partial", "last_text": text, "last_event_at": _t.monotonic()})
        self.handler._schedule_local_stt_event("partial", text)

    def on_line_text_changed(self, event: Any) -> None:
        text = self._text_from_event(event)
        import time as _t
        self.handler._heartbeat.update({"state": "partial", "last_event": "partial", "last_text": text, "last_event_at": _t.monotonic()})
        self.handler._schedule_local_stt_event("partial", text)
```

In `on_line_completed`:

```python
    def on_line_completed(self, event: Any) -> None:
        text = self._text_from_event(event)
        import time as _t
        self.handler._heartbeat.update({"state": "completed", "last_event": "completed", "last_text": text, "last_event_at": _t.monotonic()})
        self.handler._schedule_local_stt_event("completed", text)
```

- [ ] **Step 5: Increment `audio_frames` in `receive()`**

In `LocalSTTInputMixin.receive`, after the `add_audio` call succeeds, increment the counter:

```python
        try:
            self._local_stt_stream.add_audio(audio_float.tolist(), self.local_stt_sample_rate)
            self._heartbeat["audio_frames"] += 1
        except Exception as e:
            logger.debug("Dropping local STT audio frame: %s", e)
```

- [ ] **Step 6: Add `_log_heartbeat` method and async heartbeat loop**

Add these methods to `LocalSTTInputMixin`:

```python
    def _log_heartbeat(self) -> None:
        """Emit one heartbeat log line with current Moonshine state."""
        import time as _t
        h = self._heartbeat
        age = _t.monotonic() - h["last_event_at"]
        text_snippet = (h["last_text"] or "")[:40]
        logger.info(
            "[Moonshine] state=%s  last_event=%s  age=%.1fs  frames=%d  text=%r",
            h["state"],
            h["last_event"],
            age,
            h["audio_frames"],
            text_snippet,
        )
        if h["state"] == "idle" and age > 10.0 and h["audio_frames"] > 0:
            logger.warning(
                "[Moonshine] idle for %.1fs with %d audio frames received — possible thread-lock or model stall",
                age,
                h["audio_frames"],
            )

    async def _moonshine_heartbeat_loop(self) -> None:
        """Log Moonshine state every second while the stream is active."""
        import asyncio as _asyncio
        from robot_comic.config import config as _cfg
        while self._local_stt_stream is not None and _cfg.MOONSHINE_HEARTBEAT:
            self._log_heartbeat()
            await _asyncio.sleep(1.0)
```

- [ ] **Step 7: Start heartbeat after `stream.start()` in `_build_local_stt_stream`**

At the end of `_build_local_stt_stream`, after `stream.start()` and the attribute assignments, add:

```python
        from robot_comic.config import config as _cfg
        if _cfg.MOONSHINE_HEARTBEAT and self._local_loop is not None:
            import asyncio as _asyncio
            self._heartbeat_future = _asyncio.run_coroutine_threadsafe(
                self._moonshine_heartbeat_loop(), self._local_loop
            )
```

- [ ] **Step 8: Cancel the heartbeat in `shutdown`**

In `LocalSTTInputMixin.shutdown`, before the `_close_local` call, add:

```python
        if self._heartbeat_future is not None:
            self._heartbeat_future.cancel()
            self._heartbeat_future = None
```

- [ ] **Step 9: Run tests to confirm they pass**

```
pytest tests/test_local_stt_realtime.py -v
```
Expected: all PASS including the two new tests.

- [ ] **Step 10: Run full test suite**

```
pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 11: Commit**

```bash
git add src/robot_comic/local_stt_realtime.py tests/test_local_stt_realtime.py
git commit -m "feat: add env-gated Moonshine STT heartbeat log (MOONSHINE_HEARTBEAT=true)"
```

---

## Self-Review

**Spec coverage check:**
- [x] Fix 1 (Gemini Live tag stripping + prosody section) → Tasks 2 + 3
- [x] Fix 2 (servo speed + live slider) → Tasks 5 + 6
- [x] Fix 3 (video streaming disabled by default) → Task 2
- [x] Fix 4 (TTS brooklyn fast) → Task 4
- [x] Fix 5 (Moonshine heartbeat) → Task 7
- [x] Fix 6 (protobuf warning) → Task 1

**Parallelisation notes for subagent dispatch:**
- Task 1 must run first (adds config flags all other tasks depend on).
- Tasks 2, 3, 4, 5 can run in parallel after Task 1 (non-overlapping files except gemini_live.py for Tasks 2+3 — dispatch those sequentially or as one agent).
- Task 6 must follow Task 5 (needs `set_speed_factor` to exist on `MovementManager`).
- Task 7 can run in parallel with Tasks 3, 4, 5.
