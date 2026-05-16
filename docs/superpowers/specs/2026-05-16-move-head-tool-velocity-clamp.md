# move_head Tool Rate-Limit + Idempotency Guard

**Date:** 2026-05-16
**Branch:** `claude/fix-move-head-tool-velocity-clamp`
**Related issues:** #308 (continuous head_tracker velocity clamp — different path), #272 (emotion keyframe velocity), #264 (greet-scan snap)

## 1. Observed hardware behavior

During 2026-05-16 hardware validation, Gemini called `move_head direction=left`
twice within ~3 seconds. The operator's chassis slammed against the cowling on
the second call. Both `tool.execute` spans reported `outcome=success`, so the
tool dispatch path is healthy; the impact is in the resulting motion.

## 2. Audit findings

### 2a. `MovementManager` already runs two safety layers every tick

`src/robot_comic/moves.py:960-977` applies, on every tick of the 60 Hz control
loop and against every move source:

- `clamp_head_pose()` — clamps roll/pitch/yaw to the safe envelope
  (yaw ±45°, pitch −30°/+25°, roll ±20° by default).
- `cap_head_velocity()` — caps per-axis angular step to 1.5 rad/s (default).

These are global and unconditional. They engage for tool calls, dances,
breathing, and face-tracking. So the impact is **not** a missing global safety
layer — that already exists.

### 2b. Why the head still slams on rapid LLM tool calls

Walking the path for two `move_head direction=left` calls 1 second apart:

1. Call #1 captures `start_head_pose = current` (≈neutral yaw 0°), builds
   `GotoQueueMove(start=0°, target=+40°, duration=motion_duration_s)`, appends
   to `move_queue`. Worker dequeues it, head starts swinging toward +40°.
2. ~1 s later, call #2 fires. The head is now somewhere near +40° (or still
   in-flight). The tool captures `start_head_pose = current_head_pose` —
   crucially, this is **not** the pose the in-flight move will end at; it's
   the current sensor pose. The tool builds a SECOND
   `GotoQueueMove(start=<wherever the head is right now>, target=+40°,
   duration=...)` and appends.
3. Worker finishes move #1 — head is at +40°. Worker pops move #2 from queue
   and **records its own `move_start_time = now`**, then samples
   `evaluate(t=0)`. The captured `start` from step 2 is now stale (head has
   moved past it), so move #2's `t=0` output is a **step-back** to the older
   start pose, after which it ramps forward again to +40°.

The per-tick velocity cap absorbs much of the step-back into a 0.2–0.3 s
slew, but the head still reverses direction at near-max safe velocity into a
region close to the cowling. With a third "left" call piling in, the reversal
gets larger because the head has only crept halfway back.

The MovementManager cannot easily fix this from inside its loop: the
GotoQueueMove was authored with a now-stale start pose, and the loop doesn't
inspect `start` vs `current` once a move starts. The defect lives at the
**tool boundary**: the LLM emitted aggressive sequential calls and the tool
dutifully serialised them.

### 2c. #308 status

Issue #308 documents the same root cause on a different motion path
(continuous head_tracker as a secondary-move offset). Per the task scope and
`feedback_test_infra_agent_scope_creep` discipline, this PR does **not**
touch the tracker path. The shared helpers in `motion_safety.py` already
expose the global clamp / velocity-cap functions; the tool fix here reuses
none of them — it operates one level above, as an admission control on the
tool dispatch itself.

## 3. Chosen clamp shape

Two-layer defensive fix at the tool boundary, fully encapsulated in
`MoveHead`:

### 3a. Inter-call cooldown (rate limit)

Track the wall-clock time (`time.monotonic()`) of the last successful
`move_head` invocation. If the new call arrives within
`MOVE_HEAD_MIN_INTERVAL_S` (default 0.6 s) of the previous one, return a
soft error to the LLM **without** queueing a move. The error message names
the tool and the cooldown so the LLM learns to space its calls.

Rationale for 0.6 s: matches the default `motion_duration_s` (1.0 s) minus a
safety margin so a single typical move can complete (or nearly complete)
before the next is accepted. Configurable via
`REACHY_MINI_MOVE_HEAD_MIN_INTERVAL_S` for operators tuning their persona.

### 3b. Same-direction de-duplication

If the new call has the same `direction` as the last accepted call and the
robot's head is already within `MOVE_HEAD_AT_TARGET_TOL_RAD` (default
≈5°) of the previous target yaw/pitch, return a no-op success
(`{"status": "already looking <direction>"}`). Avoids stacking "left, left,
left" into a single sustained motion.

### Acceptance behaviour

The MoveHead tool keeps its current LLM signature (single `direction`
parameter, no new field). Errors returned to the LLM are short, factual, and
distinct so the model can self-correct ("move_head rate-limited; try again
in 0.4 s" / "already looking left; ignoring duplicate").

State is per-instance (the tool registry instantiates each tool once at
startup), guarded by a `threading.Lock` to handle concurrent
dispatch from the BackgroundToolManager.

## 4. Out of scope

- Issue #308 (head_tracker path) — fixed separately.
- Changing the LLM tool spec / signature.
- Tuning per-direction amplitude (e.g., reducing the 40° yaw default).
- Touching `MovementManager`, `motion_safety.py`, `GotoQueueMove`, or other
  tools.
- Re-mapping the absolute targets to deltas from current pose (would be a
  larger semantic change).

## 5. Test plan

All unit tests, no hardware. New tests added to
`tests/tools/test_move_head.py`:

1. **First call queues a move** (regression — existing behaviour preserved).
2. **Second call within cooldown returns rate-limit error and does NOT
   queue a move.**
3. **Second call after cooldown succeeds and queues normally.**
4. **Same-direction call within cooldown when robot is already at target
   returns the dedupe message and does NOT queue.**
5. **Different-direction call within cooldown still rate-limits** (cooldown
   protects against direction-switch slams too).
6. **Cooldown is configurable via env var** (set to 0 → calls always
   succeed back-to-back; restored to default for other tests via fixture).
7. **Existing tests** (`test_move_head_queues_eased_goto_move`,
   `test_move_head_still_allows_down`, schema test) continue to pass.

Time is injected via `monotonic_clock` callable on the MoveHead instance so
tests can advance time deterministically without `time.sleep`.

## 6. Files touched

- `src/robot_comic/tools/move_head.py` — add cooldown + dedupe state, env
  knobs, lock.
- `tests/tools/test_move_head.py` — extend coverage per §5.

Diff target ~150–250 LOC. No cross-cutting changes.
