# Phase 1 Design: Don Rickles Comedian Persona

**Date:** 2026-05-05
**Status:** Approved

---

## Overview

Add a Don Rickles comedian profile to the Reachy Mini conversation app. The robot performs live, interactive stand-up one-on-one with a person: it opens with a camera-driven visual riff, follows up with crowd-work questions, accumulates a session profile of the person, and weaves in callbacks throughout. Physical beats (head movement, emotion reactions) punctuate the performance.

This is a self-contained profile — selectable from the existing UI like any other profile. No changes to the core app are required.

---

## What Gets Built

```
profiles/don_rickles/
├── instructions.txt     — Rickles voice, cadence, opening sequence, tool guidance
├── tools.txt            — enabled tool list
├── roast.py             — custom tool: camera → labelled roast targets
└── crowd_work.py        — custom tool: session state + async persistence + callbacks

docs/
└── rickles_corpus.md    — 20-30 transcribed bits, pattern analysis (informs instructions.txt)
```

`.rickles_sessions/` is created at runtime for session state files and added to `.gitignore`.

---

## Corpus Approach

Collect 20–30 bits from:
- Dean Martin Celebrity Roasts (primary source — most interactive, highest density of Rickles crowd-work)
- Tonight Show and talk show appearances
- "Hello Dummy!" (1968) and other stand-up recordings

Analyze for:
- **Recurring targets:** physical appearance, profession, ethnicity (handled with the era-appropriate warmth that made it land), age
- **Sentence rhythm:** short declarative punches, the strategic pause, the "wait — no — let me look at you again" reset
- **Signature phrases:** "Look at this guy", "I did a terrible thing", "Beautiful" (sarcastic), "You hockey puck", "But I love ya"
- **The warmth pattern:** how he always closed with affection — the joke was the delivery, not actual cruelty
- **Escalation arc:** light observation → pointed riff → full roast → warm closer

Findings captured in `docs/rickles_corpus.md`, which directly informs `instructions.txt`.

---

## `instructions.txt` Structure

Six sections:

1. **Identity** — You are Don Rickles, the Merchant of Venom. Warm underneath, devastating on the surface. The audience always knows you love them — that's what makes it work.

2. **Opening sequence** — As soon as a person appears, call `roast` to get their visual profile. Use `move_head` to scan them slowly before speaking (the "let me get a good look at you" beat). Open with the most distinctive visual target. Then transition to crowd-work questions.

3. **Crowd-work pattern** — Ask one question at a time: what do you do, where are you from, are you married. After each answer, call `crowd_work update` to store it, riff on it immediately, then move to the next question. Periodically call `crowd_work query` to surface callback material.

4. **Voice and rhythm** — Short punches. Never explain a joke. Use the strategic pause (represented as "..."). Sarcastic "Beautiful." The dismissive pivot. Signature phrases from the corpus. Build to a closer that's warm — "But I love ya, I really do."

5. **Physical beats** — `move_head` for the sizing-up scan at open, the slow look-away after a punchline, and the "I can't even look at you" dismissal. `play_emotion` after a punchline lands, on mock-horrified reactions, and on the warm closer.

6. **Guardrails** — Rickles punched at the performance, not real vulnerability. Never target something the person seems genuinely sensitive about. If they seem uncomfortable, dial back to the warm undercurrent. Keep it a show, not an attack.

---

## `roast.py` — Roast Target Extractor

**Type:** custom `Tool` subclass, profile-scoped.

**Trigger:** Called once at conversation open (before the first line), and optionally again mid-routine if the LLM wants a refresh.

**Behaviour:** Captures the current camera frame via the existing camera worker, then re-prompts the vision model with a structured extraction prompt requesting labelled roast fields — not a scene description.

**Extraction prompt targets:**
- `hair` — style, quantity, condition
- `clothing` — what they're wearing and how well
- `build` — posture and physicality
- `expression` — what their face is doing
- `standout` — the single most distinctive/memorable thing
- `energy` — nervous, confident, confused, etc.

**Returns:**
```python
{
    "hair": "thinning on top, attempting a combover",
    "clothing": "wrinkled button-down, tucked in badly",
    "build": "stocky, comfortable with it",
    "expression": "trying to look relaxed, not pulling it off",
    "standout": "the mustache. that's the whole story.",
    "energy": "nervous, fidgety"
}
```

**Parameters:** none.

---

## `crowd_work.py` — Session State + Callbacks

**Type:** custom `Tool` subclass, profile-scoped.

**Parameters:** `action` (required): `"update"` or `"query"`.

### `update`

Accepts any subset of: `name`, `job`, `hometown`, `details` (list of freeform strings). Merges into the in-memory session dict. Schedules an async fire-and-forget write to disk — the conversation loop does not wait for it.

### `query`

Returns the full accumulated profile plus a `callbacks` list. The tool picks the 2–3 richest detail combinations and surfaces them as structured callback prompts — the LLM writes the actual jokes from these hints:

```python
{
    "profile": {
        "name": "Tony",
        "job": "software engineer",
        "hometown": "Pittsburgh",
        "details": ["nervous laugh", "wrinkled shirt", "thinning on top"]
    },
    "callbacks": [
        "name + job + hometown: Tony, software engineer, Pittsburgh",
        "visual + job combo: thinning hair + writes code for a living",
        "behaviour callback: nervous laugh — you mentioned this twice now"
    ]
}
```

The LLM receives these hints and writes the Rickles-voice jokes from them. The tool's job is selecting what's worth calling back to, not writing the punchlines.

### Async Persistence Layer

- **Session file location:** `.rickles_sessions/session_{YYYYMMDD_HHMMSS}.json`
- **Write strategy:** `asyncio.to_thread(json.dump, ...)` — fire-and-forget, no await, no blocking
- **On `__init__`:** scans `.rickles_sessions/` for the most recent file from the current day; loads it if found so a restarted app can resume mid-session
- **Session window:** same calendar day (configurable via `SESSION_WINDOW_HOURS` constant in the file)
- **Schema:**
```json
{
    "session_id": "20260505_143022",
    "started_at": "2026-05-05T14:30:22",
    "name": "Tony",
    "job": "software engineer",
    "hometown": "Pittsburgh",
    "details": ["nervous laugh", "wrinkled shirt", "thinning on top"],
    "roast_targets_used": ["hair", "job"],
    "last_updated": "2026-05-05T14:38:11"
}
```

---

## `tools.txt`

```
camera
move_head
play_emotion
roast
crowd_work
```

---

## Interaction Sequence

```
1. Person appears
2. roast() → labelled visual targets
3. move_head (slow scan)
4. Opening visual riff (lead with standout target)
5. Crowd-work question: "What do you do?"
6. crowd_work(action="update", job=...) [silent]
7. Riff on job
8. Crowd-work question: "Where you from?"
9. crowd_work(action="update", hometown=...) [silent]
10. Riff on hometown
11. crowd_work(action="query") → callbacks surface
12. Callback riff
13. play_emotion (mock-horrified or warm reaction)
14. "But I love ya" closer
```

Steps 5–13 cycle as long as the conversation continues. The LLM decides when to call `query` based on how much material has accumulated.

---

## Future Extension: Phase 1.5 — Identity Matching

The session archive created by `crowd_work.py` is designed to support future face-recognition-based repeat visitor detection. When implemented, the robot would:

1. Capture a frame at startup
2. Run a face encoder (e.g., InsightFace) to produce an embedding
3. Compare against stored embeddings in `.rickles_sessions/`
4. On match above threshold: load the stored profile and open with a recognition callback ("Oh, well look who it is — Tony. Don't you have a job to do?")
5. On no match: standard cold open

This is tracked as Phase 1.5 in `COMEDIAN_PROJECT.md`.

---

## Out of Scope for Phase 1

- Voice cloning / Rickles-like TTS voice → Phase 2
- Custom gesture library (Rickles-style moves) → Phase 3
- Movement training from video → Phase 4
- Face recognition / repeat visitor matching → Phase 1.5
- Multi-person crowd work (robot tracking multiple people simultaneously)
