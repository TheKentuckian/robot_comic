# Comedian Persona Project

A multi-phase project to give Reachy Mini the personality, voice, and physical presence of a stand-up comedian — with live crowd-work interactivity and camera-driven roasting.

---

## Phase 1 — Comedian Persona + Crowd Work Loop

**Goal:** A working comedian profile that can do live, interactive one-on-one stand-up using the existing codebase.

**Comedian candidates:** Andrew Dice Clay, Don Rickles, Bill Hicks (to be decided). Key criteria: enough recorded material to build a solid corpus, a distinctive voice/cadence, and a style that lends itself to audience interaction.

**Corpus approach:** Collect transcripts, interviews, and recorded sets to distill the comedian's signature phrases, cadence patterns, topics, and roasting style into a detailed `instructions.txt` prompt.

**Crowd work loop:**
- Use the existing `camera` tool to capture a frame of whoever the robot is talking to
- Use the vision model to generate a brief physical/contextual description of the person
- Incorporate that description into the routine (light roasting, callbacks, crowd-work riffs)
- Ask the person what they do for a living, where they're from, etc., and riff on their answers — as real comedians do

**What this requires:**
- A new profile folder (`profiles/comedian/`) with a rich `instructions.txt` and `tools.txt`
- Optionally, a custom `roast` tool that formalizes the camera → describe → riff pipeline
- Prompt engineering to capture the comedian's voice, topics, and interactivity patterns

**Deliverable:** A fully functional comedian profile, selectable from the UI like any other profile.

---

## Phase 1.5 — Repeat Visitor Identity Matching *(future sub-phase)*

**Goal:** The robot wakes up, takes a photo, recognises someone it has spoken to before, and opens with a callback rather than a cold start.

> *"Oh, well look who it is — Tony. Don't you have a job to do? Like sitting underneath a steamroller?"*

The crowd-work session state (Phase 1) already persists profiles to disk. Identity matching is the missing link that ties a returning face to a stored profile.

**Approach:**
- On each conversation start, capture a camera frame and run a lightweight face encoder (e.g., `face_recognition`, InsightFace) to produce an embedding
- Compare against stored embeddings in the session archive; if similarity exceeds a threshold, load that person's profile and surface their best callback hooks
- If no match, start fresh — standard Phase 1 cold open

**What this requires:**
- Face encoding and embedding storage alongside the JSON session files
- A similarity-matching step at startup (before the first line of conversation)
- A new section in `instructions.txt` for the "well look who it is" recognition opener
- Tuning the match threshold to avoid false positives (misidentifying someone as a known person)

**Note:** This is deliberately scoped as a sub-phase rather than folded into Phase 1 — it has no robot hardware dependencies beyond the camera and can be developed and tested independently.

---

## Phase 2 — Voice Cloning + Cadence

**Goal:** Replace the generic TTS voice with one that sounds like (or is strongly reminiscent of) the chosen comedian.

**Approach options:**
- **ElevenLabs voice cloning** — clone from audio samples, requires a new backend integration or a post-processing TTS layer
- **OpenAI voice cloning** — if/when supported via the Realtime API
- **Custom HF TTS model** — fine-tune or prompt a model with comedian audio samples via the existing HF backend

**Key challenge:** The existing backends (HF, OpenAI, Gemini) expose a fixed voice catalog with no cloning API today. Phase 2 likely requires adding a new backend or a TTS post-processing step.

**What this requires:**
- A new `ConversationHandler` subclass (or TTS middleware) for the chosen cloning service
- Audio sample collection and preparation (source from public recordings)
- Voice registration/cloning via the chosen service's API

---

## Phase 3 — Custom Gesture Library

**Goal:** A library of Reachy Mini movements that reflect the comedian's physical style — signature gestures, timing, emphasis moves.

**Approach:** Study recorded stand-up sets for recurring physical patterns (pointing, shrugging, leaning in, big arm gestures). Hand-craft these as `GotoQueueMove` sequences (following the existing `sweep_look.py` pattern) and register them as callable tools.

**What this requires:**
- New move definitions in a `comedian_moves.py` tool file within the profile
- Moves mapped to semantic names the LLM can call (`point_at_audience`, `shrug`, `lean_in`, etc.)
- Prompt engineering to tell the LLM when to trigger each gesture

---

## Phase 4 — Movement Training from Video (Research Phase)

**Goal:** Automatically extract pose sequences from comedian video footage and generate a library of authentic physical gestures.

**Approach (speculative):**
- Run pose estimation (e.g., MediaPipe, OpenPose) over stand-up video to extract joint trajectories
- Map human body poses to Reachy Mini's joint space
- Cluster and label recurring gestures to build a parameterized move library
- Potentially fine-tune the LLM's gesture-selection behavior with the new library

**What this requires:**
- A separate ML/data pipeline (effectively a standalone research project)
- Significant manual review to filter and label extracted poses
- Joint-space mapping work between human proportions and Reachy Mini's kinematics

**Note:** This phase is intentionally kept vague — the scope and feasibility depend heavily on what Phase 3 produces and what the robot's kinematics can realistically express.
