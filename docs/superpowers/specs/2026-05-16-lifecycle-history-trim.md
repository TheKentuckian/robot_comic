# Lifecycle Hook #5 — `history_trim.trim_history_in_place` on the composable path

**Branch:** `claude/lifecycle-history-trim`
**Epic:** #337 (pipeline refactor) — Deferred lifecycle hooks
**Date:** 2026-05-16
**Author:** sub-agent (manager-driven)

## Background — why the doc's "orchestrator-level" placement is correct,
## and why every composable triple currently grows history without bound

`PIPELINE_REFACTOR.md` line 315 lists `history_trim.trim_history_in_place`
as a deferred lifecycle hook and proposes the new home as "Orchestrator-
level in `ComposablePipeline._run_llm_loop_and_speak`". The doc's
description of the legacy call site ("Inside `_call_llm` before each
request") is slightly imprecise: the trim actually runs once per user
turn in `_dispatch_completed_transcript`, *before* the LLM loop, not
inside `_call_llm` itself. The cadence is "once per user turn" in legacy
and that's what we preserve here.

`CLAUDE.md` already documents the legacy contract:

> Handlers that keep client-side chat history (`gemini_tts.py`,
> `llama_base.py`) trim to the last `REACHY_MINI_MAX_HISTORY_TURNS` user
> turns (default 20, set `0` to disable) via
> `history_trim.trim_history_in_place`. Trimming runs *before* each
> request is built and cuts on user-turn boundaries so tool round-trips
> ride along. Realtime backends (OpenAI/HF/Gemini Live) manage history
> server-side and are not bounded by this knob.

### Where the trim lives today (legacy paths)

`git grep -nE "trim_history_in_place" src/` returns three legacy call
sites, all in `_dispatch_completed_transcript` (the once-per-user-turn
entry point):

| File | Line | Surrounding method | Wraps |
|------|------|--------------------|-------|
| `src/robot_comic/llama_base.py` | 506 | `_dispatch_completed_transcript` — before `_run_turn` | Llama + LocalSTT-Llama |
| `src/robot_comic/gemini_tts.py` | 365 | `_dispatch_completed_transcript` — before `_run_llm_with_tools` | Bundled GeminiTTS / LocalSTT-Gemini-TTS |
| `src/robot_comic/elevenlabs_tts.py` | 565 | `_dispatch_completed_transcript_impl` — before `_run_llm_with_tools` | LocalSTT-Llama-ElevenLabs / LocalSTT-Gemini-ElevenLabs |

All three sites:

1. Append the new `{"role": "user", ...}` entry to
   `self._conversation_history`.
2. Call `trim_history_in_place(self._conversation_history, role_key=...)`
   (the llama site uses the default `role_key="role"`; the gemini sites
   pass it explicitly for clarity since Gemini's history shape uses
   `parts` instead of `content` but still keys the role under `"role"`).
3. Then run the LLM loop.

The trim itself is defined in `src/robot_comic/history_trim.py:95`. It
reads `REACHY_MINI_MAX_HISTORY_TURNS` from the env (default 20, `0`
disables) and drops the oldest user-turn groups so at most `max_turns`
user turns remain. Assistant replies, tool-call turns, and tool-result
turns are kept together with their user turn ("tool round-trips ride
along"). System messages — if present at the front — are preserved.

### What the three composable LLM adapters call

The orchestrator (`ComposablePipeline`) maintains its own
`_conversation_history` list and passes it to each adapter's `chat()`
method. The three adapters all **bypass** the legacy trim:

- `LlamaLLMAdapter.chat()` (`adapters/llama_llm_adapter.py:55`) → saves
  the wrapped handler's `_conversation_history`, replaces it with `[]`,
  passes the orchestrator's `messages` as `extra_messages`, calls
  `handler._call_llm(extra_messages=messages)`. Even if `_call_llm` did
  trim (it doesn't — the trim is in `_dispatch_completed_transcript`,
  which the adapter bypasses), the trim would operate on the empty
  swap-in list, not on the orchestrator's history.
- `GeminiLLMAdapter.chat()` (`adapters/gemini_llm_adapter.py:77`) — same
  swap pattern, same bypass. The wrapped `GeminiTextResponseHandler`
  inherits `_dispatch_completed_transcript` from `BaseLlamaResponseHandler`
  but the adapter only invokes `_call_llm`.
- `GeminiBundledLLMAdapter.chat()` (`adapters/gemini_bundled_llm_adapter.py:123`)
  — swaps the handler's `_conversation_history` with a
  Gemini-parts-converted version of the orchestrator's `messages`,
  calls `handler._run_llm_with_tools()`. The legacy trim site for the
  bundled handler lives in `gemini_tts.py:365` (i.e. in
  `_dispatch_completed_transcript`, one level *up* from
  `_run_llm_with_tools`) — also bypassed.

### Conclusion

The "orchestrator-level" plan is the correct fix shape. There is no
path through delegation that preserves the legacy trim — the trim lives
in `_dispatch_completed_transcript`, a method the composable path does
not call. All three composable triples currently grow
`ComposablePipeline._conversation_history` unboundedly. **This PR adds
the trim as a `ComposablePipeline` orchestrator hook, not in each
adapter.** Direct mirror of Hook #4 (joke history): one site, one set
of regression tests, no per-adapter Protocol churn.

## Decision — fix shape

### Orchestrator-level, not adapter-level

Doc options:

1. **Adapter-level** (in each `chat()` implementation): three copies of
   the same call, each mutating the orchestrator's history list (which
   is awkward — the adapter receives `messages` as an argument but
   conceptually doesn't own it). Per-adapter knobs make no sense here:
   the trim is a property of the orchestrator-owned history, not the
   LLM backend.
2. **Orchestrator-level** (in `ComposablePipeline._run_llm_loop_and_speak`):
   one site for all LLM paths, exercised exactly once per user turn,
   right before the first LLM round.

This PR chooses **orchestrator-level** — exactly what
`PIPELINE_REFACTOR.md` line 315 prescribed. Rationale mirrors Hook #4:

- **One site for three adapters.** No per-backend dial; the bound is a
  property of the conversation, not the LLM.
- **Once per user turn, not once per LLM round.** Legacy fires in
  `_dispatch_completed_transcript` (per user turn), not in `_call_llm`
  (per round). The legacy doc-table phrasing "Inside `_call_llm` before
  each request" is slightly off; the *intent* is "once before the LLM
  loop starts". Placing the call at the top of
  `_run_llm_loop_and_speak` preserves the per-user-turn cadence.
- **Single home is friendlier for Phase 4e legacy retirement.** When 4e
  deletes the legacy sites, there's one new site, not three.

### Where exactly in `ComposablePipeline`

At the **top of `_run_llm_loop_and_speak`**, before the
`for _round in range(self.max_tool_rounds):` loop. Reasoning:

- The user turn has already been appended to `_conversation_history` by
  `_on_transcript_completed` (line 203) before
  `_run_llm_loop_and_speak` is invoked.
- The trim must run *before* the first `llm.chat(history, ...)` call so
  the request payload is bounded — same intent as the legacy
  "trim BEFORE building the next request" comment.
- Placing it once at the top (not inside the tool-round loop) matches
  the legacy cadence: legacy trims once per user turn, not once per
  tool round. The tool rounds within a single user turn don't add new
  user-role entries (only `tool` and follow-up `assistant` entries
  that ride along with the user turn), so the user-turn count is
  invariant inside the loop. One trim suffices.

### What gets called

`trim_history_in_place` is already a pure function on a leaf module
(`history_trim`). No new helper needed. The orchestrator calls it
directly:

```python
trim_history_in_place(self._conversation_history)
```

The default `role_key="role"` is correct — the orchestrator's history
entries use `{"role": "user|assistant|tool|system", "content": ...}`
shapes (see `composable_pipeline.py:203/243/269`), matching the llama
legacy site (`llama_base.py:506`).

System prompts at the front of the list are preserved by
`trim_history_in_place` itself (see `history_trim.py:118-123`), so the
optional system-prompt seeded in `ComposablePipeline.__init__` (line
119-120) is safe.

## Scope

| File | Change |
|------|--------|
| `src/robot_comic/composable_pipeline.py` | Add `from robot_comic.history_trim import trim_history_in_place` at module top; call `trim_history_in_place(self._conversation_history)` at the top of `_run_llm_loop_and_speak` with a Lifecycle Hook #5 comment block. ~5 LOC. |
| `tests/test_composable_pipeline.py` | Add regression tests proving the orchestrator calls `trim_history_in_place` once per user turn, before the first LLM call, with the orchestrator's history list as the argument. Cover the cap-respected behaviour end-to-end (history shrinks across user turns when over cap). |
| `src/robot_comic/composable_conversation_handler.py` | Update the module docstring's deferred-hooks list to mark Hook #5 as wired. Comment-only. ~1 LOC. |

## Files NOT touched

- `src/robot_comic/llama_base.py`, `src/robot_comic/gemini_tts.py`,
  `src/robot_comic/elevenlabs_tts.py` — legacy trim sites stay until
  Phase 4e. They are still the production path under
  `FACTORY_PATH=legacy` (the 4d default-flip hasn't happened).
- `src/robot_comic/adapters/*.py` — no per-adapter changes. The trim
  is orchestrator-owned, not adapter-owned.
- `src/robot_comic/history_trim.py` — no change. The existing helper
  already does exactly what's needed; no new public surface.
- `LLMBackend` Protocol, `ComposablePipeline.__init__` signature,
  factory wiring — no surface changes.

## Why not a configurable hook callback

A `trim_callback` parameter on `ComposablePipeline.__init__` would let
operators swap the trim policy per pipeline. But every operator wants
the same env-driven `REACHY_MINI_MAX_HISTORY_TURNS` knob — that's
already in the helper. Adding a callback parameter adds surface area
the factory would have to plumb in Phase 4d, for zero present-day
benefit. A direct call to the public helper is simpler.

## Acceptance criteria

- `ComposablePipeline` regression tests assert:
  - `trim_history_in_place` is called exactly once per user turn, with
    `pipeline._conversation_history` as the argument.
  - The call happens **before** the first `llm.chat()` of the turn.
  - The call happens for every LLM-driven user turn (Llama, Gemini-text,
    Gemini-bundled adapter shapes all flow through the same orchestrator
    code path, so a single orchestrator-level test covers all three).
  - The cap-respected end-to-end behaviour: running multiple user turns
    with `REACHY_MINI_MAX_HISTORY_TURNS=2` shrinks the history so at
    most 2 user turns remain.
- All existing tests still pass (especially the existing happy-path,
  tool-loop, and joke-history tests, none of which rely on un-trimmed
  history).
- `uvx ruff@0.12.0 check` and `format --check` green from repo root.
- `.venv/Scripts/python -m mypy --pretty <changed files>` green.
- `.venv/Scripts/python -m pytest tests/ -q` green (modulo the local-env
  `tests/vision/test_local_vision.py` collection error per the task
  brief).

## Out of scope

- Removing the legacy trim sites in `llama_base.py` / `gemini_tts.py`
  / `elevenlabs_tts.py` — Phase 4e cleanup.
- Filtering synthetic status markers (`is_synthetic_status_marker`)
  from the orchestrator's history — the legacy handlers guard at the
  *append* site (see `gemini_tts.py:376`), not at the trim site. The
  orchestrator currently appends `response.text` unconditionally; if a
  future adapter returns a synthetic-marker string the same problem
  would exist legacy-side. Tracked separately if it surfaces.
- A `max_turns` override parameter on `ComposablePipeline` — same
  reasoning as the configurable-callback rejection above.
