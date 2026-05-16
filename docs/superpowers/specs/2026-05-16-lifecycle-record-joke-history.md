# Lifecycle Hook #4 — `record_joke_history` on the composable path

**Branch:** `claude/lifecycle-record-joke-history`
**Epic:** #337 (pipeline refactor) — Deferred lifecycle hooks
**Date:** 2026-05-16
**Author:** sub-agent (manager-driven)

## Background — why the doc's "adapter post-call OR pipeline hook" is correct,
## but the capture is currently dropped on every composable triple

`PIPELINE_REFACTOR.md` line 314 lists `record_joke_history` as a deferred
lifecycle hook and proposes the new home as either
"`LlamaLLMAdapter.chat` post-call, or as a `ComposablePipeline` hook". Like
Hooks #1–3, this implies a clean relocation. It does not say the capture is
preserved through delegation. Investigation confirms it is not.

### Where the joke-history capture lives today (legacy paths)

`git grep -nE "JokeHistory|extract_punchline_via_llm" src/` returns two
legacy write sites:

| File | Lines | Surrounding method | Wraps |
|------|-------|--------------------|-------|
| `src/robot_comic/llama_base.py` | 578–594 | `_run_turn` (a.k.a. legacy `_run_response_loop`) — round 1, after `_stream_response_and_synthesize` returns | Llama + LocalSTT-Llama-ElevenLabs / Chatterbox |
| `src/robot_comic/gemini_tts.py` | 380–394 | `GeminiTTSResponseHandler.response` — after the bundled `_run_llm_with_tools` returns | Bundled GeminiTTS / LocalSTT-Gemini-TTS |

Both sites do the same three-step capture, guarded by
`config.JOKE_HISTORY_ENABLED` (default True):

1. Call `extract_punchline_via_llm(response_text, http_client, llama_url=...)`
   — a side-channel call to llama-server for `{"punchline": str|None,
   "topic": str}` extraction. Falls back to the `last_sentence_of` heuristic
   on network/parse errors.
2. If `punchline` is non-empty: `JokeHistory(default_history_path()).add(
   punchline, topic=..., persona=config.REACHY_MINI_CUSTOM_PROFILE or "")`.
3. Persist to `~/.robot-comic/joke-history.json`.

The captured entries are read back at session start by `prompts.py:128`
(`_append_joke_history`), which injects a "RECENT JOKES (DO NOT REPEAT)"
section into the system prompt to dampen Don-Rickles-style repetition.

### What the three composable LLM adapters call

- `LlamaLLMAdapter.chat()` → `handler._call_llm(extra_messages=messages)`
  directly. **Bypasses** `_run_turn`, so `llama_base.py:578–594` never
  fires.
- `GeminiLLMAdapter.chat()` → `handler._call_llm(extra_messages=messages)`
  directly. The gemini-text handler inherits `_run_turn` from
  `BaseLlamaResponseHandler` but the adapter only invokes `_call_llm`, so
  the inherited capture site is bypassed identically.
- `GeminiBundledLLMAdapter.chat()` → `handler._run_llm_with_tools()` on
  `GeminiTTSResponseHandler`. The capture lives one level up at
  `gemini_tts.py:380–394` (after `_run_llm_with_tools` returns), in
  `GeminiTTSResponseHandler.response` — which the adapter never invokes.

### Conclusion

The "post-call hook" plan is the correct fix shape. There is no path through
delegation that preserves the legacy write — the legacy write sites live in
methods (`_run_turn` / `GeminiTTSResponseHandler.response`) that the
composable path bypasses entirely. **This PR adds the capture as a
`ComposablePipeline` orchestrator hook, not in each adapter.**

## Decision — fix shape

### Orchestrator-level, not adapter-level

The doc gives two options:

1. **Adapter-level** (`LlamaLLMAdapter.chat` post-call): the per-backend
   site, three copies (one per adapter), gives each backend a chance to
   skip recording for backend-specific reasons.
2. **Orchestrator-level** (`ComposablePipeline` hook): one site for all
   LLM paths, exercised by the orchestrator immediately after it receives
   the final non-empty assistant text.

This PR chooses **orchestrator-level**. Rationale:

- **One site for three adapters.** Hook #2 (telemetry) had to land
  per-adapter because each backend reports `gen_ai.system` differently and
  the durations are timing-sensitive to *that adapter's* round-trip. Joke
  history has no such per-backend dial — the capture is purely a function
  of the final assistant text, regardless of which LLM produced it.
- **The legacy semantics are LLM-loop-final, not LLM-call-final.** The
  legacy `llama_base.py` write fires exactly once per turn, after tool
  rounds settle and the final assistant text exists. Hooking it into the
  adapter would fire it *per LLM round*, including tool-call rounds where
  `response.text` is empty, which the legacy code never did. The
  orchestrator already knows which round is the final speak-to-user
  round — it's `_speak_assistant_text`.
- **Bundled-Gemini parity.** Legacy bundled-Gemini already captures
  *after* `_run_llm_with_tools` returns (at `gemini_tts.py:380`). The
  bundled adapter returns its text from `chat()` and the orchestrator's
  `_speak_assistant_text` runs it through the same hook as the other
  triples. No special-casing.
- **Single home is friendlier for Phase 4e legacy retirement.** When 4e
  deletes the legacy write sites, there's one new site, not three.

### Where exactly in `ComposablePipeline`

Inside `_speak_assistant_text(response)`, **before** the TTS synthesize
loop and **after** the assistant-text-non-empty check. Reasoning:

- The legacy guards (`if response_text and JOKE_HISTORY_ENABLED:`) match
  the orchestrator's existing empty-text guard. Adding the hook inside
  `_speak_assistant_text` keeps the guard in one place.
- Capturing before TTS rather than after means a TTS failure mid-stream
  still records the joke. Legacy does the same — the capture in
  `llama_base.py:578–594` runs before any TTS work in the turn.

### What gets called

A new public async helper `record_joke_history(response_text, *, persona=...)`
in `src/robot_comic/joke_history.py`:

- Returns early if `not response_text` or `not
  config.JOKE_HISTORY_ENABLED`.
- Resolves `persona` (default: `config.REACHY_MINI_CUSTOM_PROFILE`).
- Opens a short-lived `httpx.AsyncClient`, calls `extract_punchline_via_llm`
  with the same default `llama_url` the gemini-tts site uses (the function's
  own default — `http://astralplane.lan:11434`).
- If extraction yields a non-empty `punchline`: `JokeHistory(
  default_history_path()).add(punchline, topic=..., persona=...)`.
- Wraps the whole body in `try / except Exception` and logs at DEBUG
  ("joke_history capture failed: %s"), matching the legacy best-effort
  semantics — joke history must never crash a turn.

Centralising the capture in `joke_history.record_joke_history` keeps the
orchestrator readable (one `await record_joke_history(text)` line) and
gives the legacy sites a clean target for Phase 4e to migrate to (the doc
calls out a `Phase 4e` cleanup pass; legacy can opt into this helper later
to deduplicate the code, but this PR leaves legacy untouched).

## Scope

| File | Change |
|------|--------|
| `src/robot_comic/joke_history.py` | Add public async helper `record_joke_history(response_text, *, persona=None)` that owns the httpx client lifecycle, the config guard, and the best-effort try/except. ~25 LOC. |
| `src/robot_comic/composable_pipeline.py` | Call `record_joke_history(text)` from `_speak_assistant_text` immediately after the empty-text guard and before the TTS synthesize loop. ~3 LOC. |
| `tests/test_joke_history.py` | Add tests for the new `record_joke_history` helper: disabled-config skip, empty-text skip, extraction success path, extraction-returns-None skip, extraction exception swallowed. |
| `tests/test_composable_pipeline.py` | Add regression tests proving the orchestrator calls `record_joke_history` after non-empty assistant text, does NOT call it on empty/whitespace text, and does NOT call it on tool-call-only rounds. |

## Files NOT touched

- `src/robot_comic/llama_base.py`, `src/robot_comic/gemini_tts.py` —
  legacy capture sites stay until Phase 4e. They are still the production
  path under `FACTORY_PATH=legacy` (the 4d default-flip hasn't happened).
- `src/robot_comic/adapters/*.py` — no per-adapter changes. The capture
  is orchestrator-owned, not adapter-owned.
- `src/robot_comic/composable_conversation_handler.py` — the wrapper
  doesn't need changes; the pipeline already fires the hook.
- `src/robot_comic/prompts.py` — read-side (`_append_joke_history`)
  already works on whatever the legacy or composable path wrote; no
  change.
- `LLMBackend` Protocol, `ComposablePipeline.__init__` signature, factory
  wiring — no surface changes.

## Why a top-level helper, not a `ComposablePipeline` method or a hook
## callback

Two reasons:

1. **Testability.** The helper is a plain async function on a leaf module
   (`joke_history`); tests can mock `extract_punchline_via_llm` and
   `JokeHistory.add` independently and assert on calls without spinning
   up a full pipeline. The orchestrator-side test only proves "the
   pipeline called the helper at the right moment".
2. **No new Protocol parameter.** A configurable hook callback on
   `ComposablePipeline.__init__` would be operator-tunable but adds API
   surface area at the orchestrator boundary that the factory would have
   to plumb in Phase 4d. Since *every* operator wants the same capture
   semantics (legacy already shipped it as on-by-default), a hard-coded
   call to the public helper is simpler than a configurable hook.

## Acceptance criteria

- New `record_joke_history` helper has unit-test coverage for:
  - Disabled config → no extraction call, no `JokeHistory.add` call.
  - Empty text → no extraction call.
  - Extraction yields non-empty punchline → `JokeHistory.add` called with
    the right args.
  - Extraction yields `None` punchline (setup/banter detection) → no
    `JokeHistory.add`.
  - Extraction or add raises → exception swallowed, no crash.
- `ComposablePipeline` test asserts:
  - Non-empty assistant text → `record_joke_history` called with that text.
  - Empty / whitespace text → not called.
  - Tool-call-only round → not called (capture fires *after* the final
    speak round).
- All existing tests still pass.
- `uvx ruff@0.12.0 check` and `format --check` green from repo root.
- `.venv/Scripts/python -m mypy --pretty <changed files>` green.
- `.venv/Scripts/python -m pytest tests/ -q` green (modulo the local-env
  `tests/vision/test_local_vision.py` collection error per the task
  brief).

## Out of scope

- Removing the legacy capture sites in `llama_base.py` / `gemini_tts.py`
  — Phase 4e cleanup.
- Migrating the legacy sites to the new `record_joke_history` helper —
  would touch two production paths and require regression coverage on
  both. The helper is *available* for them in Phase 4e but the migration
  is its own PR.
- Adding a `llama_url` override to the helper — the function's default
  already matches the legacy gemini-tts site's behaviour
  (`http://astralplane.lan:11434`). The legacy llama_base site passes
  `self._llama_cpp_url`, but in practice that resolves to the same
  default in every shipping config (and the call falls back to a
  heuristic on network errors anyway).
- The pre-existing `coroutine never awaited` RuntimeWarnings flagged by
  a previous agent in `tests/test_llama_base.py`. Confirmed pre-existing
  on `main` and unrelated to this PR's code paths. Flagged in the PR body
  as "weird but pre-existing".
