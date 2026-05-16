# Phase 5 — duplicate `tts.synthesize` storm investigation (2026-05-16)

**Date:** 2026-05-16
**Status:** Research + fix shipped on `claude/fix-duplicate-tts-synthesize-storm`.
**Tracks:** epic #391; follow-up to Phase 5f.2 (#422) which added the
short-fragment filter.
**Trigger:** ONE turn during 2026-05-16 hardware validation produced three
near-simultaneous `tts.synthesize` spans:

```
16:58:44  tts.synthesize  char_count=96   (1)
16:58:45  tts.synthesize  char_count=96   (2 — identical length, likely duplicate)
16:58:45  tts.synthesize  char_count=137  (3)
```

Operator heard "the same TTS message twice with a brief pause inbetween."
Not reproduced on subsequent tests in the same session.

---

## §1 — Phase A: root-cause trace

### §1.1 — Where `tts.synthesize` spans come from

The span is opened **inside the TTS adapter**, not the orchestrator:

- `elevenlabs_tts.py:969-976` opens `tts.synthesize` on every call to
  `_stream_tts_to_queue`.
- `gemini_live.py:1030` opens it on the bundled-live path (not in play here
  — composable mode is the affected pipeline).
- `llama_base.py:391/447/657/681` opens it on the bundled llama path.

`ElevenLabsTTSAdapter.synthesize` (`adapters/elevenlabs_tts_adapter.py:125`)
just delegates to `_stream_tts_to_queue`, so:

> **one `ComposablePipeline._speak_assistant_text` invocation → one
> `tts.synthesize` span.**

### §1.2 — Where `_speak_assistant_text` is called from

Only from `ComposablePipeline._run_llm_loop_and_speak` (line 528). That
helper is only called from `_on_transcript_completed` (line 499). The
loop body does `await self._speak_assistant_text(response); return`
after the first non-tool LLM response, so a single transcript dispatch
produces **at most one TTS span** from the orchestrator side.

Therefore: **three TTS spans in one turn = three independent
`_on_transcript_completed` invocations that all passed the dedup
short-circuits and dispatched.**

### §1.3 — Concurrency model — how 3 invocations can co-exist

The STT callbacks are not serialised. From
`adapters/moonshine_listener.py:321`:

```python
loop.call_soon_threadsafe(lambda: asyncio.create_task(_fire()))
```

Every Moonshine `on_line_completed` event creates a **new asyncio task**
that awaits `on_completed(text)` — i.e. `_on_transcript_completed`. Three
listener events fired close together → three concurrent
`_on_transcript_completed` tasks scheduled on the same event loop.

`FasterWhisperSTTAdapter._transcribe_and_dispatch` does `await cb(text)`
in series within the `_process_vad_chunk` chain (lines 362-386), so a
single feed_audio call cannot launch concurrent transcripts. **But**
`feed_audio` itself is called from a separate task (the audio capture
loop), so a second `feed_audio` arriving while the first transcribe is
still running can fire its own end event the moment the first one
yields. In practice the `await asyncio.to_thread(_transcribe_sync)`
release point lets the loop schedule subsequent transcripts.

**Bottom line:** both backends can fire multiple `_on_transcript_completed`
tasks within the same hundreds-of-ms window.

### §1.4 — The dedup short-circuit — what it catches and what it misses

`composable_pipeline.py:429-434`:

```python
now = time.perf_counter()
if transcript == self._last_completed_transcript and now - self._last_completed_at < 0.75:
    logger.debug("Ignoring duplicate transcript: %s", transcript)
    return
self._last_completed_transcript = transcript
self._last_completed_at = now
```

The check (line 430) and the set (lines 433-434) are pure synchronous
Python — no `await` between them — so the read-then-write is atomic
**within** an asyncio task. Two concurrent tasks that arrive at this
line with the **same text** will be serialised by the event loop's
single-tick scheduling: task A runs through to the next `await`
(`set_listening(False)` is sync; the first `await` is the
`output_queue.put` at line 476), updating the cache; task B then runs,
sees the cached value, returns early. So the **text-equality + window
dedup is concurrency-safe** for *identical* strings.

The miss is **strict text equality**. Moonshine's streaming transcriber
routinely refines its final hypothesis as audio drains — a single
utterance can fire 2-3 `on_line_completed` events with hypotheses like:

- `"Hey Rickles"` (12 chars)
- `"Hey Rickles right"` (18 chars)
- `"Hey Rickles right?"` (19 chars)

All three pass the short-fragment filter (5f.2: ≥2 words, ≥8 chars).
All three pass text-equality dedup (the strings differ). All three
reach the LLM. The LLM, being largely deterministic for prompts that
differ only in trailing punctuation or a 1-word tail, often returns the
**same** or very similar response — explaining the **96/96/137**
char_count fingerprint.

faster-whisper has a similar exposure: the webrtcvad-gated chunker can
emit close-back-to-back `end` events when speech amplitude oscillates
around the silence threshold; each gets its own transcribe + dispatch.

### §1.5 — Other potential short-circuits (ruled out)

Re-read each guard before dispatch:

- **Empty transcript** (line 415): only catches blanks.
- **Short-fragment filter** (5f.2, lines 424-426): catches `"You"`,
  `"Hi"`. Does NOT catch multi-word hypotheses ≥8 chars.
- **Pause-controller HANDLED** (lines 480-487): only fires if a pause
  command matches. Irrelevant here.
- **Welcome-gate WAITING** (lines 490-494): only fires before the wake
  name has been spoken. Once gate is GATED (post-wake), it's a no-op.
  At 16:58 the robot was deep in conversation, so gate was open.

Nothing else stops a same-utterance variant burst.

### §1.6 — Existing test coverage

`tests/test_composable_pipeline.py` has dedup coverage for the **same
text within window** case
(`test_completed_callback_suppresses_duplicate_within_window` at line
1441) but **no coverage for**:

- Concurrent `_on_transcript_completed` invocations.
- Near-duplicate (differs by trailing punctuation / one word) within
  the window.

The 5f.2 short-fragment tests (line 540+) exercise the ordering relative
to dedup but only for short text.

### §1.7 — Reproducibility

The bug requires:

1. Multiple `on_line_completed` events for one utterance with
   slightly-different hypotheses — a Moonshine streaming behaviour that
   depends on input audio amplitude envelope. Not deterministic from a
   pure unit test.
2. The asyncio loop scheduling the resulting tasks within the
   sub-second window.

We **can** reproduce in a test by feeding three close-spaced text
variants directly to `_on_transcript_completed` (the asyncio
serialisation makes this deterministic without hardware), and verifying
how many LLM calls happen. That's how we land the regression guard.

---

## §2 — Phase B verdict: ship a defensive fix (B3 = looser dedup + warn log)

The trace gives us a clear root cause and a clean fix surface. Shipping
the orchestrator-only fix here is preferable to deferring, for two
reasons:

1. The fault tree is fully sketched and the fix is one helper function
   + a 2-line predicate change.
2. The fix has no exposure to legitimate user input — close-back-to-back
   transcripts that look like edit-variants of the same utterance are
   precisely the pathological case; legitimate consecutive user lines
   are separated by the user's own breathing pause (well over 1s).

### §2.1 — Chosen fix

Replace the text-equality dedup at `composable_pipeline.py:430` with a
similarity-based dedup that catches edit-variants of the same utterance,
and **widen the window** modestly so the burst is captured even when
LLM/TTS round-trip is slow:

```python
DEDUP_WINDOW_S = 2.0  # was 0.75
DEDUP_SIMILARITY_THRESHOLD = 0.85  # difflib.SequenceMatcher ratio
```

Predicate:

```python
def _is_near_duplicate(a: str, b: str) -> bool:
    """True if two completed transcripts are likely the same utterance."""
    if a == b:
        return True
    if not a or not b:
        return False
    # Cheap length-pre-filter: ratio is upper-bounded by 2L/(La+Lb).
    la, lb = len(a), len(b)
    if min(la, lb) / max(la, lb) < DEDUP_SIMILARITY_THRESHOLD:
        return False
    return SequenceMatcher(a=a, b=b).ratio() >= DEDUP_SIMILARITY_THRESHOLD
```

On a hit, log at **WARNING** (not DEBUG) with both strings so the
hardware operator can spot the pattern in journalctl:

```
logger.warning(
    "Suppressing near-duplicate transcript within %.1fs window: prev=%r new=%r",
    DEDUP_WINDOW_S, self._last_completed_transcript, transcript,
)
```

The exact-match case keeps its DEBUG line (high-volume on the legacy
quick-fire path; we don't want to spam journal). Near-duplicate logging
is intentionally louder so we can confirm the fix is firing on the
next hardware session.

### §2.2 — Window widening (0.75s → 2.0s)

The 0.75s window came from the legacy `LocalSTTInputMixin` (line 596).
That value implicitly assumed the listener fires its second variant
quickly. On the 16:58 storm, the three spans landed across **≥1s**
(16:58:44 → 16:58:45), so the 0.75s window **may have already expired**
between event 1 and event 3. 2.0s is comfortable headroom without
crossing into legitimate-conversational-pause territory (typical
back-to-back human-conversation turn cadence is ≥2.5s with the comedian
persona's deliberate pauses making it much longer).

### §2.3 — Why not Option B2 (in-flight lock)

Option B2 (queue/drop subsequent transcripts while one LLM round-trip is
running) was considered. Rejected because:

- It changes semantics for legitimate quick user follow-ups ("wait — no,
  I meant the *other* dance"). The user often *intends* to interrupt
  themselves; a hard lock turns that into a swallowed turn.
- It papers over the actual issue (Moonshine hypothesis variants) by
  hiding it behind a timing gate. If the LLM round-trip is fast (cached
  / no tools), the lock barely covers the window and the storm leaks
  through anyway.
- The similarity dedup is a more **precise** fix — it catches the actual
  pathological pattern (near-identical text) without rejecting
  legitimate-but-fast turns.

### §2.4 — Files touched (fix)

- `src/robot_comic/composable_pipeline.py` — add stdlib `difflib`
  import, two module-level constants (`DEDUP_WINDOW_S`,
  `DEDUP_SIMILARITY_THRESHOLD`), a private `_is_near_duplicate` helper,
  and the predicate swap in `_on_transcript_completed`.
- `tests/test_composable_pipeline.py` — new tests covering:
  - Exact-duplicate within window still dropped (regression of existing
    behaviour).
  - Near-duplicate (trailing punctuation) within window dropped.
  - Near-duplicate (extra trailing word) within window dropped.
  - Different text within window still dispatched.
  - Identical text outside window still dispatched.
  - WARNING log fires for near-duplicate; DEBUG fires for exact.

Expected diff: ~80-150 LOC across both files.

### §2.5 — Out of scope

- No STT-adapter changes. The listener-burst behaviour is an upstream
  property of Moonshine; we accept it at the orchestrator boundary.
- No new dependencies — `difflib` is stdlib.
- No telemetry counter for "near-duplicate suppressed" (could be a
  follow-up if the pattern recurs).

### §2.6 — Rollback plan

If a future hardware session shows legitimate fast turns being
swallowed, the operator dials can revert behaviour with one constant:

- Set `DEDUP_SIMILARITY_THRESHOLD = 1.0` to fall back to text equality.
- Set `DEDUP_WINDOW_S = 0.75` to restore the legacy window.

No env-var plumbing in this PR; the constants are at module top for
trivial on-device edits.

---

## §3 — Test plan

Whole-repo invocation per
`docs/superpowers/memory/feedback_ruff_check_whole_repo_locally.md`:

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
/venvs/apps_venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

New tests in `tests/test_composable_pipeline.py`:

1. `test_near_duplicate_trailing_punctuation_suppressed` — `"Hey Rickles"`
   then `"Hey Rickles."` within window → LLM called once.
2. `test_near_duplicate_extra_trailing_word_suppressed` — `"Hey Rickles"`
   then `"Hey Rickles right"` within window → LLM called once.
3. `test_distinct_transcripts_both_dispatched` — `"hello"` then
   `"goodbye"` within window → LLM called twice (no false positive).
4. `test_near_duplicate_outside_window_both_dispatched` — same text
   pair separated by 3s → LLM called twice.
5. `test_near_duplicate_logs_warning` — capture log; assert WARNING
   for near-dup; assert DEBUG for exact-dup (regression of the existing
   log behaviour).

## §4 — Rollout

- Branch: `claude/fix-duplicate-tts-synthesize-storm`.
- PR title: `fix(phase-5): similarity-based dedup to break Moonshine variant-burst TTS storm`.
- Reference 2026-05-16 16:58 hardware observation in the PR body.
- No env-var changes.
