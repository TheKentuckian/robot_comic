# Phase 5f.3 — content-similarity echo filter

**Date:** 2026-05-16
**Status:** Spec — implementation on `claude/phase-5f-3-echo-content-similarity-filter`.
**Tracks:** epic #391; follow-up to Phase 5f (#416) + 5f.1 (#420) + 5f.2 (#422).
**Predecessor:** Phase 5f.2 (`#422`) — added orchestrator-side
`MIN_TRANSCRIPT_WORDS=2` / `MIN_TRANSCRIPT_CHARS=8` length filter plus a
`DEFAULT_ECHO_COOLDOWN_MS` bump 300 → 800. That closed the
single-word / single-fragment arm of the cascade.

---

## §1 — Hardware-observed cascade after 5f.2 (2026-05-16)

5f.2 successfully kills the `"You"` / `"Thank you"` style echo turn. The
remaining failure mode caught on the chassis is **longer-multi-sentence
self-echo**:

```
role=assistant content="Okay, well I didn't vote for you."
turn.outcome=interrupted   # 5f.2's 800ms cooldown lapses before the next mic frame
role=user      content="Okay, well I didn't vote for you."   # 32 chars, 7 words
turn.outcome=interrupted
role=assistant content="Oh."
turn.outcome=interrupted
role=assistant content="Alright, alright"
turn.outcome=interrupted
…
```

A 32-char / 7-word transcript clears both 5f.2 length filters, dispatches as
a fresh user turn, and Gemini responds — restarting the loop. Each cycle is
slower than pre-5f.2 (~20 s rather than sub-second) because the longer
utterance + cooldown bought us breathing room, but it's still real and the
operator hears Don cutting himself off ("Oh.", "Alright, alright" mid-line)
because barge-in fires when a new transcript arrives mid-TTS.

### Root cause (unchanged from 5f.2; new dimension)

The chassis enclosure leaks speaker output to the mic with enough fidelity
that faster-whisper transcribes Don's *own utterance* back, only mildly
corrupted, as if the user said it. Length-based filters cannot catch this
case — the content is by construction *exactly* the kind of input the
orchestrator expects.

### Why the existing guards miss

- `_speaking_until` cooldown (5f.2 = 800 ms) covers room-acoustic reverb
  *tail*, not the actual speaker playback window. A multi-sentence
  utterance lasts seconds; the deadline tracks the last frame out of the
  TTS queue, not the wall-clock playback end. Once playback ends the
  deadline lapses cleanly — but the mic is still hot for incoming
  acoustic echo of the *last* phrase of the utterance.
- 5f.2's length filter is purely structural — it can't see "this
  transcript matches what I just said."
- The duplicate-suppression window (0.75 s on the same exact string)
  is much shorter than the round-trip and works on exact string match.
  faster-whisper introduces word-error noise that defeats `==`.

## §2 — Decision

Add a **third filter layer** in `ComposablePipeline._on_transcript_completed`,
sequenced AFTER the existing 5f.2 length filter and BEFORE the
duplicate-suppression window:

> If the incoming transcript is substantially similar to one of the last
> N assistant utterances, drop it as a likely echo of our own speech.

### Why a similarity check (and not e.g. transcript-confidence weighting)

- faster-whisper's confidence is exposed but not trivially calibrated.
  Echoed *speech* (Don's voice playing into the mic) is high-confidence
  to the model — confidence is no help here.
- A `difflib.SequenceMatcher` ratio is stdlib (no new deps), order-aware,
  and forgiving of word-error noise on either side. The chassis acoustic
  channel introduces minor edit-distance differences ("Okay well I didn't
  vote for you" → "Okay well I didn't vote for you" passes 0.95+; "I
  think he's very young" → "I think he is very young" passes 0.85+).
- We use a *small* ring buffer (last 5 assistant utterances) so the
  filter is bounded and doesn't grow with session length. The most
  recent utterance dominates in practice; older entries catch the case
  where Don speaks a couple of sentences and the mic picks up an
  earlier-in-the-buffer phrase after a beat.

### Shape

Module-level constants (top of `composable_pipeline.py`, in the same
"on-device tuning is a one-line change" idiom as 5f.2):

```python
ECHO_HISTORY_MAXLEN = 5
ECHO_SIMILARITY_THRESHOLD = 0.65
```

State (instance attribute, initialized in `__init__`):

```python
self._recent_assistant_texts: collections.deque[str] = collections.deque(
    maxlen=ECHO_HISTORY_MAXLEN
)
```

Recording site — in `_speak_assistant_text`, immediately after the
existing `self._conversation_history.append({"role": "assistant", ...})`:

```python
self._recent_assistant_texts.append(_normalize_for_echo_check(text))
```

Filter site — in `_on_transcript_completed`, after the 5f.2 length
filter and before the duplicate-suppression `==` check:

```python
if self._recent_assistant_texts:
    needle = _normalize_for_echo_check(transcript)
    best = max(
        difflib.SequenceMatcher(None, needle, candidate).ratio()
        for candidate in self._recent_assistant_texts
    )
    if best >= ECHO_SIMILARITY_THRESHOLD:
        logger.debug(
            "dropping likely echo of own speech: ratio=%.2f, transcript=%r",
            best,
            transcript[:60],
        )
        return
```

### Normalization rules

`_normalize_for_echo_check(text: str) -> str` returns a comparison-only
form:

1. Lowercase.
2. Strip all Unicode punctuation (`unicodedata.category(c)[0] != "P"`
   filter; equivalently `re.sub(r"[^\w\s]", " ", s)` for the ASCII-heavy
   reality of our utterances).
3. Collapse runs of whitespace to a single space.
4. Strip leading/trailing whitespace.

The original (un-normalized) transcript is the one that would be dispatched
if it passes the filter — normalization is purely for similarity comparison.

### Trade-off (acknowledged)

A legitimate user mimicry or direct quote of Don's recent line is dropped.
Examples:

- Operator says "I didn't vote for you" right after Don finishes the same
  line — drops with ratio ≈ 0.90.
- User echoes back a punchline to riff.

This is acceptable because:

1. The cascade alternative is unrecoverable without operator intervention.
2. The persona is comedian-monologue-heavy; the conversational pattern is
   user *responds* to Don, not *quotes* Don.
3. The threshold (0.65) leaves room: a user paraphrase ("yeah well I
   voted for the other guy") will score well below 0.5 because both
   length and word overlap differ.
4. The history window (5) auto-evicts; once Don has spoken a few new
   lines the user can quote freely.

If this proves intrusive in field testing, the threshold and maxlen are
both module-level one-line changes.

## §3 — What 5f.3 does NOT do

- Does NOT change the 5f.2 length filter (`MIN_TRANSCRIPT_WORDS` /
  `MIN_TRANSCRIPT_CHARS`) — that layer stays as the structural pre-filter.
- Does NOT change `DEFAULT_ECHO_COOLDOWN_MS`.
- Does NOT touch `faster_whisper_stt_adapter.py` or
  `moonshine_stt_adapter.py` — the adapter surface is stable.
- Does NOT tune webrtcvad aggressiveness or frame thresholds.
- Does NOT add per-persona similarity thresholds.
- Does NOT add transcript-confidence weighting (separate concern; complex
  to calibrate).

## §4 — Files touched

- `src/robot_comic/composable_pipeline.py` — add `difflib`/`collections`
  imports, two module-level constants, `_normalize_for_echo_check`
  helper, ring-buffer attribute, recording site in
  `_speak_assistant_text`, filter site in `_on_transcript_completed`.
- `tests/test_composable_pipeline.py` — new tests covering the
  similarity-drop happy path, paraphrase pass-through, ring-buffer
  eviction, normalization correctness, and ordering relative to the
  5f.2 length filter + duplicate-suppression window.

Expected diff: ~200-300 LOC (filter + helper + tests). Hard ceiling 600 LOC.

## §5 — Test plan

New tests in `tests/test_composable_pipeline.py`, grouped under a
"Phase 5f.3 — content-similarity echo filter" header:

1. `test_exact_echo_of_assistant_text_dropped` — pipeline speaks "Hello
   there folks", then STT emits "Hello there folks" — dropped; history
   only contains the original assistant turn.
2. `test_minor_word_error_echo_dropped` — assistant: "Okay well I didn't
   vote for you" / STT: "okay well i didnt vote for you" (punctuation
   stripped, case folded, contraction split) — dropped at ratio ≥ 0.85.
3. `test_paraphrase_not_dropped` — assistant: "Okay well I didn't vote
   for you" / user: "yeah well I voted for the other guy" — passes; the
   LLM is invoked.
4. `test_echo_against_older_assistant_turn` — speak A, speak B, speak C;
   STT emits an echo of A — still dropped (within `maxlen=5`).
5. `test_evicted_assistant_text_no_longer_blocks` — speak `maxlen+1`
   different assistant turns, then STT emits an echo of the *first*
   one — passes (proves ring eviction).
6. `test_normalization_strips_punctuation_and_case` — direct call to
   `_normalize_for_echo_check` for the obvious transforms.
7. `test_similarity_filter_runs_after_length_filter` — STT emits a
   short transcript ("ok") that also matches an assistant turn — drops
   via the length filter (5f.2), not the similarity filter. Proves
   ordering and that the length filter's free returns are still hit
   when both would apply.
8. `test_similarity_filter_runs_before_duplicate_suppression` — speak
   "alright alright alright", then STT emits the echo twice; both drops
   should go through the similarity arm (not the dup-window), so
   `_last_completed_transcript` remains the previous *real* user turn.

Existing 5f.2 tests must continue to pass unchanged.

Whole-repo invocation per
`docs/superpowers/memory/feedback_ruff_check_whole_repo_locally.md`:

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes src/robot_comic/composable_pipeline.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

## §6 — Rollout

- PR title: `fix(phase-5f-3): drop transcripts similar to recent assistant speech`
- Reference the 2026-05-16 hardware finding (multi-sentence cascade
  observed post-5f.2) in the PR body; supersedes the long tail of #422.
- No env-var changes; constants are at module top so operator can patch
  for tuning if needed.
- Next iteration if this still leaks: webrtcvad aggressiveness bump,
  hardware AEC investigation, or transcript-confidence weighting.
