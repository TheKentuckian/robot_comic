# Design: Chatterbox TTS Sentence Pipeline Parallelism

**Date:** 2026-05-10
**Status:** Approved, awaiting implementation
**Issue:** TBD

---

## Problem

`_synthesize_and_enqueue` in `chatterbox_tts.py` processes sentences strictly sequentially: it awaits the full Chatterbox TTS response for sentence N before firing the request for sentence N+1. This means the server GPU sits idle during audio playback, and the client sits idle during generation — no overlap at all.

For a typical 4-sentence response with ~3s generation time and ~1.5s playback time per sentence:

```
current:  [gen 1][play 1][gen 2][play 2][gen 3][play 3]  → ~13.5s total
fixed:    [gen 1][play 1]                                  → ~7.5s total
               [gen 2]   [play 2]
                    [gen 3]        [play 3]
```

Each sentence boundary eliminates one full generation-cycle of dead time. On a 4-sentence response that is ~3 boundary × ~3s ≈ 9s saved.

---

## Solution

Fire all sentence TTS requests as concurrent `asyncio` tasks at the start of the call, then `await` them in submission order for correct playback sequence.

```python
# Before (sequential):
for sentence in sentences:
    pcm = await self._call_chatterbox_tts(sentence, exaggeration=e, cfg_weight=c)
    if pcm:
        for frame in self._pcm_to_frames(pcm):
            await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))

# After (pipelined):
tasks = [
    asyncio.create_task(self._call_chatterbox_tts(s, exaggeration=e, cfg_weight=c))
    for s, e, c in sentence_param_triples
]
for task in tasks:
    pcm = await task          # await in submission order → correct playback sequence
    if pcm:
        for frame in self._pcm_to_frames(pcm):
            await self.output_queue.put((_OUTPUT_SAMPLE_RATE, frame))
```

Silence segments (`seg.silence_ms`) are injected inline between the task groups they belong to — they are instant (just `np.zeros`), so they do not disrupt pipelining.

---

## Implementation

**Single file**: `src/robot_comic/chatterbox_tts.py`

**Method**: `_synthesize_and_enqueue` (line ~327)

**Steps**:

1. Flatten the current two-level loop (segments → sentences) into a single pass that collects `(sentence, exaggeration, cfg_weight, preceding_silence_ms)` tuples.
2. For each tuple, either enqueue silence immediately (instant) or create a `Task` via `asyncio.create_task(_call_chatterbox_tts(...))`.
3. Await tasks in submission order and enqueue PCM as each resolves.

**Nothing else changes**:
- `_call_chatterbox_tts` — untouched (retry logic, error handling, gain all intact)
- `_pcm_to_frames`, `_silence_pcm` — untouched
- `output_queue` and all consumers — untouched
- `translate()` call and segment structure — untouched

**Error handling**: `_call_chatterbox_tts` already catches all exceptions and returns `None`. `await task` on a task that returned `None` is handled by the existing `if pcm:` guard. No new error handling needed.

**Ordering guarantee**: `asyncio.create_task()` fires requests in iteration order (server receives them in sequence); `await` in the same order ensures playback is always in correct sentence order regardless of which TTS call finishes first.

**Concurrency**: No semaphore needed. Chatterbox queues requests server-side; concurrent fires just pre-populate that queue. The server processes one at a time (GPU-bound).

---

## Testing

1. Run with a multi-sentence LLM response and observe that sentence 2 begins playing immediately after sentence 1 finishes (no audible gap / pause).
2. Verify sentence order is never scrambled (longer sentence 1 should not be skipped in favour of a faster sentence 2).
3. Confirm TTS errors still degrade gracefully (one failed sentence skips silently, rest play correctly).

---

## Out of Scope

- Fine-tuning Chatterbox on Don Rickles audio (parked — may evaluate alternative TTS models with better zero/one-shot cloning instead)
- Chatterbox server-side streaming investigation (lower priority; model generates audio all-at-once so true mid-generation streaming is unlikely)
- Gain auto-calibration (issue #53, separate track)
