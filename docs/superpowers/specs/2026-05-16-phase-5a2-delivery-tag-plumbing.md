# Phase 5a.2 — Delivery-tag + first-audio-marker plumbing

**Branch:** `claude/phase-5a-2-delivery-tag-plumbing`
**Epic:** #391 (Phase 5)
**Date:** 2026-05-16
**Author:** sub-agent (Phase 5 manager-driven)
**Predecessors:** Phase 5a.1 (PR #392, just merged) — established the small-PR cadence for surviving TODOs
**Spec template:** mirrors `docs/superpowers/specs/2026-05-16-phase-5a1-echo-guard-persona-reset.md`

This sub-phase ships two related plumbing additions in one PR, both Phase 4 carry-forward TODOs flagged in `docs/superpowers/specs/2026-05-16-phase-5-exploration.md` §3.2 and §3.3:

1. **Channel A — delivery tags** from `LLMResponse` through `ComposablePipeline._speak_assistant_text` into `TTSBackend.synthesize(tags=...)`. TTS adapters consume from the `tags=` param when non-empty and fall back to their existing text-parsing path when it's empty.
2. **Channel B — first-audio marker** on `TTSBackend.synthesize(first_audio_marker=...)`. Adapters append the wallclock timestamp of the first PCM frame they yield; the orchestrator allocates a per-turn list and can read the timestamp out for telemetry.

Both are pure channels — this PR builds the wire but does NOT populate the producer side beyond defaults (LLM adapters keep returning empty `delivery_tags`; the orchestrator does not yet emit new telemetry from the marker).

## Channel A — Delivery tags

### Current state audit

| Surface | Today | Lines |
|---|---|---|
| `TTSBackend.synthesize(text, tags=())` Protocol | Already accepts `tags` | `backends.py:192-207` |
| `LLMResponse` | Has `text`, `tool_calls`, `metadata: dict` — no typed tags field | `backends.py:69-84` |
| `ComposablePipeline._speak_assistant_text` | Calls `self.tts.synthesize(text)` — drops tags entirely; TODO at lines 278-283 spells out the gap | `composable_pipeline.py:260-285` |
| `ElevenLabsTTSAdapter.synthesize(text, tags=())` | Forwards `tags_list = list(tags) if tags else None` to legacy `_stream_tts_to_queue(text, tags=tags_list)` — already wired, just receives empty tuple in production | `elevenlabs_tts_adapter.py:113-159` |
| `GeminiTTSAdapter.synthesize(text, tags=())` | `del tags` (accepted + dropped); parses tags from the text via `extract_delivery_tags` per sentence | `gemini_tts_adapter.py:142-205` |
| `ChatterboxTTSAdapter.synthesize(text, tags=())` | `del tags` (accepted + dropped); chatterbox handler derives delivery from active persona, not from caller tags | `chatterbox_tts_adapter.py:85-143` |
| `LlamaLLMAdapter.chat()` returns `LLMResponse(text=text, tool_calls=...)` | No tag population | `llama_llm_adapter.py:55-92` |
| `GeminiLLMAdapter.chat()` returns `LLMResponse(text=text, tool_calls=...)` | No tag population | `gemini_llm_adapter.py:78-119` |
| `GeminiBundledLLMAdapter.chat()` returns `LLMResponse(text=text, tool_calls=())` | No tag population | `gemini_bundled_llm_adapter.py:124-163` |

### Operator decision (do not relitigate)

Adapter-only fix:

- Adapters that today parse delivery tags from the LLM text keep doing so when the orchestrator passes `tags=()` (the default). When the orchestrator passes a non-empty `tags=(...,)`, the adapter consumes from the param instead.
- No persona prompt changes. LLMs keep embedding `[fast]` / `[slow]` / `[short pause]` markers in text; adapters that parse continue to parse when no structured tags arrive.

### Change shape

1. **`LLMResponse.delivery_tags: tuple[str, ...] = ()`** — new typed field with empty default. Docstring documents that it's optional delivery hints and that TTS adapters fall back to text-based extraction when empty.

2. **LLM adapters** keep returning empty `delivery_tags` (i.e. rely on the default). No production code path changes in the adapters in this PR — we are building the channel only. Verified each adapter's `LLMResponse(...)` call site:
   - `LlamaLLMAdapter`: `return LLMResponse(text=text, tool_calls=tool_calls)` (`llama_llm_adapter.py:92`) — the new field defaults to `()` so this compiles unchanged.
   - `GeminiLLMAdapter`: `return LLMResponse(text=text, tool_calls=tool_calls)` (`gemini_llm_adapter.py:119`) — same.
   - `GeminiBundledLLMAdapter`: `return LLMResponse(text=text, tool_calls=())` (`gemini_bundled_llm_adapter.py:163`) — same.

3. **`ComposablePipeline._speak_assistant_text`** — pass `tags=response.delivery_tags` to `self.tts.synthesize(...)`. Remove the TODO comment at lines 278-283 (or shrink to a brief pointer to this spec).

4. **TTS adapter pattern: consume-or-fallback.**

   `ElevenLabsTTSAdapter` is already correctly wired — it forwards `tags_list = list(tags) if tags else None`. The legacy `_stream_tts_to_queue` interprets `None` (and `[]`) as "no tags" and `["fast",...]` as a tag list, so passing `tags=("fast",)` from the orchestrator already produces the right delivery there. No code change needed; one new test pins the behaviour.

   `GeminiTTSAdapter.synthesize` — replace `del tags` with: if `tags` is non-empty, **prefer the param** as the per-call delivery cue (apply once to every sentence's `build_tts_system_instruction`); otherwise fall back to per-sentence `extract_delivery_tags(sentence)` as today. Special-case: `SHORT_PAUSE_TAG` ("short pause") in the param triggers the pre-sentence silence frame on the *first* sentence only — paralleling the legacy per-sentence behaviour without doubling silence on multi-sentence text.

   `ChatterboxTTSAdapter.synthesize` — the chatterbox handler reads delivery from the active persona, not from caller tags. The wrapped `_synthesize_and_enqueue(text)` has no `tags=` parameter at all. For this PR, the adapter accepts `tags` for Protocol compliance and **logs at DEBUG** when a non-empty `tags` arrives so future audits can spot the channel is in use; the param is otherwise dropped. Updated docstring documents the gap. **No structural refactor** of the legacy chatterbox handler — that's out of scope (also called out in the brief's "if an adapter can't cleanly do consume-or-fallback" guard).

### Why no LLM tag population in this PR?

The brief is explicit: we are building the channel, not the producer. The natural producers (structured outputs from LLMs that surface delivery hints out-of-band) are a separate concern — current LLMs embed tags in text and the text-parsing fallback handles that path. A future PR that wires a producer (e.g. function-calling delivery hints from Gemini) will populate `delivery_tags` and immediately benefit from this channel.

## Channel B — First-audio marker

### Current state audit

| Surface | Today |
|---|---|
| `TTSBackend.synthesize(text, tags=())` Protocol | No marker parameter |
| Legacy `ElevenLabsTTSResponseHandler._stream_tts_to_queue(text, first_audio_marker, tags)` | `first_audio_marker: list[float]` prefilled by the caller with `time.perf_counter()`; callee on first chunk computes delta, calls `telemetry.record_tts_first_audio`, then `first_audio_marker.clear()` | `elevenlabs_tts.py:905-1000` |
| `_dispatch_completed_transcript` legacy caller | Allocates `first_audio_marker: list[float] = [time.perf_counter()]` per turn | `elevenlabs_tts.py:432-441` |

The legacy contract is "caller pre-fills with start ts; callee uses it for delta calc". The brief asks for a different, **orthogonal** contract for the orchestrator channel: "caller passes empty list; callee appends a single timestamp on first frame; caller reads it out." Both contracts can coexist — legacy stays as-is, orchestrator uses the new contract.

### Change shape

1. **Extend `TTSBackend.synthesize` signature** with `first_audio_marker: list[float] | None = None`. Docstring: "If provided, the adapter appends `time.monotonic()` to this list on the first PCM frame it yields. The append is single-shot per call — adapters check `len(first_audio_marker) == 0` before appending. Enables the orchestrator to record per-turn first-audio latency without subscribing to internal events."

2. **Each TTS adapter** records the marker on first frame yield. Pattern:
   ```python
   _marker_appended = False
   ...
   # in the yield loop, before each yield:
   if first_audio_marker is not None and not _marker_appended:
       first_audio_marker.append(time.monotonic())
       _marker_appended = True
   yield frame
   ```
   `_marker_appended` rather than `len(first_audio_marker)` because the orchestrator might inspect the list mid-stream; the local flag is the single-shot guard.

   Choice of clock: `time.monotonic()` — matches the rest of the composable orchestrator (the `_marker_appended` local flag means we don't depend on any specific clock semantics for the guard; the timestamp value is for the consumer to interpret). Legacy `_stream_tts_to_queue` uses `time.perf_counter()` — but that's for delta calc against a pre-stamped start ts. The orchestrator's channel is a fresh wallclock read for "when did first audio leave the TTS adapter," so `time.monotonic()` is the natural pick.

3. **`ComposablePipeline._speak_assistant_text`** — allocate `first_audio_marker: list[float] = []` per call, pass it to `self.tts.synthesize(..., first_audio_marker=...)`. Do NOT add new telemetry consumers in this PR — populate only. The orchestrator can read `first_audio_marker[0]` after the iterator yields its first frame, but we don't wire that read to anything yet.

### Why no telemetry consumer in this PR?

The brief is explicit: this PR builds the channel. The existing legacy `record_tts_first_audio` continues to fire from `_stream_tts_to_queue` (which `ElevenLabsTTSAdapter` delegates into), so per-turn first-audio latency telemetry is already covered for the ElevenLabs path. Bringing the same telemetry to the other adapters is a separate PR — wire the consumer once the channel is proven.

## TTS Protocol shape after this PR

```python
def synthesize(
    self,
    text: str,
    tags: tuple[str, ...] = (),
    first_audio_marker: list[float] | None = None,
) -> AsyncIterator[AudioFrame]:
```

## Files touched

- `src/robot_comic/backends.py` — `LLMResponse.delivery_tags` field; `TTSBackend.synthesize` signature.
- `src/robot_comic/composable_pipeline.py` — `_speak_assistant_text` passes `tags=` and `first_audio_marker=`; remove TODO.
- `src/robot_comic/adapters/elevenlabs_tts_adapter.py` — add `first_audio_marker` param + appender; tags param already works.
- `src/robot_comic/adapters/gemini_tts_adapter.py` — add `first_audio_marker` param + appender; consume-or-fallback for tags.
- `src/robot_comic/adapters/chatterbox_tts_adapter.py` — add `first_audio_marker` param + appender; log non-empty tags at DEBUG; docstring update.
- `tests/adapters/test_elevenlabs_tts_adapter.py` — tests for tags-from-param and first-audio-marker.
- `tests/adapters/test_gemini_tts_adapter.py` — tests for consume-or-fallback and first-audio-marker.
- `tests/adapters/test_chatterbox_tts_adapter.py` — tests for first-audio-marker; tags-debug-log.
- `tests/test_composable_pipeline.py` — test that the orchestrator threads `delivery_tags` to TTS and allocates the marker.
- `tests/test_backends_protocols.py` — test for `LLMResponse.delivery_tags` default and that the marker channel round-trips on a mock TTS.

## Non-goals

- **No LLM-side tag production.** LLM adapters keep returning empty `delivery_tags`.
- **No new telemetry consumers.** The orchestrator allocates `first_audio_marker` and doesn't read it; no `record_tts_first_audio` migration to other adapters in this PR.
- **No persona prompt changes.** LLMs keep embedding `[fast]`-style markers in text.
- **No legacy handler refactor.** `_stream_tts_to_queue`'s caller-prefilled `first_audio_marker` semantics survive untouched — that channel is orthogonal to the new orchestrator-allocated channel.
- **No `ChatterboxTTSResponseHandler` tags-parameter retrofit.** The legacy handler has no tags arg; adding one would be a structural change to the legacy code, which the brief instructs to STOP-and-report rather than do.

## Acceptance criteria

- `LLMResponse.delivery_tags` exists with default `()`; `dataclass(frozen=True)` is preserved.
- `TTSBackend.synthesize` signature includes `first_audio_marker: list[float] | None = None`; all three adapter `synthesize` methods accept it and append on first yield.
- `ComposablePipeline._speak_assistant_text` calls `self.tts.synthesize(text, tags=response.delivery_tags, first_audio_marker=marker)` with `marker: list[float] = []`.
- `GeminiTTSAdapter` consumes from `tags=` param when non-empty, falls back to text parsing when empty.
- `ChatterboxTTSAdapter` logs non-empty `tags` at DEBUG; does not raise.
- `ElevenLabsTTSAdapter` continues to forward `tags` to the legacy method (no behavioural change there).
- New regression tests pass; existing tests still pass.
- `uvx ruff@0.12.0 check .` + `format --check .` green from repo root.
- `mypy --pretty` green on changed source files.
