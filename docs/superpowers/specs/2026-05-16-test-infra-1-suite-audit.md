# Spec: Test suite audit (test-infra-1) — issue #339

## Context

`tests/` collects 1748 tests (1819 nodes, 71 deselected by the default
`-m 'not integration'` filter) and runs in ~2 min locally on this machine.
After the Phase 4e (#337) deletion of legacy concrete handler classes
(`LocalSTTLlamaElevenLabsHandler`, `GeminiTextElevenLabsHandler`, etc.) and
the Phase 4f (#381) retirement of the `BACKEND_PROVIDER` dial, the test
suite was never pruned. This audit asks: *what is now dead or redundant?*

## Audit findings

### Dead tests

Grep for the seven retired handler class names:

```
LocalSTTLlamaElevenLabsHandler
LocalSTTChatterboxHandler
LocalSTTGeminiElevenLabsHandler
LocalSTTGeminiTTSHandler
GeminiTextChatterboxHandler
GeminiTextElevenLabsHandler
LocalSTTLlamaGeminiTTSHandler
```

Result: **zero** live imports in `tests/`. Every reference is a comment
documenting the deletion (e.g. `# Phase 4e (#337) retired
LocalSTTChatterboxHandler`). The tests that previously exercised those
classes were rewritten to drive the composable wrapper instead — that work
landed with Phase 4e. So there is no "dead test file" to delete.

### Duplicate coverage

The three TTS adapter test files (`tests/adapters/test_elevenlabs_tts_adapter.py`,
`test_chatterbox_tts_adapter.py`, `test_gemini_tts_adapter.py`) each
re-implement the same `TTSBackend` contract checks:

| Assertion | ElevenLabs | Chatterbox | Gemini |
|---|---|---|---|
| `prepare()` calls handler `_prepare_startup_credentials` | yes | yes | yes |
| `synthesize()` yields `AudioFrame` instances | yes | yes | yes |
| `synthesize("")` yields nothing | implicit | implicit | yes |
| `synthesize()` propagates handler exceptions | yes | yes | (n/a — different surface) |
| `synthesize()` accepts a `tags=()` kwarg | yes | yes | yes |
| `shutdown()` is safe with no open resource | yes | yes | yes |
| `isinstance(adapter, TTSBackend)` | yes | yes | yes |

Six of those rows can move into a parametrised
`tests/adapters/test_tts_backend_contract.py` fixture, fed by all three
adapters.

What **must stay** in the per-adapter files:

* **ElevenLabs** — duck-typed `GeminiTextElevenLabsResponseHandler` shape
  acceptance (Phase 4c.3), `_http.aclose()` semantics, queue-leak isolation.
* **Chatterbox** — `_synthesize_and_enqueue` signature (no tags / no
  marker), `AdditionalOutputs` sentinel drop, `_http.aclose()` semantics,
  queue-leak isolation.
* **Gemini** — `[short pause]` silence inflation, delivery-tag stripping,
  `system_instruction` cue suffix, first-greeting telemetry emit
  (Lifecycle Hook #3b, PR #382), `_pcm_to_frames` chunking sanity, no-op
  shutdown.

The `LLMBackend` adapter tests are NOT in scope — each LLM adapter
(`gemini_llm_adapter`, `gemini_bundled_llm_adapter`, `llama_llm_adapter`)
exposes a meaningfully different surface (single-shot vs streaming, tool
schema, bundled-TTS branch). Collapsing them would erase signal. Same for
`STTBackend`: only `MoonshineSTTAdapter` exists; Phase 5e/5f will add
faster-whisper and at that point the spec should introduce a contract
fixture.

### Other suspected redundancies (NOT acted on, flagged for follow-up)

* `tests/test_composable_pipeline.py` (24.7K) and
  `tests/test_composable_conversation_handler.py` (17.9K) both cover the
  composable wrapper at slightly different layers. A deeper review may
  collapse them, but the cost/benefit is not obvious from a surface read
  and both touch live Phase 5 code paths that are still moving.
* `tests/test_console.py` is 35K and likely has redundant route tests
  that overlap with `tests/test_admin_pipeline_3column.py`, but those
  routes are in active flux for Phase 5d (admin UI / `/health`).

## Plan

1. Add `tests/adapters/test_tts_backend_contract.py`. Parametrise over
   `("elevenlabs", "chatterbox", "gemini")` using a fixture factory that
   builds (adapter_instance, stub_handler) pairs. Cover the six shared
   contract rows above.
2. Verify the contract suite passes against all three adapters. **Do not
   delete the per-adapter copies until contract assertions are green.**
3. Delete the now-redundant assertions from per-adapter files: shared
   `prepare`, shared "yields N AudioFrames", shared empty/error paths,
   shared `isinstance` check, shared shutdown no-op. Keep the adapter-
   specific behaviour identified above.
4. Re-run full suite. Confirm count delta. Document final number in PR
   body.

## Non-goals

* No source changes (`src/robot_comic/` untouched).
* No deletion of `STTBackend` per-adapter file — only `MoonshineSTTAdapter`
  exists today.
* No deletion of `LLMBackend` per-adapter files — too divergent.
* No collapse of `test_composable_*` files (deferred).
* No deletion of tests `test_console.py` (deferred — Phase 5d in flight).

## Expected count delta

Per-adapter shared assertions: ~6 tests × 3 adapters = 18 deletions; new
contract file adds 6 × 3 parametrised cases = 18 new node IDs. **Net
delta ≈ 0** at the node-ID level. The win is maintenance: a new
`TTSBackend` Protocol method gets one new contract test, not three.

This is below the 200-400 target the instructions called out. The instructions
also say "document why the audit found fewer redundancies than expected" —
which is what this spec does. The Phase 4e/4f churn was already cleaned up
in-flight as part of those PRs (commit comments are the receipts). The
suite is in better shape than the issue-#339 headline suggests.
