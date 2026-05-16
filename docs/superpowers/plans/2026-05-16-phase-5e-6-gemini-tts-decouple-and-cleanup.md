# Phase 5e.6 — TDD Plan

**Spec:** `docs/superpowers/specs/2026-05-16-phase-5e-6-gemini-tts-decouple-and-cleanup.md`
**Branch:** `claude/phase-5e-6-gemini-tts-decouple-and-cleanup`
**Scope:** `phase-5e-6`

---

## Commit 1 — leaf guard on `GeminiTTSResponseHandler`

**Type:** `feat(phase-5e-6): add _startup_credentials_ready guard to GeminiTTSResponseHandler`

**RED:** Add `test_prepare_startup_credentials_is_idempotent` to
`tests/test_gemini_tts_handler.py`. Construct the handler, patch
`google.genai.Client`, call `_prepare_startup_credentials` twice,
assert the patched class is constructed exactly once and
`handler._client` is unchanged on the second call. Run the test —
expect RED (the constructor fires twice without the guard).

**GREEN:** Wrap `GeminiTTSResponseHandler._prepare_startup_credentials`
(`src/robot_comic/gemini_tts.py:242`) body in:

```python
if getattr(self, "_startup_credentials_ready", False):
    return
... existing body ...
self._startup_credentials_ready = True
```

Add a short docstring comment mirroring 5e.5's leaf-guard rationale.
Re-run the test — expect GREEN.

## Commit 2 — rewrite `_build_composable_gemini_tts` without mixin host

**Type:** `refactor(phase-5e-6): rewrite _build_composable_gemini_tts without mixin host`

**RED:** Add three new factory tests to
`tests/test_handler_factory.py` mirroring 5e.5's trio:

- `test_moonshine_gemini_tts_uses_standalone_moonshine_adapter` —
  `result.pipeline.stt._handler is None`.
- `test_moonshine_gemini_tts_wires_should_drop_frame_callback` —
  `result.pipeline.stt._should_drop_frame is not None` and returns
  `False`.
- `test_moonshine_gemini_tts_passes_deps_to_pipeline` —
  `result.pipeline.deps is mock_deps`.

Run them — expect RED on all three (today's builder constructs the
host-coupled adapter and passes no `deps` to the pipeline).

**GREEN:**

1. In `src/robot_comic/handler_factory.py`:
   - Delete `_LocalSTTGeminiTTSHost` (lines 138-142).
   - Drop the now-unused `from robot_comic.local_stt_realtime import
     LocalSTTInputMixin` line (71).
   - Rewrite `_build_composable_gemini_tts` to match the 5e.2-5e.5
     pattern (plain handler + standalone STT + `deps` + `welcome_gate`
     into pipeline). Use the same comment shape as the 5e.5 sibling.

Re-run the three new factory tests — expect GREEN. Re-run the
existing `test_moonshine_gemini_tts_routes_to_composable` — expect
GREEN (the plain `GeminiTTSResponseHandler` is still what gets
wrapped, so the `isinstance` assertion is satisfied).

## Commit 3 — remove `_clear_queue` legacy `_tts_handler` mirror

**Type:** `refactor(phase-5e-6): remove _clear_queue legacy _tts_handler mirror from wrapper`

**RED:** In `tests/test_composable_conversation_handler.py`:

- Delete `test_clear_queue_assignment_propagates_to_tts_handler`.
- Delete the `_tts_handler is None` assertion from
  `test_clear_queue_assignment_handles_none` (keep the test, but
  it now only asserts the wrapper-side and pipeline-side state).
- Add a new test `test_clear_queue_assignment_does_not_touch_tts_handler`
  that asserts `wrapper._tts_handler._clear_queue` is **not** set
  to the callback (i.e. setting `_clear_queue` doesn't mirror onto
  the handler). Run — expect RED (today's setter does mirror onto
  the handler).

**GREEN:** Edit `src/robot_comic/composable_conversation_handler.py`
`_clear_queue.setter` (lines 89-114):

- Drop the `if getattr(self, "_tts_handler", None) is not None: ...
  self._tts_handler._clear_queue = callback` branch.
- Update the docstring to reflect single-mirror-only (pipeline) and
  reference the 5e.6 cleanup.

Re-run — expect GREEN.

Also update the module-level docstring (lines 10-14) to drop the
"`self._tts_handler` for the `_clear_queue` mirroring shim" reference.

## Commit 4 — mark `LocalSTTInputMixin` hybrid-only + update PIPELINE_REFACTOR.md

**Type:** `docs(phase-5e-6): mark LocalSTTInputMixin as hybrid-only; update PIPELINE_REFACTOR.md`

No new tests. Doc/comment-only changes.

1. In `src/robot_comic/local_stt_realtime.py`, extend the
   `LocalSTTInputMixin` class docstring (line 234-235) with a note:

   ```
   Post-Phase-5e (2026-05-16): this mixin is **hybrid-only** — the
   five composable triples (Moonshine + llama/gemini-text/gemini-bundled
   ×  ElevenLabs/Chatterbox/Gemini-TTS) migrated off the mixin via
   5e.2-5e.6. The two remaining consumers are
   :class:`LocalSTTOpenAIRealtimeHandler` and
   :class:`LocalSTTHuggingFaceRealtimeHandler` (Phase 4c-tris Option B
   "legacy forever" hybrids — bundled LLM+TTS half lives inside the
   realtime websocket and doesn't decompose into the Protocol triple).
   New composable triples should use the standalone
   :class:`MoonshineSTTAdapter` + :class:`ComposablePipeline` shape
   from `handler_factory._build_composable_*`.
   ```

2. In `PIPELINE_REFACTOR.md`, update the Phase 5 status table — mark
   5e ✅ Done with a brief summary mentioning the six sub-phase PRs
   (#405 / #407 / #409 / #411 / #412 / 5e.6).

## Verification (post Commit 4)

From repo root:

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/gemini_tts.py \
    src/robot_comic/handler_factory.py \
    src/robot_comic/composable_conversation_handler.py \
    src/robot_comic/local_stt_realtime.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

Re-run any of the known-flake set if they fail; don't fix them.

## Push

```
git push -u origin claude/phase-5e-6-gemini-tts-decouple-and-cleanup
```

Do NOT open the PR. Manager session opens.
