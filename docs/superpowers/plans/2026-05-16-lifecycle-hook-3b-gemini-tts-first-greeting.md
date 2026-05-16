# TDD plan — Lifecycle Hook #3b (`GeminiTTSAdapter.first_greeting.tts_first_audio`)

**Spec:** `docs/superpowers/specs/2026-05-16-lifecycle-hook-3b-gemini-tts-first-greeting.md`
**Branch:** `claude/lifecycle-hook-3b-gemini-tts-first-greeting`

One commit per task. Match the 4a/4b cadence.

## Task 1 — Spec + plan + doc fix

**Goal:** Land the spec, this plan, and the small `PIPELINE_REFACTOR.md`
`#TBD → #381` fixup in one commit (no code change yet).

**Files:**

- `docs/superpowers/specs/2026-05-16-lifecycle-hook-3b-gemini-tts-first-greeting.md` (new)
- `docs/superpowers/plans/2026-05-16-lifecycle-hook-3b-gemini-tts-first-greeting.md` (new — this file)
- `PIPELINE_REFACTOR.md` (line 24: `#TBD (manager fixes on merge)` → `#381 (commit 8873fa2)`)

**Verification:**

- `uvx ruff@0.12.0 check .` — green (docs-only commit, but enforce baseline).
- `uvx ruff@0.12.0 format --check .` — green.

**Commit:** `docs(pipeline-refactor): spec + plan for lifecycle hook 3b + 4f row fix (#337)`

## Task 2 — Failing regression test through `GeminiTTSAdapter`

**Goal:** Add a test that proves `first_greeting.tts_first_audio` fires for
the composable Gemini-TTS path. Test must fail before Task 3 lands.

**File:** `tests/adapters/test_gemini_tts_adapter.py`

**Test:** `test_synthesize_emits_first_greeting_tts_first_audio_on_first_frame`

Shape:

- Patch `robot_comic.adapters.gemini_tts_adapter.telemetry.emit_first_greeting_audio_once`
  (or `robot_comic.telemetry.emit_first_greeting_audio_once` depending on
  import shape) with a `MagicMock`.
- Build `_StubGeminiTTSHandler(tts_results=[_pcm_bytes(2400)])` (one 100 ms
  chunk).
- Run `[f async for f in adapter.synthesize("Hello.")]`.
- Assert the mock was called at least once and that `len(out) == 1`.

A *second* assertion: after a second call to `synthesize("World.")` on the
same adapter, the mock has still been called (we don't require dedupe at the
adapter layer because the helper itself dedupes).

**Verification:**

- `.venv/Scripts/pytest tests/adapters/test_gemini_tts_adapter.py::test_synthesize_emits_first_greeting_tts_first_audio_on_first_frame -q`
  — RED (helper never called).

**Commit:** part of Task 3 (see "Commit cadence" below — Task 2 and Task 3
land together so the repo never has a known-failing test on the branch).

## Task 3 — Minimum implementation: call the helper inside the yield loop

**Goal:** Make Task 2's test green with the smallest change.

**File:** `src/robot_comic/adapters/gemini_tts_adapter.py`

Inside the inner `_pcm_to_frames` loop in `synthesize`, immediately before
the `yield`, call `_emit_first_greeting_audio_once()` (delegating to
`telemetry.emit_first_greeting_audio_once`). Import `telemetry` lazily inside
the function to match the legacy `gemini_tts.py:411` pattern.

```python
for frame in GeminiTTSResponseHandler._pcm_to_frames(pcm_bytes):
    from robot_comic import telemetry as _telemetry

    _telemetry.emit_first_greeting_audio_once()
    yield AudioFrame(samples=frame, sample_rate=GEMINI_TTS_OUTPUT_SAMPLE_RATE)
```

Also: extend the adapter's `synthesize` docstring with a one-line note about
the emit and link to spec #321 + Lifecycle Hook #3b.

**Verification:**

- `.venv/Scripts/pytest tests/adapters/test_gemini_tts_adapter.py -q` — green.
- `.venv/Scripts/pytest tests/ -q` — green (with the documented
  `--ignore tests/vision/test_local_vision.py` if it collection-errors
  locally, and `--deselect` for the Windows audio flake).
- `uvx ruff@0.12.0 check .` — green.
- `uvx ruff@0.12.0 format --check .` — green.
- `.venv/Scripts/mypy --pretty src/robot_comic/adapters/gemini_tts_adapter.py tests/adapters/test_gemini_tts_adapter.py`
  — green.

**Commit:** `fix(adapters): GeminiTTSAdapter emits first_greeting.tts_first_audio on first PCM frame (#337) (lifecycle hook 3b)`

## Push + PR

- Push `claude/lifecycle-hook-3b-gemini-tts-first-greeting` to `origin`.
- Open PR titled `fix(adapters): GeminiTTSAdapter emits first_greeting.tts_first_audio (#337) (lifecycle hook 3b)`.

## Commit cadence note

Two commits total:

1. spec + plan + doc table fixup.
2. test + minimum implementation (lands together so HEAD is always green).
