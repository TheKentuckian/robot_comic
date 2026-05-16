# Phase 5e.4 — Migrate `(moonshine, gemini, chatterbox)` off `LocalSTTInputMixin`

**Date:** 2026-05-16
**Status:** Spec — implementation on
`claude/phase-5e-4-gemini-chatterbox-decouple`.
**Tracks:** epic #391; established pattern from
`docs/superpowers/specs/2026-05-16-phase-5e-2-llama-elevenlabs-decouple.md`
§4.
**Predecessor:** Phase 5e.3 (`#409`, migrated `(moonshine, llama,
chatterbox)` off the mixin + landed the per-handler
`_startup_credentials_ready` guard on
:class:`ChatterboxTTSResponseHandler`).

---

## §1 — Scope

Mechanical mirror of 5e.2 / 5e.3 for the `(moonshine, gemini, chatterbox)`
triple. **No new design decisions.** The Protocol, `ComposablePipeline`
host-concern landing, `MoonshineSTTAdapter` standalone wiring, and
`_clear_queue` double-mirror in `ComposableConversationHandler` are
all already in place from 5e.2 and unchanged here.

This PR:

1. Adds the per-handler `_startup_credentials_ready` idempotency guard
   to `GeminiTextChatterboxResponseHandler._prepare_startup_credentials`
   so the dual-adapter `prepare` calls don't re-instantiate the
   `GeminiLLMClient` (which wraps a `genai.Client` constructed in the
   underlying SDK) on every call.
2. Rewrites `_build_composable_gemini_chatterbox` to construct a plain
   `GeminiTextChatterboxResponseHandler` (no mixin shell), a standalone
   `MoonshineSTTAdapter(should_drop_frame=...)`, and passes `deps` +
   `welcome_gate` into `ComposablePipeline` — identical shape to 5e.3's
   `_build_composable_llama_chatterbox`.
3. Deletes `_LocalSTTGeminiChatterboxHost` (only call site is the
   builder we're rewriting).
4. Adds factory tests mirroring 5e.3: standalone STT, `should_drop_frame`
   wired, `pipeline.deps is mock_deps`.

The other two remaining triples (`gemini_elevenlabs`, `gemini_tts`)
keep their `_LocalSTT*Host` mixin hosts unchanged. 5e.5–5e.6 retire
them one at a time.

## §2 — Gemini-text-leaf-specific notes

### §2.1 — `_prepare_startup_credentials` guard placement

`GeminiTextChatterboxResponseHandler._prepare_startup_credentials`
chains through `ChatterboxTTSResponseHandler` and
`BaseLlamaResponseHandler` (`src/robot_comic/gemini_text_handlers.py:70-87`):

```python
async def _prepare_startup_credentials(self) -> None:
    await ChatterboxTTSResponseHandler._prepare_startup_credentials(self)
    # Reassigns self._gemini_llm with a new GeminiLLMClient on every call.
    ...
    self._gemini_llm = GeminiLLMClient(api_key=api_key, model=model)
    logger.info(...)
```

5e.3 already added the guard on `ChatterboxTTSResponseHandler` so the
first half of the chain is idempotent. The leaf still needs its own
guard because its `super()`-chain succeeds (the chatterbox guard
returns early) but the leaf-body's `self._gemini_llm = GeminiLLMClient(...)`
runs on every call regardless, leaking a fresh SDK client.

Post-5e.4:

```python
async def _prepare_startup_credentials(self) -> None:
    if getattr(self, "_startup_credentials_ready", False):
        return
    await ChatterboxTTSResponseHandler._prepare_startup_credentials(self)
    from robot_comic.config import config
    from robot_comic.gemini_llm import GeminiLLMClient
    from robot_comic.gemini_text_base import _DEFAULT_GEMINI_LLM_MODEL

    api_key = getattr(config, "GEMINI_API_KEY", None) or "DUMMY"
    model = getattr(config, "GEMINI_LLM_MODEL", _DEFAULT_GEMINI_LLM_MODEL)
    self._gemini_llm = GeminiLLMClient(api_key=api_key, model=model)
    logger.info(...)
    self._startup_credentials_ready = True
```

Flag is set only on the success path — failed runs re-attempt the
full chain. Matches 5e.2 / 5e.3 contract verbatim.

### §2.2 — Factory builder rewrite

Identical shape to `_build_composable_llama_chatterbox`. The
`GeminiTextChatterboxResponseHandler` subclasses
`ChatterboxTTSResponseHandler`, so the `ChatterboxTTSAdapter` Protocol
surface and the `host._speaking_until` echo-guard read both work
unchanged (handler inherits `_enqueue_audio_frame` from
`BaseLlamaResponseHandler` via the chatterbox base).

The LLM adapter is `GeminiLLMAdapter` (not `LlamaLLMAdapter`) — that
is the only triple-specific substitution in the builder body.

### §2.3 — Out of scope

- Other triples (5e.5-5e.6).
- Touching `LocalSTTInputMixin`, `ComposablePipeline`,
  `MoonshineSTTAdapter`, `STTBackend` Protocol, or
  `BaseRealtimeHandler`.
- The `_clear_queue` double-mirror (5e.6 cleanup).
- Heartbeat / `MOONSHINE_DIAG` instrumentation (still punted, per 5e.1).

## §3 — Tests

New / updated assertions in `tests/test_handler_factory.py`:

- `test_moonshine_gemini_chatterbox_uses_standalone_moonshine_adapter` —
  `result.pipeline.stt._handler is None`.
- `test_moonshine_gemini_chatterbox_wires_should_drop_frame_callback` —
  `result.pipeline.stt._should_drop_frame is not None` and returns
  `False` at default (`_speaking_until` is 0.0).
- `test_moonshine_gemini_chatterbox_passes_deps_to_pipeline` —
  `result.pipeline.deps is mock_deps`.

Existing `test_gemini_llm_chatterbox_returns_gemini_chatterbox_handler`
(`tests/test_handler_factory_gemini_llm.py`) keeps passing (asserts
`isinstance(_tts_handler, GeminiTextChatterboxResponseHandler)`; the
plain handler is what the wrapper now wraps).

New tests in `tests/test_chatterbox_tts.py` (mirror 5e.3's idempotency
test for the leaf):

- `test_gemini_chatterbox_prepare_startup_credentials_is_idempotent` —
  second call must NOT re-instantiate `self._gemini_llm`. Also
  validates the inherited chatterbox guard short-circuits the chained
  `super()` call.

## §4 — Verification

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/gemini_text_handlers.py \
    src/robot_comic/handler_factory.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

Known flakes (do NOT fix; re-run if hit):

- `test_huggingface_realtime::test_run_realtime_session_passes_allocated_session_query`
- `test_openai_realtime::test_openai_excludes_head_tracking_when_no_head_tracker`
- `test_handler_factory::test_moonshine_openai_realtime_output`,
  `test_moonshine_hf_output`

## §5 — Risk

Low. The pattern was proven by 5e.2 and re-proven by 5e.3. The leaf
handler shares the `ChatterboxTTSResponseHandler` ancestry, so the
inherited guard handles the heavy warmup chain; the new leaf guard
only adds idempotency around the `GeminiLLMClient` reassignment plus
the `logger.info` line.

The `super()`-chain interaction with 5e.3's chatterbox guard is benign:
the leaf calls `ChatterboxTTSResponseHandler._prepare_startup_credentials`
explicitly (not `super()`); the chatterbox method checks its own flag
and returns early on the second call — exactly the cooperation 5e.3
designed for.

## §6 — Diff budget

Target: ≤300 LOC across all touched files (well under the 600 LOC
trip-wire from the brief).

- `gemini_text_handlers.py`: +4 LOC (guard).
- `handler_factory.py`: ~net parity (delete class + rewrite builder).
- `tests/test_handler_factory.py`: +50 LOC (three new tests).
- `tests/test_chatterbox_tts.py`: +50 LOC (one idempotency test).

Total: ~110 LOC. Pattern is mechanical; if it grows past 300 LOC,
stop and report.
