# Phase 5e.5 — Migrate `(moonshine, gemini, elevenlabs)` off `LocalSTTInputMixin`

**Date:** 2026-05-16
**Status:** Spec — implementation on
`claude/phase-5e-5-gemini-elevenlabs-decouple`.
**Tracks:** epic #391; established pattern from
`docs/superpowers/specs/2026-05-16-phase-5e-2-llama-elevenlabs-decouple.md`
§4.
**Predecessor:** Phase 5e.4 (`#411`, migrated `(moonshine, gemini,
chatterbox)` off the mixin + landed the per-handler
`_startup_credentials_ready` guard on
:class:`GeminiTextChatterboxResponseHandler`).

---

## §1 — Scope

Mechanical mirror of 5e.2 / 5e.3 / 5e.4 for the `(moonshine, gemini,
elevenlabs)` triple. **No new design decisions.** The Protocol,
`ComposablePipeline` host-concern landing, `MoonshineSTTAdapter`
standalone wiring, `_clear_queue` double-mirror in
`ComposableConversationHandler`, and the
`_maybe_build_welcome_gate` factory helper are all already in place
from 5e.2 and unchanged here.

This PR:

1. Adds the per-handler `_startup_credentials_ready` idempotency guard
   to `GeminiTextElevenLabsResponseHandler._prepare_startup_credentials`
   so duplicate calls from the dual-adapter `prepare` chain don't
   re-instantiate the bundled `genai.Client` (built inside
   :class:`ElevenLabsTTSResponseHandler._prepare_startup_credentials`)
   or the new `GeminiLLMClient` wrapper at the leaf body.
2. Rewrites `_build_composable_gemini_elevenlabs` to construct a plain
   `GeminiTextElevenLabsResponseHandler` (no mixin shell), a standalone
   `MoonshineSTTAdapter(should_drop_frame=...)`, and passes `deps` +
   `welcome_gate` into `ComposablePipeline` — identical shape to 5e.4's
   `_build_composable_gemini_chatterbox`.
3. Deletes `_LocalSTTGeminiElevenLabsHost` (only call site is the
   builder we're rewriting).
4. Adds factory tests mirroring 5e.4: standalone STT,
   `should_drop_frame` wired, `pipeline.deps is mock_deps`.

The single remaining triple (`gemini_tts`) keeps its
`_LocalSTTGeminiTTSHost` mixin host unchanged. 5e.6 retires it.

## §2 — Gemini-elevenlabs-leaf-specific notes

### §2.1 — `_prepare_startup_credentials` guard placement

`GeminiTextElevenLabsResponseHandler._prepare_startup_credentials`
(`src/robot_comic/gemini_text_handlers.py:163-189`) does **not** use
`super()` — it explicitly calls
`ElevenLabsTTSResponseHandler._prepare_startup_credentials(self)`
because the diamond is resolved by hand (the docstring at lines
118-125 explains why: the bases are parallel hierarchies, not
cooperative-`super` ancestors).

This means the inherited base method is invoked unconditionally
every call — and the base method itself has **no** idempotency guard
(`elevenlabs_tts.py:333-344`), so a duplicate call would:

- Build a fresh `genai.Client` (the base's `self._client = genai.Client(...)`)
- Build a fresh `httpx.AsyncClient` (the base's `self._http = httpx.AsyncClient(timeout=30.0)`)
- Re-run `tool_manager.start_up(...)`
- Build a fresh `GeminiLLMClient` (the leaf's `self._gemini_llm = GeminiLLMClient(...)`)

The leaf-only guard wraps the **whole leaf body** — base call +
leaf-specific reassignments — so the second invocation is a cheap
no-op. This matches the 5e.4 pattern verbatim (the chatterbox leaf
also explicitly calls
`ChatterboxTTSResponseHandler._prepare_startup_credentials(self)`).

Post-5e.5:

```python
async def _prepare_startup_credentials(self) -> None:
    if getattr(self, "_startup_credentials_ready", False):
        return
    await ElevenLabsTTSResponseHandler._prepare_startup_credentials(self)
    if self._http is None:
        import httpx
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(...))
    self.tool_manager.start_up(tool_callbacks=[self._handle_tool_notification])

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
full chain. Matches 5e.2 / 5e.3 / 5e.4 contract verbatim.

### §2.2 — Factory builder rewrite

Identical shape to `_build_composable_gemini_chatterbox`. The
`GeminiTextElevenLabsResponseHandler` subclasses
`ElevenLabsTTSResponseHandler` so the `ElevenLabsTTSAdapter` Protocol
surface and `host._speaking_until` echo-guard read both work
unchanged (the leaf inherits `_speaking_until` semantics from the
ElevenLabs base, same as 5e.2's `LlamaElevenLabsTTSResponseHandler`).

The LLM adapter is `GeminiLLMAdapter` (already used by the legacy
builder); the only triple-specific substitution remains the LLM
adapter (same as 5e.4).

### §2.3 — Out of scope

- The last remaining triple (5e.6, `gemini_tts`).
- Touching `LocalSTTInputMixin`, `ComposablePipeline`,
  `MoonshineSTTAdapter`, `STTBackend` Protocol, or
  `BaseRealtimeHandler`.
- The `_clear_queue` double-mirror (5e.6 cleanup).
- Heartbeat / `MOONSHINE_DIAG` instrumentation (still punted, per 5e.1).

## §3 — Tests

New / updated assertions in `tests/test_handler_factory.py`:

- `test_moonshine_gemini_elevenlabs_uses_standalone_moonshine_adapter` —
  `result.pipeline.stt._handler is None`.
- `test_moonshine_gemini_elevenlabs_wires_should_drop_frame_callback` —
  `result.pipeline.stt._should_drop_frame is not None` and returns
  `False` at default (`_speaking_until` is 0.0).
- `test_moonshine_gemini_elevenlabs_passes_deps_to_pipeline` —
  `result.pipeline.deps is mock_deps`.

Existing `test_gemini_llm_elevenlabs_returns_gemini_elevenlabs_handler`
(`tests/test_handler_factory_gemini_llm.py`) keeps passing (asserts
`isinstance(_tts_handler, GeminiTextElevenLabsResponseHandler)`; the
plain handler is what the wrapper now wraps).

New leaf-handler idempotency test in `tests/test_elevenlabs_tts.py`
(mirror 5e.4's leaf test in `tests/test_chatterbox_tts.py`):

- `test_gemini_elevenlabs_prepare_startup_credentials_is_idempotent` —
  patch `genai.Client` and `GeminiLLMClient`; call
  `_prepare_startup_credentials` twice; assert each constructor was
  invoked exactly once, the `_client` / `_http` / `_gemini_llm`
  instances survive across calls, and `tool_manager.start_up` fired
  once.

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
- `test_gemini_turn_buffers_transcripts_and_schedules_motion_reset`

## §5 — Risk

Low. The pattern was proven by 5e.2 and re-proven by 5e.3 / 5e.4.
The leaf shares the explicit-base-call shape that 5e.4's chatterbox
leaf used, and the guard wrapping the whole body is the documented
mitigation for that shape.

The base `ElevenLabsTTSResponseHandler._prepare_startup_credentials`
has no idempotency guard of its own (unlike `ChatterboxTTSResponseHandler`
post-5e.3), so this PR's leaf guard is the only thing standing
between duplicate `prepare` calls and a `genai.Client` /
`httpx.AsyncClient` leak. That base method has no other migrated
call sites today — when `LlamaElevenLabsTTSResponseHandler` runs it
goes through that handler's own guard (5e.2 added one to the llama
leaf instead of the base). 5e.6 may add a base-level guard once
every triple is migrated, but that's a future-proofing decision, not
a 5e.5 requirement.

## §6 — Diff budget

Target: ≤300 LOC across all touched files (well under the 500 LOC
trip-wire from the brief).

- `gemini_text_handlers.py`: +4 LOC (guard at top + flag at bottom).
- `handler_factory.py`: ~net parity (delete `_LocalSTTGeminiElevenLabsHost`
  + rewrite `_build_composable_gemini_elevenlabs`).
- `tests/test_handler_factory.py`: +50 LOC (three new tests).
- `tests/test_elevenlabs_tts.py`: +50 LOC (one idempotency test).

Total: ~110 LOC. Pattern is mechanical; if it grows past 300 LOC,
stop and report.
