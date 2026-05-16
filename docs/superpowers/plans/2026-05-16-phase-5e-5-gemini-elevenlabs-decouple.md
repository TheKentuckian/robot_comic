# TDD plan — Phase 5e.5 (moonshine, gemini, elevenlabs) decouple

Spec: `docs/superpowers/specs/2026-05-16-phase-5e-5-gemini-elevenlabs-decouple.md`.
Branch: `claude/phase-5e-5-gemini-elevenlabs-decouple`.

Mechanical mirror of 5e.4 for the gemini-elevenlabs leaf. RED → GREEN,
conventional commits scoped `phase-5e-5`.

## Step 1 — Idempotency guard on `GeminiTextElevenLabsResponseHandler` (RED → GREEN)

New test in `tests/test_elevenlabs_tts.py` (the elevenlabs tests file
gains the leaf-handler idempotency check; the leaf inherits from
`ElevenLabsTTSResponseHandler` so keeping the idempotency test in the
elevenlabs file mirrors the inheritance):

- `test_gemini_elevenlabs_prepare_startup_credentials_is_idempotent` —
  patch `google.genai.Client` and `GeminiLLMClient` so each
  constructor is a counted spy; mock `tool_manager.start_up`;
  call `_prepare_startup_credentials` twice; assert each constructor
  was invoked exactly once, `_client` / `_http` / `_gemini_llm`
  identity survives across calls, and `tool_manager.start_up` fired
  once.

Implement: wrap the whole leaf body in
`src/robot_comic/gemini_text_handlers.py:163-189` with a
`getattr(self, "_startup_credentials_ready", False)` early return +
set `self._startup_credentials_ready = True` at the end (success
path only).

Commit: `feat(phase-5e-5): add _startup_credentials_ready guard to GeminiTextElevenLabsResponseHandler`.

## Step 2 — Factory rewrite + delete `_LocalSTTGeminiElevenLabsHost` (RED → GREEN)

Failing tests (`tests/test_handler_factory.py`, mirror 5e.4's trio):

- `test_moonshine_gemini_elevenlabs_uses_standalone_moonshine_adapter` —
  `result.pipeline.stt._handler is None`.
- `test_moonshine_gemini_elevenlabs_wires_should_drop_frame_callback` —
  `_should_drop_frame is not None` and `_should_drop_frame() is False`.
- `test_moonshine_gemini_elevenlabs_passes_deps_to_pipeline` —
  `result.pipeline.deps is mock_deps`.

Existing `test_gemini_llm_elevenlabs_returns_gemini_elevenlabs_handler`
(`tests/test_handler_factory_gemini_llm.py`) keeps passing.

Implement: rewrite `_build_composable_gemini_elevenlabs` in
`handler_factory.py` as a mechanical mirror of
`_build_composable_gemini_chatterbox` (substitute
`GeminiTextElevenLabsResponseHandler` for
`GeminiTextChatterboxResponseHandler` and `ElevenLabsTTSAdapter` for
`ChatterboxTTSAdapter`).
Delete `_LocalSTTGeminiElevenLabsHost`.

Commit: `refactor(phase-5e-5): rewrite _build_composable_gemini_elevenlabs without mixin host`.

## Step 3 — Lint / format / type / test (verification)

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/gemini_text_handlers.py \
    src/robot_comic/handler_factory.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

Re-run known-flake tests if hit; do not fix them in this PR.

## Step 4 — Push

```
git push -u origin claude/phase-5e-5-gemini-elevenlabs-decouple
```

Manager opens the PR.
