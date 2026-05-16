# TDD plan — Phase 5e.4 (moonshine, gemini, chatterbox) decouple

Spec: `docs/superpowers/specs/2026-05-16-phase-5e-4-gemini-chatterbox-decouple.md`.
Branch: `claude/phase-5e-4-gemini-chatterbox-decouple`.

Mechanical mirror of 5e.3 for the gemini-chatterbox leaf. RED → GREEN,
conventional commits scoped `phase-5e-4`.

## Step 1 — Idempotency guard on `GeminiTextChatterboxResponseHandler` (RED → GREEN)

New test in `tests/test_chatterbox_tts.py` (the chatterbox tests file
gains the leaf-handler idempotency check; the leaf inherits from
`ChatterboxTTSResponseHandler` so keeping both idempotency tests in
one file mirrors the inheritance):

- `test_gemini_chatterbox_prepare_startup_credentials_is_idempotent` —
  patch `_probe_llama_health`, `_warmup_llm_kv_cache`, `_warmup_tts`
  with AsyncMocks (covers the inherited chatterbox guard's protected
  surface); patch `GeminiLLMClient` so its constructor is a counted
  spy; call `_prepare_startup_credentials` twice; assert
  `GeminiLLMClient` was instantiated exactly once, the same
  `_gemini_llm` instance survives, and the chatterbox warmups also
  fired only once.

Implement: add `_startup_credentials_ready` check at function top + set
to `True` at function bottom in
`src/robot_comic/gemini_text_handlers.py:70-87`.

Commit: `feat(phase-5e-4): add _startup_credentials_ready guard to GeminiTextChatterboxResponseHandler`.

## Step 2 — Factory rewrite + delete `_LocalSTTGeminiChatterboxHost` (RED → GREEN)

Failing tests (`tests/test_handler_factory.py`, mirror 5e.3's trio):

- `test_moonshine_gemini_chatterbox_uses_standalone_moonshine_adapter` —
  `result.pipeline.stt._handler is None`.
- `test_moonshine_gemini_chatterbox_wires_should_drop_frame_callback` —
  `_should_drop_frame is not None` and `_should_drop_frame() is False`.
- `test_moonshine_gemini_chatterbox_passes_deps_to_pipeline` —
  `result.pipeline.deps is mock_deps`.

Existing `test_gemini_llm_chatterbox_returns_gemini_chatterbox_handler`
(`tests/test_handler_factory_gemini_llm.py`) keeps passing.

Implement: rewrite `_build_composable_gemini_chatterbox` in
`handler_factory.py` as a mechanical mirror of
`_build_composable_llama_chatterbox` (substitute
`GeminiLLMAdapter` for `LlamaLLMAdapter` and
`GeminiTextChatterboxResponseHandler` for
`ChatterboxTTSResponseHandler`).
Delete `_LocalSTTGeminiChatterboxHost`.

Commit: `refactor(phase-5e-4): rewrite _build_composable_gemini_chatterbox without mixin host`.

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
git push -u origin claude/phase-5e-4-gemini-chatterbox-decouple
```

Manager opens the PR.
