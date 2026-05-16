# TDD plan — Phase 5e.3 (moonshine, llama, chatterbox) decouple

Spec: `docs/superpowers/specs/2026-05-16-phase-5e-3-llama-chatterbox-decouple.md`.
Branch: `claude/phase-5e-3-llama-chatterbox-decouple`.

Mechanical mirror of 5e.2. RED → GREEN, conventional commits scoped
`phase-5e-3`.

## Step 1 — Idempotency guard on `ChatterboxTTSResponseHandler` (RED → GREEN)

New test file `tests/test_chatterbox_tts.py`:

- `test_prepare_startup_credentials_is_idempotent` — patch
  `_probe_llama_health`, `_warmup_llm_kv_cache`, `_warmup_tts` with
  counted AsyncMocks; patch `super()._prepare_startup_credentials`
  call surface via mocking `tool_manager.start_up` so the base path
  is cheap. Call `_prepare_startup_credentials` twice; assert each
  warmup ran exactly once and `_http` is the same instance.

Implement: add `_startup_credentials_ready` check at function top + set
to `True` at function bottom in
`src/robot_comic/chatterbox_tts.py:125-144`.

Commit: `feat(phase-5e-3): add _startup_credentials_ready guard to ChatterboxTTSResponseHandler`.

## Step 2 — Factory rewrite + delete `_LocalSTTLlamaChatterboxHost` (RED → GREEN)

Failing tests (`tests/test_handler_factory.py`, mirror 5e.2's trio):

- `test_moonshine_llama_chatterbox_uses_standalone_moonshine_adapter` —
  `result.pipeline.stt._handler is None`.
- `test_moonshine_llama_chatterbox_wires_should_drop_frame_callback` —
  `_should_drop_frame is not None` and `_should_drop_frame() is False`.
- `test_moonshine_llama_chatterbox_passes_deps_to_pipeline` —
  `result.pipeline.deps is mock_deps`.

Existing `test_moonshine_chatterbox_default_llama_routes_to_composable`
keeps passing.

Implement: rewrite `_build_composable_llama_chatterbox` in
`handler_factory.py` as a mechanical mirror of
`_build_composable_llama_elevenlabs` (substitute
`ChatterboxTTSAdapter` for `ElevenLabsTTSAdapter` and
`ChatterboxTTSResponseHandler` for `LlamaElevenLabsTTSResponseHandler`).
Delete `_LocalSTTLlamaChatterboxHost`.

Commit: `refactor(phase-5e-3): rewrite _build_composable_llama_chatterbox without mixin host`.

## Step 3 — Lint / format / type / test (verification)

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/chatterbox_tts.py \
    src/robot_comic/handler_factory.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

Re-run known-flake tests if hit; do not fix them in this PR.

## Step 4 — Push

```
git push -u origin claude/phase-5e-3-llama-chatterbox-decouple
```

Manager opens the PR.
