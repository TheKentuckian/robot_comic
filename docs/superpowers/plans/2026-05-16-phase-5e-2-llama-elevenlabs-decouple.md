# TDD plan — Phase 5e.2 (moonshine, llama, elevenlabs) decouple

Spec: `docs/superpowers/specs/2026-05-16-phase-5e-2-llama-elevenlabs-decouple.md`.
Branch: `claude/phase-5e-2-llama-elevenlabs-decouple`.

RED → GREEN → REFACTOR. Conventional commits scoped `phase-5e-2`.

## Step 1 — Extend `STTBackend` Protocol (RED → GREEN)

Add `on_partial` and `on_speech_started` optional kwargs to
`STTBackend.start` in `backends.py`. Update the existing in-memory
test reference implementations in `tests/test_backends.py` (or
wherever the Protocol tests live) so they keep compiling.

Test first: a small assertion that `STTBackend.start` accepts the new
kwargs without raising.

Commit: `feat(phase-5e-2): extend STTBackend.start with on_partial/on_speech_started`.

## Step 2 — `MoonshineSTTAdapter` standalone-mode callbacks (RED → GREEN)

Add failing tests (`tests/adapters/test_moonshine_stt_adapter.py`):

- `test_standalone_start_invokes_on_partial_for_partial_event`
- `test_standalone_start_invokes_on_speech_started_for_started_event`
- `test_standalone_start_does_not_invoke_partial_for_completed`
- `test_standalone_start_does_not_invoke_speech_started_for_partial`
- `test_standalone_start_with_no_partial_callback_does_not_crash_on_partial_event`

Implement: extend `MoonshineSTTAdapter.start` + `_start_standalone` to
accept the new kwargs and route events accordingly.

Commit: `feat(phase-5e-2): route partial + speech-started events on standalone MoonshineSTTAdapter`.

## Step 3 — `should_drop_frame` callback (RED → GREEN)

Add failing tests:

- `test_standalone_should_drop_frame_when_callback_returns_true`
- `test_standalone_should_drop_frame_when_callback_returns_false`
- `test_standalone_should_drop_frame_callback_not_called_when_handler_provided`

Implement: add `should_drop_frame` kwarg to `MoonshineSTTAdapter.__init__`.
Consult in `feed_audio` standalone branch only.

Commit: `feat(phase-5e-2): add should_drop_frame callback to standalone MoonshineSTTAdapter`.

## Step 4 — `ComposablePipeline` host-concern landing (RED → GREEN)

Add failing tests (`tests/test_composable_pipeline.py`):

- `test_speech_started_callback_opens_turn_span`
- `test_speech_started_callback_calls_set_listening_true_when_deps_provided`
- `test_speech_started_callback_calls_head_wobbler_reset_when_deps_provided`
- `test_speech_started_callback_calls_clear_queue_when_set`
- `test_speech_started_no_deps_no_op`
- `test_partial_callback_publishes_user_partial_to_output_queue`
- `test_partial_callback_does_not_publish_empty_string`
- `test_completed_callback_publishes_user_to_output_queue_when_deps_provided`
- `test_completed_callback_records_user_transcript_when_deps_provided`
- `test_completed_callback_calls_set_listening_false_when_deps_provided`
- `test_completed_callback_suppresses_duplicate_within_window`
- `test_completed_callback_pause_controller_handled_drops_transcript`
- `test_completed_callback_pause_controller_dispatch_proceeds`
- `test_completed_callback_welcome_gate_waiting_drops_on_no_match`
- `test_completed_callback_welcome_gate_waiting_opens_on_match_and_dispatches`
- `test_completed_callback_welcome_gate_gated_dispatches_immediately`

Implement on `ComposablePipeline`:

- New `__init__` kwargs: `deps`, `welcome_gate`. New fields.
- New `_on_speech_started` async method.
- New `_on_partial_transcript` async method.
- Extend `_on_transcript_completed` with duplicate-suppression,
  user-publish, set_listening(False), record_user_transcript,
  pause-controller, welcome-gate, stt-infer span close.
- Wire all three callbacks through `start_up` to `self.stt.start(...)`.

All "when deps provided" branches gracefully no-op when `deps is None`,
preserving today's behaviour for callers that haven't migrated.

Commit: `feat(phase-5e-2): land mixin host concerns on ComposablePipeline behind deps kwarg`.

## Step 5 — `_clear_queue` pipeline mirror (RED → GREEN)

Test (`tests/test_composable_conversation_handler.py`):

- `test_clear_queue_setter_mirrors_onto_pipeline`.
- Existing `_tts_handler` mirror test stays unchanged (still required
  for un-migrated triples).

Implement: extend the setter in `composable_conversation_handler.py`
to also write to `self.pipeline._clear_queue`.

Commit: `feat(phase-5e-2): mirror _clear_queue onto ComposablePipeline in wrapper`.

## Step 6 — Idempotency guard on `LlamaElevenLabsTTSResponseHandler` (RED → GREEN)

Test (`tests/test_llama_elevenlabs_tts.py` or new file if absent):

- `test_prepare_startup_credentials_is_idempotent`.

Implement: add `_startup_credentials_ready` check + flip at end of
`LlamaElevenLabsTTSResponseHandler._prepare_startup_credentials`.

Commit: `fix(phase-5e-2): guard LlamaElevenLabsTTSResponseHandler._prepare_startup_credentials idempotency`.

## Step 7 — Factory rewrite + delete `_LocalSTTLlamaElevenLabsHost` (RED → GREEN)

Failing tests (`tests/test_handler_factory.py`):

- `test_moonshine_llama_elevenlabs_uses_standalone_moonshine_adapter` —
  assert `result.pipeline.stt._handler is None`.
- `test_moonshine_llama_elevenlabs_wires_should_drop_frame` — assert
  `result.pipeline.stt._should_drop_frame is not None`.
- Existing `test_moonshine_elevenlabs_default_llama_routes_to_composable`
  passes unchanged (asserts `isinstance(_tts_handler,
  LlamaElevenLabsTTSResponseHandler)`).

Implement: rewrite `_build_composable_llama_elevenlabs`; add
`_maybe_build_welcome_gate` helper; delete `_LocalSTTLlamaElevenLabsHost`.

Commit: `refactor(phase-5e-2): rewrite _build_composable_llama_elevenlabs without mixin host`.

## Step 8 — Lint / format / type / test (verification)

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/composable_pipeline.py \
    src/robot_comic/adapters/moonshine_stt_adapter.py \
    src/robot_comic/backends.py \
    src/robot_comic/handler_factory.py
.venv/bin/python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py
```

Fix any issues; commit per fix. Re-run known-flake tests if they hit.

## Step 9 — Push

```
git push -u origin claude/phase-5e-2-llama-elevenlabs-decouple
```

Do NOT open the PR — manager opens it.
