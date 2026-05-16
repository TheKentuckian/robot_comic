# Phase 5c.2 — TDD plan: move `apply_personality` onto `ComposablePipeline`

**Spec:** `docs/superpowers/specs/2026-05-16-phase-5c2-apply-personality-to-pipeline.md`
**Branch:** `claude/phase-5c-2-apply-personality-to-pipeline`
**Commit scope prefix:** `phase-5c-2`

TDD cadence: each task adds the failing test(s) and the production
change together in one commit. The plan is structured so each commit
leaves the test suite green and lint/types clean.

## Commit 1: `feat(phase-5c-2): add reset_per_session_state to TTSBackend Protocol`

**Why first:** the pipeline's `apply_personality` (Commit 2) calls
`await self.tts.reset_per_session_state()`. The Protocol contract has to
exist before the pipeline can rely on it.

**Production:**
- `src/robot_comic/backends.py`: add
  ```python
  async def reset_per_session_state(self) -> None:
      """..."""
      return None
  ```
  to the `TTSBackend` Protocol, with a docstring explaining the "default
  is no-op, opt-in per-adapter" semantics.

**Tests:**
- `tests/test_backends_protocols.py`:
  - Extend `_MockTTS` with `async def reset_per_session_state(self): return None`
    so `test_mock_tts_satisfies_protocol` keeps passing.
  - Add a new test `test_tts_protocol_reset_per_session_state_default_is_noop`
    that constructs a Protocol-typed adapter (the existing `_MockTTS` is
    a satisfier without an override, so it inherits nothing — write a
    minimal "structural-only" class that omits the method and confirm it
    does NOT pass `isinstance(..., TTSBackend)`; then a subclass that
    inherits from `TTSBackend` and inherits the default returns None).

**Lint/format/types:**
- `uvx ruff@0.12.0 check .`
- `uvx ruff@0.12.0 format --check .`
- `mypy --pretty --show-error-codes src/robot_comic/backends.py`

## Commit 2: `feat(phase-5c-2): implement reset_per_session_state on all three TTS adapters`

**Why next:** the Phase 5a.1 regression test
(`tests/test_composable_persona_reset.py`) currently asserts the wrapper
zeroes the three echo-guard fields. After 5c.2 the adapter is the call
site; this commit lets the adapter satisfy the test.

**Production:**
- `src/robot_comic/adapters/elevenlabs_tts_adapter.py`: add
  `reset_per_session_state` method (5-line hasattr-guarded setattr loop).
- `src/robot_comic/adapters/chatterbox_tts_adapter.py`: same.
- `src/robot_comic/adapters/gemini_tts_adapter.py`: same.

**Tests:**
- `tests/adapters/test_tts_backend_contract.py`:
  - Extend `_ElevenLabsStub`, `_ChatterboxStub`, `_GeminiStub` to
    initialise `_speaking_until=0.0`, `_response_start_ts=0.0`,
    `_response_audio_bytes=0` as instance attributes (so the hasattr
    guard finds them).
  - Add a parametric test
    `test_reset_per_session_state_zeros_echo_guard_fields` that sets the
    three fields to non-zero values on the stub, awaits
    `adapter.reset_per_session_state()`, then asserts all three are
    zero.
  - Add a parametric test
    `test_reset_per_session_state_no_op_when_fields_absent` that builds
    a stub-replacement object lacking the three fields (e.g. constructed
    via `MagicMock(spec=[])`), wraps it in the adapter type, and
    confirms the call neither raises nor creates the fields. (For
    type-strictness in the wrapper construction, the test casts via
    `# type: ignore[arg-type]`.)

**Lint/format/types:** repo-root ruff + adapter-scoped mypy.

## Commit 3: `feat(phase-5c-2): move apply_personality from wrapper onto ComposablePipeline`

**Why next:** the new pipeline method exists and the adapter contract is
ready; this commit lights up the new call path.

**Production:**
- `src/robot_comic/composable_pipeline.py`:
  - Add imports for `set_custom_profile` (from `robot_comic.config`) and
    `get_session_instructions` (from `robot_comic.prompts`).
  - Add `async def apply_personality(self, profile)` method with the
    body shown in the spec.

**Tests:**
- `tests/test_composable_pipeline.py`:
  - `test_apply_personality_resets_history_and_reseeds_system_prompt`
    — pre-populate history with system + user; monkeypatch
    `set_custom_profile` and `get_session_instructions`; assert final
    history is `[{"role": "system", "content": "fresh"}]`.
  - `test_apply_personality_awaits_tts_reset_per_session_state` —
    a `_TrackingTTS` that records `reset_per_session_state` was awaited;
    assert exactly one await on the success path.
  - `test_apply_personality_returns_failure_message_on_set_custom_profile_error`
    — monkeypatch `set_custom_profile` to raise; assert return string
    contains "Failed to apply personality" and the exception text;
    assert `reset_per_session_state` was NOT awaited; assert
    `reset_history` was NOT called (history is untouched).
  - `test_apply_personality_orders_reset_history_before_tts_reset_and_reseed`
    — sequence-recording mock TTS + spy on `reset_history`; assert
    ordering: `reset_history`, then `tts.reset_per_session_state`, then
    history-append. Mirrors Phase 5a.1's "test_orders_history_reset_..."
    test from the wrapper level.

**Lint/format/types:** repo-root ruff + pipeline-scoped mypy.

## Commit 4: `refactor(phase-5c-2): shrink wrapper.apply_personality to pass-through; drop _reset_tts_per_session_state`

**Why next:** the pipeline now owns the behaviour; the wrapper sheds the
duplicate.

**Production:**
- `src/robot_comic/composable_conversation_handler.py`:
  - Replace `async def apply_personality` body with
    `return await self.pipeline.apply_personality(profile)`.
  - Delete the `_reset_tts_per_session_state` method (no longer
    called).
  - Drop now-unused imports:
    `from robot_comic.config import set_custom_profile`,
    `from robot_comic.prompts import get_session_instructions`.
  - Update the class-level docstring's lifecycle-hook bullet for
    `_speaking_until` (point to `ComposablePipeline.apply_personality`
    instead of "this class").
  - Update the `get_available_voices` docstring reference to
    `_reset_tts_per_session_state` (now "still consulted by the
    `_clear_queue` setter — Phase 5d will revisit").
  - Leave `self._tts_handler = tts_handler` and the `_clear_queue`
    setter mirror untouched (Phase 5d territory).

**Tests:**
- `tests/test_composable_conversation_handler.py`:
  - Delete:
    - `test_apply_personality_resets_history_and_reseeds`
    - `test_apply_personality_invokes_tts_reset_helper`
    - `test_apply_personality_skips_tts_reset_on_set_custom_profile_failure`
    - `test_apply_personality_returns_failure_message_on_error`
  - Add:
    - `test_apply_personality_forwards_to_pipeline` — assert the wrapper
      `await`s `self.pipeline.apply_personality(profile)` and returns
      its result verbatim. Use `AsyncMock` on
      `wrapper.pipeline.apply_personality` returning a sentinel string.

**Lint/format/types:** repo-root ruff + scoped mypy.

## Commit 5: `test(phase-5c-2): retarget persona-reset regression at pipeline + adapter`

**Why next:** the existing regression test in
`test_composable_persona_reset.py` constructs a wrapper with a real
ElevenLabs handler and a MagicMock pipeline. With the wrapper now
forwarding to a mock pipeline, the test would no longer exercise the
real adapter/handler reset. Retarget at `ComposablePipeline` directly
with a real adapter wrapping the real handler.

**Production:** (none; tests only)

**Tests:**
- `tests/test_composable_persona_reset.py`:
  - `test_apply_personality_clears_tts_handler_echo_guard_state` —
    construct a real `ElevenLabsTTSResponseHandler`, wrap it in a real
    `ElevenLabsTTSAdapter`, construct a `ComposablePipeline` with stubs
    for STT and LLM and the real adapter as `tts`, set the three
    echo-guard fields non-zero, await `pipeline.apply_personality(...)`,
    assert all three fields are zero on the real handler.
  - `test_apply_personality_no_op_on_handler_without_echo_guard_fields`
    — replace the MagicMock-based wrapper test with one that constructs
    a `_GeminiStub`-shaped handler (no echo-guard fields), wraps it in
    `GeminiTTSAdapter`, awaits
    `pipeline.apply_personality(...)`, and asserts the call neither
    raises nor adds the fields.
  - Update the module docstring to reflect the new shape (the test now
    exercises pipeline + adapter, not the wrapper directly).

**Lint/format/types:** repo-root ruff + scoped mypy.

## Commit 6: `docs(phase-5c-2): note migration in 5a.1 spec + bump exploration memo §2.3 status`

**Why last:** with the code shipped, drop a one-paragraph "Phase 5c.2
moved this onto `ComposablePipeline`" note into the 5a.1 spec
("Migration note" subsection) and tick the §2.3 status in the
exploration memo from "wrapper owns this; Phase 5c.2 will fix" to
"5c.2 (#TBD) shipped — `apply_personality` now on
`ComposablePipeline`."

**Production:** docs only.

**Tests:** none.

**Lint/format/types:** none required for docs.

## Verification gates between commits

After each commit:

```
uvx ruff@0.12.0 check .
uvx ruff@0.12.0 format --check .
.venv/bin/mypy --pretty --show-error-codes \
    src/robot_comic/backends.py \
    src/robot_comic/composable_pipeline.py \
    src/robot_comic/composable_conversation_handler.py \
    src/robot_comic/adapters/
.venv/bin/python -m pytest tests/ -q \
    --ignore=tests/vision/test_local_vision.py
```

Per the brief, 1–3 pre-existing xdist flakes
(`test_huggingface_realtime`, `test_gemini_live`,
`test_openai_realtime`, `test_handler_factory`) may surface in the full
run. If they don't reproduce in serial isolation (`-p no:xdist`),
they're pre-existing — note in the report and don't fix here.

## Open questions / decisions taken upfront

1. **`reset_per_session_state` default body** — `return None` (no-op),
   not `raise NotImplementedError`. Voice methods raise because they're
   operator-facing; per-session reset is internal cleanup where a no-op
   is correct for backends without state to reset.
2. **Sync vs async** — async. Matches the rest of the Protocol's
   per-session lifecycle surface and lets future adapters await
   network/IO without a Protocol breaking change.
3. **`_tts_handler` retained** — yes, because `_clear_queue.setter`
   still mirrors onto it. Phase 5d's `ConversationHandler` ABC shrink
   is the right home for that cleanup.
4. **Adapter implementation locality** — per-adapter method, not a
   shared mixin/helper. The three bodies are identical today but the
   field knowledge belongs on the adapter, and a future per-adapter
   field divergence is easier to apply when the body lives there.
5. **Test split** — wrapper tests assert "forwards to pipeline";
   pipeline tests assert behavioural outcomes (history reset, system
   re-seed, TTS reset, error path); adapter contract tests assert the
   reset zeros the fields when present.
