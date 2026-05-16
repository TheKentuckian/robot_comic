# Phase 5c.2 — Move `apply_personality` onto `ComposablePipeline`

**Date:** 2026-05-16
**Epic:** #391 (Phase 5)
**Predecessor:** Phase 5c.1 (PR #399 — `TTSBackend` voice methods)
**Exploration anchor:** `docs/superpowers/specs/2026-05-16-phase-5-exploration.md` §2.3 + §3.1
**Branch:** `claude/phase-5c-2-apply-personality-to-pipeline`
**Status:** Spec for implementation.

## Motivation

`ComposableConversationHandler.apply_personality`
(`composable_conversation_handler.py:172-199`) does pipeline-shaped state
surgery from the wrapper:

1. `set_custom_profile(profile)` — global profile dial flip.
2. `self.pipeline.reset_history(keep_system=False)` — wipe conversation
   history including the system prompt.
3. `self._reset_tts_per_session_state()` (Phase 5a.1) — clear echo-guard
   accumulators (`_speaking_until`, `_response_start_ts`,
   `_response_audio_bytes`) on the wrapped TTS handler via
   `hasattr`-guarded `setattr`.
4. Re-seed `pipeline._conversation_history` with a fresh
   `{"role": "system", "content": get_session_instructions()}`.
5. Return a confirmation string.

Steps 2 and 4 reach directly into `self.pipeline._conversation_history`;
step 3 reaches into `self._tts_handler`. Both are surgery the wrapper
performs *through* the pipeline rather than *on* the pipeline. The wrapper
exists to translate FastRTC handler shape ↔ pipeline shape; persona-switch
state management is not FastRTC-shaped.

Phase 5c.2 moves the personality logic onto `ComposablePipeline` where it
belongs structurally. The wrapper becomes a thin pass-through. The TTS
per-session-reset moves onto `TTSBackend` Protocol so the pipeline can
trigger it without holding a legacy-handler reference.

## Scope (this PR)

1. **Add `apply_personality(profile: str | None) -> str` to
   `ComposablePipeline`** in `src/robot_comic/composable_pipeline.py`.
   Behaviour mirrors today's wrapper:
   - `set_custom_profile(profile)` (catch + return failure message).
   - `self.reset_history(keep_system=False)`.
   - `await self.tts.reset_per_session_state()`.
   - Append fresh `{"role": "system", "content": get_session_instructions()}`
     to `self._conversation_history`.
   - Return `f"Applied personality {profile!r}. Conversation history reset."`.

2. **Add `reset_per_session_state()` to the `TTSBackend` Protocol** in
   `src/robot_comic/backends.py` as an `async` method with a default
   body that is a clean no-op (pipeline-debug stubs or future TTS
   backends without per-session state opt out for free — unlike the
   voice methods, where the default raises). The legacy contract is
   "best-effort cleanup," not "must be supported."

3. **Implement `reset_per_session_state()` on each TTS adapter**
   (`ElevenLabsTTSAdapter`, `ChatterboxTTSAdapter`, `GeminiTTSAdapter`)
   as a thin forward that clears `_speaking_until`, `_response_start_ts`,
   and `_response_audio_bytes` on the wrapped handler with `hasattr`
   guards — same pattern the wrapper's `_reset_tts_per_session_state`
   uses today.

4. **Shrink the wrapper's `apply_personality`** to a thin pass-through:
   ```python
   async def apply_personality(self, profile: str | None) -> str:
       return await self.pipeline.apply_personality(profile)
   ```

5. **Delete the wrapper's `_reset_tts_per_session_state` helper.** Its
   only caller was the wrapper's own `apply_personality`; the
   per-session-state reset is now per-adapter.

6. **Extend the parametric contract suite** at
   `tests/adapters/test_tts_backend_contract.py` with a
   `reset_per_session_state` contract test: when the wrapped handler has
   the three echo-guard fields, the adapter zeroes them; when the
   handler lacks them, the call is a clean no-op.

7. **Migrate the wrapper tests** in
   `tests/test_composable_conversation_handler.py`:
   - The three existing
     `test_apply_personality_{resets_history_and_reseeds,
     invokes_tts_reset_helper,
     skips_tts_reset_on_set_custom_profile_failure,
     returns_failure_message_on_error}` tests pin behaviour that now
     lives on the pipeline. Replace them with one focused
     "forwards apply_personality to pipeline" test on the wrapper, and
     move the behavioural assertions to `tests/test_composable_pipeline.py`.

8. **Add `test_composable_pipeline.py` cases** covering
   `pipeline.apply_personality`:
   - History is reset (system entry plus user turn → only fresh system).
   - System prompt re-seeded from `get_session_instructions()`.
   - `tts.reset_per_session_state` is awaited exactly once on the
     success path, after `reset_history`.
   - When `set_custom_profile` raises, the method returns a failure
     string and does NOT call `reset_history` or
     `tts.reset_per_session_state`.

9. **Update `tests/test_composable_persona_reset.py`** — the existing
   regression test constructs a wrapper with a real
   `ElevenLabsTTSResponseHandler` and asserts the three echo-guard
   fields zero out after `wrapper.apply_personality(...)`. After 5c.2,
   that behaviour fires through `pipeline.apply_personality` →
   `tts.reset_per_session_state` (the adapter's forward). The test
   stays meaningful — it just needs a real `ComposablePipeline` with a
   real `ElevenLabsTTSAdapter` in place of the MagicMock pipeline. The
   adapter's `reset_per_session_state` is the call site that zeroes the
   fields, so this is the right level of regression coverage.

   Two test cases in that file:
   - `test_apply_personality_clears_tts_handler_echo_guard_state` —
     rewrite to use a real pipeline + adapter, drive
     `wrapper.apply_personality` (which forwards), assert the same
     end-state on the real handler instance.
   - `test_apply_personality_no_op_on_handler_without_echo_guard_fields`
     — equivalent reframe using a Gemini-shaped stub handler wrapped by
     `GeminiTTSAdapter`.

10. **Extend `tests/test_backends_protocols.py::_MockTTS`** with a
    `reset_per_session_state` method (or omit it and rely on the
    Protocol default — see "Protocol default semantics" below). Either
    way the `isinstance(_MockTTS(), TTSBackend)` assertion must keep
    passing.

## Non-goals (explicit)

- Do NOT touch `ConversationHandler` ABC (`apply_personality` is still
  an abstract method on the ABC; the wrapper still satisfies it via the
  pass-through). The ABC shrink is Phase 5d.
- Do NOT touch bundled-realtime handlers (`HuggingFaceRealtimeHandler`,
  `OpenaiRealtimeHandler`, `GeminiLiveHandler`,
  `LocalSTT*RealtimeHandler`) — they don't use `ComposablePipeline`.
- Do NOT modify voice-method behaviour or `TTSBackend` voice methods
  (Phase 5c.1 just shipped those; leave them).
- Do NOT drop the `_tts_handler` reference from the wrapper — see
  "`_tts_handler` retention decision" below.
- Do NOT add tag-channel work, factory STT decoupling, or any other
  sub-phase's deliverables. **Stay within 5c.2's scope.** Reference
  `docs/superpowers/memory/feedback_test_infra_agent_scope_creep.md`.

## `_tts_handler` retention decision

The brief asks: drop the `self._tts_handler` reference from the wrapper
if (and only if) nothing else on the wrapper still uses it after 5c.2.

**Audit (`Grep _tts_handler src/robot_comic/composable_conversation_handler.py`):**

| Line | Site | Purpose | Survives 5c.2? |
|---|---|---|---|
| `__init__` (51) | `self._tts_handler = tts_handler` | Store reference | depends |
| `_clear_queue.setter` (95-96) | `self._tts_handler._clear_queue = callback` | Mirror queue-flush callback onto legacy host so `LocalSTTInputMixin` listener finds it | **YES — Phase 5d** |
| `_reset_tts_per_session_state` (226) | `handler = getattr(self, "_tts_handler", None)` | Per-session echo-guard reset | NO — moved to adapter |
| `get_available_voices` (240, 244) | Doc reference only — actual call routes through `self.pipeline.tts` | comment | docs only |

**Decision: KEEP the `_tts_handler` reference.** The `_clear_queue`
mirroring shim (lines 76-96, untouched by Phase 5c.1) still needs it.
Per the Phase 5c.1 spec (lines 191-192) and the exploration memo §5.2,
the `_clear_queue` mirror is explicitly Phase 5d territory — "if 5d
gets descoped, leaving the mirror in place is fine — it's five lines,
well-commented, and works."

Phase 5c.2 deletes only the `_reset_tts_per_session_state` helper from
the wrapper and updates the `get_available_voices` docstring reference;
the `_tts_handler` field and the `_clear_queue` setter stay verbatim.

## Protocol default semantics

The voice methods on `TTSBackend` (Phase 5c.1) default-raise
`NotImplementedError` because "voice switching" is a feature the wrapper
explicitly surfaces to operators, and a backend that silently no-ops
the call would create a confusing UI experience (admin clicks "change
voice," nothing happens).

`reset_per_session_state` is different. It's an internal cleanup hook
the orchestrator fires opportunistically; a backend that has no
per-session state to clean is a perfectly valid no-op. So the default
body is `return None` rather than `raise NotImplementedError`. This
keeps `@runtime_checkable` happy (the attribute is still declared at
Protocol level) and lets future TTS backends without echo-guard state
inherit the default without ceremony.

Note on `@runtime_checkable` + default-impl: per the Phase 5c.1 spec
"Risks / surprises," `@runtime_checkable` treats every declared method
as required for `isinstance` checks even when defaulted. Test stubs
that satisfy `TTSBackend` structurally (no inheritance — `_MockTTS`,
the contract suite's per-adapter stubs) need a
`reset_per_session_state` method to keep passing
`isinstance(stub, TTSBackend)`. For the contract stubs the trivial
implementation is "store a flag the test can check." For `_MockTTS`,
an `async def reset_per_session_state(self): return None` line is
enough.

## Sync vs async choice

`reset_per_session_state` is `async` for three reasons:

1. It composes cleanly into the pipeline's `apply_personality`, which
   is already `async` (matches the wrapper's `apply_personality`
   signature on the `ConversationHandler` ABC).
2. The Phase 5c.1 spec hits this exact decision for `change_voice` and
   lands on async — consistent shape across the per-session lifecycle
   surface.
3. Pure attribute writes today, but a future adapter (e.g. one that
   needs to await a websocket-server "session-reset" message) can
   implement async without a Protocol breaking change.

The adapters' implementations are pure `setattr` calls and don't
actually await anything, but the `async` declaration costs nothing and
matches the rest of the Protocol.

## Pipeline `apply_personality` shape

```python
# src/robot_comic/composable_pipeline.py — new method on ComposablePipeline

async def apply_personality(self, profile: str | None) -> str:
    """Switch persona: reset history, clear TTS per-session state, re-seed system prompt.

    See ``docs/superpowers/specs/2026-05-16-phase-5c2-apply-personality-to-pipeline.md``
    for the why. Mirrors the legacy ``ComposableConversationHandler.apply_personality``
    body that lived here before the move.
    """
    try:
        set_custom_profile(profile)
    except Exception as exc:
        logger.error("Error applying personality %r: %s", profile, exc)
        return f"Failed to apply personality: {exc}"
    self.reset_history(keep_system=False)
    await self.tts.reset_per_session_state()
    self._conversation_history.append(
        {"role": "system", "content": get_session_instructions()}
    )
    return f"Applied personality {profile!r}. Conversation history reset."
```

New imports in `composable_pipeline.py`:

```python
from robot_comic.config import set_custom_profile
from robot_comic.prompts import get_session_instructions
```

Both already appear in `composable_conversation_handler.py`; lift-and-shift.

## Wrapper `apply_personality` after 5c.2

```python
async def apply_personality(self, profile: str | None) -> str:
    """Forward to ``ComposablePipeline.apply_personality``.

    Persona-switch state surgery (history reset, per-session TTS state
    reset, system-prompt re-seed) lives on the pipeline as of Phase
    5c.2; the wrapper just satisfies the ``ConversationHandler`` ABC
    contract by forwarding through.
    """
    return await self.pipeline.apply_personality(profile)
```

The class-level docstring's lifecycle-hook bullet for
`_speaking_until` echo-guard reset stays but updates the location pointer
to `ComposablePipeline.apply_personality`.

## Adapter `reset_per_session_state` shape

Mirrors today's wrapper helper — same fields, same `hasattr` guard, same
no-op-on-missing semantics. Each adapter:

```python
async def reset_per_session_state(self) -> None:
    """Clear per-session echo-guard accumulators on the wrapped handler.

    Pipeline calls this from ``apply_personality`` so persona switch is
    a hard cut on listening state (legacy parity site:
    ``elevenlabs_tts.py:558-560`` per-turn reset; this is the
    per-persona complement that was wired in Phase 5a.1 from the
    wrapper and moves here in Phase 5c.2).

    Defensively guarded — handlers without echo-guard state
    (``GeminiTTSResponseHandler`` today) are a clean no-op.
    """
    for field, value in (
        ("_speaking_until", 0.0),
        ("_response_start_ts", 0.0),
        ("_response_audio_bytes", 0),
    ):
        if hasattr(self._handler, field):
            setattr(self._handler, field, value)
```

Three identical bodies. The implementation lives on the adapter rather
than a shared mixin because:

- TTS adapters share no base class today (each wraps a different
  duck-typed handler Protocol).
- A free-function helper in `adapters/__init__.py` or a `_utils.py`
  would invert the locality (the field knowledge belongs on the
  adapter, not in a separate utilities namespace).
- The body is 5 lines; duplication cost is low. Future per-adapter
  variation (e.g. ElevenLabs gets an additional field) is easier to
  apply when the body lives on the adapter rather than in a shared
  helper.

## Files touched

| File | Change |
|---|---|
| `src/robot_comic/backends.py` | Add `reset_per_session_state` default-no-op method to `TTSBackend`. |
| `src/robot_comic/composable_pipeline.py` | Add `apply_personality` method; new imports for `set_custom_profile` + `get_session_instructions`. |
| `src/robot_comic/composable_conversation_handler.py` | Shrink `apply_personality` to pass-through; delete `_reset_tts_per_session_state` helper; update docstrings (class-level + voice-method comment that referenced `_reset_tts_per_session_state`). |
| `src/robot_comic/adapters/elevenlabs_tts_adapter.py` | Add `reset_per_session_state`. |
| `src/robot_comic/adapters/chatterbox_tts_adapter.py` | Add `reset_per_session_state`. |
| `src/robot_comic/adapters/gemini_tts_adapter.py` | Add `reset_per_session_state`. |
| `tests/test_composable_pipeline.py` | Add 4 new test cases for `apply_personality`. |
| `tests/test_composable_conversation_handler.py` | Replace 4 `apply_personality_*` tests with 1 "forwards to pipeline" test. |
| `tests/test_composable_persona_reset.py` | Rewrite both cases to drive `pipeline.apply_personality` via a real adapter (preserves end-to-end coverage with concrete handler instances). |
| `tests/adapters/test_tts_backend_contract.py` | Add 2 contract tests (zeroes-fields-when-present + no-op-when-absent). Extend each per-adapter stub with `_speaking_until`/`_response_start_ts`/`_response_audio_bytes` + `reset_per_session_state` no-op aware. |
| `tests/test_backends_protocols.py` | Add `reset_per_session_state` to `_MockTTS` (keeps `isinstance` check passing). |

No `handler_factory.py` changes — the wrapper constructor signature is
unchanged; the factory still feeds `tts_handler=host` for the
`_clear_queue` mirror to find.

## Test plan

- `tests/adapters/test_tts_backend_contract.py` — 6 new assertions
  (2 tests × 3 adapters).
- `tests/test_composable_pipeline.py` — 4 new `apply_personality_*`
  tests (history reset, system re-seed, TTS reset awaited, error path).
- `tests/test_composable_conversation_handler.py` — net -3 tests (4
  removed, 1 added).
- `tests/test_composable_persona_reset.py` — both tests rewritten to
  use real `ComposablePipeline` + real adapter; assertions on
  end-state of the wrapped handler unchanged.
- `tests/test_backends_protocols.py::test_mock_tts_satisfies_protocol`
  — still green after `_MockTTS` extension.
- Lint: `uvx ruff@0.12.0 check .` and `uvx ruff@0.12.0 format --check .`
  from repo root.
- Types: `mypy --pretty --show-error-codes src/robot_comic/backends.py
  src/robot_comic/composable_pipeline.py
  src/robot_comic/composable_conversation_handler.py
  src/robot_comic/adapters/`.
- Full suite: `python -m pytest tests/ -q
  --ignore=tests/vision/test_local_vision.py`.

## Risks / surprises

- **`runtime_checkable` + default no-op**: same gotcha as 5c.1 — the
  method is required attribute even with a default body. `_MockTTS`,
  contract stubs, and `StubTTS` instances must declare the method (or
  inherit it). Stubs that don't pass `isinstance` are unaffected.
- **End-to-end regression test moves through the adapter**: the
  Phase 5a.1 regression test was structured to assert against the
  wrapper's direct mutation of the wrapped handler. After 5c.2 the same
  outcome is delivered via the adapter forwarding through pipeline →
  adapter → wrapped handler. The test stays meaningful but the failure
  surface points one level deeper. Documented in the test's docstring.
- **Pipeline now imports `config.set_custom_profile` and
  `prompts.get_session_instructions`**: these were wrapper-only imports
  before. No circular-import risk (both modules are leaves with respect
  to the pipeline). The wrapper drops these imports in the same PR.
- **The `_clear_queue` mirror still references `_tts_handler`**: future
  contributors reading the wrapper post-5c.2 may wonder why a "thin
  wrapper" still holds the legacy handler. The docstring on
  `__init__` and on `_clear_queue.setter` make this explicit — the
  field exists for the FastRTC/`LocalSTTInputMixin` barge-in wiring
  only, and Phase 5d is the place to address it.
