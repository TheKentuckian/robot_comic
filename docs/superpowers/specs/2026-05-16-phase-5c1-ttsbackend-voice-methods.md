# Phase 5c.1 — Extend `TTSBackend` Protocol with voice/personality methods

**Date:** 2026-05-16
**Epic:** #391 (Phase 5)
**Predecessor exploration:** `docs/superpowers/specs/2026-05-16-phase-5-exploration.md` §2.3
**Status:** Spec for implementation.

## Motivation

`ComposableConversationHandler` forwards `get_available_voices`,
`get_current_voice`, and `change_voice` to a legacy `*ResponseHandler`
instance held by reference (`composable_conversation_handler.py:237-247`).
That reference exists *solely* to satisfy these three methods — every
other wrapper concern routes through `self._pipeline`. The forwarding
target (`self._tts_handler`) is the same concrete object the factory
also feeds to the TTS adapter, so the data lives on the right object;
the wrapper just doesn't have a Protocol-blessed way to reach it.

The `TTSBackend` Protocol (`backends.py:185-235`) declares `prepare`,
`synthesize`, `shutdown` — no voice methods. Adding them lets the wrapper
forward through `self._pipeline.tts` and stop reaching into the legacy
handler for voice queries. Phase 5c.2 will follow up by moving
`apply_personality` onto `ComposablePipeline`; Phase 5d may then drop
the `_tts_handler` reference entirely.

## Scope (this PR)

1. Extend `TTSBackend` Protocol with three voice methods, each with a
   default body that raises `NotImplementedError`.
2. Implement the three methods on each TTS adapter
   (`ElevenLabsTTSAdapter`, `ChatterboxTTSAdapter`, `GeminiTTSAdapter`)
   as thin forwards to the wrapped legacy handler.
3. Update `ComposableConversationHandler` to forward
   `get_available_voices` / `get_current_voice` / `change_voice` through
   `self._pipeline.tts` instead of `self._tts_handler`. Keep the
   `_tts_handler` reference (still used by `_reset_tts_per_session_state`
   in `apply_personality`; Phase 5c.2 / 5d will revisit).
4. Extend the parametric `tests/adapters/test_tts_backend_contract.py`
   suite with voice-method contract assertions that run against every
   registered TTS adapter.

## Non-goals (explicit)

- Do NOT move `apply_personality` onto `ComposablePipeline` (Phase 5c.2).
- Do NOT delete `_tts_handler` from the wrapper (Phase 5c.2 / 5d).
- Do NOT touch the bundled-realtime handlers (`HuggingFaceRealtimeHandler`,
  `OpenaiRealtimeHandler`, `GeminiLiveHandler`) — they don't use TTS
  adapters.
- Do NOT remove voice methods from `ConversationHandler` ABC (Phase 5d).

## Legacy-handler signature audit

All three TTS-holding handler classes implement matching signatures:

| Method | Signature | Behavior |
| --- | --- | --- |
| `async get_available_voices(self) -> list[str]` | async, returns list | ElevenLabs: `list(ELEVENLABS_AVAILABLE_VOICES)`; Chatterbox: HTTP-fetch predefined voices (falls back to current on error); Gemini: `list(GEMINI_TTS_AVAILABLE_VOICES)`. |
| `def get_current_voice(self) -> str` | **sync**, returns str | ElevenLabs: `_voice_override` or env `ELEVENLABS_VOICE` or profile config; Chatterbox: `self._chatterbox_voice`; Gemini: `_voice_override` or default (with validity check). |
| `async change_voice(self, voice: str) -> str` | async, returns confirmation str | All three: `self._voice_override = voice; return f"Voice changed to {voice}."` — no validation, no rejection. |

Source citations:

- `src/robot_comic/elevenlabs_tts.py:499-538`
- `src/robot_comic/chatterbox_tts.py:306-324`
- `src/robot_comic/gemini_tts.py:335-349`

The contract surface is therefore uniform — no per-adapter signature
quirk to accommodate. The `change_voice` contract is "set the override
and return a confirmation string"; unknown voices are NOT rejected
(they're stored as-is and the next synthesis call resolves them or falls
back). The parametric test must assert this lenient behavior, not raise-on-unknown.

## Protocol extension

`src/robot_comic/backends.py:185-235` gains three methods, each with a
default body that raises `NotImplementedError`. The default body lets
adapters that genuinely don't support voice switching (today: none, but
future Whisper-only or pipeline-debug TTS stubs) opt out of the contract
without subclassing the Protocol.

```python
@runtime_checkable
class TTSBackend(Protocol):
    # ... existing prepare / synthesize / shutdown ...

    async def get_available_voices(self) -> list[str]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support voice listing"
        )

    def get_current_voice(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support voice queries"
        )

    async def change_voice(self, voice: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support voice switching"
        )
```

## `@runtime_checkable` + default-impl interaction

Python's `@runtime_checkable` Protocols treat every declared method
(including those with default implementations) as a required attribute
for `isinstance()` checks. Verified locally:

```python
@runtime_checkable
class P(Protocol):
    def foo(self) -> int: ...
    def bar(self) -> int:
        raise NotImplementedError

class OnlyFoo:                              # no bar method
    def foo(self) -> int: return 1

isinstance(OnlyFoo(), P)  # False
```

**Implication:** test stubs that satisfy `TTSBackend` today via
structural duck-typing (no inheritance) will fail `isinstance(stub,
TTSBackend)` after this PR unless they add the three new methods. Two
in-repo stubs are affected:

- `tests/test_backends_protocols.py::_MockTTS` — used by
  `test_mock_tts_satisfies_protocol`. Add three trivial methods.
- `tests/adapters/test_tts_backend_contract.py` per-adapter stubs
  (`_ElevenLabsStub`, `_ChatterboxStub`, `_GeminiStub`) — used by the
  contract suite's `isinstance` assertion. Add three trivial methods to
  each.

Stubs that don't undergo `isinstance` checks (`StubTTS` in
`test_composable_conversation_handler.py`, `_RecordingTTS` in
`test_composable_pipeline.py`) are unaffected because Python's structural
typing only matters at the actual check site.

The `_make_wrapper()` helper in
`tests/test_composable_conversation_handler.py` builds the wrapper with
`pipeline=MagicMock(); tts_handler=MagicMock()`. After this PR the
wrapper forwards voice calls through `self._pipeline.tts` (a
`MagicMock` attribute of the pipeline mock); the test updates wire
`wrapper.pipeline.tts.get_current_voice` etc. and stop wiring through
`wrapper._tts_handler` for the voice methods. The `_tts_handler` mock
stays in place because `_reset_tts_per_session_state` (called from
`apply_personality`) still reads from it.

## Adapter implementations

Each adapter forwards to its already-wrapped handler reference
(`self._handler`):

```python
# Pattern (identical across all three adapters):
async def get_available_voices(self) -> list[str]:
    return await self._handler.get_available_voices()

def get_current_voice(self) -> str:
    return self._handler.get_current_voice()

async def change_voice(self, voice: str) -> str:
    return await self._handler.change_voice(voice)
```

The duck-typed handler Protocols
(`_ElevenLabsCompatibleHandler`, `_GeminiTTSCompatibleHandler`) gain the
three method declarations so mypy structural matching covers the new
adapter calls. `ChatterboxTTSAdapter` types its handler as the concrete
class `ChatterboxTTSResponseHandler` via `TYPE_CHECKING` import, so no
Protocol update is needed there.

## Wrapper update

`composable_conversation_handler.py:237-247` becomes:

```python
async def get_available_voices(self) -> list[str]:
    """Forward to the pipeline's TTS adapter."""
    return await self.pipeline.tts.get_available_voices()

def get_current_voice(self) -> str:
    """Forward to the pipeline's TTS adapter."""
    return self.pipeline.tts.get_current_voice()

async def change_voice(self, voice: str) -> str:
    """Forward to the pipeline's TTS adapter."""
    return await self.pipeline.tts.change_voice(voice)
```

`self._tts_handler` is NOT deleted — it's still consulted by
`_reset_tts_per_session_state` (called from `apply_personality`).
Phase 5c.2 will move that work and Phase 5d may delete the reference.

## Contract test additions

Add to `tests/adapters/test_tts_backend_contract.py` (extending the
existing parametric suite — same `ADAPTERS` registry):

```python
@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_get_available_voices_returns_list_of_strings(build) -> None:
    adapter, _ = build()
    voices = await adapter.get_available_voices()
    assert isinstance(voices, list)
    for v in voices:
        assert isinstance(v, str)


@pytest.mark.parametrize("build", ADAPTERS)
def test_get_current_voice_returns_string(build) -> None:
    adapter, _ = build()
    voice = adapter.get_current_voice()
    assert isinstance(voice, str)


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_change_voice_returns_resolved_id(build) -> None:
    adapter, _ = build()
    result = await adapter.change_voice("SomeVoice")
    assert isinstance(result, str)


@pytest.mark.parametrize("build", ADAPTERS)
@pytest.mark.asyncio
async def test_change_voice_to_unknown_voice_does_not_raise(build) -> None:
    """All three legacy handlers store unknown voices as-is — they don't
    validate at change-time, only at synthesis-time."""
    adapter, _ = build()
    result = await adapter.change_voice("definitely-not-a-real-voice-id-12345")
    assert isinstance(result, str)
```

Each stub in the contract suite gains the three voice methods (returning
`[]`, the empty string, and a confirmation string respectively) so the
`isinstance` assertion keeps passing and the new contract tests have
something to exercise.

## Files touched

| File | Change |
| --- | --- |
| `src/robot_comic/backends.py` | Add 3 default-impl methods to `TTSBackend` Protocol. |
| `src/robot_comic/adapters/elevenlabs_tts_adapter.py` | Add 3 forward methods; extend `_ElevenLabsCompatibleHandler` Protocol. |
| `src/robot_comic/adapters/chatterbox_tts_adapter.py` | Add 3 forward methods. |
| `src/robot_comic/adapters/gemini_tts_adapter.py` | Add 3 forward methods; extend `_GeminiTTSCompatibleHandler` Protocol. |
| `src/robot_comic/composable_conversation_handler.py` | Forward voice methods through `self.pipeline.tts` instead of `self._tts_handler`. Update docstrings. |
| `tests/adapters/test_tts_backend_contract.py` | Add 4 new parametric tests; extend each stub with 3 voice methods. |
| `tests/test_backends_protocols.py` | Extend `_MockTTS` with 3 voice methods. |
| `tests/test_composable_conversation_handler.py` | Update 3 wrapper tests to mock `pipeline.tts` instead of `_tts_handler`. |

## Test plan

- `tests/adapters/test_tts_backend_contract.py` — extended; 12 new parametric assertions (4 tests × 3 adapters).
- `tests/test_backends_protocols.py::test_mock_tts_satisfies_protocol` — still green after `_MockTTS` extension.
- `tests/test_composable_conversation_handler.py::test_{get_current,get_available,change}_voice_delegates` — updated to assert forwarding via `pipeline.tts`.
- Per-adapter integration tests in `tests/adapters/test_*_adapter.py` — unchanged; they test adapter-specific behavior, not the Protocol surface.
- Lint: `uvx ruff@0.12.0 check .` and `uvx ruff@0.12.0 format --check .` from repo root.
- Types: `mypy --pretty --show-error-codes src/robot_comic/backends.py src/robot_comic/composable_conversation_handler.py src/robot_comic/adapters/`.
- Full suite: `python -m pytest tests/ -q --ignore=tests/vision/test_local_vision.py`.

## Risks / surprises

- **`runtime_checkable` + default-impl forces test-stub updates.** The
  brief assumed defaults would let stubs go untouched; Python's Protocol
  semantics require the attribute to be present even when defaulted. The
  fix is small (4 stubs × 3 trivial methods) and the alternative
  (drop `@runtime_checkable` from `TTSBackend`) would lose the
  `test_adapter_satisfies_tts_backend_protocol` guard.
- **`change_voice` contract is lenient.** Unknown voices don't raise.
  The contract test pins this; a future hardening PR (validate at
  change-time vs synthesis-time) is out of scope.
- **No factory wiring changes.** The wrapper still receives
  `tts_handler` via its constructor (the factory still wires it). Phase
  5c.2 / 5d will revisit.
