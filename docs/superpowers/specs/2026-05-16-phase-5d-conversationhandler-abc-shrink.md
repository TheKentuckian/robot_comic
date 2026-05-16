# Phase 5d — Shrink `ConversationHandler` ABC to FastRTC-shim role

**Date:** 2026-05-16
**Epic:** #391 (Phase 5)
**Predecessors:** Phase 5c.1 (PR #399 — `TTSBackend` voice methods),
Phase 5c.2 (PR #401 — `apply_personality` onto `ComposablePipeline`)
**Exploration anchor:** `docs/superpowers/specs/2026-05-16-phase-5-exploration.md` §2.4
**Branch:** `claude/phase-5d-conversationhandler-abc-shrink`
**Status:** Spec for implementation.

## Motivation

Operator chose "shrink, not collapse" per the exploration memo §2.4 / §5.3 q.4.
Phase 5c.1 moved the three voice methods (`get_available_voices`,
`get_current_voice`, `change_voice`) onto the `TTSBackend` Protocol, and
Phase 5c.2 moved `apply_personality` onto `ComposablePipeline`. The ABC's
abstract declarations for those four methods are now duplicates — the
real surface lives elsewhere. Removing them shrinks the ABC to its
FastRTC-shim role: lifecycle + per-frame I/O.

The concrete implementations on `ComposableConversationHandler` and
`BaseRealtimeHandler` STAY — callers still expect them via duck-typing
through `headless_personality_ui.py`'s REST routes — but the
`@abstractmethod` enforcement disappears.

## Scope (this PR)

1. **Drop four `@abstractmethod` declarations** from
   `ConversationHandler` (`src/robot_comic/conversation_handler.py`):
   - `apply_personality`
   - `get_available_voices`
   - `get_current_voice`
   - `change_voice`

2. **Update the ABC docstring** to reflect the FastRTC-shim role.

3. **Keep `deps: ToolDependencies` annotation** on the ABC. The audit
   (below) found one production site that reads `handler.deps` through
   the ABC type (`console.py:1496`); removing the annotation would
   break mypy type-checking there.

4. **Keep `_clear_queue: Callable[[], None] | None`** on the ABC.
   `LocalStream.__init__` (`console.py:271`) writes to it for barge-in
   plumbing; the wrapper's `_clear_queue.setter` mirrors it onto the
   underlying TTS handler (`composable_conversation_handler.py:89-104`).
   This is the documented "leaky abstraction" the exploration memo §2.4
   point 3 flagged — out of scope for this PR (would touch the
   `_LocalSTTInputMixin` barge-in path).

5. **Keep `output_queue`** on the ABC — `LocalStream.clear_audio_queue`
   rebinds it for barge-in (`console.py:1386`) and `emit()` semantically
   pulls from it.

6. **Update `headless_personality_ui.py` calls to use duck-typing.** The
   module takes `handler: ConversationHandler` and calls
   `handler.apply_personality(sel)`, `handler.get_available_voices()`,
   `handler.get_current_voice()`, and `handler.change_voice(voice)`.
   Once those methods are no longer on the ABC, mypy will fail on those
   four call sites. Resolve with `getattr` lookups that fail gracefully
   (matches the existing pattern at line 287 for `get_current_voice`),
   or with narrow `# type: ignore[attr-defined]` comments if `getattr`
   gymnastics would obscure intent.

## Non-goals (explicit)

- Do NOT collapse the ABC entirely — operator chose shrink.
- Do NOT drop `_clear_queue` mirror in `ComposableConversationHandler`
  — that's barge-in plumbing the FastRTC `LocalStream` still writes to.
- Do NOT drop the wrapper's `_tts_handler` reference — `_clear_queue.setter`
  still uses it (confirmed in 5c.2 audit).
- Do NOT touch bundled-realtime handlers' concrete implementations of
  the four methods — they keep working as-is. The methods are still
  present and called by `headless_personality_ui.py`.
- Do NOT remove `deps` field from the ABC — see audit below.
- Do NOT add Phase 5e (factory STT decouple) or 5f (faster-whisper)
  scaffolding.

## `deps: ToolDependencies` field audit

**Grep result** (`handler\.deps` across `src/`):

| File | Line | Site | Through ABC type? |
|---|---|---|---|
| `src/robot_comic/console.py` | 1496 | `head_wobbler = self.handler.deps.head_wobbler` in `_play_loop` | **YES** — `self.handler` typed as `Optional[ConversationHandler]` (`console.py:235`) |

That's the only read site outside concrete subclasses that own `deps`
themselves. Removing the annotation from the ABC would make
`self.handler.deps` fail mypy in `console.py:1496` — either with a
type: ignore or a `getattr`. Both add noise for a 1-line annotation
removal.

**Decision:** KEEP the `deps: ToolDependencies` annotation on the ABC.

Per the brief's instruction ("If anything reads `handler.deps` through
the ABC type, document and leave it"), this is the documented outcome.
A future Phase 5e/5f or a dedicated `console.py` refactor could narrow
the field down (e.g. introduce a `HandlerWithDeps` Protocol the
console-side reads against) but that's larger than 5d's scope and not
worth the churn for a single attribute access.

## Bundled-realtime handler audit

`HuggingFaceRealtimeHandler`, `OpenaiRealtimeHandler`, `GeminiLiveHandler`
all inherit from `BaseRealtimeHandler` (`base_realtime.py:90`) which
implements all four removed-from-ABC methods concretely:

| Method | `BaseRealtimeHandler` site |
|---|---|
| `apply_personality` | `base_realtime.py:322-382` |
| `get_available_voices` | `base_realtime.py:1233-1235` |
| `get_current_voice` | `base_realtime.py:317-320` |
| `change_voice` | `base_realtime.py:305-315` |

Subclass overrides exist (e.g. `GeminiLiveHandler.apply_personality` at
`gemini_live.py:586`, `OpenaiRealtimeHandler.get_available_voices` at
`openai_realtime.py:91`) but none rely on the ABC's `@abstractmethod`
enforcement to discover that the parent provides a fallback. The shrink
is purely a metadata change for these handlers.

The two hybrid handlers (`LocalSTTOpenAIRealtimeHandler`,
`LocalSTTHuggingFaceRealtimeHandler` in `local_stt_realtime.py`) inherit
from `BaseRealtimeHandler` via the same path and are not affected.

## Wrapper audit

`ComposableConversationHandler` (`composable_conversation_handler.py`)
implements all four methods as forwarders (5c.1 + 5c.2 already moved
the actual work elsewhere):

| Method | Wrapper site | Forwards to |
|---|---|---|
| `apply_personality` | `:180-196` | `self.pipeline.apply_personality(profile)` |
| `get_available_voices` | `:198-210` | `self.pipeline.tts.get_available_voices()` |
| `get_current_voice` | `:212-214` | `self.pipeline.tts.get_current_voice()` |
| `change_voice` | `:216-218` | `self.pipeline.tts.change_voice(voice)` |

These bodies stay verbatim post-5d. Only the ABC's `@abstractmethod`
marker disappears.

## Caller audit (the duck-typing surface)

| Caller | Call site | Through `ConversationHandler` type? | Fix needed? |
|---|---|---|---|
| `headless_personality_ui.py:286` | `await handler.apply_personality(sel)` | YES (param at `:49`) | YES — `getattr` or type ignore |
| `headless_personality_ui.py:316` | `await handler.get_available_voices()` | YES | YES |
| `headless_personality_ui.py:335` | `handler.get_current_voice()` | YES | YES |
| `headless_personality_ui.py:361` | `await handler.change_voice(voice)` | YES | YES |
| `headless_personality_ui.py:287-288` | `getattr(handler, "get_current_voice", None)` | already duck-typed | NO |
| `console.py` | (none for voice/personality methods) | — | NO |
| `main.py` | (none — only `set_ws_client` and ws-routing) | — | NO |

Fix pattern for `headless_personality_ui.py`: use `getattr` lookups that
fall back to a safe default, mirroring the existing line 287-288
pattern. This is the most explicit and avoids `# type: ignore` noise.

Example for `apply_personality`:

```python
apply_personality = getattr(handler, "apply_personality", None)
if not callable(apply_personality):
    return JSONResponse(
        {"ok": False, "error": "handler does not support personality switching"},
        status_code=501,
    )
status = await apply_personality(sel)
```

For `get_available_voices` / `get_current_voice` (already have fallback
paths to `get_available_voices_for_provider()` / a default voice), the
existing exception-catching code path already handles missing methods
gracefully via `try/except Exception`. The added `getattr` indirection
makes the absence path explicit and lets mypy resolve types correctly.

## Updated ABC

```python
class ConversationHandler(ABC):
    """FastRTC-shim ABC for realtime conversation backends.

    Defines lifecycle (``start_up``/``shutdown``), per-frame I/O
    (``receive``/``emit``), and per-peer cloning (``copy``) — the surface
    FastRTC's ``AsyncStreamHandler`` consumes. Voice methods
    (``get_available_voices``/``get_current_voice``/``change_voice``) live
    on :class:`~robot_comic.backends.TTSBackend` (Phase 5c.1) and
    persona-switch state surgery lives on
    :class:`~robot_comic.composable_pipeline.ComposablePipeline.apply_personality`
    (Phase 5c.2). Concrete handlers (the composable wrapper, the
    bundled-realtime handlers) still implement those methods directly
    for callers that reach for them via duck-typing
    (``headless_personality_ui.py`` REST routes); they are no longer
    ABC-enforced.

    See ``docs/superpowers/specs/2026-05-16-phase-5d-conversationhandler-abc-shrink.md``
    for the shrink rationale.
    """

    deps: ToolDependencies
    output_queue: asyncio.Queue[QueueItem]
    _clear_queue: Callable[[], None] | None

    @abstractmethod
    def copy(self) -> ConversationHandler: ...

    @abstractmethod
    async def start_up(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    async def receive(self, frame: AudioFrame) -> None: ...

    @abstractmethod
    async def emit(self) -> HandlerOutput: ...
```

## Test impacts

**Search for tests asserting ABC abstractness** (`pytest.raises(TypeError)`
related to `ConversationHandler`): none found. The existing positive
test (`test_composable_conversation_handler.py:39-41`,
`isinstance(wrapper, ConversationHandler)`) keeps passing because the
wrapper still subclasses `ConversationHandler` and the abstract methods
that remain (`copy`/`start_up`/`shutdown`/`receive`/`emit`) are all
implemented concretely.

The 9 `assert isinstance(result, ComposableConversationHandler)`
assertions across `tests/test_handler_factory*.py` and
`tests/test_phase_5b_*.py` are unaffected — they assert wrapper type,
not ABC abstractness.

**New test:** add a focused test that the ABC no longer enforces the
four removed methods. A test class subclassing `ConversationHandler`
that implements only the five remaining abstract methods should
instantiate without `TypeError`. This pins the shrink so a future PR
accidentally re-adding `@abstractmethod` to one of the moved methods
fails.

## Files touched

| File | Change |
|---|---|
| `src/robot_comic/conversation_handler.py` | Drop 4 `@abstractmethod` declarations; update docstring. |
| `src/robot_comic/headless_personality_ui.py` | Reach for voice/personality methods via `getattr` so mypy still type-checks against the post-shrink ABC. |
| `tests/test_conversation_handler_abc.py` | NEW: pin that the ABC enforces only the five FastRTC-shim methods. |

No changes to `composable_conversation_handler.py`, `base_realtime.py`,
or any concrete handler — those keep their implementations.

## Test plan

- New `tests/test_conversation_handler_abc.py` — assert that a minimal
  subclass implementing only the five remaining abstract methods can be
  instantiated.
- Existing `tests/test_composable_conversation_handler.py` — all 20
  tests still pass (verified baseline).
- Existing factory-mode and 5b-dispatcher-wiring `isinstance(...,
  ComposableConversationHandler)` assertions — unaffected.
- Lint: `uvx ruff@0.12.0 check .` and `uvx ruff@0.12.0 format --check .`
  from repo root.
- Types: `mypy --pretty --show-error-codes
  src/robot_comic/conversation_handler.py
  src/robot_comic/composable_conversation_handler.py
  src/robot_comic/base_realtime.py
  src/robot_comic/headless_personality_ui.py`.
- Full suite: `python -m pytest tests/ -q
  --ignore=tests/vision/test_local_vision.py`.

## Risks / surprises

- **`headless_personality_ui.py` REST routes lose ABC typing for four
  methods.** The `getattr` lookups + fallback paths preserve runtime
  behavior identically (concrete handlers still implement the methods)
  and yield clearer error responses if a hypothetical future
  ConversationHandler subclass omits a method. The `headless_personality_ui.py`
  test surface is small (no direct unit tests of these routes today);
  manual verification on the admin UI is the regression check, but the
  duck-typing change is mechanical and low-risk.
- **`deps` annotation stays.** This is a documented partial retreat from
  the §2.4 "misplaced field" critique. Removing it is a follow-up that
  needs a console-side refactor (introduce a `HandlerWithDeps` Protocol
  or move `head_wobbler` access elsewhere).
- **`_clear_queue` mirror stays.** Per exploration memo §5.2 and the
  brief's non-goals: "Do NOT drop `_clear_queue` mirror in
  `ComposableConversationHandler`."
- **Bundled-realtime handlers untouched.** They've been out of scope
  since Phase 0; the shrink doesn't change runtime behavior for them.
