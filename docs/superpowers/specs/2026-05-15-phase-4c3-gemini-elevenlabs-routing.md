# Phase 4c.3 — Composable routing for `(moonshine, elevenlabs, gemini)`

**Date:** 2026-05-15
**Scope:** Broaden `ElevenLabsTTSAdapter`'s constructor annotation to a Protocol that captures the duck-typed surface it actually uses; add a new `_build_composable_gemini_elevenlabs` helper in `handler_factory.py`; route the `(moonshine, elevenlabs, gemini)` triple through `ComposableConversationHandler` when `REACHY_MINI_FACTORY_PATH=composable`; tests under `tests/`.
**Epic:** #337 — Pipeline refactor (Option C, incremental retirement)
**Predecessors:** Phase 4c.2 (#362 / `a91acb2`) — `GeminiLLMAdapter` landed; `(moonshine, chatterbox, gemini)` routes through `ComposableConversationHandler` under `FACTORY_PATH=composable`.
**Successors:** Phase 4c.4 — `(moonshine, elevenlabs, gemini-fallback)` via `LocalSTTGeminiElevenLabsHandler`; 4c.5 covers gemini-bundled TTS.

## Background

`FACTORY_PATH=composable` already covers three triples:

- `(moonshine, llama, elevenlabs)` — Phase 4b.
- `(moonshine, llama, chatterbox)` — Phase 4c.1.
- `(moonshine, chatterbox, gemini)` — Phase 4c.2.

The next triple to migrate is `(moonshine, elevenlabs, gemini)`. The legacy class is `GeminiTextElevenLabsHandler` (`gemini_text_handlers.py`), which composes:

- `LocalSTTInputMixin` — moonshine listener (already adapted by `MoonshineSTTAdapter`).
- `GeminiTextElevenLabsResponseHandler(GeminiTextResponseHandler, ElevenLabsTTSResponseHandler)` — Gemini-API LLM + ElevenLabs TTS via a diamond MRO.

Both halves already have adapters from previous phases:

- **LLM half** — `GeminiLLMAdapter` (4c.2). `GeminiTextResponseHandler._call_llm` is the same signature/return-shape as `BaseLlamaResponseHandler._call_llm`, and Gemini-native tool-call shapes are already normalised to llama-server shape inside `gemini_llm.py::GeminiLLMClient.call_completion`. The adapter is fully reusable for this triple.
- **TTS half** — `ElevenLabsTTSAdapter` (4a/4b). Its constructor is currently annotated `handler: "ElevenLabsTTSResponseHandler"`, but `GeminiTextElevenLabsResponseHandler` inherits from `ElevenLabsTTSResponseHandler` — so the bare class-level annotation would already accept it, save for one subtlety covered below.
- **STT half** — `MoonshineSTTAdapter` (Phase 3).

The 4b helper `_build_composable_llama_elevenlabs` already passes the legacy handler to `ElevenLabsTTSAdapter` via `cast(Any, legacy)` because the legacy class there (`LocalSTTLlamaElevenLabsHandler`) is *not* a subclass of `ElevenLabsTTSResponseHandler` — it has the same duck-typed surface but a parallel inheritance chain. The cast is the TODO this sub-phase pays down: replace the cast-driven duck-type with a real Protocol so we get back type-checking *and* the new gemini triple wires up without a cast.

### Why `cast(Any, legacy)` works for 4b but is a smell

The 4b cast hides a real correctness invariant — the adapter calls these four members on the handler:

| Member | How it's used |
|--------|---------------|
| `_prepare_startup_credentials()` | `await handler._prepare_startup_credentials()` in `prepare()` |
| `output_queue` | Swapped with a temp queue at the start of `synthesize()`, restored in `finally` |
| `_stream_tts_to_queue(text, tags=...)` | Awaited inside the streaming task |
| `_http` | Checked + closed in `shutdown()`; assigned `None` after close |

A `cast(Any, ...)` says "trust me". A Protocol says "here are the four attributes I depend on; the type checker enforces it everywhere". The latter is strictly better for a refactor whose entire point is making adapter→handler contracts explicit.

## Goal

1. Define a `_ElevenLabsCompatibleHandler` Protocol in `elevenlabs_tts_adapter.py` capturing the four-member duck-typed surface.
2. Broaden the constructor annotation from `handler: "ElevenLabsTTSResponseHandler"` to `handler: "_ElevenLabsCompatibleHandler"`.
3. Add `_build_composable_gemini_elevenlabs` in `handler_factory.py` mirroring the 4c.2 helper.
4. Route `(moonshine, elevenlabs, gemini)` through `ComposableConversationHandler` when `FACTORY_PATH=composable`; default `legacy` keeps `GeminiTextElevenLabsHandler` — zero behaviour change.

## Out of scope (deferred)

| Item | Sub-phase |
|------|-----------|
| `(moonshine, elevenlabs, gemini-fallback)` — `LocalSTTGeminiElevenLabsHandler` | 4c.4 |
| `(moonshine, gemini_tts)` — `LocalSTTGeminiTTSHandler` | 4c.5 (needs `GeminiTTSAdapter`) |
| Sibling `HybridRealtimePipeline` for `LocalSTT*RealtimeHandler` | 4c-tris |
| Flip default to `composable` | 4d |
| Delete legacy handlers | 4e |
| Retire `BACKEND_PROVIDER` | 4f |
| Lifecycle hooks: telemetry, boot-timeline events (#321), joke history, history trim, echo-guard timestamps | Per-hook follow-up PRs |
| Removing the `cast(Any, legacy)` from `_build_composable_llama_elevenlabs` | Bonus if mypy is happy — see "Bonus cleanup" below |

## Design

### 1. The Protocol — `_ElevenLabsCompatibleHandler`

Lives in `src/robot_comic/adapters/elevenlabs_tts_adapter.py` as a module-private Protocol. Module-private because nothing outside this adapter has a legitimate reason to import it; the adapter owns the contract.

```python
from typing import Any, Protocol
import asyncio


class _ElevenLabsCompatibleHandler(Protocol):
    """Duck-typed surface ``ElevenLabsTTSAdapter`` needs on its wrapped handler.

    Captures only the four members the adapter actually touches:

    - ``_prepare_startup_credentials()`` — awaited from ``prepare()``.
    - ``output_queue`` — read + reassigned (swapped with a temp queue for the
      duration of one ``synthesize()`` call, then restored).
    - ``_stream_tts_to_queue(text, tags=...)`` — the streaming entry point.
    - ``_http`` — optional ``httpx.AsyncClient`` closed in ``shutdown()``.

    Concrete satisfiers (today) are ``ElevenLabsTTSResponseHandler`` and any
    subclass — including the Gemini-text MRO diamond subclass
    ``GeminiTextElevenLabsResponseHandler``. The 4b
    ``LlamaElevenLabsTTSResponseHandler`` also matches structurally even
    though it is in a parallel inheritance chain (no relation to
    ``ElevenLabsTTSResponseHandler``).
    """

    output_queue: asyncio.Queue[Any]
    _http: Any  # httpx.AsyncClient | None — typed Any to avoid importing httpx here.

    async def _prepare_startup_credentials(self) -> None: ...

    async def _stream_tts_to_queue(
        self,
        text: str,
        first_audio_marker: list[float] | None = None,
        tags: list[str] | None = None,
    ) -> bool: ...
```

Notes:

- `_http` typed `Any` because the only operation the adapter performs is `await http.aclose()` after a `getattr(..., None)` guard — pinning `httpx.AsyncClient | None` would force the adapter to import httpx, which is unnecessary coupling.
- The Protocol is NOT marked `@runtime_checkable`. We don't use `isinstance(handler, _ElevenLabsCompatibleHandler)` anywhere; mypy structural matching is the only contract surface, and runtime-checkable Protocols have a measurable import-time cost we don't need to pay.
- The method signature of `_stream_tts_to_queue` mirrors the legacy method exactly (including `first_audio_marker` even though the adapter doesn't pass it today — the legacy method's signature is still part of the contract for future first-audio-marker plumbing, tracked as a known gap in the adapter's module docstring).

### 2. Broadened constructor annotation

```python
class ElevenLabsTTSAdapter:
    """Adapter exposing ``ElevenLabsTTSResponseHandler`` as ``TTSBackend``."""

    def __init__(self, handler: "_ElevenLabsCompatibleHandler") -> None:
        """Wrap a pre-constructed handler instance."""
        self._handler = handler
```

The `TYPE_CHECKING` import of `ElevenLabsTTSResponseHandler` is removed (no longer referenced). The Protocol replaces it.

`self._handler` becomes implicitly typed `_ElevenLabsCompatibleHandler`. All existing `self._handler._stream_tts_to_queue(...)` / `self._handler.output_queue = ...` / `self._handler._prepare_startup_credentials()` / `getattr(self._handler, "_http", None)` calls type-check against the Protocol.

### 3. `_build_composable_gemini_elevenlabs` in `handler_factory.py`

Mirrors `_build_composable_gemini_chatterbox` (4c.2) verbatim with two substitutions: `ChatterboxTTSAdapter` → `ElevenLabsTTSAdapter`, and `GeminiTextChatterboxHandler` → `GeminiTextElevenLabsHandler`.

```python
def _build_composable_gemini_elevenlabs(**handler_kwargs: Any) -> Any:
    """Construct the composable (moonshine, elevenlabs, gemini) pipeline.

    Builds a legacy ``GeminiTextElevenLabsHandler`` (the adapters delegate
    into it), wraps it with the three Phase 3/4 adapters, composes them into
    a ``ComposablePipeline`` seeded with the current session instructions,
    and returns a ``ComposableConversationHandler`` whose ``build`` closure
    re-runs the same construction. FastRTC's ``copy()`` per-peer cloning
    invokes the closure for fresh state on each new peer.

    The ElevenLabs TTS half is shared with Phase 4b's llama variant; the
    LLM half is the same ``GeminiLLMAdapter`` from Phase 4c.2.  No new
    adapter is introduced — only the routing.
    """
    from robot_comic.prompts import get_session_instructions
    from robot_comic.adapters import (
        GeminiLLMAdapter,
        MoonshineSTTAdapter,
        ElevenLabsTTSAdapter,
    )
    from robot_comic.composable_pipeline import ComposablePipeline
    from robot_comic.gemini_text_handlers import GeminiTextElevenLabsHandler
    from robot_comic.composable_conversation_handler import ComposableConversationHandler

    def _build() -> ComposableConversationHandler:
        legacy = GeminiTextElevenLabsHandler(**handler_kwargs)
        stt = MoonshineSTTAdapter(legacy)
        llm = GeminiLLMAdapter(legacy)
        tts = ElevenLabsTTSAdapter(legacy)
        pipeline = ComposablePipeline(
            stt,
            llm,
            tts,
            system_prompt=get_session_instructions(),
        )
        return ComposableConversationHandler(
            pipeline=pipeline,
            tts_handler=legacy,
            deps=handler_kwargs["deps"],
            build=_build,
        )

    return _build()
```

No `cast(Any, legacy)` needed because `GeminiTextElevenLabsHandler` (via `GeminiTextElevenLabsResponseHandler` → `ElevenLabsTTSResponseHandler`) satisfies the new Protocol structurally and nominally.

### 4. Factory branch wiring

Inside the existing `LLM_BACKEND=gemini` arm in `handler_factory.py`, in the `output_backend == AUDIO_OUTPUT_ELEVENLABS` block (currently around line 254), prepend a composable check:

```python
if output_backend == AUDIO_OUTPUT_ELEVENLABS:
    # Phase 4c.3 (#337): gemini+elevenlabs is routed through
    # ComposableConversationHandler when FACTORY_PATH=composable.
    # Default FACTORY_PATH=legacy keeps the existing
    # GeminiTextElevenLabsHandler selection below.
    if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
        logger.info(
            "HandlerFactory: selecting ComposableConversationHandler "
            "(%s → %s, llm=%s, factory_path=composable)",
            input_backend,
            output_backend,
            LLM_BACKEND_GEMINI,
        )
        return _build_composable_gemini_elevenlabs(**handler_kwargs)
    from robot_comic.gemini_text_handlers import GeminiTextElevenLabsHandler

    logger.info(
        "HandlerFactory: selecting GeminiTextElevenLabsHandler (%s → %s, llm=%s)",
        input_backend,
        output_backend,
        LLM_BACKEND_GEMINI,
    )
    return GeminiTextElevenLabsHandler(**handler_kwargs)
```

### 5. Resolution of open design questions from the briefing

| Question | Resolution |
|----------|-----------|
| Type-annotation broadening: Protocol vs Any? | **Protocol** — a four-method/two-attribute surface is small enough that a Protocol is cheap (<20 LOC) and preserves mypy strict-mode coverage everywhere the adapter is used. `Any` would lose type-checking on the `synthesize`/`prepare`/`shutdown` methods that rely on the handler's surface. The Protocol also documents intent: future readers see exactly which legacy-handler methods are part of the adapter contract. |
| `prepare()` reentrancy — does the gemini+elevenlabs adapter chain double-init the genai client? | The composable factory wraps a *single* `GeminiTextElevenLabsHandler` instance with all three adapters. When `pipeline.start_up()` runs (in `ComposableConversationHandler.start_up`), each adapter's `prepare()` is called once. `MoonshineSTTAdapter.prepare()` is a no-op (it calls `_start_local_stt_pipeline` which has its own idempotency guard). `GeminiLLMAdapter.prepare()` calls `handler._prepare_startup_credentials()`, which for `GeminiTextElevenLabsResponseHandler` ([`gemini_text_handlers.py:151-177`](../../../src/robot_comic/gemini_text_handlers.py)) (a) calls `ElevenLabsTTSResponseHandler._prepare_startup_credentials` (which sets up httpx + the genai client), (b) initialises the BaseLlama httpx if missing, (c) starts the tool manager, (d) replaces `_gemini_llm` with a `GeminiLLMClient`. Then `ElevenLabsTTSAdapter.prepare()` calls the same `_prepare_startup_credentials()` again — that's the double-init concern. But `ElevenLabsTTSResponseHandler._prepare_startup_credentials` already guards: `if self._http is None: self._http = httpx.AsyncClient(...)` (line 339-340). The genai-client init inside ElevenLabs's `_prepare_startup_credentials` is unconditional (`elevenlabs_tts.py:336-345`) — but **it constructs a fresh `genai.Client` and overwrites the previous one**; the previous client (also created by the same method) is GC'd. No socket leak because `genai.Client` opens connections lazily. The diamond override in `GeminiTextElevenLabsResponseHandler._prepare_startup_credentials` does the same thing — it calls `ElevenLabsTTSResponseHandler._prepare_startup_credentials` *first* (creating a `genai` client), then constructs a `GeminiLLMClient` (a different class). The `genai` client created in ElevenLabs init lives on `self._gemini` (or wherever ElevenLabs stores it); `GeminiLLMClient` lives on `self._gemini_llm`. Calling `_prepare_startup_credentials` twice means: ElevenLabs init runs twice (re-creates `self._gemini`), then `GeminiLLMClient` is constructed twice (re-creates `self._gemini_llm`). The second instantiation is wasteful but correct — no socket leak, no observable behaviour delta from the operator's perspective. **Fixing the double-init in this PR is out of scope** because the same double-init exists today on `(moonshine, llama, elevenlabs)` via the 4b helper (where `GeminiLLMAdapter.prepare` would also re-prep) and on `(moonshine, chatterbox, gemini)` via 4c.2. Pinning this with a test would require either restructuring the adapter prep semantics (a Phase 4 lifecycle-hook concern) or accepting double-init as part of the legacy compatibility surface. We document it and move on. |
| `change_voice` / `get_available_voices` — does `GeminiTextElevenLabsResponseHandler` expose them like `ElevenLabsTTSResponseHandler` does? | Yes — `GeminiTextElevenLabsResponseHandler` explicitly delegates all three voice methods to `ElevenLabsTTSResponseHandler` via diamond shims ([`gemini_text_handlers.py:139-149`](../../../src/robot_comic/gemini_text_handlers.py)). `get_current_voice` is sync, `get_available_voices` / `change_voice` are async — same signatures as the base. The wrapper's `change_voice` and `get_available_voices` forward to `_tts_handler` which is the legacy class instance; the diamond shims route those calls to `ElevenLabsTTSResponseHandler`'s implementation. No additional plumbing needed. |

### 6. Why no new adapter is needed

This is the cheapest sub-phase in 4c precisely because every component already exists:

- `MoonshineSTTAdapter` — Phase 3, untouched.
- `GeminiLLMAdapter` — Phase 4c.2, untouched.
- `ElevenLabsTTSAdapter` — Phase 4a/4b, **only the constructor annotation is broadened**. Runtime behaviour is unchanged.

The unique work is (a) the Protocol definition + annotation swap and (b) the factory helper + composable-gate. No new test infrastructure beyond what 4c.2 set up.

### 7. Bonus cleanup (optional, in scope)

`_build_composable_llama_elevenlabs` (`handler_factory.py:374`) currently uses `cast(Any, legacy)` to satisfy the old `ElevenLabsTTSResponseHandler` annotation against `LocalSTTLlamaElevenLabsHandler` (a structurally compatible but inheritance-unrelated class). With the new Protocol in place, the cast is unnecessary — `LocalSTTLlamaElevenLabsHandler` matches `_ElevenLabsCompatibleHandler` structurally.

If mypy is green after removing `cast(Any, legacy)` and the `Any` import becomes unused, drop both. Otherwise leave the cast for now and follow up in a separate cleanup PR.

## Files Changed

| File | Change |
|------|--------|
| `src/robot_comic/adapters/elevenlabs_tts_adapter.py` | EDIT — define `_ElevenLabsCompatibleHandler` Protocol; broaden constructor annotation; remove the `TYPE_CHECKING` import of `ElevenLabsTTSResponseHandler`. ~25 LOC delta. |
| `src/robot_comic/handler_factory.py` | EDIT — composable gate inside the `LLM_BACKEND_GEMINI` + `AUDIO_OUTPUT_ELEVENLABS` arm; new `_build_composable_gemini_elevenlabs` helper. ~50 LOC delta. Possibly remove the `cast(Any, legacy)` in `_build_composable_llama_elevenlabs` as bonus cleanup. |
| `tests/adapters/test_elevenlabs_tts_adapter.py` | EDIT — add a test that a `GeminiTextElevenLabsResponseHandler`-shaped stub instance is accepted by `ElevenLabsTTSAdapter`. Existing tests must still pass. |
| `tests/test_handler_factory_factory_path.py` | EDIT — add legacy-path regression guard + four composable-path tests for `(moonshine, elevenlabs, gemini)`. Update the existing `test_composable_path_with_gemini_llm_unchanged` to cover a still-unmigrated triple, OR rename to make explicit it's about the legacy default. |

No changes to `composable_pipeline.py`, `composable_conversation_handler.py`, the ABC, `main.py`, or the underlying `gemini_text_handlers.py` / `elevenlabs_tts.py` classes.

## Success Criteria

- `FACTORY_PATH=legacy` (default) + `(moonshine, elevenlabs, gemini)` → `GeminiTextElevenLabsHandler` (bit-for-bit current behaviour).
- `FACTORY_PATH=composable` + same triple → `ComposableConversationHandler` whose pipeline holds `MoonshineSTTAdapter`, `GeminiLLMAdapter`, `ElevenLabsTTSAdapter` — all wrapping a single `GeminiTextElevenLabsHandler` instance.
- `wrapper.copy()` returns a different wrapper whose `_tts_handler` is a different `GeminiTextElevenLabsHandler` instance.
- `(moonshine, llama, elevenlabs)` (4b), `(moonshine, llama, chatterbox)` (4c.1), `(moonshine, chatterbox, gemini)` (4c.2) still return their composable wrappers.
- Every unmigrated triple still returns its legacy concrete class.
- Existing `ElevenLabsTTSAdapter` unit tests still pass (no behaviour regression from the type-annotation broadening).
- New tests cover the broadened annotation + the new factory dispatch.
- Bundled-realtime modes ignore `FACTORY_PATH`.
- `uvx ruff@0.12.0 check` / `format --check` / `.venv/bin/mypy --pretty` / `pytest tests/ -q --ignore=tests/vision/test_local_vision.py` all green from the repo root.

## Test Plan

### Unit test addition (`tests/adapters/test_elevenlabs_tts_adapter.py`)

A single new test pins the broadened annotation: a stub that mimics `GeminiTextElevenLabsResponseHandler`'s shape (no inheritance from `ElevenLabsTTSResponseHandler`, but exposes the four Protocol members) is accepted by the adapter constructor and works through one `synthesize()` round-trip.

| Test | Asserts |
|------|---------|
| `test_adapter_accepts_duck_typed_gemini_elevenlabs_handler_shape` | A stub with no `ElevenLabsTTSResponseHandler` parent that exposes `_prepare_startup_credentials`, `output_queue`, `_stream_tts_to_queue`, and `_http` works end-to-end through `prepare()` → `synthesize()` → `shutdown()`. |

The existing 11 tests in this file already exercise the four Protocol members against `_StubElevenLabsHandler` (which is itself a duck-typed stub — note the existing tests already use `# type: ignore[arg-type]` because the stub doesn't inherit from `ElevenLabsTTSResponseHandler`). Post-broadening, the `type: ignore` comments become unused and should be removed; `warn_unused_ignores = true` in `pyproject.toml` makes mypy flag them.

### Factory dispatch tests (additions to `tests/test_handler_factory_factory_path.py`)

| Test | Asserts |
|------|---------|
| `test_legacy_path_returns_legacy_handler_for_gemini_elevenlabs` | `FACTORY_PATH=legacy` + (gemini, elevenlabs) → `GeminiTextElevenLabsHandler`. |
| `test_composable_path_returns_wrapper_for_gemini_elevenlabs` | `FACTORY_PATH=composable` + same triple → `ComposableConversationHandler` with `GeminiTextElevenLabsHandler` as `_tts_handler`. |
| `test_composable_path_wires_three_adapters_for_gemini_elevenlabs` | `pipeline.stt/llm/tts` are `MoonshineSTTAdapter`, `GeminiLLMAdapter`, `ElevenLabsTTSAdapter`; all wrap the same legacy instance. |
| `test_composable_path_seeds_system_prompt_for_gemini_elevenlabs` | `pipeline._conversation_history[0]` is the patched `get_session_instructions` value. |
| `test_composable_path_copy_constructs_fresh_legacy_for_gemini_elevenlabs` | `copy()` produces a new wrapper + a new `GeminiTextElevenLabsHandler`. |

The existing `test_composable_path_with_gemini_llm_unchanged` test currently patches `GeminiTextElevenLabsHandler` and asserts the legacy class is returned for `(moonshine, elevenlabs, gemini)` — that assertion **becomes wrong** once 4c.3 lands. Replace it with an equivalent test that covers an *actually-still-unmigrated* gemini triple (none of `chatterbox`/`elevenlabs` remain), OR delete it. Easier: delete it. The legacy-path regression guard `test_legacy_path_returns_legacy_handler_for_gemini_elevenlabs` covers the same property for the default path.

### What we don't add tests for

- Hardware barge-in (operator validates on robot after merge).
- End-to-end transcript → audio path with real Gemini/ElevenLabs servers (covered by the legacy handler's existing test suite; this PR doesn't change those flows).
- Voice switching propagation through the wrapper — covered by Phase 4a wrapper tests; the legacy handler exposes the same voice surface via diamond shims so existing tests pass unmodified.
- The double-init of `_prepare_startup_credentials` discussed in §5 — out of scope; existed since 4b, fix is a lifecycle-hook follow-up.

## Migration Notes

- Default behaviour unchanged — operators with `REACHY_MINI_FACTORY_PATH` unset (or `legacy`) keep `GeminiTextElevenLabsHandler`.
- Operators who set `FACTORY_PATH=composable` for the gemini+elevenlabs triple (zero today on main) see `ComposableConversationHandler`. Opt-in.
- Reverting is one env-var flip.
- 4c.4: `LocalSTTGeminiElevenLabsHandler` (the gemini-fallback variant) — same shape, reuses the same three adapters.

## Risks

- **Protocol vs concrete class subtle type drift.** If a future change to `ElevenLabsTTSResponseHandler` adds a new method that `ElevenLabsTTSAdapter` calls, the Protocol must be updated in sync. Mypy will catch the call against the unbroadened Protocol member-list at the adapter call site, so the failure mode is "mypy red", not "silent runtime breakage". Acceptable.
- **`type: ignore[arg-type]` removal in `test_elevenlabs_tts_adapter.py`.** Existing tests pass `_StubElevenLabsHandler` instances which don't inherit from `ElevenLabsTTSResponseHandler` — they currently use `# type: ignore[arg-type]` for two assertions. Post-broadening these become "unused ignore" with `warn_unused_ignores = true`. The plan removes them in the same step that introduces the Protocol; we do not allow the `type: ignore` comments to leak across the broadening.
- **`test_composable_path_with_gemini_llm_unchanged` becomes stale.** Currently asserts elevenlabs+gemini → legacy `GeminiTextElevenLabsHandler`. With this PR it must be either deleted (preferred) or rewritten to cover a different unmigrated combo. The test deletion is part of the same commit as the factory wiring.

## After-merge follow-ups (out of scope for 4c.3)

- 4c.4: `LocalSTTGeminiElevenLabsHandler` — same shape. Reuses the broadened `ElevenLabsTTSAdapter` annotation + `GeminiLLMAdapter`. The legacy handler in `elevenlabs_tts.py:LocalSTTGeminiElevenLabsHandler` is *not* a subclass of `ElevenLabsTTSResponseHandler` — confirm structural Protocol match before wiring.
- 4c.5: build `GeminiTTSAdapter` for the gemini-bundled triple.
- Lifecycle hooks: per-PR rollout per the operating manual.
- `_prepare_startup_credentials` double-init cleanup: tracked as a Phase 4 lifecycle concern when the adapter prep semantics are revisited.
