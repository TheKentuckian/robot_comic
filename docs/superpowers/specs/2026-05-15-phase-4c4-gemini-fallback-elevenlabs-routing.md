# Phase 4c.4 — Composable routing for `(moonshine, elevenlabs, gemini-fallback)`

**Date:** 2026-05-15
**Scope:** Add a composable-path gate inside the `(moonshine, *, elevenlabs)` fallthrough arm of `handler_factory.py` (the "gemini-fallback" arm) and reuse Phase 4c.3's `_build_composable_gemini_elevenlabs` helper for it. Under `REACHY_MINI_FACTORY_PATH=legacy` (default), the existing `LocalSTTGeminiElevenLabsHandler` (aliased `LocalSTTElevenLabsHandler`) is still returned — zero behaviour change for default users. Tests under `tests/test_handler_factory_factory_path.py`.
**Epic:** #337 — Pipeline refactor (Option C, incremental retirement)
**Predecessors:** Phase 4c.3 (#364 / `eb45e14`) — `_build_composable_gemini_elevenlabs` helper landed and the `_ElevenLabsCompatibleHandler` Protocol broadened `ElevenLabsTTSAdapter` to accept the diamond-MRO `GeminiTextElevenLabsResponseHandler`.
**Successors:** Phase 4c.5 — `GeminiTTSAdapter` + `(moonshine, gemini_tts)` routing.

## Background

`FACTORY_PATH=composable` already covers four triples:

- `(moonshine, llama, elevenlabs)` — Phase 4b.
- `(moonshine, llama, chatterbox)` — Phase 4c.1.
- `(moonshine, chatterbox, gemini)` — Phase 4c.2.
- `(moonshine, elevenlabs, gemini)` — Phase 4c.3 (#364).

The next triple to migrate is `(moonshine, elevenlabs, gemini-fallback)`. The legacy class is `LocalSTTGeminiElevenLabsHandler` (`elevenlabs_tts.py:1053`), aliased as `LocalSTTElevenLabsHandler` for backward compatibility.

### What is "gemini-fallback"?

The exploration memo §1 enumerates the factory dispatch matrix:

> - `(moonshine, elevenlabs, llama)` → `LocalSTTLlamaElevenLabsHandler`
> - `(moonshine, chatterbox, gemini)` → `GeminiTextChatterboxHandler`
> - `(moonshine, elevenlabs, gemini)` → `GeminiTextElevenLabsHandler`
> - `(moonshine, chatterbox, *)` → `LocalSTTChatterboxHandler` (llama by default)
> - `(moonshine, gemini_tts, *)` → `LocalSTTGeminiTTSHandler`
> - `(moonshine, elevenlabs, gemini-fallback)` → `LocalSTTGeminiElevenLabsHandler`

The literal source-of-truth dispatch in `handler_factory.py` shows the "gemini-fallback" arm is the **outer-`if input_backend == AUDIO_INPUT_MOONSHINE` fallthrough** at lines 314–322:

```python
if input_backend == AUDIO_INPUT_MOONSHINE:
    if output_backend == AUDIO_OUTPUT_ELEVENLABS:
        from robot_comic.elevenlabs_tts import LocalSTTGeminiElevenLabsHandler
        logger.info(...)
        return LocalSTTGeminiElevenLabsHandler(**handler_kwargs)
```

Walking the control flow:

1. `_llm_backend = getattr(config, "LLM_BACKEND", LLM_BACKEND_LLAMA)` (line 181 — default `llama`).
2. `_llm_backend == LLM_BACKEND_LLAMA` arm matches `output == elevenlabs` → returns `LocalSTTLlamaElevenLabsHandler` at line 207.
3. `_llm_backend == LLM_BACKEND_GEMINI` arm matches `output == elevenlabs` → returns `GeminiTextElevenLabsHandler` (or composable wrapper under 4c.3) at line 276.
4. **Both arms return**, so the outer-`if` arm at line 314 (`LocalSTTGeminiElevenLabsHandler`) is **only reached when `LLM_BACKEND` is neither `"llama"` nor `"gemini"`**.

`config.py:218-221` defines exactly two `LLM_BACKEND_*` constants and `config.py:1031` sets the runtime default to `LLM_BACKEND_LLAMA` if the env var is unset. The `getattr` default at factory line 181 mirrors that.

**So the "gemini-fallback" branch is effectively unreachable in production**: it fires only when an operator sets `REACHY_MINI_LLM_BACKEND` to a non-`llama`/`gemini` string (typo, empty after strip, removed value). It also fires for any test that constructs `config` without setting `LLM_BACKEND` (the `getattr` default catches it).

The name "fallback" captures exactly this: it's the legacy code's safety net for unknown `LLM_BACKEND` values. Historically (pre-PR #215) it was the main path for elevenlabs+moonshine before the explicit `LLM_BACKEND` arms were carved out.

### How `LocalSTTGeminiElevenLabsHandler` differs from `GeminiTextElevenLabsHandler`

Both share the high-level triple semantics — Moonshine STT + ElevenLabs TTS + Gemini-API LLM. The differences live in the LLM call path:

| Aspect | `GeminiTextElevenLabsHandler` (4c.3 target) | `LocalSTTGeminiElevenLabsHandler` (4c.4 target) |
|--------|----------------------------------------------|--------------------------------------------------|
| MRO | `LocalSTTInputMixin, GeminiTextElevenLabsResponseHandler` | `LocalSTTInputMixin, ElevenLabsTTSResponseHandler` |
| Diamond-MRO base | `GeminiTextResponseHandler, ElevenLabsTTSResponseHandler` | (no diamond — only `ElevenLabsTTSResponseHandler`) |
| `_call_llm` | **Inherited from `GeminiTextResponseHandler`** (via `BaseLlamaResponseHandler`). Returns `(text, raw_tool_calls, raw_msg)`. | **Not defined.** |
| LLM client field | `self._gemini_llm: GeminiLLMClient` — wraps `genai.Client` and normalises tool-call shape to llama-server-compatible dicts before returning from `_call_llm`. | `self._client: genai.Client` (raw) — `ElevenLabsTTSResponseHandler._prepare_startup_credentials` initialises it directly. |
| LLM loop | `BaseLlamaResponseHandler._call_llm` → handles history, tools, retries. | `ElevenLabsTTSResponseHandler._run_llm_with_tools` (line 743) — a self-contained Gemini-specific tool loop that calls `self._client.aio.models.generate_content(...)` directly. |
| `_prepare_startup_credentials` | Overridden in `GeminiTextElevenLabsResponseHandler` to call ElevenLabs prep, then construct `GeminiLLMClient`. | Inherited from `ElevenLabsTTSResponseHandler`. Constructs `genai.Client` directly — no `GeminiLLMClient`. |

The empirical confirmation (manually verified locally):

```
>>> from robot_comic.elevenlabs_tts import LocalSTTGeminiElevenLabsHandler
>>> [c.__name__ for c in LocalSTTGeminiElevenLabsHandler.__mro__]
['LocalSTTGeminiElevenLabsHandler', 'LocalSTTInputMixin', 'ElevenLabsTTSResponseHandler',
 'AsyncStreamHandler', 'StreamHandlerBase', 'ConversationHandler', 'ABC', 'object']
>>> hasattr(LocalSTTGeminiElevenLabsHandler, '_call_llm')
False
>>> hasattr(LocalSTTGeminiElevenLabsHandler, '_run_llm_with_tools')
True
```

### Implications for composable wiring

`GeminiLLMAdapter.chat` (Phase 4c.2, untouched) calls `self._handler._call_llm(extra_messages=messages)`. `LocalSTTGeminiElevenLabsHandler` does not implement `_call_llm`. **`GeminiLLMAdapter` cannot wrap `LocalSTTGeminiElevenLabsHandler` directly.**

Three options:

1. **Build a new `LocalSTTGeminiElevenLabsLLMAdapter`** that wraps `_run_llm_with_tools` instead of `_call_llm`. ~120 LOC of new code.
2. **Modify `GeminiLLMAdapter`** to detect which method is available and dispatch. Violates the briefing's "Do not modify any adapter" rule.
3. **For the composable path, instantiate `GeminiTextElevenLabsHandler` instead of `LocalSTTGeminiElevenLabsHandler`**, and reuse the existing 4c.3 helper as-is. Zero new adapter code. The legacy-path remains untouched and still returns `LocalSTTGeminiElevenLabsHandler` for bit-for-bit compatibility.

**Resolution: option 3.** Rationale:

- The composable triple's identity is `(moonshine, elevenlabs, gemini-LLM)` regardless of which concrete legacy class hosts the LLM call. `GeminiTextElevenLabsHandler` is a fully equivalent host that exposes the `_call_llm` surface the existing adapter expects.
- The `LocalSTTGeminiElevenLabsHandler` arm is a fallback-for-unknown-`LLM_BACKEND`-values path. In production, operators either set `LLM_BACKEND=gemini` (already routed by 4c.3) or accept the default `llama` (routed to the llama-elevenlabs arm). The "gemini-fallback" composable path is opt-in via the same `FACTORY_PATH=composable` flag, and choosing it for an unknown-`LLM_BACKEND` value means the operator is electing the composable Gemini behaviour anyway.
- The legacy path is preserved untouched: under `FACTORY_PATH=legacy` (default), the factory still returns `LocalSTTGeminiElevenLabsHandler` from the outer-`if` arm at line 314. Zero behaviour change for default users — the operator-facing contract is intact.
- Phase 4e will retire `LocalSTTGeminiElevenLabsHandler` and the entire dispatch fallthrough; consolidating onto `GeminiTextElevenLabsHandler` early matches that future shape.
- The briefing's open question 3 ("Is the existing `_build_composable_gemini_elevenlabs` reusable as-is?") answers: **yes**, the helper is byte-identical because both composable paths construct the same `GeminiTextElevenLabsHandler` legacy.

The alternative (option 1, new adapter) carries cost without benefit: a ~120 LOC `_run_llm_with_tools`-wrapping adapter that will be deleted in Phase 4e (when `LocalSTTGeminiElevenLabsHandler` is removed). The composable path is already gemini-flavoured; there is no operator-observable distinction in behaviour between "gemini LLM via `GeminiLLMClient`" and "gemini LLM via raw `genai.Client`" beyond the tool-call-shape normalisation that `GeminiLLMClient` provides — which is the *correct* path for tool dispatch.

## Goal

1. Add a composable-path gate inside the outer-`if input_backend == AUDIO_INPUT_MOONSHINE` arm of `handler_factory.py` (around line 314, the `output_backend == AUDIO_OUTPUT_ELEVENLABS` block) that:
   - When `FACTORY_PATH == FACTORY_PATH_COMPOSABLE` → calls the existing 4c.3 `_build_composable_gemini_elevenlabs(**handler_kwargs)`.
   - Otherwise (default `legacy`) → returns `LocalSTTGeminiElevenLabsHandler(**handler_kwargs)` (current behaviour).
2. Reuse the existing 4c.3 helper unchanged. No new helper, no new adapter.
3. Pin both branches with tests in `tests/test_handler_factory_factory_path.py`.

## Out of scope (deferred)

| Item | Sub-phase |
|------|-----------|
| `(moonshine, gemini_tts)` — `LocalSTTGeminiTTSHandler` | 4c.5 (needs `GeminiTTSAdapter`) |
| Sibling `HybridRealtimePipeline` for `LocalSTT*RealtimeHandler` | 4c-tris |
| Flip default to `composable` | 4d |
| Delete legacy handlers (including `LocalSTTGeminiElevenLabsHandler`) | 4e |
| Retire `BACKEND_PROVIDER` | 4f |
| Lifecycle hooks: telemetry, boot-timeline events (#321), joke history, history trim, echo-guard timestamps | Per-hook follow-up PRs |
| Building a `_run_llm_with_tools`-wrapping adapter (option 1 above) | Out of scope; option 3 consolidates onto the existing path |

## Design

### 1. Factory branch wiring

The outer-`if input_backend == AUDIO_INPUT_MOONSHINE` arm at lines 293–322 of `handler_factory.py` currently looks like:

```python
if input_backend == AUDIO_INPUT_MOONSHINE:
    if output_backend == AUDIO_OUTPUT_CHATTERBOX:
        ...
    if output_backend == AUDIO_OUTPUT_GEMINI_TTS:
        ...
    if output_backend == AUDIO_OUTPUT_ELEVENLABS:
        from robot_comic.elevenlabs_tts import LocalSTTGeminiElevenLabsHandler
        logger.info(
            "HandlerFactory: selecting LocalSTTGeminiElevenLabsHandler (%s → %s)",
            input_backend, output_backend,
        )
        return LocalSTTGeminiElevenLabsHandler(**handler_kwargs)
    ...
```

The 4c.4 change prepends a composable check inside the elevenlabs sub-block, mirroring 4c.1/4c.2/4c.3:

```python
if output_backend == AUDIO_OUTPUT_ELEVENLABS:
    # Phase 4c.4 (#337): the "gemini-fallback" arm for elevenlabs is
    # reached when LLM_BACKEND is neither "llama" nor "gemini"; for
    # the composable path we route through the same builder used by
    # the LLM_BACKEND=gemini arm (4c.3) because the underlying triple
    # is the same (moonshine + elevenlabs + Gemini-API LLM), just
    # reached through a different dispatch condition. The composable
    # path's GeminiLLMAdapter requires a handler with _call_llm
    # (i.e. GeminiTextElevenLabsHandler), which the 4c.3 builder
    # already constructs.
    if getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY) == FACTORY_PATH_COMPOSABLE:
        logger.info(
            "HandlerFactory: selecting ComposableConversationHandler "
            "(%s → %s, llm=gemini-fallback, factory_path=composable)",
            input_backend,
            output_backend,
        )
        return _build_composable_gemini_elevenlabs(**handler_kwargs)
    from robot_comic.elevenlabs_tts import LocalSTTGeminiElevenLabsHandler

    logger.info(
        "HandlerFactory: selecting LocalSTTGeminiElevenLabsHandler (%s → %s)",
        input_backend,
        output_backend,
    )
    return LocalSTTGeminiElevenLabsHandler(**handler_kwargs)
```

### 2. Helper reuse — no new code

The existing `_build_composable_gemini_elevenlabs(**handler_kwargs)` (Phase 4c.3, `handler_factory.py:492-537`) is reused verbatim. It already:

- Imports `GeminiTextElevenLabsHandler`, `GeminiLLMAdapter`, `MoonshineSTTAdapter`, `ElevenLabsTTSAdapter`, `ComposablePipeline`, `ComposableConversationHandler`.
- Wraps the legacy `GeminiTextElevenLabsHandler` in the three adapters, all sharing the same instance.
- Returns a `ComposableConversationHandler` with a `_build` closure for FastRTC `copy()`.

The `_ElevenLabsCompatibleHandler` Protocol (4c.3) already accepts `GeminiTextElevenLabsResponseHandler` structurally. No adapter modifications needed.

### 3. Resolution of open design questions from the briefing

| Question | Resolution |
|----------|-----------|
| What distinguishes "gemini" from "gemini-fallback" in the legacy factory? | **`LLM_BACKEND` value.** `LLM_BACKEND=gemini` → the inner `LLM_BACKEND_GEMINI` arm at factory line 229 (handles elevenlabs at line 254 → `GeminiTextElevenLabsHandler`). `LLM_BACKEND` neither `llama` nor `gemini` (typo / unset to non-default / empty) → falls through both inner arms to the outer `input_backend == AUDIO_INPUT_MOONSHINE` arm at line 293 → for elevenlabs at line 314 → `LocalSTTGeminiElevenLabsHandler`. Default `LLM_BACKEND` is `llama`, not `gemini-fallback`; the "gemini-fallback" name is descriptive of the historical hardcoded-Gemini behaviour of `ElevenLabsTTSResponseHandler`, not a literal config value. |
| Does `LocalSTTGeminiElevenLabsHandler.start_up` and tool-loop differ from `GeminiTextElevenLabsHandler`'s? | **Yes.** `LocalSTTGeminiElevenLabsHandler` lacks `_call_llm`; it uses `ElevenLabsTTSResponseHandler._run_llm_with_tools` which directly invokes `self._client.aio.models.generate_content(...)`. `GeminiTextElevenLabsHandler` exposes `_call_llm` via `GeminiTextResponseHandler` and uses `self._gemini_llm = GeminiLLMClient(...)` which normalises tool-call shapes before they reach the orchestrator. The two paths produce equivalent operator-observable behaviour but route through different code surfaces. **For the composable wiring we MUST use `GeminiTextElevenLabsHandler`** because `GeminiLLMAdapter.chat` calls `_call_llm`. |
| Is the existing `_build_composable_gemini_elevenlabs` from 4c.3 reusable as-is? | **Yes.** This sub-phase is purely a factory-dispatch addition. The 4c.3 helper is byte-identical to what 4c.4 needs. The two triples collapse to identical composable wiring; the only difference is the dispatch condition that picks the helper. |

### 4. Why this approach over building a new adapter

Building `LocalSTTGeminiElevenLabsLLMAdapter` (option 1 from §Implications above) would:

- Add ~120 LOC of new adapter wrapping `_run_llm_with_tools`, which has a different return shape (single string instead of `(text, tool_calls, msg)`). To match the `LLMResponse` Protocol the adapter would need to: (a) parse the string for inline tool-call markers — but `_run_llm_with_tools` already dispatches tools internally so there are none to expose; (b) return empty `tool_calls` and the response string only — which means the orchestrator's `_run_llm_loop_and_speak` tool-round logic is bypassed entirely, breaking the composable pipeline's tool model.
- Be deleted in Phase 4e anyway when `LocalSTTGeminiElevenLabsHandler` is retired.
- Doubly duplicate Gemini-API client construction code that already lives in `GeminiLLMClient`.

The composable representation is *intentionally* a normalised representation: one adapter per LLM API. Keeping two adapters for the same API to preserve a fallback-dispatch quirk would be a design regression. Phase 4e will retire both `LocalSTTGeminiElevenLabsHandler` and this dispatch arm; the composable path can land its target shape now.

### 5. Edge cases handled

- **`FACTORY_PATH` env-var unset**: `getattr(config, "FACTORY_PATH", FACTORY_PATH_LEGACY)` defaults to `legacy`. The legacy arm is taken. `LocalSTTGeminiElevenLabsHandler` is returned. Zero behaviour change.
- **`LLM_BACKEND` typo or empty after strip**: `_llm_backend == LLM_BACKEND_LLAMA` and `_llm_backend == LLM_BACKEND_GEMINI` both fail, control falls through to the outer `input_backend == AUDIO_INPUT_MOONSHINE` arm. Under `FACTORY_PATH=legacy` this returns `LocalSTTGeminiElevenLabsHandler` (current behaviour). Under `FACTORY_PATH=composable` this returns the 4c.3 composable wrapper. Both branches are tested in §Test Plan.
- **`FACTORY_PATH=composable` + `LLM_BACKEND=gemini` + elevenlabs**: 4c.3's gate at line 259 fires first and returns the composable wrapper. The 4c.4 gate at line 314 is unreachable. No double-dispatch.
- **`FACTORY_PATH=composable` + `LLM_BACKEND=llama` + elevenlabs**: 4b's gate at line 190 fires first and returns the llama-elevenlabs composable wrapper. The 4c.4 gate is unreachable.
- **Voice / personality methods**: the composable wrapper forwards `change_voice` / `get_available_voices` / `get_current_voice` to the underlying `GeminiTextElevenLabsHandler`, which (via the diamond-MRO shim at `gemini_text_handlers.py:139-149`) routes those calls to `ElevenLabsTTSResponseHandler` — same as 4c.3. No additional plumbing needed.

## Files Changed

| File | Change |
|------|--------|
| `src/robot_comic/handler_factory.py` | EDIT — prepend composable gate inside the outer-`if input_backend == AUDIO_INPUT_MOONSHINE` arm's `output_backend == AUDIO_OUTPUT_ELEVENLABS` block (around line 314). Reuses existing `_build_composable_gemini_elevenlabs` helper unchanged. ~15 LOC delta. |
| `tests/test_handler_factory_factory_path.py` | EDIT — add a Phase 4c.4 section with legacy-path regression guard + composable-path dispatch tests for the gemini-fallback triple. |

**No changes** to: any adapter module, `composable_pipeline.py`, `composable_conversation_handler.py`, the `ConversationHandler` ABC, `main.py`, `elevenlabs_tts.py`, `gemini_text_handlers.py`, `gemini_llm.py`, `config.py`, or the `prompts` module.

**No new files.** The helper, the adapter Protocol, the adapter modules, and the legacy class are all reused unchanged.

## Success Criteria

- `FACTORY_PATH=legacy` (default) + `LLM_BACKEND` not `llama`/`gemini` + `(moonshine, elevenlabs)` → `LocalSTTGeminiElevenLabsHandler` (bit-for-bit current behaviour).
- `FACTORY_PATH=composable` + same dispatch trigger → `ComposableConversationHandler` whose pipeline holds `MoonshineSTTAdapter`, `GeminiLLMAdapter`, `ElevenLabsTTSAdapter`, all wrapping a single `GeminiTextElevenLabsHandler` instance.
- `wrapper.copy()` returns a different wrapper whose `_tts_handler` is a different `GeminiTextElevenLabsHandler` instance.
- All previously-migrated triples ((moonshine, llama, elevenlabs) — 4b; (moonshine, llama, chatterbox) — 4c.1; (moonshine, chatterbox, gemini) — 4c.2; (moonshine, elevenlabs, gemini) — 4c.3) still return their composable wrappers.
- Every unmigrated triple still returns its legacy concrete class.
- Existing tests still pass (no behaviour regression).
- New tests cover both branches of the new gate.
- Bundled-realtime modes ignore `FACTORY_PATH`.
- `uvx ruff@0.12.0 check` / `format --check` / `.venv/bin/mypy --pretty` / `pytest tests/ -q --ignore=tests/vision/test_local_vision.py` all green from the repo root.

## Test Plan

### Factory dispatch tests (additions to `tests/test_handler_factory_factory_path.py`)

Mirrors the 4c.3 section structure. The dispatch trigger is `LLM_BACKEND` set to a non-default value (e.g. an empty string after strip, or any string other than `"llama"`/`"gemini"`). We use a clearly-invalid sentinel like `"unknown"` to exercise the fallthrough path explicitly.

| Test | Asserts |
|------|---------|
| `test_legacy_path_returns_legacy_handler_for_gemini_fallback_elevenlabs` | `FACTORY_PATH=legacy` + `LLM_BACKEND="unknown"` + (moonshine, elevenlabs) → `LocalSTTGeminiElevenLabsHandler`. |
| `test_composable_path_returns_wrapper_for_gemini_fallback_elevenlabs` | `FACTORY_PATH=composable` + same dispatch → `ComposableConversationHandler` with `GeminiTextElevenLabsHandler` as `_tts_handler` (NOT `LocalSTTGeminiElevenLabsHandler` — composable path consolidates onto the `_call_llm`-capable class). |
| `test_composable_path_wires_three_adapters_for_gemini_fallback_elevenlabs` | `pipeline.stt/llm/tts` are `MoonshineSTTAdapter`, `GeminiLLMAdapter`, `ElevenLabsTTSAdapter`; all wrap the same `GeminiTextElevenLabsHandler` instance. |
| `test_composable_path_seeds_system_prompt_for_gemini_fallback_elevenlabs` | `pipeline._conversation_history[0]` is the patched `get_session_instructions` value. |
| `test_composable_path_copy_constructs_fresh_legacy_for_gemini_fallback_elevenlabs` | `copy()` produces a new wrapper + a new `GeminiTextElevenLabsHandler`. |

All five tests follow the existing `_fake_cls` patching style. The composable-path tests patch `robot_comic.gemini_text_handlers.GeminiTextElevenLabsHandler` (NOT `robot_comic.elevenlabs_tts.LocalSTTGeminiElevenLabsHandler`) because the composable path constructs the gemini-text class. The legacy-path test patches `robot_comic.elevenlabs_tts.LocalSTTGeminiElevenLabsHandler`.

### What we don't add tests for

- Hardware barge-in (operator validates on robot after merge).
- End-to-end transcript → audio path with real Gemini/ElevenLabs servers (covered by the legacy handler's existing test suite; this PR doesn't change those flows).
- Voice switching propagation through the wrapper — covered by Phase 4a wrapper tests; `GeminiTextElevenLabsHandler` exposes the same voice surface via diamond shims so existing tests pass unmodified.

## Migration Notes

- Default behaviour unchanged — operators with `REACHY_MINI_FACTORY_PATH` unset (or `legacy`) keep `LocalSTTGeminiElevenLabsHandler` for the fallback dispatch trigger.
- Operators who set `FACTORY_PATH=composable` AND somehow trigger the gemini-fallback arm (`LLM_BACKEND` typo or unrecognised value) see a `ComposableConversationHandler` wrapping `GeminiTextElevenLabsHandler` — semantically equivalent to setting `LLM_BACKEND=gemini` and using the 4c.3 path. Effectively no operator should land here in practice; the path exists for robustness.
- Reverting is one env-var flip (`FACTORY_PATH=legacy`).
- Phase 4e will delete `LocalSTTGeminiElevenLabsHandler` and this entire dispatch fallthrough.

## Risks

- **The composable path silently swaps the legacy class.** A test that asserts the `_tts_handler` is a `LocalSTTGeminiElevenLabsHandler` instance under `FACTORY_PATH=composable` would fail. This is by design (see §Implications) and pinned by the new test `test_composable_path_returns_wrapper_for_gemini_fallback_elevenlabs` (asserts the `_tts_handler` is `GeminiTextElevenLabsHandler`, not the original legacy class). No such cross-class assertion exists in the codebase today.
- **`LLM_BACKEND="unknown"` sentinel may collide with future config validation.** If `config.py` adds a runtime allowlist for `LLM_BACKEND`, the test would break. Acceptable — the validation should land with its own tests; this PR's test only needs *any* value that triggers the fallthrough, which can be reselected at that time.
- **Operator surprise** if someone is intentionally relying on the legacy `LocalSTTGeminiElevenLabsHandler` behaviour (e.g. for the `_run_llm_with_tools` quirks) and flips `FACTORY_PATH=composable`. There is no evidence anyone does. The `LocalSTTGeminiElevenLabsHandler` arm is unreachable under default `LLM_BACKEND=llama`, the documented operator setting.

## After-merge follow-ups (out of scope for 4c.4)

- 4c.5: `LocalSTTGeminiTTSHandler` — needs a new `GeminiTTSAdapter`.
- Lifecycle hooks: per-PR rollout per the operating manual.
- Phase 4e: delete `LocalSTTGeminiElevenLabsHandler` + the gemini-fallback dispatch fallthrough.
