**Phase 4a (#337) landed 2026-05-15 as PR #355 → commit c8de597 on main.**

## What's in main now

- `src/robot_comic/composable_conversation_handler.py` — `ComposableConversationHandler(ConversationHandler)` wrapping `ComposablePipeline`. ABC-complete (copy, start_up, shutdown, receive, emit, apply_personality, change_voice, get_available_voices, get_current_voice). NOT yet routed by the factory.
- `tests/test_composable_conversation_handler.py` — 13 tests, all green.
- `docs/superpowers/specs/2026-05-15-phase-4a-composable-conversation-handler.md` — spec.
- `docs/superpowers/plans/2026-05-15-phase-4a-composable-conversation-handler.md` — TDD plan.

## Confirmed Phase 4 plan (Option C, EXPANDED from memo's original 5–7 sessions)

- **4a** — wrapper. ✅ DONE
- **4b** — factory dual path behind `REACHY_MINI_FACTORY_PATH=legacy|composable`, default `legacy`. First triple: `(moonshine, llama, elevenlabs)`.
- **4c** — expand to remaining composable triples. Includes building `ChatterboxTTSAdapter` and `GeminiLLMAdapter` (operator chose to ship Gemini adapter in Phase 4, not defer to 5).
- **4c-tris** — sibling `HybridRealtimePipeline` class for `LocalSTTOpenAIRealtimeHandler` + `LocalSTTHuggingFaceRealtimeHandler` (memo's STT/LLM/TTS Protocol doesn't fit; needs separate design memo first).
- **4d** — flip default to `composable`.
- **4e** — delete legacy concrete handlers + orphan `LocalSTTLlamaGeminiTTSHandler` (confirmed unreachable from factory today — dead code, not migrated).
- **4f** — retire `BACKEND_PROVIDER` and `LOCAL_STT_RESPONSE_BACKEND` config dials (touches main.py, profiles, env templates, deploy scripts — operator chose to include in Phase 4, not defer to 5).

## Lifecycle hooks deliberately NOT plumbed in 4a

Each gets its own small follow-up PR between 4b and 4d:

- `telemetry.record_llm_duration`
- boot-timeline supporting events (#321)
- `record_joke_history` (`llama_base.py:553-568`)
- `history_trim.trim_history_in_place`
- `_speaking_until` echo-guard timestamps (`elevenlabs_tts.py:471-473`)

Total Phase 4 estimate: ~10–14 sessions for the full stack.

## Remote-execution container bootstrap

The `/venvs/apps_venv` path in `CLAUDE.md` is a local-machine convention and does not exist in the remote execution environment. To bootstrap a fresh container:

```
apt-get install -y libgirepository1.0-dev
uv sync --frozen --all-extras --group dev
# tests: .venv/bin/pytest tests/ -q
```
