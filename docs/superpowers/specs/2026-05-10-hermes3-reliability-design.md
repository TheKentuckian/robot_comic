# Hermes3 Reliability — Prompt Hardening + Retry-with-Nudge

**Date:** 2026-05-10  
**Scope:** `src/robot_comic/chatterbox_tts.py`  
**Issue:** #52

## Background

The Chatterbox TTS pipeline uses Hermes3 8B via Ollama `/api/chat` for text generation and tool dispatch. Two reliability issues have already been addressed:

- **Token bloat** (`bbf526e`): `_trim_tool_spec()` caps tool descriptions so total prompt stays under ~1224 tokens, preventing Hermes3 from reverting to its "I am Hermes" persona.
- **Text-format tool calls** (`becccef`): regex fallback catches `{function:greet, action:scan}` emitted in the content field instead of the structured `tool_calls` field.

This spec covers speculative hardening to make tool-call behaviour more consistent — not fixing a specific observed failure.

## Design

### 1. Prompt Hardening

Inject a short Hermes3/Ollama-specific tool-use addendum directly in `_call_llm()`, appended to the system message at call time. This keeps persona instructions cleanly in profile `instructions.txt` files and model-plumbing instructions in the handler.

The addendum instructs Hermes3 to:
- Always emit tool calls via the structured `tool_calls` field
- Never embed tool calls as text in the content field
- If a tool call is required, output only the tool call — no explanatory prose alongside it

The addendum is ~5 lines, keeping the combined prompt well under the ~1224-token budget restored by `_trim_tool_spec()`.

**Where:** Appended to `system_prompt` string inside `_call_llm()`, not in any profile file.

### 2. Parser Improvements

Two targeted fixes to the text-format fallback in `_call_llm()`:

**A. JSON-in-content detection**  
After the existing `_TEXT_TOOL_CALL_RE` check, attempt `json.loads()` on the content field. If it parses to a dict with a recognisable tool-call structure (has a `"name"` or `"function"` key), extract tool name and arguments. This catches OpenAI-style JSON Hermes3 occasionally emits in the content field.

**B. Arg parser fix**  
`_parse_text_tool_args` currently splits on `,` which breaks when argument values contain commas. New order:
1. Try `json.loads()` on the args substring — handles `{"action": "scan"}`
2. Fall back to the existing `key:val` comma-split heuristic

Both changes stay inside existing function boundaries — no new files or abstractions.

### 3. Retry-with-Nudge

After a successful network response that is semantically empty (no `tool_calls` and no meaningful text), fire one ephemeral retry:

1. Build a temporary messages list: `existing_history + empty_assistant_turn + {"role": "user", "content": "Please use a tool call now."}`
2. POST to Ollama with the ephemeral messages — **not saved to `_conversation_history`**
3. Apply the same text-format and JSON-in-content detection to the nudge response
4. If the nudge response is also empty, log a warning and return `("", [])` — no crash, no TTS garbage spoken

**Trigger condition:** `not tool_calls and not text` after parsing.  
**Max nudges per turn:** 1 (controlled by a local `nudge_attempted` bool inside the attempt loop).  
**Relationship to existing retries:** Orthogonal. `_LLM_MAX_RETRIES` handles network/HTTP failures; the nudge handles semantic emptiness on a successful response.

## Files Changed

| File | Change |
|------|--------|
| `src/robot_comic/chatterbox_tts.py` | All three changes — addendum injection, parser fix, nudge logic |

No new files. No changes to profiles, config, or other handlers.

## Success Criteria

- Hermes3 uses structured `tool_calls` more consistently across turns
- Malformed text-format tool calls in JSON format are caught and dispatched
- Arg values containing commas parse correctly
- When Hermes3 returns an empty response, one nudge recovers the turn without polluting history
- No increase in average latency on turns where Hermes3 responds correctly
