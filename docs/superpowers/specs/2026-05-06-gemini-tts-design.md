# Gemini TTS Integration Design

**Date:** 2026-05-06
**Status:** Approved

---

## Overview

Add `gemini_tts` as a selectable output backend under the existing `local_stt` (Custom audio path) flow. Moonshine STT runs on-device as before; when the user selects Gemini TTS as the output option, transcripts are routed to a Gemini text model for response + tool handling, then to `gemini-3.1-flash-tts-preview` for voice synthesis. Audio returns at 24kHz PCM, same as the rest of the pipeline.

This is Option A of three discussed approaches — the fastest path to ship and test, using the established mixin composition pattern. Option C (fully modular pipeline with independent `AUDIO_INPUT_BACKEND`/`AUDIO_OUTPUT_BACKEND` config dimensions) is deferred to end of task queue.

---

## What Gets Built

```
src/robot_comic/
└── gemini_tts.py                  — GeminiTTSResponseHandler + LocalSTTGeminiTTSHandler

Modified:
├── local_stt_realtime.py          — extract _dispatch_completed_transcript() hook
├── config.py                      — add "gemini_tts" to LOCAL_STT_RESPONSE_BACKEND_CHOICES
├── main.py                        — dispatch case for gemini_tts response backend
├── console.py                     — credential check + persistence for gemini_tts
└── static/
    ├── index.html                 — enable disabled Gemini TTS output card
    └── main.js                    — enable card, update journeyMeta(), setSelectedLocalSTTOutput()

tests/
├── test_gemini_tts_handler.py     — unit tests for GeminiTTSResponseHandler
└── test_local_stt_dispatch_hook.py — regression tests for existing mixin dispatch
```

---

## Architecture

### Handler Composition

```
ConversationHandler (ABC)
  └── GeminiTTSResponseHandler     — response cycle: history, LLM, tools, TTS
        └── LocalSTTGeminiTTSHandler(LocalSTTInputMixin, GeminiTTSResponseHandler)
              ^-- mixin contributes Moonshine STT input
```

`GeminiTTSResponseHandler` extends `ConversationHandler` directly (not `BaseRealtimeHandler`) — there is no WebSocket connection.

`LocalSTTGeminiTTSHandler` is the concrete class registered with `main.py` and `console.py`. It is instantiated when `BACKEND_PROVIDER=local_stt` and `LOCAL_STT_RESPONSE_BACKEND=gemini_tts`.

### Mixin Refactor (`local_stt_realtime.py`)

`_handle_local_stt_event` currently inlines the "completed transcript → OpenAI WebSocket" dispatch. Extract this into an overrideable method:

```python
async def _dispatch_completed_transcript(self, transcript: str) -> None:
    """Send a completed transcript to the response backend. Override in subclasses."""
    # existing OpenAI realtime path (unchanged)
    await self.connection.conversation.item.create(...)
    await self._safe_response_create(...)
```

The two existing handlers (`LocalSTTOpenAIRealtimeHandler`, `LocalSTTHuggingFaceRealtimeHandler`) inherit this default — zero behavior change. `LocalSTTGeminiTTSHandler` overrides it to call the Gemini response cycle instead.

---

## Data Flow

### Per-Utterance Cycle

```
Microphone audio
  → LocalSTTInputMixin.receive()       — feeds Moonshine
  → _handle_local_stt_event("completed", transcript)
  → _dispatch_completed_transcript(transcript)   ← overridden in LocalSTTGeminiTTSHandler
      1. Append user message to conversation history
      2. Call gemini-2.5-flash with history + system prompt + tool specs
      3. Execute tool calls (loop until final text)
      4. Append assistant response to history
      5. Call gemini-3.1-flash-tts-preview (voice: Algenib)
      6. Decode base64 PCM blob → chunk into ~2400-sample (100ms at 24kHz) frames
      7. Push frames to output_queue
  → LocalStream.play_loop() reads output_queue → robot speaker
```

### Conversation History

Maintained as a `list[dict]` with `{"role": "user"|"model", "parts": [...]}` format (Gemini SDK shape). Scoped to the session — initialized on `start_up()`, cleared on `shutdown()`. Tool call/result round-trips are included as intermediate history entries.

### Voice Styling

TTS tag injection is handled at the profile layer, not in the handler. The Gemini text model system prompt (from `instructions.txt`) directs the model to embed inline audio tags in its responses. For Don Rickles, instructions should include guidance to use `[fast]`, `[annoyance]`, `[amusement]`, `[aggression]`, and similar tags naturally. For profiles that don't specify tag behavior, no tags are injected.

This keeps the handler tag-agnostic and the styling configurable per personality.

---

## Configuration

### `config.py` changes

```python
GEMINI_TTS_OUTPUT = "gemini_tts"   # constant for the new response backend choice
LOCAL_STT_RESPONSE_BACKEND_CHOICES = (OPENAI_BACKEND, HF_BACKEND, GEMINI_TTS_OUTPUT)
```

Default voice constant added alongside existing voice constants:
```python
GEMINI_TTS_DEFAULT_VOICE = "Algenib"
```

LLM model for response generation (separate from TTS model):
```python
GEMINI_TTS_LLM_MODEL = "gemini-2.5-flash"   # text reasoning + tool calls
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"  # voice synthesis
```
Both as module-level constants in `gemini_tts.py` (not env-configurable initially — can be promoted later).

No new env vars. The handler reuses `GEMINI_API_KEY` already in `Config`.

### Credential logic

- `local_stt` + `gemini_tts` output → requires `GEMINI_API_KEY`
- `_has_required_key()`, `_requirement_name()`, and `_set_backend()` in `console.py` updated to cover this combination

---

## Audio

- **Input to TTS:** Text string (with optional inline tags)
- **Model:** `gemini-3.1-flash-tts-preview`
- **Voice:** `Algenib` (configurable via voice override)
- **Output format:** 24,000 Hz, mono, 16-bit signed PCM (matches existing pipeline — no resampling needed)
- **Response shape:** base64-encoded blob in `response.candidates[0].content.parts[0].inline_data.data`
- **Chunking:** decode → numpy int16 array → split into ~2400-sample (100ms) frames → push to `output_queue`

---

## Error Handling

### TTS API errors

Per official docs, the TTS API occasionally returns `500` with stray text tokens. Wrap each TTS call:
- Up to 3 retries, 0.5s backoff between attempts
- On all retries exhausted: log warning, push `AdditionalOutputs({"role": "assistant", "content": "[TTS error — could not generate audio]"})` to the queue so the UI shows something

### LLM errors

Tool call failures are caught, appended to history as tool result error messages, and execution continues — same pattern as the existing Gemini Live handler. Network errors during the LLM call: log and no-op (user can speak again).

---

## Admin UI

### `index.html`

Remove `is-disabled` class and `disabled` attribute from the Gemini Flash 3.1 TTS output card. Update the `<small>` description from placeholder text to accurate description.

### `main.js`

- `setSelectedLocalSTTOutput()`: currently normalizes any non-HF value to `openai`; update to also recognize `gemini_tts` as a valid selection
- `journeyMeta()` LOCAL_STT_BACKEND branch: add `gemini_tts` case with appropriate copy ("Speech comes back through Gemini 3.1 Flash TTS with Rickles voice.")
- `saveBackendConfig()`: already reads from `local-stt-response` hidden select — no change needed there
- Backend validation in `_set_backend()`: `gemini_tts` must be added to the accepted choices

---

## Testing

### `tests/test_gemini_tts_handler.py`

- Conversation history: user messages appended, model responses appended, tool round-trips included
- `_dispatch_completed_transcript()`: mocked Gemini clients, verifies LLM called then TTS called in sequence
- PCM chunking: given a known-size blob, verifies correct number of frames pushed to output_queue
- TTS retry logic: mock 500 errors, verify 3 attempts made, then error output pushed
- Voice override: verify `get_current_voice()` returns override when set

### `tests/test_local_stt_dispatch_hook.py`

- Verify `LocalSTTOpenAIRealtimeHandler._dispatch_completed_transcript()` still calls `self.connection.conversation.item.create()` (regression)
- Verify `LocalSTTHuggingFaceRealtimeHandler` same (inherits OpenAI path via MRO, regression)
- Verify `LocalSTTGeminiTTSHandler._dispatch_completed_transcript()` calls Gemini, not the WebSocket

---

## Don Rickles Profile Update

`profiles/don_rickles/instructions.txt` should receive a new section explaining how to use inline TTS tags for expressiveness when running on the Gemini TTS backend. Suggested additions:

- Use `[fast]` for rapid-fire insult delivery
- Use `[annoyance]` and `[aggression]` for the contempt peaks
- Use `[amusement]` immediately after a particularly sharp line (Rickles always loved his own jokes)
- Keep tags sparse — one per sentence maximum, never adjacent

---

## What This Unblocks

Once shipped, the Gemini TTS card in the admin UI becomes selectable under Custom audio path. The Don Rickles profile gains access to expressive voice delivery that the current Gemini Live / OpenAI voice paths can't match. Future output providers (ElevenLabs etc.) follow the same pattern: new constant in `LOCAL_STT_RESPONSE_BACKEND_CHOICES`, new handler class composing `LocalSTTInputMixin`, credential wiring, UI card.
