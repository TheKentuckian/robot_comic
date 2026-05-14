# Audio Backend Matrix

This document describes the **modular audio pipeline** introduced in issue #54.
The goal is to decouple the speech-to-text (STT / input) backend from the
text-to-speech (TTS / output) backend so any supported pair can be used without
requiring a dedicated handler class for every combination.

> **Status**: Config scaffold only (PR #54). Handler splitting — the code that
> actually wires the selected input/output backends together at runtime — is
> deferred to a follow-up PR.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REACHY_MINI_AUDIO_INPUT_BACKEND` | _(derived from `BACKEND_PROVIDER`)_ | STT source to use |
| `REACHY_MINI_AUDIO_OUTPUT_BACKEND` | _(derived from `BACKEND_PROVIDER`)_ | TTS sink to use |

**Both variables must be set together.** Setting only one triggers a warning and
falls back to the `BACKEND_PROVIDER`-derived defaults.

---

## Valid Values

### Input (STT) backends

| Value | Description |
|---|---|
| `moonshine` | Local Moonshine streaming STT (on-robot) |
| `openai_realtime_input` | OpenAI Realtime API (server-side VAD + STT) |
| `gemini_live_input` | Gemini Live API (server-side VAD + STT) |
| `hf_input` | Hugging Face realtime endpoint (server-side VAD + STT) |

### Output (TTS) backends

| Value | Description |
|---|---|
| `chatterbox` | Local Chatterbox TTS service |
| `gemini_tts` | Gemini TTS API (`gemini-3.1-flash-tts-preview`) |
| `elevenlabs` | ElevenLabs TTS API |
| `openai_realtime_output` | OpenAI Realtime API (server-side TTS) |
| `gemini_live_output` | Gemini Live API (server-side TTS, bundled with input) |
| `hf_output` | Hugging Face realtime endpoint (server-side TTS, bundled with input) |

---

## Supported Combinations (currently implemented)

These combinations map 1:1 to an existing handler class and work today.

| Input | Output | Existing handler | Notes |
|---|---|---|---|
| `hf_input` | `hf_output` | `HuggingFaceRealtimeHandler` | Default (`BACKEND_PROVIDER=huggingface`) |
| `openai_realtime_input` | `openai_realtime_output` | `OpenaiRealtimeHandler` | `BACKEND_PROVIDER=openai` |
| `gemini_live_input` | `gemini_live_output` | `GeminiLiveHandler` | `BACKEND_PROVIDER=gemini` |
| `moonshine` | `chatterbox` | `LocalSTTChatterboxHandler` | `local_stt` + `LOCAL_STT_RESPONSE_BACKEND=chatterbox` |
| `moonshine` | `gemini_tts` | `LocalSTTGeminiTTSHandler` | `local_stt` + `LOCAL_STT_RESPONSE_BACKEND=gemini_tts` |
| `moonshine` | `elevenlabs` | `LocalSTTElevenLabsHandler` | `local_stt` + `LOCAL_STT_RESPONSE_BACKEND=elevenlabs` |
| `moonshine` | `openai_realtime_output` | `LocalSTTOpenAIRealtimeHandler` | `local_stt` + `LOCAL_STT_RESPONSE_BACKEND=openai` |
| `moonshine` | `hf_output` | `LocalSTTHuggingFaceRealtimeHandler` | `local_stt` + `LOCAL_STT_RESPONSE_BACKEND=huggingface` |

### Backwards compatibility

When neither `REACHY_MINI_AUDIO_INPUT_BACKEND` nor
`REACHY_MINI_AUDIO_OUTPUT_BACKEND` is set, the values are **derived
automatically** from `BACKEND_PROVIDER` via `derive_audio_backends()`.
Existing deployments need no configuration changes.

---

## Aspirational Combinations (not yet implemented)

These combos would require a new handler class or a pipeline orchestrator to
split the input/output stages. Setting them today logs a `WARNING` and falls
back to the `BACKEND_PROVIDER`-derived defaults.

| Input | Output | Blocker |
|---|---|---|
| `gemini_live_input` | `chatterbox` | Gemini Live audio I/O is bundled; splitting requires custom handler |
| `gemini_live_input` | `elevenlabs` | Same as above |
| `hf_input` | `chatterbox` | HF realtime audio is bundled; splitting requires custom handler |
| `moonshine` | `gemini_live_output` | Gemini Live output requires the full Gemini Live connection |
| `openai_realtime_input` | `chatterbox` | OpenAI Realtime audio is bundled; splitting requires custom handler |
| `openai_realtime_input` | `gemini_tts` | Same as above |
| _(any cross-pair not listed above)_ | _(any)_ | No handler exists |

---

## Path Forward

The config scaffold (this PR) establishes the vocabulary and validation rules.
A follow-up PR should:

1. Introduce a **pipeline orchestrator** (e.g. `audio_pipeline.py`) that, given
   `AUDIO_INPUT_BACKEND` and `AUDIO_OUTPUT_BACKEND`, constructs a handler by
   composing STT and TTS mixins rather than selecting a monolithic class.
2. Refactor existing bundled handlers (`GeminiLiveHandler`, etc.) to expose
   separate `input_mixin` / `output_mixin` interfaces.
3. Update `main.py` handler selection to use the orchestrator when both
   `AUDIO_INPUT_BACKEND` and `AUDIO_OUTPUT_BACKEND` are explicitly set.
4. Add admin UI dropdowns for both backends (currently the UI only exposes
   `BACKEND_PROVIDER`).
