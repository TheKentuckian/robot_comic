"""Phase 1-Protocol adapters over the existing handler classes.

These adapters present the ``backends.STTBackend / LLMBackend / TTSBackend``
contracts on top of the legacy class hierarchy without touching the legacy
classes themselves. ``ComposablePipeline`` (Phase 2) can then drive a
production pipeline by composing adapter instances instead of inheriting
from one of the bundled handler classes.

Adapter pattern: each adapter holds an instance of an existing handler /
mixin and proxies Protocol method calls to the handler's internals. This is
deliberately a thin layer so the new path shares the underlying API code
with the legacy path (no duplication of llama-server calls, ElevenLabs
streaming, Moonshine setup, etc.).

Phase 4 will replace the delegation with cleanly extracted implementations
that don't depend on the legacy classes; the adapter API stays stable so
``ComposablePipeline`` doesn't notice.
"""

from robot_comic.adapters.gemini_llm_adapter import GeminiLLMAdapter
from robot_comic.adapters.llama_llm_adapter import LlamaLLMAdapter
from robot_comic.adapters.moonshine_stt_adapter import MoonshineSTTAdapter
from robot_comic.adapters.chatterbox_tts_adapter import ChatterboxTTSAdapter
from robot_comic.adapters.elevenlabs_tts_adapter import ElevenLabsTTSAdapter


__all__ = [
    "ChatterboxTTSAdapter",
    "ElevenLabsTTSAdapter",
    "GeminiLLMAdapter",
    "LlamaLLMAdapter",
    "MoonshineSTTAdapter",
]
