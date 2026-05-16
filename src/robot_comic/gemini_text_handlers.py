"""Gemini text-LLM + local-TTS diamond response handler classes.

These response-handler bases use the Gemini API for the LLM step and a
local TTS backend for speech synthesis, making them useful when
llama-server is unavailable or when higher-quality Gemini reasoning is
preferred.

Handler matrix
--------------
  Gemini text LLM + Chatterbox TTS  → ``GeminiTextChatterboxResponseHandler``
  Gemini text LLM + ElevenLabs TTS  → ``GeminiTextElevenLabsResponseHandler``

The composable factory (``handler_factory.py``) composes
``LocalSTTInputMixin`` over these bases via private host subclasses; the
adapters then wrap the composed instance to drive the
:class:`~robot_comic.composable_pipeline.ComposablePipeline`.

The Gemini TTS combination (Moonshine + bundled Gemini LLM + Gemini TTS)
lives on :class:`~robot_comic.gemini_tts.GeminiTTSResponseHandler` — no
diamond is needed there because the Gemini API natively bundles LLM and
TTS through one client.
"""

from __future__ import annotations
import logging

from robot_comic.chatterbox_tts import ChatterboxTTSResponseHandler
from robot_comic.elevenlabs_tts import ElevenLabsTTSResponseHandler
from robot_comic.gemini_text_base import GeminiTextResponseHandler


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chatterbox TTS + Gemini LLM
# ---------------------------------------------------------------------------


class GeminiTextChatterboxResponseHandler(GeminiTextResponseHandler, ChatterboxTTSResponseHandler):
    """Gemini text LLM + Chatterbox TTS voice output.

    MRO: GeminiTextChatterboxResponseHandler
         → GeminiTextResponseHandler  (overrides _stream_llm_deltas / _call_llm)
         → ChatterboxTTSResponseHandler  (supplies _synthesize_and_enqueue,
                                          _prepare_startup_credentials for TTS)
         → BaseLlamaResponseHandler  (tool dispatch, history, output queue)

    ``_prepare_startup_credentials`` chain:
    1. ``GeminiTextResponseHandler._prepare_startup_credentials`` calls
       ``super()._prepare_startup_credentials()`` which reaches
       ``ChatterboxTTSResponseHandler._prepare_startup_credentials`` (which in
       turn calls ``BaseLlamaResponseHandler._prepare_startup_credentials``).
    2. Then ``GeminiTextResponseHandler`` adds ``GeminiLLMClient`` init.
    3. Result: both the Chatterbox HTTP client and the Gemini client are ready.
    """

    _BACKEND_LABEL = "gemini_text_chatterbox"
    _TTS_SYSTEM = "chatterbox"

    def copy(self) -> "GeminiTextChatterboxResponseHandler":
        """Return a new instance with the same configuration."""
        return type(self)(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def _prepare_startup_credentials(self) -> None:
        # ChatterboxTTSResponseHandler._prepare_startup_credentials sets up
        # httpx client, probes llama health (skippable), and warms TTS.
        # We call it first so the Chatterbox client is ready before Gemini init.
        await ChatterboxTTSResponseHandler._prepare_startup_credentials(self)
        # Now initialise the Gemini LLM client on top.
        from robot_comic.config import config
        from robot_comic.gemini_llm import GeminiLLMClient
        from robot_comic.gemini_text_base import _DEFAULT_GEMINI_LLM_MODEL

        api_key = getattr(config, "GEMINI_API_KEY", None) or "DUMMY"
        model = getattr(config, "GEMINI_LLM_MODEL", _DEFAULT_GEMINI_LLM_MODEL)
        self._gemini_llm = GeminiLLMClient(api_key=api_key, model=model)
        logger.info(
            "GeminiTextChatterboxResponseHandler initialised: llm=%s tts=chatterbox voice=%s",
            model,
            self.get_current_voice(),
        )


# ---------------------------------------------------------------------------
# ElevenLabs TTS + Gemini LLM
# ---------------------------------------------------------------------------


class GeminiTextElevenLabsResponseHandler(GeminiTextResponseHandler, ElevenLabsTTSResponseHandler):
    """Gemini text LLM + ElevenLabs TTS voice output.

    MRO: GeminiTextElevenLabsResponseHandler
         → GeminiTextResponseHandler  (overrides _stream_llm_deltas / _call_llm)
         → ElevenLabsTTSResponseHandler  (supplies _synthesize_and_enqueue,
                                          _prepare_startup_credentials for TTS)
         → (AsyncStreamHandler, ConversationHandler)

    Note: ``ElevenLabsTTSResponseHandler`` does *not* inherit from
    ``BaseLlamaResponseHandler``; it is a parallel hierarchy (AsyncStreamHandler
    + ConversationHandler).  ``GeminiTextResponseHandler`` brings in
    ``BaseLlamaResponseHandler``.  To resolve the diamond, we explicitly call
    each ``_prepare_startup_credentials`` in order rather than relying on
    cooperative super() chains — the same technique used by
    ``GeminiTextChatterboxResponseHandler`` above.
    """

    _BACKEND_LABEL = "gemini_text_elevenlabs"
    _TTS_SYSTEM = "elevenlabs"

    def copy(self) -> "GeminiTextElevenLabsResponseHandler":
        """Return a new instance with the same configuration."""
        return type(self)(
            self.deps,
            self.sim_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    # ------------------------------------------------------------------ #
    # MRO diamond shims: BaseLlamaResponseHandler (via GeminiTextResponseHandler)
    # appears earlier in the MRO than ElevenLabsTTSResponseHandler, so its
    # NotImplementedError stubs win the lookup for voice methods.
    # Delegate to the ElevenLabs implementation.
    # NOTE: _synthesize_and_enqueue is intentionally NOT delegated —
    # ElevenLabsTTSResponseHandler uses a different response loop
    # (_run_llm_with_tools / _stream_tts_to_queue) and does not implement
    # the BaseLlamaResponseHandler.synth interface. Wiring this combination
    # end-to-end requires more than a shim; tracked separately.
    # ------------------------------------------------------------------ #

    def get_current_voice(self) -> str:
        """Delegate to the ElevenLabs implementation (diamond MRO shim)."""
        return ElevenLabsTTSResponseHandler.get_current_voice(self)

    async def get_available_voices(self) -> list[str]:
        """Delegate to the ElevenLabs implementation (diamond MRO shim)."""
        return await ElevenLabsTTSResponseHandler.get_available_voices(self)

    async def change_voice(self, voice: str) -> str:
        """Delegate to the ElevenLabs implementation (diamond MRO shim)."""
        return await ElevenLabsTTSResponseHandler.change_voice(self, voice)

    async def _prepare_startup_credentials(self) -> None:
        # ElevenLabsTTSResponseHandler._prepare_startup_credentials creates the
        # Gemini (text-LLM) client for its own use *and* an httpx client for
        # ElevenLabs API calls.  We call it first, then replace its genai
        # client with our GeminiLLMClient wrapper.
        await ElevenLabsTTSResponseHandler._prepare_startup_credentials(self)
        # Initialise BaseLlamaResponseHandler's httpx client if not done yet
        # (ElevenLabsTTS uses its own httpx; we need the llama_base one for
        # any inherited machinery that references self._http).
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0))
        self.tool_manager.start_up(tool_callbacks=[self._handle_tool_notification])

        from robot_comic.config import config
        from robot_comic.gemini_llm import GeminiLLMClient
        from robot_comic.gemini_text_base import _DEFAULT_GEMINI_LLM_MODEL

        api_key = getattr(config, "GEMINI_API_KEY", None) or "DUMMY"
        model = getattr(config, "GEMINI_LLM_MODEL", _DEFAULT_GEMINI_LLM_MODEL)
        self._gemini_llm = GeminiLLMClient(api_key=api_key, model=model)
        logger.info(
            "GeminiTextElevenLabsResponseHandler initialised: llm=%s tts=elevenlabs voice=%s",
            model,
            self.get_current_voice(),
        )
