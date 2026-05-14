"""Gemini text-generation base handler for local-STT pipelines.

Subclass of ``BaseLlamaResponseHandler`` that replaces the LLM step
(``_stream_llm_deltas`` and ``_call_llm``) with ``GeminiLLMClient`` calls
while inheriting the full tool-dispatch, TTS orchestration, and history-
management machinery from the base class unchanged.

Concrete subclasses supply ``_synthesize_and_enqueue()`` (TTS half) via an
additional mixin — exactly as ``LocalSTTChatterboxHandler`` and
``LocalSTTElevenLabsHandler`` do.
"""

from __future__ import annotations
import uuid
import logging
from typing import Any, Optional, AsyncGenerator

from robot_comic.config import config
from robot_comic.prompts import get_session_instructions
from robot_comic.gemini_llm import GeminiLLMClient
from robot_comic.llama_base import BaseLlamaResponseHandler
from robot_comic.tools.core_tools import ToolDependencies, get_active_tool_specs


logger = logging.getLogger(__name__)

_DEFAULT_GEMINI_LLM_MODEL = "gemini-2.5-flash"


class GeminiTextResponseHandler(BaseLlamaResponseHandler):
    """``BaseLlamaResponseHandler`` with the LLM step wired to the Gemini API.

    The TTS half (``_synthesize_and_enqueue``) is intentionally *not* provided
    here — concrete subclasses mix in the relevant TTS implementation.

    All tool dispatch, history trim, telemetry, and output-queue management is
    inherited unchanged from ``BaseLlamaResponseHandler``.
    """

    _BACKEND_LABEL: str = "gemini_text"
    _TTS_SYSTEM: str = "unknown"  # overridden by concrete subclasses

    def __init__(
        self,
        deps: ToolDependencies,
        sim_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        """Initialise handler state; concrete subclasses must supply TTS via a mixin."""
        super().__init__(
            deps=deps,
            sim_mode=sim_mode,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )
        self._gemini_llm: GeminiLLMClient | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def _prepare_startup_credentials(self) -> None:
        """Initialise the Gemini LLM client and the shared HTTP client."""
        await super()._prepare_startup_credentials()
        api_key = getattr(config, "GEMINI_API_KEY", None) or "DUMMY"
        model = getattr(config, "GEMINI_LLM_MODEL", _DEFAULT_GEMINI_LLM_MODEL)
        self._gemini_llm = GeminiLLMClient(api_key=api_key, model=model)
        logger.info(
            "GeminiTextResponseHandler initialised: model=%s",
            model,
        )

    # ------------------------------------------------------------------ #
    # LLM overrides                                                        #
    # ------------------------------------------------------------------ #

    def _build_llm_messages(
        self,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        """Build (messages, tools, system_instruction) for a Gemini call.

        Mirrors the logic in ``BaseLlamaResponseHandler._stream_llm_deltas``
        so both methods stay in sync without duplication.
        """
        system_prompt = get_session_instructions()
        tool_specs = get_active_tool_specs(self.deps)
        # Convert tool specs to the OpenAI-style format expected by GeminiLLMClient
        chat_tools = [{"type": "function", **{k: v for k, v in t.items() if k != "type"}} for t in tool_specs]
        messages = list(self._conversation_history)
        if extra_messages:
            messages = messages + extra_messages
        logger.info(
            "_build_llm_messages: profile=%r tools=%d sys_chars=%d",
            getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
            len(tool_specs),
            len(system_prompt),
        )
        return messages, chat_tools, system_prompt

    async def _stream_llm_deltas(
        self,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Override: stream deltas from the Gemini API instead of llama-server.

        Yields the same delta shapes as the llama_base implementation so all
        downstream accumulation in ``_stream_response_and_synthesize`` works
        without modification.
        """
        assert self._gemini_llm is not None, (
            "GeminiLLMClient not initialised — call _prepare_startup_credentials first"
        )

        messages, chat_tools, system_prompt = self._build_llm_messages(extra_messages)

        async for delta in self._gemini_llm.stream_completion(
            messages=messages,
            tools=chat_tools,
            system_instruction=system_prompt,
        ):
            yield delta

    async def _call_llm(
        self,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Override: call the Gemini API (non-streaming) for follow-up tool rounds.

        Returns ``(text, tool_calls, raw_message)`` exactly as the base class.
        """
        assert self._gemini_llm is not None, (
            "GeminiLLMClient not initialised — call _prepare_startup_credentials first"
        )

        messages, chat_tools, system_prompt = self._build_llm_messages(extra_messages)

        text, tool_calls, raw_msg = await self._gemini_llm.call_completion(
            messages=messages,
            tools=chat_tools,
            system_instruction=system_prompt,
        )

        # Ensure every tool call has a non-empty id (matches llama_base behaviour).
        for tc in tool_calls:
            if not tc.get("id"):
                tc["id"] = str(uuid.uuid4())[:8]

        return text, tool_calls, raw_msg
