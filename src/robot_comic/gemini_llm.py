"""Cloud Gemini text-generation LLM client.

Provides ``GeminiLLMClient`` — an async wrapper around the google-genai SDK
that presents the same delta-shape contract as ``BaseLlamaResponseHandler``'s
``_stream_llm_deltas`` / ``_call_llm`` so the handler layer can swap the LLM
step without touching TTS, tool dispatch, or history management.

Delta shapes yielded by ``stream_completion``:
  {"type": "text_delta",    "content": str}
  {"type": "tool_call_delta", "index": int, "id": str, "name": str, "arguments": str}
  {"type": "finish_reason",  "finish_reason": str}

``call_completion`` returns ``(text, tool_calls, raw_message_dict)`` exactly as
``BaseLlamaResponseHandler._call_llm`` does, so follow-up tool-round logic can
be shared verbatim.
"""

from __future__ import annotations
import json
import uuid
import asyncio
import logging
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from robot_comic.gemini_live import _openai_tool_specs_to_gemini
from robot_comic.gemini_retry import (
    compute_backoff,
    is_rate_limit_error,
    describe_quota_failure,
    extract_retry_after_seconds,
)


logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash"
_LLM_MAX_RETRIES = 4
_LLM_RETRY_BASE_DELAY = 1.0


def _openai_messages_to_gemini(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[Any]]:
    """Convert OpenAI-style messages list to (system_instruction, contents).

    The system message (role="system") is extracted into *system_instruction*.
    All other messages are converted to ``types.Content`` objects.

    Tool-result messages (role="tool") are converted to function-response
    parts attached to a "user" role Content, matching the Gemini convention.
    """
    system_instruction: str | None = None
    contents: list[Any] = []

    for msg in messages:
        role: str = msg.get("role", "user")
        if role == "system":
            system_instruction = msg.get("content") or ""
            continue

        if role == "tool":
            # OpenAI tool result: {"role": "tool", "content": <json_str>, "tool_call_id": <id>}
            # Gemini expects function_response parts in a "user" Content.
            # We can't know the function name from tool_call_id alone, so we use
            # a generic name; the model handles it by position in the conversation.
            content_str = msg.get("content") or "{}"
            try:
                result_dict: dict[str, Any] = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                result_dict = {"result": content_str}
            fn_name = msg.get("name") or "tool_result"
            part = types.Part(
                function_response=types.FunctionResponse(
                    name=fn_name,
                    response=result_dict,
                )
            )
            contents.append(types.Content(role="user", parts=[part]))
            continue

        if role == "assistant":
            gemini_role = "model"
            text_content = msg.get("content") or ""
            parts: list[Any] = []
            if text_content:
                parts.append(types.Part(text=text_content))
            # Carry embedded tool_calls as function_call parts so the Gemini
            # context window stays coherent across multi-turn tool chains.
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                fn_name_tc = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args_dict: dict[str, Any] = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}
                else:
                    args_dict = raw_args or {}
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=fn_name_tc,
                            args=args_dict,
                        )
                    )
                )
            if not parts:
                parts.append(types.Part(text=""))
            contents.append(types.Content(role=gemini_role, parts=parts))
            continue

        # Default: user message
        text_content_user = msg.get("content") or ""
        contents.append(types.Content(role="user", parts=[types.Part(text=text_content_user)]))

    return system_instruction, contents


class GeminiLLMClient:
    """Async Gemini text-generation client with streaming and non-streaming variants.

    Parameters
    ----------
    api_key:
        Gemini / Google AI Studio API key.
    model:
        Gemini model ID to use (default: ``"gemini-2.5-flash"``).

    """

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        """Initialise the client with *api_key* and the target *model*."""
        self._api_key = api_key
        self._model = model
        self._client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------ #
    # Streaming                                                            #
    # ------------------------------------------------------------------ #

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_instruction: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream LLM completion as deltas matching the llama_base shape.

        Yields dicts with ``type`` in:
        - ``"text_delta"``     — ``{"type": "text_delta", "content": str}``
        - ``"tool_call_delta"``— ``{"type": "tool_call_delta", "index": int,
                                    "id": str, "name": str, "arguments": str}``
        - ``"finish_reason"``  — ``{"type": "finish_reason", "finish_reason": str}``

        Parameters
        ----------
        messages:
            OpenAI-style conversation history (role/content dicts).
        tools:
            OpenAI-style tool specs (``{"type": "function", "name": ..., ...}``).
        system_instruction:
            Override system prompt; when ``None`` the first system message in
            *messages* is used (or no system instruction if none is present).

        """
        # Convert format
        _sys_from_msgs, contents = _openai_messages_to_gemini(messages)
        effective_system = system_instruction if system_instruction is not None else _sys_from_msgs

        function_declarations = _openai_tool_specs_to_gemini(tools)
        tools_config: list[Any] = (
            [types.Tool(function_declarations=function_declarations)]  # type: ignore[arg-type]
            if function_declarations
            else []
        )

        gen_config = types.GenerateContentConfig(
            system_instruction=effective_system or None,
            tools=tools_config or None,
        )

        delay = _LLM_RETRY_BASE_DELAY
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                # Track accumulated tool-call fragments keyed by index
                tool_calls_seen: dict[int, bool] = {}
                tc_index = 0

                async for chunk in await self._client.aio.models.generate_content_stream(
                    model=self._model,
                    contents=contents,
                    config=gen_config,
                ):
                    candidates = getattr(chunk, "candidates", None) or []
                    if not candidates:
                        continue
                    candidate = candidates[0]
                    content_obj = getattr(candidate, "content", None)
                    if content_obj is None:
                        continue
                    parts_list = getattr(content_obj, "parts", None) or []

                    for part in parts_list:
                        # Text delta
                        text = getattr(part, "text", None)
                        if text is not None and text != "":
                            yield {"type": "text_delta", "content": text}

                        # Function call (tool call delta)
                        fc = getattr(part, "function_call", None)
                        if fc is not None:
                            idx = tc_index
                            if idx not in tool_calls_seen:
                                tool_calls_seen[idx] = True
                                tc_index += 1
                            tc_id = str(uuid.uuid4())[:8]
                            fn_name_stream: str = getattr(fc, "name", "") or ""
                            raw_args_stream = getattr(fc, "args", {}) or {}
                            args_str = json.dumps(raw_args_stream)
                            yield {
                                "type": "tool_call_delta",
                                "index": idx,
                                "id": tc_id,
                                "name": fn_name_stream,
                                "arguments": args_str,
                            }

                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason is not None:
                        # Gemini finish_reason is an enum; convert to lowercase string
                        fr_str = (
                            str(finish_reason.name).lower()
                            if hasattr(finish_reason, "name")
                            else str(finish_reason).lower()
                        )
                        yield {"type": "finish_reason", "finish_reason": fr_str}

                return  # success

            except Exception as exc:
                rate_limited = is_rate_limit_error(exc)
                is_retryable = rate_limited or "503" in str(exc) or "UNAVAILABLE" in str(exc)
                if not is_retryable or attempt == _LLM_MAX_RETRIES - 1:
                    if rate_limited and attempt == _LLM_MAX_RETRIES - 1:
                        logger.error(
                            "Gemini LLM rate-limited (quota=%s) after %d attempts; giving up",
                            describe_quota_failure(exc),
                            _LLM_MAX_RETRIES,
                        )
                    raise
                retry_after = extract_retry_after_seconds(exc) if rate_limited else None
                delay_s = compute_backoff(attempt, delay, retry_after)
                if rate_limited:
                    logger.warning(
                        "Gemini LLM 429 (quota=%s, attempt %d/%d); sleeping %.1fs before retry",
                        describe_quota_failure(exc),
                        attempt + 1,
                        _LLM_MAX_RETRIES,
                        delay_s,
                    )
                else:
                    logger.warning(
                        "Gemini LLM stream attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1,
                        _LLM_MAX_RETRIES,
                        str(exc).split("\n")[0],
                        delay_s,
                    )
                await asyncio.sleep(delay_s)
                delay *= 2

    # ------------------------------------------------------------------ #
    # Non-streaming (for follow-up tool rounds)                           #
    # ------------------------------------------------------------------ #

    async def call_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_instruction: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Non-streaming completion — used for follow-up tool-result passes.

        Returns
        -------
        (text, tool_calls, raw_message_dict)
            Mirrors ``BaseLlamaResponseHandler._call_llm`` exactly.

        """
        text_parts: list[str] = []
        tool_calls_by_idx: dict[int, dict[str, Any]] = {}

        async for delta in self.stream_completion(messages, tools, system_instruction):
            if delta["type"] == "text_delta":
                text_parts.append(delta["content"])
            elif delta["type"] == "tool_call_delta":
                idx = delta["index"]
                if idx not in tool_calls_by_idx:
                    tool_calls_by_idx[idx] = {
                        "index": idx,
                        "id": delta.get("id", "") or str(uuid.uuid4())[:8],
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if delta.get("id"):
                    tool_calls_by_idx[idx]["id"] = delta["id"]
                if delta.get("name"):
                    tool_calls_by_idx[idx]["function"]["name"] = delta["name"]
                if delta.get("arguments") is not None:
                    existing = tool_calls_by_idx[idx]["function"]["arguments"]
                    # Gemini returns the full JSON in one shot; don't double-append.
                    if not existing:
                        tool_calls_by_idx[idx]["function"]["arguments"] = delta["arguments"]

        text = "".join(text_parts).strip()
        tool_calls: list[dict[str, Any]] = []
        for tc in tool_calls_by_idx.values():
            args_str = tc["function"]["arguments"]
            if isinstance(args_str, str):
                try:
                    tc["function"]["arguments"] = json.loads(args_str)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse tool_call[%d] arguments: %r",
                        tc.get("index"),
                        args_str[:100],
                    )
                    tc["function"]["arguments"] = {}
            tool_calls.append(tc)

        if not text and not tool_calls:
            logger.warning("GeminiLLMClient.call_completion: empty response")

        raw_msg: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            raw_msg["tool_calls"] = tool_calls
        return text, tool_calls, raw_msg
