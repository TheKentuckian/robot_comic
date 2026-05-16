import re
import sys
import logging
from pathlib import Path

from robot_comic.config import (
    AUDIO_OUTPUT_GEMINI_TTS,
    DEFAULT_PROFILES_DIRECTORY,
    config,
    get_default_voice_for_provider,
)


logger = logging.getLogger(__name__)


PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"

# Regex that matches the entire "## GEMINI TTS DELIVERY TAGS" section up to the
# next "##" heading or end-of-string.  re.DOTALL lets "." cross newlines.
_TTS_SECTION_RE = re.compile(
    r"## GEMINI TTS DELIVERY TAGS\b.*?(?=\r?\n##|\Z)",
    re.DOTALL,
)


def _uses_gemini_tts(audio_output_backend: str) -> bool:
    """Return True when the active audio output backend renders Gemini TTS delivery tags.

    Delivery tags are consumed only by the Gemini TTS audio output backend
    (``AUDIO_OUTPUT_GEMINI_TTS``).  Every other audio output (Chatterbox,
    ElevenLabs, OpenAI Realtime, Gemini Live, Hugging Face …) ignores the
    section and would only waste tokens.
    """
    return audio_output_backend == AUDIO_OUTPUT_GEMINI_TTS


def _strip_tts_section(instructions: str) -> str:
    """Remove the GEMINI TTS DELIVERY TAGS section from *instructions*.

    Collapses any resulting triple-blank lines to a single blank line and
    strips leading/trailing whitespace.
    """
    result = _TTS_SECTION_RE.sub("", instructions)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _filter_delivery_tags(instructions: str) -> str:
    """Conditionally strip the GEMINI TTS DELIVERY TAGS section.

    The section is kept when:
    - ``REACHY_MINI_FORCE_DELIVERY_TAGS=1`` is set (debugging override), OR
    - the active audio backend actually consumes the tags (Gemini TTS paths).

    It is stripped when the active backend is anything else (Gemini Live,
    Chatterbox, HuggingFace, OpenAI, ElevenLabs, …).
    """
    force: bool = getattr(config, "FORCE_DELIVERY_TAGS", False)
    if force:
        return instructions

    audio_output_backend: str = getattr(config, "AUDIO_OUTPUT_BACKEND", "")

    if _uses_gemini_tts(audio_output_backend):
        # Tags are consumed — leave the section in place.
        return instructions

    return _strip_tts_section(instructions)


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    # pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')
    # Allow slashes for subdirectories
    pattern = re.compile(r"^\[([a-zA-Z0-9/_-]+)\]$")

    lines = content.split("\n")
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return "\n".join(expanded_lines)


def _append_joke_history(instructions: str) -> str:
    """Append the joke history "don't repeat" section if the feature is enabled.

    Returns *instructions* unchanged when the feature is disabled or history is empty.
    """
    enabled: bool = getattr(config, "JOKE_HISTORY_ENABLED", True)
    if not enabled:
        return instructions

    try:
        from robot_comic.joke_history import JokeHistory, default_history_path

        history = JokeHistory(default_history_path())
        block = history.format_for_prompt()
        if block:
            return instructions.rstrip() + "\n\n" + block
    except Exception as exc:
        logger.debug("Could not load joke history for prompt injection: %s", exc)

    return instructions


def get_session_instructions() -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.info(f"Loading default prompt from {PROMPTS_LIBRARY_DIRECTORY / 'default_prompt.txt'}")
        instructions_file = PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
    else:
        if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.info(
                "Loading prompt from external profile '%s' (root=%s)",
                profile,
                config.PROFILES_DIRECTORY,
            )
        else:
            logger.info(f"Loading prompt from profile '{profile}'")
        instructions_file = config.PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME

    try:
        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                # Expand [<name>] placeholders with content from prompts library
                expanded_instructions = _expand_prompt_includes(instructions)
                # Strip GEMINI TTS DELIVERY TAGS section for backends that do
                # not consume it (avoids wasted tokens + accidental tag leakage).
                filtered = _filter_delivery_tags(expanded_instructions)
                # Append joke history "don't repeat" block when enabled.
                return _append_joke_history(filtered)
            logger.error(f"Profile '{profile}' has empty {INSTRUCTIONS_FILENAME}")
            sys.exit(1)
        logger.error(f"Profile {profile} has no {INSTRUCTIONS_FILENAME}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load instructions from profile '{profile}': {e}")
        sys.exit(1)


def get_session_voice(default: str | None = None) -> str:
    """Resolve the voice to use for the session.

    If a custom profile is selected and contains a voice.txt, return its
    trimmed content; otherwise return the provided default or the active
    backend default voice.
    """
    fallback = get_default_voice_for_provider() if default is None else default
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return fallback
    try:
        voice_file = config.PROFILES_DIRECTORY / profile / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or fallback
    except Exception:
        pass
    return fallback
