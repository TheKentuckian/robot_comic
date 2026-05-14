"""Language dissection tool — structural scaffold for George Carlin's euphemism breakdown pattern.

The tool provides a heuristic analysis of a phrase without making any additional LLM calls.
It looks the phrase up in the curated euphemisms dictionary and, for unknown phrases, falls
back to a literal word-by-word decomposition.  The result is a structured hook the Carlin
persona can riff against.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict
from pathlib import Path

from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Resolve the bundled dictionary relative to this file so it works whether the
# package is installed editable or as a wheel.
_EUPHEMISMS_PATH = Path(__file__).parents[3] / "profiles" / "george_carlin" / "euphemisms.json"


def _load_euphemisms(path: Path = _EUPHEMISMS_PATH) -> Dict[str, Any]:
    """Load the curated euphemisms dictionary from *path*.

    Returns an empty dict if the file cannot be read so the tool degrades
    gracefully rather than raising at import time.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("euphemisms.json is not a dict — returning empty dict")
            return {}
        return data
    except FileNotFoundError:
        logger.warning("euphemisms.json not found at %s — tool will use fallback only", path)
        return {}
    except Exception as exc:
        logger.warning("Failed to load euphemisms.json: %s", exc)
        return {}


# Module-level cache — loaded once on first import.
_EUPHEMISMS: Dict[str, Any] | None = None


def _get_euphemisms() -> Dict[str, Any]:
    global _EUPHEMISMS
    if _EUPHEMISMS is None:
        _EUPHEMISMS = _load_euphemisms()
    return _EUPHEMISMS


def _make_literal_words(phrase: str) -> Dict[str, str]:
    """Return a plain dictionary gloss for each word in an unknown phrase."""
    return {word: f"plain meaning of '{word}'" for word in phrase.lower().split()}


def dissect_phrase(phrase: str, euphemisms: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a structured dissection of *phrase*.

    Looks up the normalised phrase in the curated dictionary first.  If not
    found, builds a minimal fallback entry from the individual words.

    Args:
        phrase: The target phrase to dissect, e.g. ``"thoughts and prayers"``.
        euphemisms: Override the dictionary (mainly for tests).

    Returns:
        A dict with keys ``phrase``, ``literal_words``, ``euphemism_target``,
        and ``dissection_suggestion``.

    """
    if euphemisms is None:
        euphemisms = _get_euphemisms()

    normalised = phrase.strip().lower()

    entry = euphemisms.get(normalised)
    if entry is not None:
        return {
            "phrase": phrase,
            "literal_words": entry.get("literal_words", _make_literal_words(normalised)),
            "euphemism_target": entry.get("euphemism_target", "unknown"),
            "dissection_suggestion": entry.get("dissection_suggestion", ""),
        }

    # Fallback: phrase not in dictionary — return a minimal structural scaffold.
    return {
        "phrase": phrase,
        "literal_words": _make_literal_words(normalised),
        "euphemism_target": "unknown — phrase not in curated dictionary",
        "dissection_suggestion": (
            f"The phrase '{phrase}' isn't in the standard playbook. "
            "Pull each word apart on its own: what does it literally mean, "
            "and what is the speaker using it to avoid saying?"
        ),
    }


class LanguageDissect(Tool):
    """Dissect a phrase or euphemism into its structural components for riff material.

    Returns a structured analysis — no additional LLM call is made.  The result
    gives the persona the raw material (literal meanings, what is being hidden,
    a suggested angle) to run the quote-name-strip deconstruction pattern.
    """

    name = "language_dissect"
    description = (
        "Analyse a phrase or euphemism to expose what it is hiding. "
        "Returns: the original phrase, a word-by-word literal gloss, what the phrase is "
        "euphemistically concealing, and a short dissection angle for the Carlin pattern "
        "(quote the word → name what it's doing → strip it to the plain version). "
        "No extra LLM call — pure structural scaffold. "
        "Call this whenever the conversation surfaces an institutional euphemism, "
        "a corporate buzzword, a political phrase, or any soft language worth dissecting."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "phrase": {
                "type": "string",
                "description": (
                    "The word or phrase to dissect. Use the exact wording the person said, "
                    "or a well-known euphemism (e.g. 'thoughts and prayers', "
                    "'passed away', 'human resources', 'collateral damage')."
                ),
            },
        },
        "required": ["phrase"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Dissect the supplied phrase and return structured riff material."""
        phrase: str = kwargs.get("phrase", "").strip()
        logger.info("Tool call: language_dissect phrase=%r", phrase)

        if not phrase:
            return {"error": "No phrase provided — pass a non-empty 'phrase' argument."}

        return dissect_phrase(phrase)
