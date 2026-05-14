"""Language dissection tool — structural scaffold for George Carlin's euphemism breakdown pattern.

The tool provides a heuristic analysis of a phrase without making any additional LLM calls.
It looks the phrase up in the curated euphemisms dictionary and, for unknown phrases, falls
back to a per-word decomposition via a bundled plain-English lexicon with suffix stripping.
The result is a structured hook the Carlin persona can riff against.
"""

from __future__ import annotations
import re
import json
import logging
from typing import Any, Dict, List
from pathlib import Path

from robot_comic.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Resolve the bundled dictionary relative to this file so it works whether the
# package is installed editable or as a wheel.
_EUPHEMISMS_PATH = Path(__file__).parents[3] / "profiles" / "george_carlin" / "euphemisms.json"
_LEXICON_PATH = Path(__file__).parent / "language_lexicon.json"

# Common suffixes to strip when looking up a stem in the lexicon, ordered from
# longest to shortest so multi-character endings are tried first.
_SUFFIXES = (
    "ization",
    "isation",
    "ations",
    "ation",
    "ments",
    "ness",
    "ment",
    "ity",
    "ies",
    "ize",
    "ise",
    "ers",
    "ing",
    "ous",
    "ive",
    "ble",
    "ful",
    "al",
    "ly",
    "ed",
    "er",
    "es",
    "s",
)


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


def _load_lexicon(path: Path = _LEXICON_PATH) -> Dict[str, str]:
    """Load the plain-English lexicon from *path*.

    Returns an empty dict on failure so the tool degrades to v1 fallback.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("language_lexicon.json is not a dict — returning empty dict")
            return {}
        return data
    except FileNotFoundError:
        logger.warning("language_lexicon.json not found at %s — decomposition disabled", path)
        return {}
    except Exception as exc:
        logger.warning("Failed to load language_lexicon.json: %s", exc)
        return {}


# Module-level caches — loaded once on first use.
_EUPHEMISMS: Dict[str, Any] | None = None
_LEXICON: Dict[str, str] | None = None


def _get_euphemisms() -> Dict[str, Any]:
    global _EUPHEMISMS
    if _EUPHEMISMS is None:
        _EUPHEMISMS = _load_euphemisms()
    return _EUPHEMISMS


def _get_lexicon() -> Dict[str, str]:
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = _load_lexicon()
    return _LEXICON


def _make_literal_words(phrase: str) -> Dict[str, str]:
    """Return a plain dictionary gloss for each word in an unknown phrase."""
    return {word: f"plain meaning of '{word}'" for word in phrase.lower().split()}


def _decompose_word(word: str, lexicon: Dict[str, str] | None = None) -> tuple[str, str] | None:
    """Try to find *word* (or a stemmed form) in the lexicon.

    Returns ``(root, gloss)`` on success or ``None`` if no match is found.

    The function first tries the bare word, then progressively strips known
    suffixes until the stem matches a lexicon entry.  A minimum stem length of
    three characters is enforced to avoid nonsense matches (e.g. stripping
    ``-ation`` from ``nation`` → ``n``).
    """
    if lexicon is None:
        lexicon = _get_lexicon()

    clean = re.sub(r"[^a-z]", "", word.lower())
    if not clean:
        return None

    # Exact match first.
    if clean in lexicon:
        return clean, lexicon[clean]

    # Try each suffix in order.
    for suffix in _SUFFIXES:
        if clean.endswith(suffix):
            stem = clean[: -len(suffix)]
            if len(stem) >= 3 and stem in lexicon:
                return stem, lexicon[stem]

    return None


def _decompose_phrase(phrase: str, lexicon: Dict[str, str] | None = None) -> List[Dict[str, str]]:
    """Decompose each word in *phrase* using the lexicon.

    Returns a list of per-word dicts, each containing:
    - ``word``: the original word token
    - ``root``: the matched lexicon entry key (may equal ``word``)
    - ``gloss``: the plain-English meaning
    - ``decomposed``: whether a lexicon match was found

    Words not found in the lexicon receive ``decomposed: false`` and an empty
    gloss so the caller can decide how to handle them.
    """
    if lexicon is None:
        lexicon = _get_lexicon()

    results: List[Dict[str, str]] = []
    for token in phrase.lower().split():
        match = _decompose_word(token, lexicon)
        if match is not None:
            root, gloss = match
            results.append({"word": token, "root": root, "gloss": gloss, "decomposed": "true"})
        else:
            results.append({"word": token, "root": token, "gloss": "", "decomposed": "false"})
    return results


def dissect_phrase(phrase: str, euphemisms: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a structured dissection of *phrase*.

    Looks up the normalised phrase in the curated dictionary first.  If not
    found, attempts per-word decomposition via the plain-English lexicon.
    When at least 60% of words decompose successfully the result carries
    ``decomposition_quality: "high"``; otherwise the v1 flat fallback is
    returned with ``decomposition_quality: "low"``.

    Args:
        phrase: The target phrase to dissect, e.g. ``"thoughts and prayers"``.
        euphemisms: Override the dictionary (mainly for tests).

    Returns:
        A dict with keys ``phrase``, ``literal_words``, ``euphemism_target``,
        ``dissection_suggestion``, and (for fallback paths) ``decomposition_quality``.

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

    # Fallback: phrase not in dictionary — attempt per-word decomposition.
    words = normalised.split()
    if not words:
        # Whitespace-only input: return a minimal scaffold.
        return {
            "phrase": phrase,
            "literal_words": {},
            "euphemism_target": "unknown — phrase not in curated dictionary",
            "dissection_suggestion": "No words to dissect.",
            "decomposition_quality": "low",
        }

    breakdown = _decompose_phrase(normalised)
    decomposed_count = sum(1 for item in breakdown if item["decomposed"] == "true")
    ratio = decomposed_count / len(breakdown) if breakdown else 0.0

    if ratio >= 0.6:
        # Rich per-word breakdown.
        literal_words = {
            item["word"]: (f"{item['root']} ({item['gloss']})" if item["root"] != item["word"] else item["gloss"])
            for item in breakdown
        }
        glosses = [f"{item['word']} — {item['gloss']}" for item in breakdown if item["decomposed"] == "true"]
        dissection_suggestion = (
            f"'{phrase}' isn't in the playbook, but the words give it away: "
            + "; ".join(glosses)
            + ". Strip the fancy packaging and say what you mean."
        )
        return {
            "phrase": phrase,
            "literal_words": literal_words,
            "euphemism_target": "unknown — phrase not in curated dictionary",
            "dissection_suggestion": dissection_suggestion,
            "decomposition_quality": "high",
        }

    # Low-quality fallback — v1 behavior with quality flag.
    return {
        "phrase": phrase,
        "literal_words": _make_literal_words(normalised),
        "euphemism_target": "unknown — phrase not in curated dictionary",
        "dissection_suggestion": (
            f"The phrase '{phrase}' isn't in the standard playbook. "
            "Pull each word apart on its own: what does it literally mean, "
            "and what is the speaker using it to avoid saying?"
        ),
        "decomposition_quality": "low",
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
