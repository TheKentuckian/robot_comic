"""Static-prompt manifest parser and playback helper for lifecycle audio clips.

Reads ``assets/static_prompts.toml`` (repo-relative source of truth), resolves
WAV paths for a given ``(category, persona)`` pair, and dispatches playback via
the existing ``warmup_audio._dispatch_single_wav`` chokepoint.

Public API
----------
- :func:`load_manifest` — parse and validate the TOML manifest.
- :func:`resolve_clip` — find the best matching :class:`ClipEntry` for a
  ``(category, persona)`` query, with per-persona → global fallback.
- :func:`play_static_prompt` — resolve + dispatch playback; logs + skips when
  the WAV file is absent; never raises.

Error / fallback category
~~~~~~~~~~~~~~~~~~~~~~~~~
The ``"error"`` category is intentionally routed through the local WAV
chokepoint only.  Callers MUST NOT hit the LLM or TTS from the error path —
those are the services that are down when this fires.

Pool sampling
~~~~~~~~~~~~~
Multiple clips may share the same ``(category, scope/persona)``.  When more
than one match exists, one is chosen at random.  Callers that need
deterministic selection should use :func:`resolve_clip` directly.
"""

from __future__ import annotations
import random
import logging
from typing import Literal, Optional
from pathlib import Path
from dataclasses import field, dataclass


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo-root anchor
# ---------------------------------------------------------------------------

# src/robot_comic/static_prompts.py → src/robot_comic → src → repo root
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_MANIFEST_PATH: Path = _REPO_ROOT / "assets" / "static_prompts.toml"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

ScopeT = Literal["global", "per_persona"]
GeneratorT = Literal["elevenlabs", "xtts"]

VALID_CATEGORIES = frozenset(
    {"startup", "picker", "persona_selected", "persona_switch", "pause", "resume", "shutdown", "error"}
)
VALID_SCOPES: frozenset[str] = frozenset({"global", "per_persona"})
VALID_GENERATORS: frozenset[str] = frozenset({"elevenlabs", "xtts"})


@dataclass(frozen=True)
class ClipEntry:
    """One row from the ``[[clips]]`` array in ``static_prompts.toml``."""

    category: str
    scope: ScopeT
    text: str
    output_path: str  # repo-relative string
    generator: GeneratorT
    persona: Optional[str] = field(default=None)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def resolved_path(self) -> Path:
        """Absolute path to the WAV file."""
        return _REPO_ROOT / self.output_path

    def exists(self) -> bool:
        """Return True when the WAV file is present on disk."""
        return self.resolved_path.is_file()


# ---------------------------------------------------------------------------
# Manifest loading + validation
# ---------------------------------------------------------------------------


class ManifestError(ValueError):
    """Raised when static_prompts.toml cannot be parsed or fails validation."""


def _parse_toml(path: Path) -> "dict[str, object]":
    """Parse a TOML file using tomllib (3.11+) or tomli fallback.

    tomllib is in the Python 3.11+ stdlib.  On older Python or in
    environments without the stdlib version, tomli is tried as a fallback.
    """
    # Try stdlib first (Python 3.11+); fall back to third-party tomli.
    # The try/except is written to satisfy both mypy (stdlib always wins)
    # and runtime compatibility (older Python still uses tomli).
    import sys  # noqa: PLC0415

    if sys.version_info >= (3, 11):
        import tomllib  # noqa: PLC0415
    else:
        try:
            import tomllib  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[import-not-found,no-redef]  # noqa: PLC0415
            except ImportError as exc:
                raise ManifestError(
                    "Neither tomllib (Python 3.11+) nor tomli is available. Install tomli:  uv pip install tomli"
                ) from exc

    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:
        raise ManifestError(f"Failed to parse {path}: {exc}") from exc


def _validate_entry(raw: object, index: int) -> ClipEntry:
    """Validate a raw dict and return a :class:`ClipEntry`.

    Raises :class:`ManifestError` on any validation failure.
    """
    if not isinstance(raw, dict):
        raise ManifestError(f"clips[{index}] is not a table")

    def _req(key: str) -> str:
        val = raw.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ManifestError(f"clips[{index}] missing or empty required field '{key}'")
        return val.strip()

    category = _req("category")
    if category not in VALID_CATEGORIES:
        raise ManifestError(f"clips[{index}] category={category!r} is not one of {sorted(VALID_CATEGORIES)}")

    scope_raw = _req("scope")
    if scope_raw not in VALID_SCOPES:
        raise ManifestError(f"clips[{index}] scope={scope_raw!r} is not one of {sorted(VALID_SCOPES)}")
    scope: ScopeT = scope_raw  # type: ignore[assignment]

    text = _req("text")
    output_path = _req("output_path")

    generator_raw = raw.get("generator", "elevenlabs")
    if not isinstance(generator_raw, str):
        raise ManifestError(f"clips[{index}] generator must be a string")
    generator_raw = generator_raw.strip()
    if generator_raw not in VALID_GENERATORS:
        raise ManifestError(f"clips[{index}] generator={generator_raw!r} is not one of {sorted(VALID_GENERATORS)}")
    generator: GeneratorT = generator_raw  # type: ignore[assignment]

    persona_raw = raw.get("persona")
    persona: Optional[str] = None
    if persona_raw is not None:
        if not isinstance(persona_raw, str) or not persona_raw.strip():
            raise ManifestError(f"clips[{index}] persona must be a non-empty string when present")
        persona = persona_raw.strip()

    if scope == "per_persona" and persona is None:
        raise ManifestError(f"clips[{index}] scope='per_persona' requires a 'persona' field")

    return ClipEntry(
        category=category,
        scope=scope,
        text=text,
        output_path=output_path,
        generator=generator,
        persona=persona,
    )


def load_manifest(path: Path | None = None) -> list[ClipEntry]:
    """Parse and validate ``static_prompts.toml``; return a list of :class:`ClipEntry`.

    Parameters
    ----------
    path:
        Override the default manifest path (``assets/static_prompts.toml``).
        Useful in tests.

    Raises
    ------
    ManifestError
        When the file is missing, unparseable, or fails schema validation.

    """
    manifest_path = path or _MANIFEST_PATH
    if not manifest_path.is_file():
        raise ManifestError(f"Manifest not found: {manifest_path}")

    raw = _parse_toml(manifest_path)
    clips_raw = raw.get("clips")
    if not isinstance(clips_raw, list):
        raise ManifestError("static_prompts.toml must have a [[clips]] array")

    entries: list[ClipEntry] = []
    for i, raw_entry in enumerate(clips_raw):
        entries.append(_validate_entry(raw_entry, i))

    return entries


# ---------------------------------------------------------------------------
# Clip resolution
# ---------------------------------------------------------------------------


def resolve_clip(
    clips: list[ClipEntry],
    category: str,
    *,
    persona: Optional[str] = None,
) -> ClipEntry | None:
    """Find the best-matching clip for a ``(category, persona)`` query.

    Resolution order:
    1. Per-persona match: ``scope='per_persona'`` and ``clip.persona == persona``
       (only when *persona* is non-empty).
    2. Global fallback: ``scope='global'`` with matching *category*.

    When multiple clips match at the same level, one is chosen at random (pool
    sampling).  Returns ``None`` when no match is found at either level.
    """
    if category not in VALID_CATEGORIES:
        logger.warning("resolve_clip: unknown category %r", category)
        return None

    # 1. Per-persona candidates
    if persona:
        per_persona = [
            c for c in clips if c.category == category and c.scope == "per_persona" and c.persona == persona
        ]
        if per_persona:
            return random.choice(per_persona)

    # 2. Global fallback
    global_clips = [c for c in clips if c.category == category and c.scope == "global"]
    if global_clips:
        return random.choice(global_clips)

    return None


# ---------------------------------------------------------------------------
# Playback API
# ---------------------------------------------------------------------------

# Module-level cache: loaded once on first play_static_prompt call.
_CACHED_MANIFEST: list[ClipEntry] | None = None


def _get_manifest() -> list[ClipEntry]:
    """Return cached manifest, loading on first call. Never raises — returns [] on error."""
    global _CACHED_MANIFEST
    if _CACHED_MANIFEST is not None:
        return _CACHED_MANIFEST
    try:
        _CACHED_MANIFEST = load_manifest()
    except ManifestError as exc:
        logger.warning("static_prompts: failed to load manifest: %s", exc)
        _CACHED_MANIFEST = []
    return _CACHED_MANIFEST


def play_static_prompt(
    category: str,
    *,
    persona: Optional[str] = None,
    manifest_override: list[ClipEntry] | None = None,
) -> bool:
    """Resolve and fire-and-forget play a lifecycle audio clip.

    Resolves the WAV path via :func:`resolve_clip`, then dispatches playback
    via :func:`warmup_audio._dispatch_single_wav` (the shared chokepoint used
    by all pre-rendered audio in this codebase).

    Parameters
    ----------
    category:
        One of the lifecycle category strings defined in
        :data:`VALID_CATEGORIES`.
    persona:
        Optional current persona slug (e.g. ``"don_rickles"``).  Used for
        per-persona clip lookup with global fallback.
    manifest_override:
        Inject a pre-loaded manifest (tests, regenerator).  When ``None``,
        the module-level cached manifest is used.

    Returns
    -------
    bool
        ``True`` if playback was dispatched, ``False`` otherwise (missing
        file, no player available, etc.).  Never raises.

    """
    try:
        clips = manifest_override if manifest_override is not None else _get_manifest()
        clip = resolve_clip(clips, category, persona=persona)
        if clip is None:
            logger.debug("play_static_prompt(%r, persona=%r): no clip found", category, persona)
            return False

        if not clip.exists():
            logger.info(
                "play_static_prompt(%r, persona=%r): WAV not found at %s; skipping. "
                "Run scripts/regenerate-static-prompts.py to generate missing clips.",
                category,
                persona,
                clip.output_path,
            )
            return False

        # Import here to stay on the same late-import pattern as the rest of
        # warmup_audio callers — avoids circular import at module level.
        from robot_comic.warmup_audio import _dispatch_single_wav  # noqa: PLC0415

        dispatched = _dispatch_single_wav(clip.resolved_path)
        if dispatched:
            logger.info(
                "play_static_prompt(%r, persona=%r) dispatched: %s",
                category,
                persona,
                clip.output_path,
            )
        return dispatched
    except Exception as exc:
        # play_static_prompt must NEVER raise — lifecycle hooks depend on this.
        logger.warning("play_static_prompt(%r, persona=%r) failed: %s", category, persona, exc)
        return False


def invalidate_manifest_cache() -> None:
    """Clear the module-level manifest cache (used in tests and after regeneration)."""
    global _CACHED_MANIFEST
    _CACHED_MANIFEST = None
