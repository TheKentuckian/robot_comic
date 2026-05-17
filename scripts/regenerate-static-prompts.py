#!/usr/bin/env python3
"""Regenerate static lifecycle audio clips from assets/static_prompts.toml.

Usage
-----
    python scripts/regenerate-static-prompts.py [options]

Options
-------
--tts-backend {elevenlabs,xtts}
    Override the TTS backend for ALL clips (ignores per-clip ``generator``
    field).  When omitted, each clip uses the backend declared in its own
    manifest entry.

--force
    Regenerate all clips even if the output file already exists.

--dry-run
    Print which clips would be generated (and skipped) without calling any
    TTS API.

--persona PERSONA
    Only process clips for this persona slug (e.g. ``don_rickles``).  Global
    clips are always included unless this flag is combined with
    ``--category``.

--category CATEGORY
    Only process clips in this category.

--list
    List all clips in the manifest (path, category, scope, generator,
    exists?) and exit without generating anything.

Cross-platform
--------------
This script is pure Python and works on Linux, macOS, and Windows.
No shell-specific syntax is used.

ElevenLabs backend
------------------
Requires ``ELEVENLABS_API_KEY`` env var.  Voice IDs are resolved per-persona
from ``profiles/<persona>/elevenlabs.txt`` (or ``.local.txt`` overlay).
Global clips use the ``DEFAULT_ELEVENLABS_VOICE`` env var fallback.

XTTS backend
------------
Not yet implemented.  Raises ``NotImplementedError`` pointing at issue #438.
Once the xtts adapter lands (#438), replace the ``_generate_xtts`` stub.

Idempotency
-----------
Re-running with no changes is a no-op: clips whose ``output_path`` already
exists on disk are skipped.  Use ``--force`` to overwrite.
"""

from __future__ import annotations
import os
import sys
import logging
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Bootstrap: make ``robot_comic`` importable when run as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("regenerate_static_prompts")


# ---------------------------------------------------------------------------
# Backend stubs / implementations
# ---------------------------------------------------------------------------


def _generate_elevenlabs(text: str, output_path: Path, *, voice_id: str) -> None:
    """Synthesize *text* with ElevenLabs and write WAV to *output_path*.

    Requires:
    - ``elevenlabs`` package (``uv pip install elevenlabs``)
    - ``ELEVENLABS_API_KEY`` env var

    Raises
    ------
    RuntimeError
        When the API key is missing or the synthesis call fails.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY env var is not set. "
            "Set it before running the regenerator:\n"
            "  export ELEVENLABS_API_KEY=sk-..."
        )
    try:
        from elevenlabs import ElevenLabs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("elevenlabs package not installed. Install it:\n  uv pip install elevenlabs") from exc

    client = ElevenLabs(api_key=api_key)
    audio_bytes = b"".join(client.text_to_speech.convert(text=text, voice_id=voice_id))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    logger.info("  wrote %d bytes → %s", len(audio_bytes), output_path)


def _generate_xtts(text: str, output_path: Path, *, persona: str | None) -> None:  # noqa: ARG001
    """XTTS synthesis stub — not yet implemented.

    Will be filled in once the xtts TTS adapter lands (issue #438).
    The xtts LAN service is not available yet; attempting to use this
    backend before #438 is merged will raise ``NotImplementedError``.
    """
    raise NotImplementedError(
        "XTTS TTS backend is not yet implemented in the regenerator. "
        "See https://github.com/TheKentuckian/robot_comic/issues/438 "
        "for the xtts adapter that will power this path."
    )


# ---------------------------------------------------------------------------
# Voice ID resolution
# ---------------------------------------------------------------------------

_DEFAULT_VOICE_ENV = "DEFAULT_ELEVENLABS_VOICE"
_DEFAULT_VOICE_FALLBACK = "21m00Tcm4TlvDq8ikWAM"  # Rachel — ElevenLabs default


def _resolve_elevenlabs_voice(persona: str | None) -> str:
    """Return the ElevenLabs voice ID to use for *persona* (or the default).

    Resolution order:
    1. ``profiles/<persona>/elevenlabs.local.txt`` (local override, not committed)
    2. ``profiles/<persona>/elevenlabs.txt`` (committed defaults)
    3. ``DEFAULT_ELEVENLABS_VOICE`` env var
    4. Hardcoded ``21m00Tcm4TlvDq8ikWAM`` (Rachel)

    The text files contain the raw voice ID on the first non-comment line.
    """
    if persona:
        persona_dir = _REPO_ROOT / "profiles" / persona
        for fname in ("elevenlabs.local.txt", "elevenlabs.txt"):
            p = persona_dir / fname
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        logger.debug("  voice_id from %s/%s: %s", persona, fname, line)
                        return line
    env_voice = os.environ.get(_DEFAULT_VOICE_ENV, "").strip()
    if env_voice:
        return env_voice
    return _DEFAULT_VOICE_FALLBACK


# ---------------------------------------------------------------------------
# Main regeneration logic
# ---------------------------------------------------------------------------


def _list_clips(clips: list) -> None:
    """Print a formatted table of all clips."""
    fmt = "{:<20} {:<12} {:<14} {:<12} {:<8} {}"
    header = fmt.format("category", "scope", "persona", "generator", "exists?", "output_path")
    print(header)
    print("-" * len(header))
    for c in clips:
        exists_marker = "YES" if c.exists() else "no"
        print(fmt.format(c.category, c.scope, c.persona or "-", c.generator, exists_marker, c.output_path))


def regenerate(
    *,
    tts_backend_override: str | None,
    force: bool,
    dry_run: bool,
    persona_filter: str | None,
    category_filter: str | None,
    list_only: bool,
) -> int:
    """Main regeneration driver. Returns exit code (0 = ok, 1 = errors)."""
    from robot_comic.static_prompts import ManifestError, load_manifest  # noqa: PLC0415

    try:
        clips = load_manifest()
    except ManifestError as exc:
        logger.error("Cannot load manifest: %s", exc)
        return 1

    if list_only:
        _list_clips(clips)
        return 0

    # Apply filters
    filtered = clips
    if persona_filter:
        filtered = [c for c in filtered if c.scope == "global" or c.persona == persona_filter]
    if category_filter:
        filtered = [c for c in filtered if c.category == category_filter]

    n_skipped = n_generated = n_errors = 0

    for clip in filtered:
        effective_generator = tts_backend_override or clip.generator

        # Idempotency: skip existing unless --force
        if clip.exists() and not force:
            logger.debug("SKIP (exists): %s", clip.output_path)
            n_skipped += 1
            continue

        logger.info(
            "%-20s %-12s persona=%-14s generator=%s  → %s",
            clip.category,
            clip.scope,
            clip.persona or "-",
            effective_generator,
            clip.output_path,
        )

        if dry_run:
            n_generated += 1
            continue

        try:
            out = clip.resolved_path
            out.parent.mkdir(parents=True, exist_ok=True)

            if effective_generator == "elevenlabs":
                voice_id = _resolve_elevenlabs_voice(clip.persona)
                _generate_elevenlabs(clip.text, out, voice_id=voice_id)
            elif effective_generator == "xtts":
                _generate_xtts(clip.text, out, persona=clip.persona)
            else:
                raise ValueError(f"Unknown generator: {effective_generator!r}")

            n_generated += 1
        except Exception as exc:
            logger.error("FAILED %s: %s", clip.output_path, exc)
            n_errors += 1

    summary = f"Done. generated={n_generated} skipped={n_skipped} errors={n_errors}" + (
        " (dry-run — no files written)" if dry_run else ""
    )
    if n_errors:
        logger.error(summary)
        return 1
    logger.info(summary)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tts-backend",
        choices=["elevenlabs", "xtts"],
        default=None,
        metavar="{elevenlabs,xtts}",
        help="Override TTS backend for all clips (default: use per-clip 'generator' field).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all clips even if output file already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without calling any TTS API.",
    )
    p.add_argument(
        "--persona",
        default=None,
        metavar="PERSONA",
        help="Only process clips for this persona slug (e.g. don_rickles).",
    )
    p.add_argument(
        "--category",
        default=None,
        metavar="CATEGORY",
        help="Only process clips in this category (e.g. shutdown).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all clips and their status, then exit.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    return regenerate(
        tts_backend_override=args.tts_backend,
        force=args.force,
        dry_run=args.dry_run,
        persona_filter=args.persona,
        category_filter=args.category,
        list_only=args.list,
    )


if __name__ == "__main__":
    sys.exit(main())
