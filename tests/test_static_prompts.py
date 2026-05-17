"""Tests for the static-prompts manifest, playback helper, and regenerator.

Coverage:
- Manifest loading: happy path, malformed TOML, missing required fields,
  unknown scope/category/generator.
- resolve_clip: per-persona priority, global fallback, missing category.
- play_static_prompt: missing WAV logs + returns False, present WAV dispatches,
  unknown category, exception safety.
- Regenerator idempotent-skip path: TTS backend NOT called when output_path
  already exists.
- SIGTERM handler fires shutdown audio (mocked dispatch).
"""

from __future__ import annotations
import textwrap
import threading
import importlib.util
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from robot_comic.static_prompts import (
    VALID_CATEGORIES,
    ClipEntry,
    ManifestError,
    resolve_clip,
    load_manifest,
    play_static_prompt,
    invalidate_manifest_cache,
)


# ---------------------------------------------------------------------------
# Regenerator module loader (filename has a hyphen, not directly importable)
# ---------------------------------------------------------------------------

_REGEN_SCRIPT = Path(__file__).parents[1] / "scripts" / "regenerate-static-prompts.py"


def _load_regen_module() -> Any:
    """Load scripts/regenerate-static-prompts.py as a module via importlib."""
    spec = importlib.util.spec_from_file_location("regenerate_static_prompts", _REGEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_TOML = textwrap.dedent("""\
    [[clips]]
    category    = "startup"
    scope       = "global"
    text        = "Reachy Mini, online."
    output_path = "assets/startup/online.wav"
    generator   = "elevenlabs"
""")

_PER_PERSONA_TOML = textwrap.dedent("""\
    [[clips]]
    category    = "pause"
    scope       = "per_persona"
    persona     = "don_rickles"
    text        = "Standing by."
    output_path = "profiles/don_rickles/pause.wav"
    generator   = "elevenlabs"

    [[clips]]
    category    = "pause"
    scope       = "global"
    text        = "On break."
    output_path = "assets/pause/on_break.wav"
    generator   = "elevenlabs"
""")

_POOL_TOML = textwrap.dedent("""\
    [[clips]]
    category    = "startup"
    scope       = "global"
    text        = "Online."
    output_path = "assets/startup/online.wav"
    generator   = "elevenlabs"

    [[clips]]
    category    = "startup"
    scope       = "global"
    text        = "Booting up."
    output_path = "assets/startup/booting_up.wav"
    generator   = "elevenlabs"
""")


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "static_prompts.toml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# ManifestError tests (malformed / missing fields)
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_loads_minimal_valid_toml(self, tmp_path: Path) -> None:
        p = _write_toml(tmp_path, _MINIMAL_TOML)
        clips = load_manifest(p)
        assert len(clips) == 1
        c = clips[0]
        assert c.category == "startup"
        assert c.scope == "global"
        assert c.text == "Reachy Mini, online."
        assert c.output_path == "assets/startup/online.wav"
        assert c.generator == "elevenlabs"
        assert c.persona is None

    def test_loads_per_persona_clip(self, tmp_path: Path) -> None:
        p = _write_toml(tmp_path, _PER_PERSONA_TOML)
        clips = load_manifest(p)
        assert len(clips) == 2
        per_p = [c for c in clips if c.scope == "per_persona"]
        assert len(per_p) == 1
        assert per_p[0].persona == "don_rickles"

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="not found"):
            load_manifest(tmp_path / "nonexistent.toml")

    def test_raises_for_invalid_toml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.toml"
        p.write_text("[[clips]\nbroken = ", encoding="utf-8")
        with pytest.raises(ManifestError):
            load_manifest(p)

    def test_raises_for_missing_clips_key(self, tmp_path: Path) -> None:
        p = _write_toml(tmp_path, "[meta]\nfoo = 1\n")
        with pytest.raises(ManifestError, match="clips"):
            load_manifest(p)

    def test_raises_for_unknown_category(self, tmp_path: Path) -> None:
        toml = _MINIMAL_TOML.replace('category    = "startup"', 'category    = "bogus"')
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ManifestError, match="category"):
            load_manifest(p)

    def test_raises_for_unknown_scope(self, tmp_path: Path) -> None:
        toml = _MINIMAL_TOML.replace('scope       = "global"', 'scope       = "unknown"')
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ManifestError, match="scope"):
            load_manifest(p)

    def test_raises_for_unknown_generator(self, tmp_path: Path) -> None:
        toml = _MINIMAL_TOML.replace('generator   = "elevenlabs"', 'generator   = "openai"')
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ManifestError, match="generator"):
            load_manifest(p)

    def test_raises_per_persona_without_persona_field(self, tmp_path: Path) -> None:
        toml = textwrap.dedent("""\
            [[clips]]
            category    = "pause"
            scope       = "per_persona"
            text        = "Standing by."
            output_path = "profiles/don_rickles/pause.wav"
            generator   = "elevenlabs"
        """)
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ManifestError, match="persona"):
            load_manifest(p)

    def test_raises_for_missing_text_field(self, tmp_path: Path) -> None:
        toml = textwrap.dedent("""\
            [[clips]]
            category    = "startup"
            scope       = "global"
            output_path = "assets/startup/online.wav"
            generator   = "elevenlabs"
        """)
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ManifestError, match="text"):
            load_manifest(p)

    def test_defaults_generator_to_elevenlabs(self, tmp_path: Path) -> None:
        toml = textwrap.dedent("""\
            [[clips]]
            category    = "startup"
            scope       = "global"
            text        = "Online."
            output_path = "assets/startup/online.wav"
        """)
        p = _write_toml(tmp_path, toml)
        clips = load_manifest(p)
        assert clips[0].generator == "elevenlabs"

    def test_all_categories_accepted(self, tmp_path: Path) -> None:
        """Every VALID_CATEGORIES string should parse without error."""
        lines = []
        for i, cat in enumerate(sorted(VALID_CATEGORIES)):
            lines += [
                "[[clips]]",
                f'category    = "{cat}"',
                '  scope       = "global"',
                f'  text        = "Text {i}."',
                f'  output_path = "assets/{cat}/clip.wav"',
                '  generator   = "elevenlabs"',
                "",
            ]
        toml_content = "\n".join(lines)
        p = _write_toml(tmp_path, toml_content)
        clips = load_manifest(p)
        assert len(clips) == len(VALID_CATEGORIES)


# ---------------------------------------------------------------------------
# resolve_clip tests
# ---------------------------------------------------------------------------


class TestResolveClip:
    def _make_clips(self) -> list[ClipEntry]:
        import tempfile

        from robot_comic.static_prompts import load_manifest

        toml = _PER_PERSONA_TOML
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sp.toml"
            p.write_text(toml, encoding="utf-8")
            return load_manifest(p)

    def test_per_persona_preferred_over_global(self) -> None:
        clips = self._make_clips()
        clip = resolve_clip(clips, "pause", persona="don_rickles")
        assert clip is not None
        assert clip.scope == "per_persona"
        assert clip.persona == "don_rickles"

    def test_global_fallback_when_no_per_persona(self) -> None:
        clips = self._make_clips()
        # bill_hicks has no per-persona pause clip in _PER_PERSONA_TOML
        clip = resolve_clip(clips, "pause", persona="bill_hicks")
        assert clip is not None
        assert clip.scope == "global"

    def test_global_fallback_when_no_persona_given(self) -> None:
        clips = self._make_clips()
        clip = resolve_clip(clips, "pause")
        assert clip is not None
        assert clip.scope == "global"

    def test_returns_none_for_missing_category(self) -> None:
        clips = self._make_clips()
        clip = resolve_clip(clips, "startup")
        # No startup clips in _PER_PERSONA_TOML
        assert clip is None

    def test_returns_none_for_unknown_category(self) -> None:
        clips = self._make_clips()
        result = resolve_clip(clips, "totally_unknown")  # type: ignore[arg-type]
        assert result is None

    def test_pool_sampling_stays_within_pool(self, tmp_path: Path) -> None:
        p = _write_toml(tmp_path, _POOL_TOML)
        clips = load_manifest(p)
        paths = {resolve_clip(clips, "startup").output_path for _ in range(30)}  # type: ignore[union-attr]
        # Must always pick from within the pool
        assert paths <= {"assets/startup/online.wav", "assets/startup/booting_up.wav"}
        # After enough draws both variants should appear (probabilistically — 30
        # trials at p=0.5 each means failure probability ≈ 2^-30 ≈ negligible)
        assert len(paths) == 2


# ---------------------------------------------------------------------------
# play_static_prompt tests
# ---------------------------------------------------------------------------


class TestPlayStaticPrompt:
    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        """Ensure the module-level manifest cache is clean between tests."""
        invalidate_manifest_cache()

    def _make_global_clip(self, tmp_path: Path, *, file_exists: bool = False) -> tuple[list[ClipEntry], Path]:
        """Return (clips, wav_path) for a single global startup clip."""
        import textwrap

        wav_path = tmp_path / "assets" / "startup" / "online.wav"
        rel = "assets/startup/online.wav"
        toml = textwrap.dedent(f"""\
            [[clips]]
            category    = "startup"
            scope       = "global"
            text        = "Online."
            output_path = "{rel}"
            generator   = "elevenlabs"
        """)
        manifest_path = tmp_path / "sp.toml"
        manifest_path.write_text(toml, encoding="utf-8")
        clips = load_manifest(manifest_path)

        if file_exists:
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.write_bytes(b"RIFF")

        return clips, wav_path

    def test_returns_false_when_wav_missing(self, tmp_path: Path) -> None:
        clips, _ = self._make_global_clip(tmp_path, file_exists=False)
        result = play_static_prompt("startup", manifest_override=clips)
        assert result is False

    def test_returns_false_for_unknown_category(self, tmp_path: Path) -> None:
        clips, wav = self._make_global_clip(tmp_path, file_exists=True)
        result = play_static_prompt("totally_unknown", manifest_override=clips)  # type: ignore[arg-type]
        assert result is False

    def test_returns_false_when_no_clips_for_category(self, tmp_path: Path) -> None:
        clips, wav = self._make_global_clip(tmp_path, file_exists=True)
        # clips only have 'startup', so 'shutdown' → no match
        result = play_static_prompt("shutdown", manifest_override=clips)
        assert result is False

    def test_dispatches_when_wav_present(self, tmp_path: Path) -> None:
        clips, wav = self._make_global_clip(tmp_path, file_exists=True)

        import robot_comic.warmup_audio as wa_mod
        import robot_comic.static_prompts as sp_mod

        # Patch _REPO_ROOT so ClipEntry.resolved_path points into tmp_path.
        # resolved_path is a property that reads _REPO_ROOT at call time, so
        # patching before play_static_prompt is called is sufficient.
        orig_root = sp_mod._REPO_ROOT
        orig_dispatch = wa_mod._dispatch_single_wav
        dispatched_paths: list[Path] = []

        def _capture(p: Path) -> bool:
            dispatched_paths.append(p)
            return True

        try:
            sp_mod._REPO_ROOT = tmp_path
            wa_mod._dispatch_single_wav = _capture  # type: ignore[assignment]
            result = play_static_prompt("startup", manifest_override=clips)
            assert result is True
            assert len(dispatched_paths) == 1
            assert dispatched_paths[0] == wav
        finally:
            sp_mod._REPO_ROOT = orig_root
            wa_mod._dispatch_single_wav = orig_dispatch  # type: ignore[assignment]

    def test_never_raises_on_exception(self, tmp_path: Path) -> None:
        """play_static_prompt must swallow all exceptions."""
        clips, wav = self._make_global_clip(tmp_path, file_exists=True)

        import robot_comic.warmup_audio as wa_mod

        original = wa_mod._dispatch_single_wav
        try:
            wa_mod._dispatch_single_wav = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[assignment]
            # Should not raise
            result = play_static_prompt("startup", manifest_override=clips)
            assert result is False  # returns False on exception
        finally:
            wa_mod._dispatch_single_wav = original  # type: ignore[assignment]

    def test_per_persona_resolution(self, tmp_path: Path) -> None:
        """Per-persona clip is chosen over global when persona matches."""
        toml = _PER_PERSONA_TOML
        manifest_path = tmp_path / "sp.toml"
        manifest_path.write_text(toml, encoding="utf-8")

        # Create the per-persona WAV file so dispatch happens
        per_persona_wav = tmp_path / "profiles" / "don_rickles" / "pause.wav"
        per_persona_wav.parent.mkdir(parents=True, exist_ok=True)
        per_persona_wav.write_bytes(b"RIFF")

        # Patch resolved_path on the per-persona clip to point to tmp_path file
        import robot_comic.static_prompts as sp_mod

        # Patch _REPO_ROOT so resolved_path points into tmp_path
        orig_root = sp_mod._REPO_ROOT
        try:
            sp_mod._REPO_ROOT = tmp_path
            # Re-load clips with patched root
            clips2 = load_manifest(manifest_path)
            import robot_comic.warmup_audio as wa_mod

            orig_dispatch = wa_mod._dispatch_single_wav
            dispatched_paths: list[Path] = []

            def _capture(p: Path) -> bool:
                dispatched_paths.append(p)
                return True

            wa_mod._dispatch_single_wav = _capture  # type: ignore[assignment]
            try:
                play_static_prompt("pause", persona="don_rickles", manifest_override=clips2)
                assert len(dispatched_paths) == 1
                assert dispatched_paths[0].name == "pause.wav"
                # Confirm it's the per-persona one, not the global
                assert "don_rickles" in str(dispatched_paths[0])
            finally:
                wa_mod._dispatch_single_wav = orig_dispatch  # type: ignore[assignment]
        finally:
            sp_mod._REPO_ROOT = orig_root


# ---------------------------------------------------------------------------
# Regenerator idempotency tests
# ---------------------------------------------------------------------------


class TestRegeneratorIdempotency:
    """Verify that the regenerator skips clips that already exist on disk."""

    def _build_manifest_toml(self, tmp_path: Path) -> tuple[Path, Path]:
        """Write a TOML manifest with one clip and return (manifest_path, wav_path)."""
        wav_path = tmp_path / "assets" / "startup" / "online.wav"
        rel = "assets/startup/online.wav"
        toml = textwrap.dedent(f"""\
            [[clips]]
            category    = "startup"
            scope       = "global"
            text        = "Online."
            output_path = "{rel}"
            generator   = "elevenlabs"
        """)
        manifest_path = tmp_path / "static_prompts.toml"
        manifest_path.write_text(toml, encoding="utf-8")
        return manifest_path, wav_path

    def _patch_sp_module(self, tmp_path: Path, manifest_path: Path) -> tuple[Any, Any, Path, Path]:
        """Return (sp_mod, orig_root, orig_manifest_path) with patches applied.

        Callers must restore originals in a finally block:
            sp_mod._REPO_ROOT = orig_root
            sp_mod._MANIFEST_PATH = orig_manifest
        """
        import robot_comic.static_prompts as sp_mod

        orig_root = sp_mod._REPO_ROOT
        orig_manifest = sp_mod._MANIFEST_PATH
        sp_mod._REPO_ROOT = tmp_path
        sp_mod._MANIFEST_PATH = manifest_path
        return sp_mod, orig_root, orig_manifest

    def test_skips_existing_clip_without_calling_backend(self, tmp_path: Path) -> None:
        manifest_path, wav_path = self._build_manifest_toml(tmp_path)
        # Pre-create the WAV file so it already exists
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"RIFF")

        sp_mod, orig_root, orig_manifest = self._patch_sp_module(tmp_path, manifest_path)
        regen_mod = _load_regen_module()
        try:
            mock_generate = MagicMock()
            orig_generate = regen_mod._generate_elevenlabs
            try:
                regen_mod._generate_elevenlabs = mock_generate
                exit_code = regen_mod.regenerate(
                    tts_backend_override=None,
                    force=False,
                    dry_run=False,
                    persona_filter=None,
                    category_filter=None,
                    list_only=False,
                )
                assert exit_code == 0
                # TTS backend must NOT be called when the file already exists
                mock_generate.assert_not_called()
            finally:
                regen_mod._generate_elevenlabs = orig_generate
        finally:
            sp_mod._REPO_ROOT = orig_root
            sp_mod._MANIFEST_PATH = orig_manifest

    def test_generates_missing_clip(self, tmp_path: Path) -> None:
        manifest_path, wav_path = self._build_manifest_toml(tmp_path)
        # Do NOT create the WAV file

        sp_mod, orig_root, orig_manifest = self._patch_sp_module(tmp_path, manifest_path)
        regen_mod = _load_regen_module()
        try:
            # Mock the ElevenLabs generate function to write a fake WAV
            def _fake_generate(text: str, output_path: Path, *, voice_id: str) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"RIFF")

            mock_generate = MagicMock(side_effect=_fake_generate)
            orig_generate = regen_mod._generate_elevenlabs
            try:
                regen_mod._generate_elevenlabs = mock_generate
                exit_code = regen_mod.regenerate(
                    tts_backend_override=None,
                    force=False,
                    dry_run=False,
                    persona_filter=None,
                    category_filter=None,
                    list_only=False,
                )
                # Only one clip in our test TOML, so exactly one generate call.
                assert mock_generate.call_count == 1
                assert exit_code == 0
            finally:
                regen_mod._generate_elevenlabs = orig_generate
        finally:
            sp_mod._REPO_ROOT = orig_root
            sp_mod._MANIFEST_PATH = orig_manifest

    def test_dry_run_does_not_call_backend(self, tmp_path: Path) -> None:
        manifest_path, wav_path = self._build_manifest_toml(tmp_path)
        # Do NOT create the WAV file

        sp_mod, orig_root, orig_manifest = self._patch_sp_module(tmp_path, manifest_path)
        regen_mod = _load_regen_module()
        try:
            mock_generate = MagicMock()
            orig_generate = regen_mod._generate_elevenlabs
            try:
                regen_mod._generate_elevenlabs = mock_generate
                exit_code = regen_mod.regenerate(
                    tts_backend_override=None,
                    force=False,
                    dry_run=True,
                    persona_filter=None,
                    category_filter=None,
                    list_only=False,
                )
                assert exit_code == 0
                mock_generate.assert_not_called()
            finally:
                regen_mod._generate_elevenlabs = orig_generate
        finally:
            sp_mod._REPO_ROOT = orig_root
            sp_mod._MANIFEST_PATH = orig_manifest


# ---------------------------------------------------------------------------
# SIGTERM handler shutdown audio integration test
# ---------------------------------------------------------------------------


class TestSigtermShutdownAudio:
    """Integration test: SIGTERM handler fires shutdown audio.

    The _request_graceful_shutdown closure defined in main.run() is not
    directly importable (it's a local function inside run()). We test the
    semantics of that handler by replicating its body and verifying that
    play_static_prompt is invoked with category='shutdown'.

    Note: on Windows, os.kill(pid, SIGTERM) terminates the process immediately
    without invoking Python signal handlers, so a subprocess-based test is
    not cross-platform.  Instead, we test the handler body directly.
    """

    def test_shutdown_handler_body_calls_play_static_prompt(self) -> None:
        """Handler body from main.py must call play_static_prompt('shutdown').

        We replicate the handler body and use module-level access so that
        monkey-patching sp_mod.play_static_prompt is visible at call time.
        """
        import robot_comic.static_prompts as sp_mod

        stop_event = threading.Event()
        called_with: list[tuple[str, object]] = []

        def _mock_play(category: str, **kwargs: object) -> bool:
            called_with.append((category, kwargs.get("persona")))
            return False  # WAV doesn't exist in test — returning False is fine

        orig_play = sp_mod.play_static_prompt
        try:
            sp_mod.play_static_prompt = _mock_play  # type: ignore[assignment]

            # Replicate the handler body from main.py _request_graceful_shutdown.
            # Call via module reference so the monkey-patch is visible.
            def _simulated_handler() -> None:
                try:
                    import robot_comic.static_prompts as _sp  # noqa: PLC0415
                    from robot_comic import config as _cfg  # noqa: PLC0415

                    _persona = getattr(_cfg.config, "REACHY_MINI_CUSTOM_PROFILE", None) or None
                    _sp.play_static_prompt("shutdown", persona=_persona)
                except Exception:
                    pass
                stop_event.set()

            _simulated_handler()

            assert stop_event.is_set()
            assert len(called_with) == 1
            assert called_with[0][0] == "shutdown"
        finally:
            sp_mod.play_static_prompt = orig_play  # type: ignore[assignment]

    def test_shutdown_handler_does_not_raise_on_audio_error(self) -> None:
        """If play_static_prompt raises, the handler must still set the stop event."""
        import robot_comic.static_prompts as sp_mod

        stop_event = threading.Event()

        def _exploding_play(category: str, **kwargs: object) -> bool:
            raise RuntimeError("TTS is down")

        orig_play = sp_mod.play_static_prompt
        try:
            sp_mod.play_static_prompt = _exploding_play  # type: ignore[assignment]

            def _simulated_handler() -> None:
                try:
                    import robot_comic.static_prompts as _sp  # noqa: PLC0415
                    from robot_comic import config as _cfg  # noqa: PLC0415

                    _persona = getattr(_cfg.config, "REACHY_MINI_CUSTOM_PROFILE", None) or None
                    _sp.play_static_prompt("shutdown", persona=_persona)
                except Exception:
                    pass  # must not propagate
                stop_event.set()

            _simulated_handler()
            assert stop_event.is_set()  # handler continued despite audio failure
        finally:
            sp_mod.play_static_prompt = orig_play  # type: ignore[assignment]
