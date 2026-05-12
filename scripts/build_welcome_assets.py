#!/usr/bin/env python3
"""Build Welcome Gate audio assets.

Two halves:

1. ``extract`` — pull per-comedian "name" clips from YouTube. Uses yt-dlp to
   grab the audio, transcribes with word-level timestamps (faster-whisper
   preferred, whisper-timestamped fallback), scans for the first configured
   phrase, then cuts and loudness-normalises a short clip to
   ``profiles/<name>/welcome_name.wav``.

2. ``generate-narrator`` / ``generate-character`` — synthesise WAVs with
   Gemini TTS for the shared narrator prompts and the original-character
   profiles (no real human to extract from).

All output is 24 kHz mono int16 WAV. The script is re-runnable: existing
outputs are skipped unless ``--force`` is passed.

Optional dependencies (install via ``pip install -e .[asset_build]``):
    faster-whisper  (preferred — fast, GPU-friendly)
    whisper-timestamped  (fallback)

External tools expected on PATH:
    yt-dlp
    ffmpeg, ffprobe
"""

from __future__ import annotations
import os
import re
import sys


# Force UTF-8 stdout/stderr so we don't crash printing transcripts / log lines
# from faster-whisper that contain non-cp1252 characters on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import time
import wave
import base64
import shutil
import hashlib
import argparse
import subprocess
import dataclasses
from typing import Any, Iterable
from pathlib import Path

import tomllib


# Local import so we share constants/auth conventions with the runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Load .env so GEMINI_API_KEY / GOOGLE_API_KEY work the same as in the app.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = REPO_ROOT / "profiles"
ASSETS_DIR = REPO_ROOT / "assets"
WELCOME_DIR = ASSETS_DIR / "welcome"
SOURCES_TOML = ASSETS_DIR / "welcome_sources.toml"

SAMPLE_RATE = 24000  # matches GEMINI_TTS_OUTPUT_SAMPLE_RATE
PAD_BEFORE = 0.10
PAD_AFTER = 0.15
NARRATOR_VOICE = "Algenib"  # easy to swap; default Gemini TTS voice

# Real-character profiles are listed in welcome_sources.toml. Everything else
# under profiles/ (minus these) gets Gemini-TTS'd from the display name.
SKIP_PROFILES = {"default", "example"}

NARRATOR_PROMPTS: dict[str, str] = {
    "welcome": "Welcome to Robot Comic. Pick a comedian, or say list for the full lineup.",
    "list_intro": "Here are your comedians...",
    "not_understood": "Didn't catch that. Try again, or say list.",
    "switching": "Switching comedians...",
}


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Word:
    """Single transcribed word with timing relative to the analysed audio."""

    text: str
    start: float
    end: float


def die(msg: str, code: int = 1) -> "None":
    """Print an error and exit."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def check_external_tools() -> None:
    """Verify yt-dlp / ffmpeg / ffprobe are on PATH; exit with guidance if not."""
    missing = [t for t in ("yt-dlp", "ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        die(
            "missing required tools on PATH: "
            + ", ".join(missing)
            + ". Install ffmpeg and `pip install -U yt-dlp` (or use your package manager)."
        )


def parse_hms(value: str | None) -> float | None:
    """Parse 'HH:MM:SS' or 'MM:SS' or seconds-as-number into seconds."""
    if not value:
        return None
    parts = value.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        die(f"invalid time {value!r}")
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 1:
        return nums[0]
    die(f"invalid time {value!r}")
    return None  # unreachable


def fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, capturing text output. Raises on non-zero exit."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


# --------------------------------------------------------------------------- #
# Profile helpers                                                             #
# --------------------------------------------------------------------------- #


def list_profiles() -> list[str]:
    """Return sorted profile directory names (excluding SKIP_PROFILES)."""
    if not PROFILES_DIR.exists():
        die(f"profiles directory not found: {PROFILES_DIR}")
    return sorted(p.name for p in PROFILES_DIR.iterdir() if p.is_dir() and p.name not in SKIP_PROFILES)


def display_name(profile: str) -> str:
    """Derive a spoken display name from a profile directory name."""
    return profile.replace("_", " ").title()


def welcome_name_path(profile: str) -> Path:
    """Path to the per-profile welcome_name.wav."""
    return PROFILES_DIR / profile / "welcome_name.wav"


def load_sources() -> dict[str, dict[str, Any]]:
    """Parse welcome_sources.toml into a {profile: {...}} dict."""
    if not SOURCES_TOML.exists():
        die(f"sources TOML not found: {SOURCES_TOML}")
    with SOURCES_TOML.open("rb") as f:
        return tomllib.load(f)


# --------------------------------------------------------------------------- #
# Transcription                                                               #
# --------------------------------------------------------------------------- #


_whisper_model: Any = None  # cached module-level — destroyed on interpreter exit


def transcribe_words(audio_path: Path) -> list[Word]:
    """Transcribe ``audio_path`` and return a flat list of word-level tokens.

    Prefers faster-whisper; falls back to whisper-timestamped. Raises
    RuntimeError if neither is installed.
    """
    global _whisper_model
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError:
        return _transcribe_whisper_timestamped(audio_path)

    if _whisper_model is None:
        print("  loading faster-whisper (medium)...")
        _whisper_model = WhisperModel("medium", device="auto", compute_type="auto")
    print("  transcribing with faster-whisper (medium)...")
    segments, _ = _whisper_model.transcribe(str(audio_path), word_timestamps=True, language="en")
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            if w.start is None or w.end is None:
                continue
            words.append(Word(text=w.word, start=float(w.start), end=float(w.end)))
    return words


def _transcribe_whisper_timestamped(audio_path: Path) -> list[Word]:
    """Fallback transcription using whisper-timestamped."""
    try:
        import whisper_timestamped as wt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("no transcription backend installed. Install with: pip install -e .[asset_build]") from exc

    print("  transcribing with whisper-timestamped (medium)...")
    model = wt.load_model("medium")
    result = wt.transcribe(model, str(audio_path), language="en")
    words: list[Word] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(Word(text=w["text"], start=float(w["start"]), end=float(w["end"])))
    return words


_PUNCT_RE = re.compile(r"[^\w']+", re.UNICODE)


def _norm_token(text: str) -> str:
    """Normalise a transcribed word for matching: lowercase, strip punctuation."""
    return _PUNCT_RE.sub("", text.lower()).strip()


def find_phrase(words: list[Word], phrase: str) -> tuple[float, float] | None:
    """Locate the first occurrence of ``phrase`` in ``words``.

    Returns ``(start, end)`` in the transcription's time base, or ``None``.
    """
    target = [t for t in (_norm_token(p) for p in phrase.split()) if t]
    if not target:
        return None
    norm = [_norm_token(w.text) for w in words]
    n = len(target)
    for i in range(len(words) - n + 1):
        if norm[i : i + n] == target:
            return words[i].start, words[i + n - 1].end
    return None


# --------------------------------------------------------------------------- #
# Audio I/O                                                                   #
# --------------------------------------------------------------------------- #


def yt_dlp_audio(url: str, dest_dir: Path) -> Path:
    """Download audio for ``url`` as WAV, returning the resulting path."""
    out_template = str(dest_dir / "source.%(ext)s")
    print(f"  downloading {url}")
    run(
        [
            "yt-dlp",
            "-x",
            "--audio-format",
            "wav",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "-o",
            out_template,
            url,
        ]
    )
    wavs = list(dest_dir.glob("source.wav"))
    if not wavs:
        die("yt-dlp produced no WAV file")
    return wavs[0]


def ffmpeg_clip_window(src: Path, dst: Path, start: float | None, end: float | None) -> None:
    """Trim ``src`` to ``[start, end]`` (full file if both None) into ``dst``."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(src), "-ac", "1", "-ar", "16000", str(dst)]
    run(cmd)


def ffmpeg_cut_and_normalise(src: Path, dst: Path, start: float, duration: float) -> None:
    """Cut ``[start, start+duration]`` from ``src``, loudnorm + downsample to ``dst``."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src),
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(dst),
    ]
    run(cmd)


def ffprobe_duration(path: Path) -> float:
    """Return the duration of an audio file in seconds."""
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    ).stdout.strip()
    return float(out)


def write_wav_pcm16(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Write raw little-endian 16-bit mono PCM to a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def transcode_to_target(src_pcm: bytes, src_rate: int) -> bytes:
    """Resample raw PCM16 mono to SAMPLE_RATE via ffmpeg pipe."""
    if src_rate == SAMPLE_RATE:
        return src_pcm
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(src_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-f",
            "s16le",
            "pipe:1",
        ],
        input=src_pcm,
        capture_output=True,
        check=True,
    )
    return proc.stdout


# --------------------------------------------------------------------------- #
# Gemini TTS                                                                  #
# --------------------------------------------------------------------------- #


def _gemini_client() -> Any:
    """Build a Gemini client using the same env vars as the runtime."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        die("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
    try:
        from google import genai  # type: ignore[import-not-found]
    except ImportError:
        die("google-genai not installed (it should be a runtime dep)")
        raise  # unreachable; satisfies type checker
    return genai.Client(api_key=api_key)


_GEMINI_MIN_INTERVAL_S = 7.0  # 10 RPM quota → 1 req per 6s; pad to 7s for safety
_GEMINI_MAX_ATTEMPTS = 4
_last_gemini_call = 0.0


class DailyQuotaExhausted(RuntimeError):
    """Raised when Gemini returns a per-day quota exhaustion. Don't retry."""


def _extract_pcm(resp: Any) -> bytes | None:
    """Pull inline audio bytes from a Gemini response, or None if absent."""
    try:
        parts = resp.candidates[0].content.parts
    except (AttributeError, IndexError, TypeError):
        return None
    for part in parts or []:
        inline = getattr(part, "inline_data", None)
        data = getattr(inline, "data", None) if inline is not None else None
        if data is None:
            continue
        return base64.b64decode(data) if isinstance(data, str) else bytes(data)
    return None


def gemini_tts(text: str, voice: str) -> bytes:
    """Synthesise ``text`` with Gemini TTS at ``voice``; return raw PCM16 bytes.

    Throttles to the 10 RPM quota and retries on 429 / empty-audio responses.
    Short inputs occasionally come back as a text candidate with no
    ``inline_data``; we retry with a slightly padded prompt in that case.
    """
    global _last_gemini_call
    from google.genai import types  # type: ignore[import-not-found]

    from robot_comic.gemini_tts import GEMINI_TTS_MODEL, GEMINI_TTS_OUTPUT_SAMPLE_RATE

    client = _gemini_client()
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))
        ),
    )

    last_exc: Exception | None = None
    for attempt in range(_GEMINI_MAX_ATTEMPTS):
        wait = _GEMINI_MIN_INTERVAL_S - (time.monotonic() - _last_gemini_call)
        if wait > 0:
            time.sleep(wait)
        # Pad very short prompts on retry — bare two-word names sometimes return text-only.
        prompt = text if attempt == 0 else f"Say: {text}."
        try:
            resp = client.models.generate_content(model=GEMINI_TTS_MODEL, contents=prompt, config=cfg)
            _last_gemini_call = time.monotonic()
            pcm = _extract_pcm(resp)
            if pcm:
                return transcode_to_target(pcm, GEMINI_TTS_OUTPUT_SAMPLE_RATE)
            last_exc = RuntimeError("response had no inline audio data")
            print(f"  attempt {attempt + 1}: empty audio — retrying with padded prompt")
        except Exception as exc:
            _last_gemini_call = time.monotonic()
            msg = str(exc)
            last_exc = exc
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if "per_day" in msg or "PerDay" in msg:
                    raise DailyQuotaExhausted(
                        "Gemini TTS daily quota exhausted — retry tomorrow or swap API keys"
                    ) from exc
                m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", msg)
                raw = float(m.group(1)) if m else 30.0
                delay = min(70.0, raw + 1.0)  # cap: 60s quota window
                print(f"  attempt {attempt + 1}: 429 rate-limit — sleeping {delay:.0f}s")
                time.sleep(delay)
                continue
            print(f"  attempt {attempt + 1}: {msg.splitlines()[0][:200]}")
    raise RuntimeError(f"gemini_tts failed after {_GEMINI_MAX_ATTEMPTS} attempts: {last_exc}")


# --------------------------------------------------------------------------- #
# Subcommands                                                                 #
# --------------------------------------------------------------------------- #


def cmd_status(_args: argparse.Namespace) -> int:
    """Report which profiles already have a welcome_name.wav."""
    sources = load_sources()
    real = set(sources.keys())
    rows: list[tuple[str, str, str]] = []
    for prof in list_profiles():
        path = welcome_name_path(prof)
        present = "OK " if path.exists() else "-- "
        kind = "real" if prof in real else "char"
        size = f"{path.stat().st_size:>7d} B" if path.exists() else "       — "
        rows.append((present, kind, f"{prof:<28}  {size}"))

    print(f"{'have':<5}{'kind':<6}profile")
    print("-" * 60)
    for have, kind, rest in rows:
        print(f"{have:<5}{kind:<6}{rest}")
    missing = sum(1 for h, _, _ in rows if h.strip() == "--")
    print(f"\n{len(rows) - missing}/{len(rows)} have welcome_name.wav")
    return 0


def _extract_one(profile: str, entry: dict[str, Any], force: bool, dry_run: bool) -> bool:
    """Run the extraction pipeline for one profile. Returns True on success."""
    out_path = welcome_name_path(profile)
    if out_path.exists() and not force:
        print(f"[{profile}] skip — {out_path.name} already exists (use --force to overwrite)")
        return True

    url = (entry.get("url") or "").strip()
    phrases: list[str] = list(entry.get("phrases") or [])
    if not url:
        print(f"[{profile}] skip — no url set in welcome_sources.toml")
        return False
    if not phrases:
        print(f"[{profile}] skip — no phrases configured")
        return False

    start_w = parse_hms(entry.get("start"))
    end_w = parse_hms(entry.get("end"))

    print(f"[{profile}] extracting")
    if dry_run:
        print(f"  dry-run: url={url} phrases={phrases} window={start_w}..{end_w}")
        return True

    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"welcome_{profile}_") as tmp:
        tmpdir = Path(tmp)
        try:
            src = yt_dlp_audio(url, tmpdir)
        except subprocess.CalledProcessError as exc:
            print(f"  yt-dlp failed: {exc.stderr.strip()[:400]}")
            return False

        analysed = src
        offset = 0.0
        if start_w is not None or end_w is not None:
            analysed = tmpdir / "window.wav"
            ffmpeg_clip_window(src, analysed, start_w, end_w)
            offset = start_w or 0.0

        try:
            words = transcribe_words(analysed)
        except RuntimeError as exc:
            print(f"  {exc}")
            return False
        if not words:
            print("  transcription returned no words")
            return False

        match: tuple[float, float] | None = None
        chosen_phrase: str | None = None
        for phrase in phrases:
            match = find_phrase(words, phrase)
            if match is not None:
                chosen_phrase = phrase
                break
        if match is None or chosen_phrase is None:
            words_preview = " ".join(w.text for w in words[:40]).strip()
            print(f"  no phrase matched. Tried: {phrases}")
            print(f"  first 40 words: {words_preview!r}")
            return False

        m_start, m_end = match
        cut_start = max(0.0, m_start - PAD_BEFORE)
        cut_end = m_end + PAD_AFTER
        duration = max(0.05, cut_end - cut_start)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_cut_and_normalise(analysed, out_path, cut_start, duration)

        src_ts = offset + m_start
        out_dur = ffprobe_duration(out_path)
        size = out_path.stat().st_size
        digest = sha256_file(out_path)
        print(f"  phrase: {chosen_phrase!r}")
        print(f"  source: {fmt_ts(src_ts)} (window-relative {fmt_ts(m_start)})")
        print(f"  output: {out_path.relative_to(REPO_ROOT)}  {out_dur:.2f}s  {size} bytes")
        print(f"  sha256: {digest}")
    return True


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract one or all profiles per welcome_sources.toml."""
    check_external_tools()
    sources = load_sources()

    if args.all:
        targets = sorted(sources.keys())
    else:
        if not args.profile:
            die("extract requires <profile> or --all")
        if args.profile not in sources:
            die(f"profile {args.profile!r} not in {SOURCES_TOML.name}")
        targets = [args.profile]

    ok = 0
    for prof in targets:
        if _extract_one(prof, sources[prof], force=args.force, dry_run=args.dry_run):
            ok += 1
    print(f"\nextracted/skipped-ok: {ok}/{len(targets)}")
    return 0 if ok == len(targets) else 1


def cmd_generate_narrator(args: argparse.Namespace) -> int:
    """Generate the four shared narrator WAVs under assets/welcome/."""
    WELCOME_DIR.mkdir(parents=True, exist_ok=True)
    for key, text in NARRATOR_PROMPTS.items():
        out = WELCOME_DIR / f"{key}.wav"
        if out.exists() and not args.force:
            print(f"[narrator] skip {key}.wav (use --force to overwrite)")
            continue
        print(f"[narrator] {key}.wav  voice={NARRATOR_VOICE}  text={text!r}")
        if args.dry_run:
            continue
        try:
            pcm = gemini_tts(text, NARRATOR_VOICE)
        except DailyQuotaExhausted as exc:
            print(f"  {exc}. Stopping; rerun once quota resets.")
            return 1
        write_wav_pcm16(out, pcm)
        print(f"  -> {out.relative_to(REPO_ROOT)}  {out.stat().st_size} bytes  sha256={sha256_file(out)}")
    return 0


def _profile_voice(profile: str) -> str:
    """Return the voice for ``profile`` from voice.txt, or NARRATOR_VOICE."""
    vfile = PROFILES_DIR / profile / "voice.txt"
    if vfile.exists():
        v = vfile.read_text(encoding="utf-8").strip()
        if v:
            return v
    return NARRATOR_VOICE


def cmd_generate_character(args: argparse.Namespace) -> int:
    """Gemini-TTS the display name for original-character profile(s)."""
    sources = load_sources()
    real = set(sources.keys())
    all_profiles = list_profiles()
    characters = [p for p in all_profiles if p not in real]

    if args.all:
        targets = characters
    else:
        if not args.profile:
            die("generate-character requires <profile> or --all")
        if args.profile not in characters:
            die(f"{args.profile!r} is not an original-character profile (or doesn't exist)")
        targets = [args.profile]

    ok = 0
    for prof in targets:
        out = welcome_name_path(prof)
        if out.exists() and not args.force:
            print(f"[{prof}] skip (use --force to overwrite)")
            ok += 1
            continue
        name = display_name(prof)
        voice = _profile_voice(prof)
        print(f"[{prof}] voice={voice}  text={name!r}")
        if args.dry_run:
            ok += 1
            continue
        try:
            pcm = gemini_tts(name, voice)
        except DailyQuotaExhausted as exc:
            print(f"  {exc}. Stopping; rerun once quota resets.")
            break
        except Exception as exc:
            print(f"  TTS failed: {exc}")
            continue
        write_wav_pcm16(out, pcm)
        print(f"  -> {out.relative_to(REPO_ROOT)}  {out.stat().st_size} bytes  sha256={sha256_file(out)}")
        ok += 1
    print(f"\ngenerated/skipped-ok: {ok}/{len(targets)}")
    return 0 if ok == len(targets) else 1


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="show what would happen")
    common.add_argument("--force", action="store_true", help="overwrite existing outputs")

    s = sub.add_parser("status", help="report welcome_name.wav state per profile")
    s.set_defaults(func=cmd_status)

    e = sub.add_parser("extract", parents=[common], help="extract YouTube name clips")
    e.add_argument("profile", nargs="?", help="profile name (omit with --all)")
    e.add_argument("--all", action="store_true", help="extract every profile in TOML")
    e.set_defaults(func=cmd_extract)

    n = sub.add_parser("generate-narrator", parents=[common], help="Gemini-TTS shared narrator WAVs")
    n.set_defaults(func=cmd_generate_narrator)

    g = sub.add_parser("generate-character", parents=[common], help="Gemini-TTS original-character name clips")
    g.add_argument("profile", nargs="?", help="profile name (omit with --all)")
    g.add_argument("--all", action="store_true", help="generate for every character profile")
    g.set_defaults(func=cmd_generate_character)

    return p


def _final_summary() -> None:
    """Remind the user to review and selectively commit WAV outputs."""
    print(
        "\nReminder: WAV outputs are gitignored. Listen to each result and "
        "`git add -f` only the ones you want to ship."
    )


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    rc = int(args.func(args))
    _final_summary()
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # ctranslate2's destructor segfaults on some Windows configs at interpreter
    # shutdown, masking our real exit code. Skip the cleanup with os._exit.
    os._exit(rc)
