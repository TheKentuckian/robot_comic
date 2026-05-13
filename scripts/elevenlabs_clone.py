#!/usr/bin/env python3
"""
ElevenLabs voice cloning helper — multi-persona.

Supports Instant Voice Cloning (IVC, the default workflow) and the older
Professional Voice Cloning (PVC) commands (kept for reference; see the
DEPRECATED notice in each subcommand's help text).

IVC usage (recommended):
    export ELEVENLABS_API_KEY=sk_...

    # Create an IVC voice using specific clip files
    python scripts/elevenlabs_clone.py create-ivc \\
        --profile don_rickles "Don Rickles" \\
        clip_buythistape1993_0130_30s.wav clip_standup_0015_30s.wav

    # Or upload every clip_*.wav in the profile's voice_prep/ dir
    python scripts/elevenlabs_clone.py create-ivc \\
        --profile don_rickles "Don Rickles" --all-clips

PVC usage (DEPRECATED — see docs/voice-cloning.md for why PVC is not used):
    python scripts/elevenlabs_clone.py create --profile don_rickles "Don Rickles" --all
    python scripts/elevenlabs_clone.py train <voice_id>
    python scripts/elevenlabs_clone.py status <voice_id> --profile don_rickles --write-config
    python scripts/elevenlabs_clone.py list

Per-profile inputs / outputs:
    Source clips:       profiles/<profile>/voice_prep/clip_*.wav  (IVC)
    Source WAVs:        profiles/<profile>/voice_prep/*.wav        (PVC)
    Private override:   profiles/<profile>/elevenlabs.local.txt   (gitignored)
    Public config:      profiles/<profile>/elevenlabs.txt         (committed)

Requirements (IVC):
    - Any ElevenLabs subscription (IVC is available on all tiers)
    - ~10 clean clips of 30s each (<=11 MB total)
"""

import os
import re
import sys
import argparse
from pathlib import Path

import requests


# Reconfigure stdout to UTF-8 so any Unicode output does not crash on
# Windows terminals whose ANSI code page is cp1252 or similar.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_BASE = "https://api.elevenlabs.io/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Default server-side upload size limit observed empirically.
DEFAULT_MAX_BYTES = 11_000_000


def get_api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("ERROR: ELEVENLABS_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    return key


def headers() -> dict:
    return {"xi-api-key": get_api_key()}


def profile_voice_prep_dir(profile: str) -> Path:
    """Return profiles/<profile>/voice_prep/, raising if it doesn't exist."""
    p = REPO_ROOT / "profiles" / profile / "voice_prep"
    if not p.is_dir():
        print(f"ERROR: voice_prep dir not found: {p}", file=sys.stderr)
        sys.exit(1)
    return p


def profile_elevenlabs_config(profile: str) -> Path:
    """Return profiles/<profile>/elevenlabs.txt path (may not exist)."""
    return REPO_ROOT / "profiles" / profile / "elevenlabs.txt"


def profile_elevenlabs_local_config(profile: str) -> Path:
    """Return profiles/<profile>/elevenlabs.local.txt path (gitignored, may not exist)."""
    return REPO_ROOT / "profiles" / profile / "elevenlabs.local.txt"


def resolve_wav_files(profile: str, file_args: list[str], use_all: bool) -> list[Path]:
    voice_prep = profile_voice_prep_dir(profile)

    if use_all:
        files = sorted(voice_prep.glob("*.wav"))
        if not files:
            print(f"ERROR: no .wav files found in {voice_prep}", file=sys.stderr)
            sys.exit(1)
        return files

    if not file_args:
        print("ERROR: no WAV files given (use --all to upload every WAV in voice_prep/)", file=sys.stderr)
        sys.exit(1)

    resolved: list[Path] = []
    for arg in file_args:
        p = Path(arg)
        if not p.is_absolute():
            p = voice_prep / arg
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        resolved.append(p)
    return resolved


def resolve_clip_files(profile: str, file_args: list[str], use_all_clips: bool) -> list[Path]:
    """Resolve IVC clip files.

    With --all-clips: globs profiles/<profile>/voice_prep/clip_*.wav.
    Otherwise: resolves each positional arg relative to voice_prep/ if not absolute.
    """
    voice_prep = profile_voice_prep_dir(profile)

    if use_all_clips:
        files = sorted(voice_prep.glob("clip_*.wav"))
        if not files:
            print(f"ERROR: no clip_*.wav files found in {voice_prep}", file=sys.stderr)
            print(
                "  Run scan_segments.py with --extract to generate clips, or pass files explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        return files

    if not file_args:
        print(
            "ERROR: no files given. Pass clip paths or use --all-clips.",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved: list[Path] = []
    for arg in file_args:
        p = Path(arg)
        if not p.is_absolute():
            p = voice_prep / arg
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)
        resolved.append(p)
    return resolved


def write_voice_id_to_local_config(profile: str, voice_id: str) -> None:
    """Set/replace `voice_id=<id>` in profiles/<profile>/elevenlabs.local.txt (gitignored)."""
    path = profile_elevenlabs_local_config(profile)
    if not path.exists():
        path.write_text(f"voice_id={voice_id}\n", encoding="utf-8")
        print(f"[OK] Created {path} with voice_id={voice_id}")
        return

    content = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*#?\s*voice_id\s*=.*$", re.MULTILINE)
    new_line = f"voice_id={voice_id}"
    if pattern.search(content):
        content = pattern.sub(new_line, content, count=1)
    else:
        if not content.endswith("\n"):
            content += "\n"
        content += f"{new_line}\n"
    path.write_text(content, encoding="utf-8")
    print(f"[OK] Updated {path} with voice_id={voice_id}")


def write_voice_id_to_config(profile: str, voice_id: str) -> None:
    """Set/replace `voice_id=<id>` in profiles/<profile>/elevenlabs.txt."""
    path = profile_elevenlabs_config(profile)
    if not path.exists():
        path.write_text(f"voice_id={voice_id}\n", encoding="utf-8")
        print(f"[OK] Created {path} with voice_id={voice_id}")
        return

    content = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*#?\s*voice_id\s*=.*$", re.MULTILINE)
    new_line = f"voice_id={voice_id}"
    if pattern.search(content):
        content = pattern.sub(new_line, content, count=1)
    else:
        if not content.endswith("\n"):
            content += "\n"
        content += f"{new_line}\n"
    path.write_text(content, encoding="utf-8")
    print(f"[OK] Updated {path} with voice_id={voice_id}")


# ---------------------------------------------------------------------------
# IVC subcommand (primary / recommended workflow)
# ---------------------------------------------------------------------------


def cmd_create_ivc(args: argparse.Namespace) -> None:
    """Create an Instant Voice Clone via POST /v1/voices/add."""
    clip_files = resolve_clip_files(args.profile, args.files, args.all_clips)

    total_bytes = sum(p.stat().st_size for p in clip_files)
    total_mb = total_bytes / (1024 * 1024)

    if total_bytes > args.max_bytes:
        limit_mb = args.max_bytes / (1024 * 1024)
        print(
            f"ERROR: combined upload size {total_mb:.1f} MB exceeds --max-bytes limit {limit_mb:.1f} MB.",
            file=sys.stderr,
        )
        print(
            "  Reduce the number of clips or trim them, then retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    description = args.description or f"IVC clone for profile {args.profile}"
    print(
        f"Creating IVC voice '{args.name}' for profile '{args.profile}' "
        f"with {len(clip_files)} clip(s) ({total_mb:.1f} MB)..."
    )

    # Build multipart payload: name + description + one file field per clip.
    multipart_files: list[tuple] = []
    open_handles = []
    try:
        for clip in clip_files:
            fh = open(clip, "rb")  # noqa: SIM115 — kept open until request completes
            open_handles.append(fh)
            multipart_files.append(("files", (clip.name, fh, "audio/wav")))

        resp = requests.post(
            f"{API_BASE}/voices/add",
            headers=headers(),
            data={"name": args.name, "description": description},
            files=multipart_files,
        )
    finally:
        for fh in open_handles:
            fh.close()

    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)

    voice_id: str = resp.json().get("voice_id", "")
    if not voice_id:
        print(f"ERROR: no voice_id in response: {resp.text}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] IVC voice created. voice_id={voice_id}")

    if args.write_profile_override:
        write_voice_id_to_local_config(args.profile, voice_id)
    else:
        print("\nTo activate this voice for the profile, run:")
        print(f"  echo 'voice_id={voice_id}' >> profiles/{args.profile}/elevenlabs.local.txt")


# ---------------------------------------------------------------------------
# PVC subcommands (DEPRECATED — kept for reference only)
# ---------------------------------------------------------------------------


def cmd_create(args: argparse.Namespace) -> None:
    """Create a new PVC voice and upload sample files.

    DEPRECATED: PVC is not used in this project. See docs/voice-cloning.md.
    """
    wav_files = resolve_wav_files(args.profile, args.files, args.all)
    total_mb = sum(p.stat().st_size for p in wav_files) / (1024 * 1024)
    print(
        f"Creating PVC voice '{args.name}' for profile '{args.profile}' "
        f"with {len(wav_files)} samples ({total_mb:.1f} MB total)..."
    )

    create_resp = requests.post(
        f"{API_BASE}/voices/pvc",
        headers=headers(),
        json={
            "name": args.name,
            "description": args.description or f"Cloned voice for profile {args.profile}",
            "language": "en",
        },
    )
    if create_resp.status_code != 200:
        print(f"ERROR creating voice: {create_resp.status_code} {create_resp.text}", file=sys.stderr)
        sys.exit(1)

    voice_id = create_resp.json().get("voice_id")
    print(f"[OK] Created PVC voice. voice_id={voice_id}")

    for wav in wav_files:
        size_mb = wav.stat().st_size / (1024 * 1024)
        print(f"Uploading {wav.name} ({size_mb:.1f} MB)...")
        with open(wav, "rb") as fh:
            up_resp = requests.post(
                f"{API_BASE}/voices/pvc/{voice_id}/samples",
                headers=headers(),
                files={"files": (wav.name, fh, "audio/wav")},
            )
        if up_resp.status_code != 200:
            print(f"  ERROR uploading {wav.name}: {up_resp.status_code} {up_resp.text}", file=sys.stderr)
            continue
        print(f"  [OK] Uploaded {wav.name}")

    print("\nAll samples uploaded. Next step:")
    print(f"  python scripts/elevenlabs_clone.py train {voice_id}")


def cmd_train(args: argparse.Namespace) -> None:
    """Start training a PVC voice.

    DEPRECATED: PVC is not used in this project. See docs/voice-cloning.md.
    """
    resp = requests.post(
        f"{API_BASE}/voices/pvc/{args.voice_id}/train",
        headers=headers(),
        json={"model_id": "eleven_multilingual_v2"},
    )
    if resp.status_code != 200:
        print(f"ERROR starting training: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] Training started for voice_id={args.voice_id}")
    print("  Training typically takes 1-4 hours.")
    print("  NOTE: /train returns 200 even when is_allowed_to_fine_tune is False.")
    print("  Always re-GET the voice to verify actual eligibility before waiting.")
    print(f"  Check status: python scripts/elevenlabs_clone.py status {args.voice_id}")


def cmd_status(args: argparse.Namespace) -> None:
    """Check PVC voice training status.

    DEPRECATED: PVC is not used in this project. See docs/voice-cloning.md.
    """
    resp = requests.get(f"{API_BASE}/voices/pvc/{args.voice_id}", headers=headers())
    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    state = data.get("training", {}).get("state", "unknown")
    progress = data.get("training", {}).get("progress", 0)
    print(f"voice_id: {args.voice_id}")
    print(f"name:     {data.get('name', 'n/a')}")
    print(f"state:    {state}")
    print(f"progress: {progress}%")

    if state == "completed":
        if args.write_config and args.profile:
            write_voice_id_to_config(args.profile, args.voice_id)
        else:
            print("\n[OK] Ready to use! Update profiles/<profile>/elevenlabs.txt:")
            print(f"  voice_id={args.voice_id}")
            print("  (or re-run with --profile <name> --write-config to do it automatically)")


def cmd_list(args: argparse.Namespace) -> None:
    """List all PVC voices on the account.

    DEPRECATED: PVC is not used in this project. See docs/voice-cloning.md.
    """
    resp = requests.get(f"{API_BASE}/voices", headers=headers())
    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    voices = resp.json().get("voices", [])
    pvc_voices = [v for v in voices if v.get("category") == "professional"]
    if not pvc_voices:
        print("No PVC voices found.")
        return
    print(f"{'voice_id':<25} {'name':<30} {'state':<40}")
    print("-" * 95)
    for v in pvc_voices:
        # fine_tuning.state is a per-model dict on newer API responses
        # (e.g. {'eleven_multilingual_v2': 'fine_tuned'}), not a plain string.
        # Stringify before f-string formatting to avoid a crash on Windows.
        raw_state = v.get("fine_tuning", {}).get("state", "n/a")
        if isinstance(raw_state, dict):
            state: str = next(iter(raw_state.values()), "n/a")
        else:
            state = str(raw_state)
        print(f"{v['voice_id']:<25} {v['name']:<30} {state:<40}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ElevenLabs voice cloning helper (IVC + legacy PVC)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # create-ivc (IVC — recommended)
    # ------------------------------------------------------------------
    p_ivc = sub.add_parser(
        "create-ivc",
        help="Create an Instant Voice Clone (IVC) — recommended workflow",
    )
    p_ivc.add_argument(
        "--profile",
        required=True,
        help="Persona profile name (e.g. don_rickles). "
        "Clip files are resolved relative to profiles/<profile>/voice_prep/.",
    )
    p_ivc.add_argument("name", help="Voice name as it will appear in ElevenLabs (e.g. 'Don Rickles')")
    p_ivc.add_argument(
        "files",
        nargs="*",
        help="Clip WAV files to upload (paths relative to profiles/<profile>/voice_prep/ "
        "unless absolute). Ignored when --all-clips is given.",
    )
    p_ivc.add_argument(
        "--all-clips",
        action="store_true",
        help="Upload every clip_*.wav in the profile's voice_prep/ dir.",
    )
    p_ivc.add_argument("--description", default=None, help="Voice description (optional)")
    p_ivc.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Abort if combined file size exceeds this many bytes "
        f"(default: {DEFAULT_MAX_BYTES:,}, the empirically observed ElevenLabs limit).",
    )
    p_ivc.add_argument(
        "--write-profile-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the returned voice_id into profiles/<profile>/elevenlabs.local.txt "
        "(gitignored). Enabled by default; pass --no-write-profile-override to skip.",
    )
    p_ivc.set_defaults(func=cmd_create_ivc)

    # ------------------------------------------------------------------
    # create (PVC — DEPRECATED)
    # ------------------------------------------------------------------
    p_create = sub.add_parser(
        "create",
        help="DEPRECATED: Create a PVC voice. Use create-ivc instead.",
    )
    p_create.add_argument(
        "--profile",
        required=True,
        help="Persona profile name (e.g. don_rickles). Resolves WAV files relative to profiles/<profile>/voice_prep/.",
    )
    p_create.add_argument("name", help="Voice name (e.g., 'Don Rickles')")
    p_create.add_argument(
        "files",
        nargs="*",
        help="WAV files to upload as samples (paths relative to profiles/<profile>/voice_prep/ unless absolute).",
    )
    p_create.add_argument("--all", action="store_true", help="Upload every .wav in the profile's voice_prep/ dir.")
    p_create.add_argument("--description", default=None, help="Voice description")
    p_create.set_defaults(func=cmd_create)

    # ------------------------------------------------------------------
    # train (PVC — DEPRECATED)
    # ------------------------------------------------------------------
    p_train = sub.add_parser("train", help="DEPRECATED: Start training a PVC voice")
    p_train.add_argument("voice_id", help="PVC voice ID to train")
    p_train.set_defaults(func=cmd_train)

    # ------------------------------------------------------------------
    # status (PVC — DEPRECATED)
    # ------------------------------------------------------------------
    p_status = sub.add_parser("status", help="DEPRECATED: Check PVC voice training status")
    p_status.add_argument("voice_id", help="PVC voice ID")
    p_status.add_argument(
        "--profile",
        default=None,
        help="Profile to update with the voice_id when status==completed (requires --write-config).",
    )
    p_status.add_argument(
        "--write-config",
        action="store_true",
        help="When state==completed, write voice_id into profiles/<profile>/elevenlabs.txt.",
    )
    p_status.set_defaults(func=cmd_status)

    # ------------------------------------------------------------------
    # list (PVC — DEPRECATED)
    # ------------------------------------------------------------------
    p_list = sub.add_parser("list", help="DEPRECATED: List all PVC voices")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
