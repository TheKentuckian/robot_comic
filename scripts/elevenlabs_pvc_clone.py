#!/usr/bin/env python3
"""
ElevenLabs Professional Voice Cloning (PVC) helper — multi-persona.

Drives the ElevenLabs PVC API end-to-end for any profile. WAV file
arguments are resolved relative to `profiles/<profile>/voice_prep/`
when the path is not absolute, so you can stay in the repo root and
type short filenames.

Usage:
    export ELEVENLABS_API_KEY=sk_...

    # Create a PVC voice for don_rickles using its voice_prep samples
    python scripts/elevenlabs_pvc_clone.py create \\
        --profile don_rickles "Don Rickles" \\
        buythistape1993.wav candidate_standup_0015.wav

    # Or upload every WAV in the profile's voice_prep dir
    python scripts/elevenlabs_pvc_clone.py create \\
        --profile don_rickles "Don Rickles" --all

    # Start training (after upload completes)
    python scripts/elevenlabs_pvc_clone.py train <voice_id>

    # Poll status; once "completed" you can auto-write voice_id into
    # profiles/<profile>/elevenlabs.txt
    python scripts/elevenlabs_pvc_clone.py status <voice_id> \\
        --profile don_rickles --write-config

    # List existing PVC voices on the account
    python scripts/elevenlabs_pvc_clone.py list

Per-profile inputs / outputs:
    Source WAVs:  profiles/<profile>/voice_prep/*.wav
    Config sink:  profiles/<profile>/elevenlabs.txt  (voice_id= line)

Requirements:
    - ElevenLabs Creator+ subscription (PVC tier)
    - 30 min to 3 hr of clean audio per persona for best results
    - Samples should be: clear speech, minimal background noise, no music
"""
import os
import sys
import argparse
import re
from pathlib import Path

import requests

API_BASE = "https://api.elevenlabs.io/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def cmd_create(args: argparse.Namespace) -> None:
    """Create a new PVC voice and upload sample files."""
    wav_files = resolve_wav_files(args.profile, args.files, args.all)
    total_mb = sum(p.stat().st_size for p in wav_files) / (1024 * 1024)
    print(f"Creating PVC voice '{args.name}' for profile '{args.profile}' "
          f"with {len(wav_files)} samples ({total_mb:.1f} MB total)...")

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
    print(f"✓ Created PVC voice. voice_id={voice_id}")

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
        print(f"  ✓ Uploaded {wav.name}")

    print(f"\nAll samples uploaded. Next step:")
    print(f"  python scripts/elevenlabs_pvc_clone.py train {voice_id}")


def cmd_train(args: argparse.Namespace) -> None:
    """Start training a PVC voice."""
    resp = requests.post(
        f"{API_BASE}/voices/pvc/{args.voice_id}/train",
        headers=headers(),
        json={"model_id": "eleven_multilingual_v2"},
    )
    if resp.status_code != 200:
        print(f"ERROR starting training: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    print(f"✓ Training started for voice_id={args.voice_id}")
    print(f"  Training typically takes 1-4 hours.")
    print(f"  Check status: python scripts/elevenlabs_pvc_clone.py status {args.voice_id}")


def write_voice_id_to_config(profile: str, voice_id: str) -> None:
    """Set/replace `voice_id=<id>` in profiles/<profile>/elevenlabs.txt."""
    path = profile_elevenlabs_config(profile)
    if not path.exists():
        # Create a minimal config if missing
        path.write_text(f"voice_id={voice_id}\n", encoding="utf-8")
        print(f"✓ Created {path} with voice_id={voice_id}")
        return

    content = path.read_text(encoding="utf-8")
    # Replace existing voice_id (commented or not), or append.
    pattern = re.compile(r"^\s*#?\s*voice_id\s*=.*$", re.MULTILINE)
    new_line = f"voice_id={voice_id}"
    if pattern.search(content):
        content = pattern.sub(new_line, content, count=1)
    else:
        if not content.endswith("\n"):
            content += "\n"
        content += f"{new_line}\n"
    path.write_text(content, encoding="utf-8")
    print(f"✓ Updated {path} with voice_id={voice_id}")


def cmd_status(args: argparse.Namespace) -> None:
    """Check PVC voice training status."""
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
            print(f"\n✓ Ready to use! Update profiles/<profile>/elevenlabs.txt:")
            print(f"  voice_id={args.voice_id}")
            print(f"  (or re-run with --profile <name> --write-config to do it automatically)")


def cmd_list(args: argparse.Namespace) -> None:
    """List all PVC voices."""
    resp = requests.get(f"{API_BASE}/voices", headers=headers())
    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    voices = resp.json().get("voices", [])
    pvc_voices = [v for v in voices if v.get("category") == "professional"]
    if not pvc_voices:
        print("No PVC voices found.")
        return
    print(f"{'voice_id':<25} {'name':<30} {'state':<15}")
    print("-" * 70)
    for v in pvc_voices:
        state = v.get("fine_tuning", {}).get("state", "n/a")
        print(f"{v['voice_id']:<25} {v['name']:<30} {state:<15}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ElevenLabs PVC voice cloning helper (multi-persona)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new PVC voice and upload samples")
    p_create.add_argument("--profile", required=True,
                          help="Persona profile name (e.g. don_rickles). "
                               "Resolves WAV files relative to profiles/<profile>/voice_prep/.")
    p_create.add_argument("name", help="Voice name (e.g., 'Don Rickles')")
    p_create.add_argument("files", nargs="*",
                          help="WAV files to upload as samples (paths relative to "
                               "profiles/<profile>/voice_prep/ unless absolute).")
    p_create.add_argument("--all", action="store_true",
                          help="Upload every .wav in the profile's voice_prep/ dir.")
    p_create.add_argument("--description", default=None, help="Voice description")
    p_create.set_defaults(func=cmd_create)

    p_train = sub.add_parser("train", help="Start training a PVC voice")
    p_train.add_argument("voice_id", help="PVC voice ID to train")
    p_train.set_defaults(func=cmd_train)

    p_status = sub.add_parser("status", help="Check PVC voice training status")
    p_status.add_argument("voice_id", help="PVC voice ID")
    p_status.add_argument("--profile", default=None,
                          help="Profile to update with the voice_id when status==completed "
                               "(requires --write-config).")
    p_status.add_argument("--write-config", action="store_true",
                          help="When state==completed, write voice_id into "
                               "profiles/<profile>/elevenlabs.txt.")
    p_status.set_defaults(func=cmd_status)

    p_list = sub.add_parser("list", help="List all PVC voices")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
