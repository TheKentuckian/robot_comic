#!/usr/bin/env python3
"""
ElevenLabs Professional Voice Cloning (PVC) helper.

Uploads Don Rickles audio samples to ElevenLabs PVC, monitors training,
and returns the voice ID to use in the don_rickles profile's elevenlabs.txt.

Usage:
    # 1. Set API key
    export ELEVENLABS_API_KEY=sk_...

    # 2. Create a new PVC voice with sample files
    python pvc_clone.py create "Don Rickles" --description "1993 stand-up" \\
        buythistape1993.wav candidate_standup_*.wav

    # 3. Start training (after upload completes)
    python pvc_clone.py train <voice_id>

    # 4. Check training status
    python pvc_clone.py status <voice_id>

    # 5. List existing PVC voices
    python pvc_clone.py list

Requirements:
    - ElevenLabs Creator+ subscription (PVC tier)
    - 30 min to 3 hr of clean audio for best results
    - Samples should be: clear speech, minimal background noise, no music
"""
import os
import sys
import argparse
import requests
from pathlib import Path

API_BASE = "https://api.elevenlabs.io/v1"


def get_api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("ERROR: ELEVENLABS_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    return key


def headers() -> dict:
    return {"xi-api-key": get_api_key()}


def cmd_create(args: argparse.Namespace) -> None:
    """Create a new PVC voice and upload sample files."""
    wav_files = [Path(p) for p in args.files]
    missing = [str(p) for p in wav_files if not p.exists()]
    if missing:
        print(f"ERROR: Files not found: {missing}", file=sys.stderr)
        sys.exit(1)

    total_mb = sum(p.stat().st_size for p in wav_files) / (1024 * 1024)
    print(f"Creating PVC voice '{args.name}' with {len(wav_files)} samples ({total_mb:.1f} MB total)...")

    # Create the PVC voice
    create_resp = requests.post(
        f"{API_BASE}/voices/pvc",
        headers=headers(),
        json={
            "name": args.name,
            "description": args.description or f"Cloned voice: {args.name}",
            "language": "en",
        },
    )
    if create_resp.status_code != 200:
        print(f"ERROR creating voice: {create_resp.status_code} {create_resp.text}", file=sys.stderr)
        sys.exit(1)

    voice_id = create_resp.json().get("voice_id")
    print(f"✓ Created PVC voice. voice_id={voice_id}")

    # Upload samples
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
    print(f"  python pvc_clone.py train {voice_id}")


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
    print(f"  Check status: python pvc_clone.py status {args.voice_id}")


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
        print(f"\n✓ Ready to use! Update profiles/don_rickles/elevenlabs.txt:")
        print(f"  voice_id={args.voice_id}")


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
    parser = argparse.ArgumentParser(description="ElevenLabs PVC voice cloning helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new PVC voice and upload samples")
    p_create.add_argument("name", help="Voice name (e.g., 'Don Rickles')")
    p_create.add_argument("files", nargs="+", help="WAV files to upload as samples")
    p_create.add_argument("--description", default=None, help="Voice description")
    p_create.set_defaults(func=cmd_create)

    p_train = sub.add_parser("train", help="Start training a PVC voice")
    p_train.add_argument("voice_id", help="PVC voice ID to train")
    p_train.set_defaults(func=cmd_train)

    p_status = sub.add_parser("status", help="Check PVC voice training status")
    p_status.add_argument("voice_id", help="PVC voice ID")
    p_status.set_defaults(func=cmd_status)

    p_list = sub.add_parser("list", help="List all PVC voices")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
