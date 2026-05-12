# Voice cloning (ElevenLabs PVC)

The ElevenLabs backend supports per-persona Professional Voice Clones. The
runtime resolves a cloned voice ID via `_resolve_voice_id()` in
`src/robot_comic/elevenlabs_tts.py`: a `voice_id=<id>` line in
`profiles/<profile>/elevenlabs.txt` takes precedence over the named-voice
catalog, so swapping personas swaps voices automatically.

## Workflow

All commands are run from the repo root with `ELEVENLABS_API_KEY` exported.
The script (`scripts/elevenlabs_pvc_clone.py`) is persona-aware: WAV
filenames are resolved relative to `profiles/<profile>/voice_prep/` unless
they are absolute paths.

### 1. Prep source audio

Drop clean WAVs into `profiles/<profile>/voice_prep/`. Aim for 30 min–3 hr
of clear speech, minimal background noise/music. Use the existing
`scan_segments.py` / `find_clean_segment.ps1` helpers (in
`profiles/don_rickles/voice_prep/`) as templates for scoring candidate
segments by amplitude consistency.

### 2. Create the clone and upload samples

```bash
python scripts/elevenlabs_pvc_clone.py create \
    --profile don_rickles "Don Rickles" \
    buythistape1993.wav candidate_standup_0015.wav
```

Or upload every WAV in the profile's `voice_prep/` directory:

```bash
python scripts/elevenlabs_pvc_clone.py create \
    --profile don_rickles "Don Rickles" --all
```

The command prints the new `voice_id` — note it for the next step.

### 3. Start training

```bash
python scripts/elevenlabs_pvc_clone.py train <voice_id>
```

Training takes 1–4 hours on ElevenLabs' side.

### 4. Poll status and write the voice ID back

```bash
python scripts/elevenlabs_pvc_clone.py status <voice_id> \
    --profile don_rickles --write-config
```

When `state == completed`, the script writes (or replaces) the
`voice_id=` line in `profiles/don_rickles/elevenlabs.txt`. The next time
the app starts with the don_rickles profile selected, the cloned voice
will be used.

### 5. List existing clones

```bash
python scripts/elevenlabs_pvc_clone.py list
```

## Cloning a new persona

The same script works for any profile:

```bash
mkdir profiles/new_persona/voice_prep
cp /path/to/source/*.wav profiles/new_persona/voice_prep/
python scripts/elevenlabs_pvc_clone.py create --profile new_persona "Name" --all
python scripts/elevenlabs_pvc_clone.py train <voice_id>
python scripts/elevenlabs_pvc_clone.py status <voice_id> \
    --profile new_persona --write-config
```

The runtime picks up `profiles/new_persona/elevenlabs.txt` automatically
when that profile is active.

## Notes

- PVC requires an ElevenLabs Creator+ subscription.
- One clone per persona — no cross-persona sharing of cloned voices.
- The `voice` field in `elevenlabs.txt` is informational once `voice_id`
  is set; the cloned voice ID is what's actually sent to the API.
