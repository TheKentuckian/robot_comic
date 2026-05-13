# Voice cloning (ElevenLabs IVC)

This project uses ElevenLabs **Instant Voice Cloning (IVC)** for per-persona
cloned voices. The Don Rickles IVC voice (`r3TOduczSXN66XmgF5em`) is the
reference example throughout this guide.

At runtime, the cloned voice ID is resolved by `_resolve_voice_id()` in
`src/robot_comic/elevenlabs_tts.py`: a `voice_id=<id>` line in
`profiles/<profile>/elevenlabs.local.txt` (gitignored) takes precedence over
`elevenlabs.txt` (committed), which in turn takes precedence over the
named-voice catalog. Swapping profiles swaps voices automatically.

---

## Why not PVC?

See [Don't use PVC in this project](#dont-use-pvc-in-this-project) below.
Short answer: ElevenLabs PVC requires a biometric captcha from the
rights-holder — archival celebrity audio cannot pass it, and the ToS prohibits
cloning public figures without permission regardless.

---

## Prerequisites

- An ElevenLabs account (any tier — IVC is available on all tiers).
- `ELEVENLABS_API_KEY` exported in your shell.
- `ffmpeg` on PATH (needed by `scan_segments.py` for clip extraction).
- Python `requests` installed (`pip install requests` or it comes via the
  project venv).

---

## Step-by-step: IVC for a new persona

### 1. Gather source audio

Put one or more long-form clean recordings in `profiles/<name>/voice_prep/`.
Aim for a total of a few minutes of clear speech, minimal background noise, no
overlapping music or other speakers.

```
profiles/don_rickles/voice_prep/
    buythistape1993_full.wav       # raw archival recording
    cavett1972_full.wav
```

No committing these files — they are large binaries. The `voice_prep/`
directory is gitignored except for the extraction scripts.

### 2. Extract clean 30-second clips

`profiles/don_rickles/voice_prep/scan_segments.py` scores windows of a
recording by amplitude consistency and mean level, then extracts the top picks
as `clip_<label>_<mmss>_<window>s.wav` files.

```bash
# Scan and extract the top 10 non-overlapping 30s windows from a recording.
python profiles/don_rickles/voice_prep/scan_segments.py \
    profiles/don_rickles/voice_prep/buythistape1993_full.wav \
    --window 30 --step 15 --top 10 \
    --non-overlap --extract \
    --out-dir profiles/don_rickles/voice_prep \
    --label buythistape1993
```

Output files will be named like `clip_buythistape1993_0130_30s.wav`
(timestamp 01:30, 30-second window).

Repeat for each source recording. Aim for ~10 clips covering different
delivery styles (enthusiastic, conversational, punchline). The combined upload
must stay under ~11 MB — see `--max-bytes` below.

**Note on ffmpeg paths**: `scan_segments.py` currently hard-codes the
Chocolatey ffmpeg path (`C:\ProgramData\chocolatey\bin\ffmpeg.exe`). If ffmpeg
is on PATH via another install, edit `FFMPEG` / `FFPROBE` at the top of the
script.

### 3. Create the IVC voice

```bash
export ELEVENLABS_API_KEY=sk_...

# Upload all clip_*.wav files in voice_prep/ at once
python scripts/elevenlabs_clone.py create-ivc \
    --profile don_rickles "Don Rickles" \
    --all-clips

# Or pass specific clips
python scripts/elevenlabs_clone.py create-ivc \
    --profile don_rickles "Don Rickles" \
    clip_buythistape1993_0130_30s.wav \
    clip_cavett1972_0220_30s.wav
```

The script:

1. Globs (or resolves) the clip files.
2. Checks the combined size against `--max-bytes` (default 11 MB) and aborts
   if exceeded — better than letting ElevenLabs reject the multipart request
   mid-upload.
3. POSTs to `https://api.elevenlabs.io/v1/voices/add` (multipart).
4. Prints the returned `voice_id`.
5. Writes `voice_id=<id>` into `profiles/<profile>/elevenlabs.local.txt`
   (gitignored) by default. Pass `--no-write-profile-override` to skip.

The Don Rickles run produced `voice_id=r3TOduczSXN66XmgF5em`.

### 4. Verify the voice is active

Start the app with the don_rickles profile selected. The startup log should
show:

```
[ElevenLabs] resolved voice_id=r3TOduczSXN66XmgF5em (from local override)
```

If you see a named-catalog voice instead, check that
`profiles/don_rickles/elevenlabs.local.txt` exists and contains
`voice_id=r3TOduczSXN66XmgF5em`.

---

## The `elevenlabs.local.txt` override pattern

ElevenLabs voice IDs are credentials — they tie to your account and can be
used to rack up charges. They must not be committed to the public repo.

The loader in `src/robot_comic/elevenlabs_tts.py` checks for
`profiles/<profile>/elevenlabs.local.txt` first (gitignored) before falling
back to `elevenlabs.txt` (committed). The `.local.txt` file uses the same
key=value format:

```ini
# profiles/don_rickles/elevenlabs.local.txt  (gitignored — never commit)
voice_id=r3TOduczSXN66XmgF5em
```

`scripts/elevenlabs_clone.py create-ivc` writes this file automatically after
a successful clone. You can also write it by hand or copy it from a secure
credential store.

The `.gitignore` pattern `elevenlabs.local.txt` was added in issue #126.

---

## Cloning a new persona

```bash
mkdir -p profiles/new_persona/voice_prep

# Drop source audio into voice_prep/
# ...

# Extract 30-second clips
python profiles/don_rickles/voice_prep/scan_segments.py \
    profiles/new_persona/voice_prep/source.wav \
    --window 30 --step 15 --top 10 --non-overlap --extract \
    --out-dir profiles/new_persona/voice_prep \
    --label source

# Create the IVC voice
python scripts/elevenlabs_clone.py create-ivc \
    --profile new_persona "Persona Name" \
    --all-clips
```

The runtime picks up `profiles/new_persona/elevenlabs.local.txt`
automatically when that profile is selected.

---

## Don't use PVC in this project

ElevenLabs **Professional Voice Cloning (PVC)** was attempted for Don Rickles
and blocked at the eligibility gate. Here is why it cannot work here and
should not be attempted for future personas:

**The captcha gate.** Before ElevenLabs will train a PVC voice, the
rights-holder must visit a web page, read a randomly-generated phrase aloud,
and ElevenLabs biometric-matches that live recording against the uploaded
samples. Archival or deceased-person audio cannot satisfy this — there is no
one to read the phrase.

**The ToS.** ElevenLabs' Terms of Service explicitly prohibit cloning the
voice of a public figure without their explicit written permission, regardless
of the audio source.

**Slot economics.** PVC slots are limited and slow (1–4 hours to train per
voice). IVC is instant and available on all tiers.

**The `create` / `train` / `status` / `list` subcommands** in
`scripts/elevenlabs_clone.py` are kept for reference with DEPRECATED labels.
Do not use them for new personas.

### Tricky PVC behaviors to be aware of

If you ever re-examine the PVC code path for any reason:

- **`/train` returns `{"status": "ok"}` even when `is_allowed_to_fine_tune`
  is `False`.**  Do not interpret the 200 response as a green light. Always
  re-GET the voice (`GET /v1/voices/pvc/<id>`) and check the actual
  `is_allowed_to_fine_tune` field before waiting for training to complete.

- **PVC voice deletion via API returns 403 `pro_voice_deletion_forbidden`.**
  ElevenLabs does not allow deleting a PVC voice slot via the API. The
  workaround is to delete every sample (`DELETE /v1/voices/<id>/samples/<sid>`)
  and rename the voice — the slot becomes reusable without being deleted.

---

## Useful surprises about the ElevenLabs API

- **Server-side transcoding.** ElevenLabs transcodes uploaded WAV files to MP3
  internally. If you retrieve the sample list after uploading, the filenames
  will have `.mp3` extensions even though you sent `.wav`. This is cosmetic —
  the audio data is intact.

- **11 MB upload limit.** The practical limit for the combined multipart body
  of a `POST /v1/voices/add` request is approximately 11 MB. The server returns
  a generic 4xx or 5xx if you exceed it without a clear message. The script
  enforces this pre-flight via `--max-bytes`.

- **IVC voice quality scales with clip diversity.** Ten varied 30-second clips
  (different energy levels, sentence lengths, pauses) outperform thirty uniform
  clips from the same passage.
