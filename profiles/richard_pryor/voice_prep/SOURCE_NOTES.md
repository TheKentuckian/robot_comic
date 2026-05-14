# Richard Pryor — IVC Voice Clone Source Notes

> **RESEARCH-ONLY DOCUMENT**
> These notes are for local development reference. No audio has been fetched
> or committed. Cloning the voice of a public figure without explicit written
> permission is prohibited by ElevenLabs Terms of Service. The cloned voice
> is never distributed publicly as part of this project.
> See `docs/voice-cloning.md` for the full policy context.

---

## Recommended sources

### Primary: Richard Pryor: Live in Concert (1979)

| Field | Detail |
|---|---|
| Title | Richard Pryor: Live in Concert |
| Year | 1979 (filmed December 10, 1978) |
| Runtime | ~78 minutes |
| Venue | Terrace Theater, Long Beach, California |
| Director | Jeff Margolis |
| YouTube URL | `https://www.youtube.com/watch?v=6jTjYAWlo0U` |
| Backup / playlist | `https://www.youtube.com/playlist?list=PLE26986C2A3AEDF70` |

**Audio quality assessment:** Shot with four fixed camera set-ups (close-up
spotlight, low-angle medium, two wide shots) using a single-source stage
microphone feed recorded in 35 mm. The close-mic boom/handheld stage mic
dominates the mix; audience noise is present but secondary — laughter erupts
between bits rather than over Pryor's voice. The 1979 recording predates
the arena-reverb era: the Terrace Theater is a mid-size venue and the room
acoustics are relatively dry compared to larger concert halls. Wide-band
YouTube encodes of this title (uploaded August 2025) show a clean vocal
centre-channel.

**Era / vocal identity:** This is Pryor at the undisputed peak of his
powers — mid-register, full dynamic range from whisper to full-voice shout,
with characteristic tight glottal articulation on consonants. The
1978-79 era represents the most stable single-identity period of his career.

**Why this over alternatives:** Live in Concert is universally cited as the
definitive recorded stand-up performance. The vocal consistency across the
78 minutes makes it ideal for IVC clip diversity. The full-length YouTube
upload (August 2025 rip) delivers a continuous, unbroken audio stream —
preferable to stitching segments from a multi-part playlist.

---

### Alternate / supplementary: Richard Pryor: Live on the Sunset Strip (1982)

| Field | Detail |
|---|---|
| Title | Richard Pryor: Live on the Sunset Strip |
| Year | 1982 (filmed December 1981 – January 1982) |
| Runtime | ~82 minutes |
| Venue | Hollywood Palladium, Los Angeles (main set); Circle Star Theater, San Carlos (select material) |
| YouTube URL | `https://www.youtube.com/watch?v=8wLwRs-r9u4` |
| Backup | `https://www.youtube.com/watch?v=Y6XaSp4ljrk` |

**Audio quality assessment:** Hollywood Palladium is a larger, livelier
room than the Terrace Theater. Audience laughter is louder and bleeds into
more pauses. The vocal centre-channel is still clean but the reverberation
tail is more pronounced. Recording engineers Biff Dawes, Jack Crymes, and
Bill Broms are credited (Wally Heider location recording), indicating a
professional multi-mic stage mix rather than a single boom.

**Era / vocal identity:** Three years on from Live in Concert; Pryor had
survived his 1980 freebasing accident. His voice is slightly lower and
huskier post-recovery — noticeably different register from 1979. **Do not
mix clips from these two specials in the same IVC submission** — combining
different vocal eras causes identity drift (analogous to issue #223 with
Rickles).

**When to use this source:** Use Sunset Strip only if the primary 1979
upload becomes unavailable or if you specifically want the post-1980 register.
Do not mix with 1979 clips.

---

## Extraction plan

### Step 1 — Download audio

```bash
# Primary source (Live in Concert 1979)
yt-dlp -x --audio-format wav \
    -o "richard_pryor_raw_live_in_concert_1979.wav" \
    "https://www.youtube.com/watch?v=6jTjYAWlo0U"

# Alternate (Sunset Strip 1982) — only if needed separately
yt-dlp -x --audio-format wav \
    -o "richard_pryor_raw_sunset_strip_1982.wav" \
    "https://www.youtube.com/watch?v=8wLwRs-r9u4"
```

Place the downloaded WAV in `profiles/richard_pryor/voice_prep/`. The
directory is gitignored for audio files — do not commit them.

### Step 2 — Extract clean clips

```bash
# Scan the first 60 minutes (laughter density is lower early in the set)
python profiles/don_rickles/voice_prep/scan_segments.py \
    profiles/richard_pryor/voice_prep/richard_pryor_raw_live_in_concert_1979.wav \
    --window 30 --step 15 --top 10 \
    --non-overlap --extract \
    --max-secs 3600 \
    --out-dir profiles/richard_pryor/voice_prep \
    --label live_in_concert_1979
```

**Target clip count:** 10 clips (ElevenLabs IVC sweet spot; see
`docs/voice-cloning.md` §"IVC voice quality scales with clip diversity").
Combined size must stay under ~11 MB — the script enforces this via
`--max-bytes` on the clone step.

Output files will be named like:
`clip_live_in_concert_1979_0130_30s.wav`

### Step 3 — Create IVC voice

Follow `docs/voice-cloning.md` §"Step-by-step: IVC for a new persona"
from step 3 onward:

```bash
python scripts/elevenlabs_clone.py create-ivc \
    --profile richard_pryor "Richard Pryor" \
    --all-clips
```

---

## Quality caveats

- **License:** The concert films are copyright Richard Pryor Enterprises /
  Columbia Pictures. YouTube uploads are third-party; access availability
  may change. ElevenLabs ToS prohibits cloning public figures' voices without
  their explicit permission. These source notes are for local research
  prototyping only — the cloned voice is not distributed publicly.

- **Audience laughter:** Live in Concert has concentrated audience eruptions
  between bits. `scan_segments.py` scores by amplitude consistency — windows
  with a sustained laughter tail will rank poorly and be deprioritized
  automatically. Prefer clips where the score shows low spread (compact
  dynamic range = Pryor talking, not the audience laughing).

- **Multi-speaker caution:** Sunset Strip opens with a brief introduction by
  Jim Brown; skip the first ~3 minutes to avoid the two-speaker zone.

- **Era mixing:** Do not combine clips from 1979 and 1982. The post-freebasing
  vocal register shift is large enough to confuse IVC identity anchoring.

---

## Recommendations for the user

1. Watch the first 10 minutes of Live in Concert before downloading — if
   the YouTube encode sounds muddy or compressed, try the Plex free stream
   (`https://watch.plex.tv/movie/richard-pryor-live-in-concert`) or the
   Netflix version for a cleaner source.

2. Use the **earlier part of the special** (first 20–40 minutes) for
   extraction. Audiences are less warmed up, laughter eruptions are shorter,
   and Pryor is in his most conversational register — less shouting — which
   gives IVC a wider vocal range sample per clip.

3. If `scan_segments.py` returns fewer than 10 non-overlapping picks with
   good scores, widen to `--step 10` and re-run without `--max-secs` to
   scan the full 78 minutes.
