# Dave Chappelle — IVC Voice Clone Source Notes

> **RESEARCH-ONLY DOCUMENT**
> These notes are for local development reference. No audio has been fetched
> or committed. Cloning the voice of a public figure without explicit written
> permission is prohibited by ElevenLabs Terms of Service. The cloned voice
> is never distributed publicly as part of this project.
> See `docs/voice-cloning.md` for the full policy context.

---

## Recommended sources

### Primary: Dave Chappelle: Sticks & Stones (2019)

| Field | Detail |
|---|---|
| Title | Dave Chappelle: Sticks & Stones |
| Year | 2019 (filmed June 13–16, 2019) |
| Runtime | ~66 minutes |
| Venue | The Tabernacle, Downtown Atlanta, Georgia |
| Director | Stan Lathan |
| Netflix URL | `https://www.netflix.com/title/81140577` |
| YouTube clips (official) | `https://www.youtube.com/watch?v=wZXoErL2124` (Netflix Is A Joke channel) |

**Audio quality assessment:** Professional Netflix-grade production: 10 Sony
4300 cameras (three jibs, two handhelds, four peds, one Steadicam) with a
dedicated multi-mic stage audio rig. Post-mixing by Levels Audio (engineers
Brian Riordan and Connor Moore) — this is the cleanest stand-up audio of
any Chappelle special. The Tabernacle is a mid-capacity venue (~2,600 seats);
room reverb is controlled. The special was nominated for an Emmy for Outstanding
Sound Mixing (Variety Special). Vocal centre-channel is exceptionally isolated
from audience noise at high-quality stream resolutions.

**Era / vocal identity:** This is Chappelle at 45-46 years old — his voice
has fully settled into its mature register: a low-to-mid baritone with a
characteristic Southern/Mid-Atlantic relaxed drawl, longer pauses, and
minimal vocal strain. All three 2019–2023 Netflix specials share this same
register. This is the canonical "current Dave Chappelle" voice.

**Why this over alternatives:**
- Sticks & Stones has the most professionally produced audio of the modern
  specials. The Levels Audio post-mix means a cleaner vocal separation than
  The Closer or The Dreamer where audience ambience is mixed more prominently.
- The Tabernacle's controlled acoustics are better than the Lincoln Theatre
  (The Dreamer, 2023) for IVC purposes — less room reverb bleeding into pauses.
- The special is 66 minutes, giving plenty of material to select 10
  non-overlapping clips.

**Important:** The full special is exclusive to Netflix. Only official clip
excerpts exist on YouTube (Netflix Is A Joke channel). Download via yt-dlp
will require a Netflix cookie session (`--cookies-from-browser firefox` or
equivalent). See yt-dlp docs on authenticated streams.

---

### Alternate: Dave Chappelle: The Closer (2021)

| Field | Detail |
|---|---|
| Title | Dave Chappelle: The Closer |
| Year | 2021 |
| Runtime | ~72 minutes |
| Venue | Undisclosed (standard Netflix production) |
| Director | Stan Lathan |
| Netflix URL | `https://www.netflix.com/title/81228510` |

**Audio quality assessment:** Same Stan Lathan / Netflix production pipeline
as Sticks & Stones. Comparable audio quality. Vocal register identical to
2019 — safe to mix clips from Sticks & Stones and The Closer in the same
IVC submission since they are the same vocal era.

**When to use this source:** If 10 clean clips from Sticks & Stones are
insufficient (e.g. too many picks rejected due to audience eruptions), add
supplementary clips from The Closer. The eras match closely enough — do not
mix with pre-2016 Chappelle material.

**Do NOT use as sole source:** The Closer attracted significant controversy
during its 2021 release cycle; the Netflix Is A Joke channel clip set for
this special is limited, making it harder to preview audio quality before
committing to a download.

---

### Not recommended: Dave Chappelle: What's in a Name? (2022)

Runtime only ~40 minutes; this is a speech at Duke Ellington School of the
Arts, not a stand-up comedy performance. The audio environment is a school
auditorium (hard surfaces, variable mic distance), and the content lacks
the comedic pacing variety needed for IVC clip diversity. Skip.

---

## Extraction plan

### Step 1 — Download audio

```bash
# Sticks & Stones (2019) — requires Netflix authentication
# Option A: browser cookie extraction (yt-dlp native)
yt-dlp -x --audio-format wav \
    --cookies-from-browser firefox \
    -o "dave_chappelle_raw_sticks_and_stones_2019.wav" \
    "https://www.netflix.com/title/81140577"

# Option B: if cookie extraction fails, export cookies.txt manually
# from a browser extension (e.g. "Get cookies.txt LOCALLY") and use:
yt-dlp -x --audio-format wav \
    --cookies cookies.txt \
    -o "dave_chappelle_raw_sticks_and_stones_2019.wav" \
    "https://www.netflix.com/title/81140577"
```

Place the downloaded WAV in `profiles/dave_chappelle/voice_prep/`. The
directory is gitignored for audio files — do not commit them.

### Step 2 — Extract clean clips

```bash
# Scan the first 50 minutes; skip the final 15 min where audience
# warm-up noise typically peaks
python profiles/don_rickles/voice_prep/scan_segments.py \
    profiles/dave_chappelle/voice_prep/dave_chappelle_raw_sticks_and_stones_2019.wav \
    --window 30 --step 15 --top 10 \
    --non-overlap --extract \
    --max-secs 3000 \
    --out-dir profiles/dave_chappelle/voice_prep \
    --label sticks_and_stones_2019
```

**Target clip count:** 10 clips. Combined size must stay under ~11 MB —
enforced by `scripts/elevenlabs_clone.py` via `--max-bytes`.

Output files will be named like:
`clip_sticks_and_stones_2019_0130_30s.wav`

If supplementing with The Closer clips, use label `the_closer_2021` and
confirm both sets land in the same output directory before the clone step.

### Step 3 — Create IVC voice

```bash
python scripts/elevenlabs_clone.py create-ivc \
    --profile dave_chappelle "Dave Chappelle" \
    --all-clips
```

Follow `docs/voice-cloning.md` §"Step-by-step: IVC for a new persona"
from step 3 onward for verification.

---

## Quality caveats

- **License:** Both Sticks & Stones and The Closer are exclusive Netflix
  originals, copyright Netflix / Dave Chappelle. ElevenLabs ToS prohibits
  cloning living public figures' voices without their explicit written
  permission. These source notes are for local research prototyping only —
  the cloned voice is not distributed publicly.

- **Netflix DRM:** Netflix streams are typically Widevine L1/L3 protected.
  yt-dlp can download Netflix content in some regions and configurations;
  results vary. If download fails, the Netflix Is A Joke YouTube channel
  provides official clips that can be assembled as partial sources (clips are
  typically 2–5 minutes, so multiple clips may be needed to reach 10 x 30s
  windows). Clip URL list: `https://www.youtube.com/playlist?list=PLXSrjGY5Tz_hmqDd7fs8hMd7fysZDwmZA`

- **Audience laughter:** The Tabernacle crowds are enthusiastic. As with
  Pryor, `scan_segments.py` amplitude consistency scoring will naturally
  de-rank windows where laughter dominates. Prefer picks with low spread
  scores, which indicate sustained Chappelle speech rather than audience
  reaction.

- **Era mixing:** Do not mix pre-2016 Chappelle material (e.g. Comedy
  Central-era specials like For What It's Worth 2004 or Killin' Them Softly
  2000). His 2000-era voice is noticeably younger, higher, and faster. The
  vocal register shift is large enough to cause IVC identity drift.

---

## Recommendations for the user

1. Preview the Netflix Is A Joke YouTube clips first (e.g.
   `https://www.youtube.com/watch?v=wZXoErL2124`) to confirm the
   audio quality before committing to a full Netflix download session.

2. Use the **first 20–40 minutes** of Sticks & Stones for extraction.
   Audience warm-up is lower early in the set; Chappelle's delivery is more
   measured and conversational (longer pauses, slower pace) which helps IVC
   capture his full dynamic range without clipping on loud punchlines.

3. If yt-dlp Netflix download fails, use the YouTube clip playlist as
   a fallback. Each clip is 2–5 minutes; you may need 4–5 clips to have
   enough raw material for `scan_segments.py` to find 10 non-overlapping
   30-second windows with good scores.

4. The special includes a bonus epilogue ("The Punchline," filmed at
   Broadway's Lunt-Fontanne Theatre). Skip this portion — different
   room acoustics and a different microphone position will introduce
   inconsistency into the IVC sample set.
