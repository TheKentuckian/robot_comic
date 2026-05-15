# Audio playback recipe for autonomous STT testing

When you place audio files (m4a, wav) next to the robot's microphone on
a dev workstation, the laptop's speakers can drive the robot's mic
end-to-end through the STT → LLM → TTS pipeline without anyone speaking
into the room. This enables fully autonomous interactive testing.

Confirmed working 2026-05-15 with `hello.m4a` and `my name is tony.m4a`
played from a Windows laptop near the robot. Moonshine STT transcribed
each cleanly and the full turn completed end-to-end through llama-server
and ElevenLabs TTS.

## Recipe — Windows (PowerShell + WPF MediaPlayer)

```powershell
Add-Type -AssemblyName PresentationCore   # once per PS session
$p = New-Object System.Windows.Media.MediaPlayer
$p.Open([uri]"<absolute-path-to-audio-file>")
Start-Sleep -Milliseconds 800             # let MediaFoundation buffer
$p.Play()
Start-Sleep -Seconds 5                    # tune to clip length
$p.Stop(); $p.Close()
```

After playback, wait ~10–15 s for the full turn (Moonshine partial →
completed, LLM call with potential retries, streaming TTS) and check the
on-robot journal for transcript / turn-outcome events. From the dev
laptop, assuming SSH access to the robot:

```bash
ssh <robot-host> 'journalctl -u reachy-app-autostart --since "20 seconds ago" -o short-iso' \
  | grep -E "on_line_completed text=|role=user content=|turn.outcome|turn.excerpt"
```

Substitute `<robot-host>` for your SSH alias / hostname.

## Recipe — macOS / Linux

`afplay` (macOS) or `aplay`/`paplay` (Linux) work as drop-in replacements:

```bash
# macOS
afplay <absolute-path-to-audio-file>

# Linux (PulseAudio)
paplay <absolute-path-to-audio-file>

# Linux (ALSA, fallback)
aplay <absolute-path-to-audio-file>
```

## Gotchas

- `NaturalDuration.HasTimeSpan` on the WPF MediaPlayer will often return
  `False` right after `Open()` even after sleeping — the player can play
  without it. Don't gate on duration; use a fixed sleep based on known
  clip length.
- Background-mode playback works: start the playback in a background
  task while you tail the robot's journal in parallel from another shell.
- Filenames with spaces work fine when wrapped in quotes
  (`"my name is tony.m4a"`).

## When this becomes unreliable

- **Overlapping utterances**: if multiple plays fire within one turn's
  processing window (~10–15 s), later audio arrives while the listener
  is still responding to the prior turn. Mid-turn audio either gets
  merged into the trailing partial transcript or is buffered and surfaces
  after the response completes — but the new turn won't cleanly resolve.
  Wait at least one full turn duration between plays for stress tests
  where each play should resolve independently.
- **Speaker/mic geometry**: the recipe assumes the laptop speakers are
  audible to the robot mic at the current room geometry. Re-verify after
  moving the rig.
