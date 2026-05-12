# find_clean_segment.ps1
# Scans audio files for the cleanest 10s speech window by analyzing amplitude variance.
# Low variance = consistent speech level. High variance = audience spikes/laughter.

param(
    [string]$InputFile,
    [int]$SegmentLength = 10,   # seconds per candidate
    [int]$StepSize = 5,         # seconds between candidates
    [int]$TopN = 5              # how many best candidates to report
)

$ffmpeg  = "C:\ProgramData\chocolatey\bin\ffmpeg.exe"
$ffprobe = "C:\ProgramData\chocolatey\bin\ffprobe.exe"

# Get total duration in seconds
$durationRaw = & $ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $InputFile 2>&1
$totalSeconds = [int][double]$durationRaw
Write-Host "Scanning $([System.IO.Path]::GetFileName($InputFile)) ($totalSeconds s) in ${SegmentLength}s windows..."

$results = @()

for ($t = 5; $t -lt ($totalSeconds - $SegmentLength); $t += $StepSize) {
    # Use astats to get RMS level and peak for this window
    $stats = & $ffmpeg -v quiet -ss $t -t $SegmentLength -i $InputFile `
        -af "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:key=lavfi.astats.Overall.Peak_level" `
        -f null - 2>&1 | Select-String "RMS_level|Peak_level"

    $rms  = $null
    $peak = $null
    foreach ($line in $stats) {
        if ($line -match "RMS_level=(-?\d+\.?\d*)") { $rms  = [double]$Matches[1] }
        if ($line -match "Peak_level=(-?\d+\.?\d*)") { $peak = [double]$Matches[1] }
    }

    if ($null -ne $rms -and $rms -gt -60) {  # skip near-silence
        $spread = $peak - $rms  # lower spread = more consistent = less crowd spikes
        $results += [PSCustomObject]@{
            StartSec  = $t
            StartTime = [TimeSpan]::FromSeconds($t).ToString("mm\:ss")
            RMS       = [math]::Round($rms, 1)
            Peak      = [math]::Round($peak, 1)
            Spread    = [math]::Round($spread, 1)
        }
    }
}

# Sort by spread (lowest = most consistent speech, least audience laughter)
$best = $results | Sort-Object Spread | Select-Object -First $TopN

Write-Host ""
Write-Host "Top $TopN cleanest segments (lowest amplitude spread = most consistent speech):"
Write-Host ("{0,-10} {1,-8} {2,-8} {3,-8}" -f "StartTime", "RMS(dB)", "Peak(dB)", "Spread")
Write-Host ("-" * 40)
foreach ($r in $best) {
    Write-Host ("{0,-10} {1,-8} {2,-8} {3,-8}" -f $r.StartTime, $r.RMS, $r.Peak, $r.Spread)
}

return $best
