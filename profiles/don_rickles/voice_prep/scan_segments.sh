#!/usr/bin/env bash
# Scans an audio file for cleanest 10s windows via volumedetect.
# Low spread (max_volume - mean_volume) = consistent speech, minimal audience spikes.
# Usage: bash scan_segments.sh <input.wav> [step_seconds]

INPUT="$1"
STEP="${2:-5}"
SEG=10

duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null)
total=$(printf "%.0f" "$duration")

echo "Scanning $(basename "$INPUT") (${total}s) in ${SEG}s windows every ${STEP}s..."
echo ""
printf "%-10s %-10s %-10s %-10s\n" "Start" "Mean(dB)" "Max(dB)" "Spread"
printf "%-10s %-10s %-10s %-10s\n" "----------" "----------" "----------" "----------"

results=()

for ((t=5; t<total-SEG; t+=STEP)); do
    stats=$(ffmpeg -v quiet -ss "$t" -t "$SEG" -i "$INPUT" \
        -af "volumedetect" -f null /dev/null 2>&1)
    mean=$(echo "$stats" | grep mean_volume | grep -oP '[-\d.]+(?= dB)')
    max=$(echo "$stats"  | grep max_volume  | grep -oP '[-\d.]+(?= dB)')

    # Skip silence (mean < -50 dB)
    if [[ -z "$mean" ]] || (( $(echo "$mean < -50" | bc -l) )); then
        continue
    fi

    spread=$(echo "$max - $mean" | bc -l)
    mm=$(printf "%02d" $((t/60)))
    ss=$(printf "%02d" $((t%60)))
    results+=("$(printf '%07.3f' "$spread")|${mm}:${ss}|$mean|$max|$spread")
done

# Sort by spread (lowest first = most consistent = cleanest)
IFS=$'\n' sorted=($(sort <<<"${results[*]}"))
unset IFS

count=0
for entry in "${sorted[@]}"; do
    IFS='|' read -r _ timestamp mean max spread <<< "$entry"
    printf "%-10s %-10s %-10s %-10s\n" "$timestamp" "$mean" "$max" "$spread"
    ((count++))
    [[ $count -ge 10 ]] && break
done
