#!/usr/bin/env python3
"""
Scan an audio file for the best N-second windows for voice clone reference.
Composite score: low amplitude spread (consistent) + high mean (energized delivery).

Examples
--------
# Old behavior: 10s windows, print top picks
python scan_segments.py cavett1972.wav --window 10 --step 10

# New: scan 30s windows (ElevenLabs PVC minimum), non-overlapping, extract top 6
python scan_segments.py cavett1972.wav --window 30 --step 15 --top 6 \\
    --non-overlap --extract --out-dir .
"""
import argparse
import os
import re
import subprocess
import sys

FFMPEG  = r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"
FFPROBE = r"C:\ProgramData\chocolatey\bin\ffprobe.exe"


def get_duration(path: str) -> float:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def measure_segment(path: str, start: float, length: float):
    r = subprocess.run(
        [FFMPEG, "-v", "info", "-ss", str(start), "-t", str(length),
         "-i", path, "-af", "volumedetect", "-f", "null", os.devnull],
        capture_output=True, text=True,
    )
    out = r.stdout + r.stderr
    mean = re.search(r"mean_volume:\s*([-\d.]+)", out)
    peak = re.search(r"max_volume:\s*([-\d.]+)", out)
    if mean and peak:
        return float(mean.group(1)), float(peak.group(1))
    return None, None


def extract_segment(src: str, start: float, length: float, dst: str) -> None:
    # 22050 mono 16-bit matches the existing candidate_*.wav format used elsewhere.
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-ss", str(start), "-t", str(length),
         "-i", src,
         "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le",
         dst],
        check=True,
    )


def scan(path, window=30, step=15, top=12, max_secs=None,
         non_overlap=False, extract=False, out_dir=None, label=None):
    total = int(get_duration(path))
    scan_to = min(total - window, max_secs) if max_secs else total - window
    base = os.path.splitext(os.path.basename(path))[0]
    tag = label or base
    print(f"Scanning {base} ({total}s) — first {scan_to}s in {window}s windows every {step}s...")

    results = []
    for t in range(5, scan_to, step):
        mean, peak = measure_segment(path, t, window)
        if mean is None or mean < -50 or peak >= -0.1:
            continue
        spread = peak - mean
        results.append((t, spread, mean, peak))

    if not results:
        print("No suitable segments found.")
        return []

    means = [m for _, _, m, _ in results]
    mn_lo, mn_hi = min(means), max(means)
    rng = (mn_hi - mn_lo) or 1
    scored = []
    for t, spread, mean, peak in results:
        mean_score = (mean - mn_lo) / rng
        composite = spread - (mean_score * 3)
        scored.append((composite, spread, t, mean, peak))
    scored.sort()

    if non_overlap:
        picked = []
        for cand in scored:
            t = cand[2]
            if all(abs(t - p[2]) >= window for p in picked):
                picked.append(cand)
            if len(picked) >= top:
                break
    else:
        picked = scored[:top]

    print(f"\nTop {len(picked)} windows (low spread + energized delivery):")
    print(f"{'Start':<10} {'Mean(dB)':<10} {'Max(dB)':<10} {'Spread':<10} {'Score':<10}")
    print("-" * 52)
    extracted = []
    for i, (composite, spread, t, mean, peak) in enumerate(picked):
        mm, ss = divmod(t, 60)
        ts = f"{mm:02d}:{ss:02d}"
        flag = " <-- TRY THIS" if i == 0 else ""
        print(f"{ts:<10} {mean:<10.1f} {peak:<10.1f} {spread:<10.1f} {composite:<10.1f}{flag}")

        if extract:
            dst_dir = out_dir or os.path.dirname(path) or "."
            os.makedirs(dst_dir, exist_ok=True)
            fname = f"clip_{tag}_{mm:02d}{ss:02d}_{window}s.wav"
            dst = os.path.join(dst_dir, fname)
            try:
                extract_segment(path, t, window, dst)
                extracted.append(dst)
                print(f"  -> {dst}")
            except subprocess.CalledProcessError as e:
                print(f"  !! extract failed for t={t}: {e}")

    if extract:
        print(f"\nExtracted {len(extracted)} clip(s).")

    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Source audio file")
    ap.add_argument("--window", type=int, default=30, help="Window length in seconds (default: 30; ElevenLabs PVC min)")
    ap.add_argument("--step", type=int, default=15, help="Step between window starts in seconds (default: 15)")
    ap.add_argument("--top", type=int, default=12, help="How many top picks to keep (default: 12)")
    ap.add_argument("--max-secs", type=int, default=None, help="Only scan the first N seconds")
    ap.add_argument("--non-overlap", action="store_true", help="Reject overlapping picks (separation >= window)")
    ap.add_argument("--extract", action="store_true", help="Also write each picked window to disk as 22050 mono WAV")
    ap.add_argument("--out-dir", default=None, help="Destination dir for extracted clips (default: source dir)")
    ap.add_argument("--label", default=None, help="Filename label for extracted clips (default: source basename)")
    args = ap.parse_args()

    scan(args.path, window=args.window, step=args.step, top=args.top,
         max_secs=args.max_secs, non_overlap=args.non_overlap,
         extract=args.extract, out_dir=args.out_dir, label=args.label)


if __name__ == "__main__":
    main()
