#!/usr/bin/env python3
"""
Scan an audio file for the best N-second windows for voice clone reference.
Composite score: low amplitude spread (consistent) + high mean (energized delivery).
"""
import subprocess, re, sys, os

FFMPEG  = r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"
FFPROBE = r"C:\ProgramData\chocolatey\bin\ffprobe.exe"

def get_duration(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())

def measure_segment(path, start, length):
    r = subprocess.run([FFMPEG, "-v", "info", "-ss", str(start), "-t", str(length),
                        "-i", path, "-af", "volumedetect", "-f", "null", "/dev/null"],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    mean = re.search(r"mean_volume:\s*([-\d.]+)", out)
    peak = re.search(r"max_volume:\s*([-\d.]+)", out)
    if mean and peak:
        return float(mean.group(1)), float(peak.group(1))
    return None, None

def scan(path, seg=10, step=10, top=12, max_secs=None):
    total = int(get_duration(path))
    scan_to = min(total - seg, max_secs) if max_secs else total - seg
    print(f"Scanning {os.path.basename(path)} ({total}s) — first {scan_to}s in {seg}s windows every {step}s...")

    results = []
    for t in range(5, scan_to, step):
        mean, peak = measure_segment(path, t, seg)
        if mean is None or mean < -50 or peak >= -0.1:  # skip silence and clipped audio
            continue
        spread = peak - mean
        mm, ss = divmod(t, 60)
        results.append((spread, f"{mm:02d}:{ss:02d}", mean, peak))

    if not results:
        print("No suitable segments found.")
        return []

    # Composite: reward low spread AND high mean (energized delivery)
    means = [m for _, _, m, _ in results]
    mn_lo, mn_hi = min(means), max(means)
    rng = (mn_hi - mn_lo) or 1
    scored = []
    for spread, ts, mean, peak in results:
        mean_score = (mean - mn_lo) / rng   # 0=quietest, 1=loudest
        composite = spread - (mean_score * 3)
        scored.append((composite, spread, ts, mean, peak))
    scored.sort()

    print(f"\nTop {top} windows (low spread + energized delivery):")
    print(f"{'Start':<10} {'Mean(dB)':<10} {'Max(dB)':<10} {'Spread':<10} {'Score':<10}")
    print("-" * 52)
    for i, (composite, spread, ts, mean, peak) in enumerate(scored[:top]):
        flag = " <-- TRY THIS" if i == 0 else ""
        print(f"{ts:<10} {mean:<10.1f} {peak:<10.1f} {spread:<10.1f} {composite:<10.1f}{flag}")

    return scored[:top]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    max_s = int(sys.argv[2]) if len(sys.argv) > 2 else None
    step  = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    if not path:
        print("Usage: python scan_segments.py <audio.wav> [max_seconds] [step_seconds]")
        sys.exit(1)
    scan(path, step=step, max_secs=max_s)
