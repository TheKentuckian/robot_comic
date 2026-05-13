#!/usr/bin/env python3
"""Convert Moonshine ONNX model files to the ONNX Runtime native .ort format.

The .ort (FlatBuffer) format skips the ONNX graph-parse step at load time,
giving roughly 3-5x faster startup on resource-constrained hardware such as a
Raspberry Pi or Reachy Mini robot.

Usage
-----
    python scripts/convert_moonshine_to_ort.py /path/to/moonshine/model/dir

    # Use the default Moonshine cache directory:
    python scripts/convert_moonshine_to_ort.py ./cache/moonshine_voice

    # Explicit optimization style (Fixed is the default; Runtime also valid):
    python scripts/convert_moonshine_to_ort.py ./cache/moonshine_voice \\
        --optimization-style Fixed

The converted .ort files are written alongside the source .onnx files in the
same directory.  The original .onnx files are left untouched.

Prerequisites
-------------
    pip install onnxruntime  # or onnxruntime-gpu / onnxruntime-silicon

The robot_comic runtime will automatically prefer .ort files over .onnx when
both are present (see ``local_stt_realtime.resolve_ort_model_path``).
"""

from __future__ import annotations
import sys
import argparse
import subprocess
from pathlib import Path


def _convert_dir(input_dir: Path, optimization_style: str) -> list[tuple[Path, Path]]:
    """Run onnxruntime's converter on *input_dir* and return (src, dst) pairs."""
    onnx_files = sorted(input_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files found in {input_dir}")

    print(f"Found {len(onnx_files)} .onnx file(s) in {input_dir}:")
    for f in onnx_files:
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    # onnxruntime.tools.convert_onnx_models_to_ort converts all .onnx files in
    # the given directory and writes .ort siblings.
    cmd = [
        sys.executable,
        "-m",
        "onnxruntime.tools.convert_onnx_models_to_ort",
        str(input_dir),
        "--optimization_style",
        optimization_style,
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"onnxruntime conversion failed with exit code {result.returncode}.\n"
            "Make sure onnxruntime is installed: pip install onnxruntime"
        )

    # Collect produced .ort files and compare sizes.
    results: list[tuple[Path, Path]] = []
    for src in onnx_files:
        dst = src.with_suffix(".ort")
        if dst.exists():
            results.append((src, dst))
        else:
            print(f"  WARNING: expected {dst} but it was not produced.")
    return results


def _print_summary(results: list[tuple[Path, Path]]) -> None:
    if not results:
        print("No .ort files were produced.")
        return
    print("\nConversion summary:")
    print(f"  {'Source (.onnx)':40s}  {'Size':>8s}    {'Output (.ort)':40s}  {'Size':>8s}  {'Ratio':>6s}")
    print("  " + "-" * 110)
    for src, dst in results:
        src_mb = src.stat().st_size / 1024 / 1024
        dst_mb = dst.stat().st_size / 1024 / 1024
        ratio = dst_mb / src_mb if src_mb else 0.0
        print(f"  {src.name:40s}  {src_mb:>7.1f}M    {dst.name:40s}  {dst_mb:>7.1f}M  {ratio:>5.2f}x")
    print()
    print("The robot_comic runtime will automatically use these .ort files.")
    print("Original .onnx files were NOT modified.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Moonshine .onnx models to .ort for faster startup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .onnx model files (e.g. ./cache/moonshine_voice).",
    )
    parser.add_argument(
        "--optimization-style",
        default="Fixed",
        choices=["Fixed", "Runtime"],
        help=(
            "ORT optimization style.  'Fixed' (default) bakes in graph optimizations "
            "at conversion time for fastest load.  'Runtime' defers some optimizations "
            "to load time (more flexible but slightly slower)."
        ),
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        results = _convert_dir(input_dir, args.optimization_style)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_summary(results)


if __name__ == "__main__":
    main()
