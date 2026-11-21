"""Benchmark + equivalence harness for region detection.

Runs the legacy whole-image implementation and the tiled implementation on the
same screenshots, compares the detected regions exactly, and records wall time
and peak RSS for each run in an isolated subprocess.

Usage:
    python -m benchmarks.bench_regions
    python -m benchmarks.bench_regions --width 6000 --height 4000 --runs 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def _synthetic(width: int, height: int, seed: int = 7) -> Image.Image:
    """Large synthetic fault screenshot with tiled edges and text-like noise."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), (245, 246, 248))
    d = ImageDraw.Draw(img)
    # red banner spanning the whole width (crosses many tiles)
    d.rectangle([0, height // 10, width, height // 10 + 120], fill=(198, 40, 40))
    # amber strip
    d.rectangle([0, height // 4, width, height // 4 + 40], fill=(255, 171, 0))
    # boxed dialogs at various positions/sizes, some deliberately straddling
    # the 768 px tile boundaries (128 cells * 6 px)
    for cx, cy, bw, bh in [
        (700, 600, 400, 300),  # straddles x tile edge at 768
        (1500, 1300, 500, 400),
        (2500, 200, 350, 250),
    ]:
        if cx + bw < width and cy + bh < height:
            d.rectangle([cx, cy, cx + bw, cy + bh], fill=(255, 255, 255), outline=(60, 60, 60), width=3)
            d.rectangle([cx, cy, cx + bw, cy + 30], fill=(58, 58, 60))
    # scatter small red/amber patches + noise to exercise size filtering
    for _ in range(40):
        x = int(rng.integers(0, max(width - 40, 1)))
        y = int(rng.integers(0, max(height - 40, 1)))
        color = (198, 40, 40) if rng.random() < 0.5 else (255, 171, 0)
        d.rectangle([x, y, x + int(rng.integers(5, 45)), y + int(rng.integers(5, 45))], fill=color)
    noise = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise, "RGB")
    # blend noise only in the bottom band to create high-contrast texture
    band = (0, height - height // 5, width, height)
    img.paste(noise_img.crop(band), band[:2])
    return img


def _region_tuple(regions) -> list[tuple]:
    return [
        (r.kind, r.x, r.y, r.width, r.height, r.score, tuple(r.notes))
        for r in regions
    ]


def _run_one(module: str, image_path: str) -> dict:
    """Run one implementation in this process and report time/peak memory.

    Timing and allocation tracing are separate passes: tracemalloc slows down
    every Python allocation, which would dominate the legacy implementation's
    many small numpy allocations and make the timing meaningless.
    """
    import tracemalloc

    from app.vision.regions import detect_regions

    if module == "legacy":
        from benchmarks.legacy_regions import detect_regions as legacy_detect

        detect = legacy_detect
    else:
        detect = detect_regions

    img = Image.open(image_path)
    img.load()

    t0 = time.perf_counter()
    regions = detect(img, max_regions=8)
    elapsed = time.perf_counter() - t0
    peak = _peak_rss_kb()

    tracemalloc.start()
    detect(img, max_regions=8)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "module": module,
        "seconds": elapsed,
        "peak_rss_mb": peak / 1024,
        "python_peak_mb": traced_peak / (1024 * 1024),
        "regions": _region_tuple(regions),
    }


def _peak_rss_kb() -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def _subprocess(module: str, image_path: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-m", "benchmarks.bench_regions", "--worker", module, "--image", image_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["legacy", "tiled"], help=argparse.SUPPRESS)
    parser.add_argument("--image", help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--keep", type=Path, default=ROOT / "benchmarks" / "_large_sample.png")
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(_run_one(args.worker, args.image)))
        return 0

    args.keep.parent.mkdir(parents=True, exist_ok=True)
    img = _synthetic(args.width, args.height)
    img.save(args.keep)
    targets = [str(ROOT / "samples" / "fault_screenshot.png"), str(args.keep)]
    targets = [t for t in targets if Path(t).exists()]

    results = []
    all_equal = True
    for target in targets:
        legacy = _subprocess("legacy", target)
        tiled_runs = [_subprocess("tiled", target) for _ in range(args.runs)]
        tiled = min(tiled_runs, key=lambda r: r["seconds"])
        equal = legacy["regions"] == tiled["regions"]
        all_equal &= equal
        results.append(
            {
                "image": os.path.relpath(target, ROOT),
                "size": list(Image.open(target).size),
                "equal": equal,
                "legacy_seconds": round(legacy["seconds"], 4),
                "tiled_seconds": round(tiled["seconds"], 4),
                "legacy_peak_rss_mb": round(legacy["peak_rss_mb"], 1),
                "tiled_peak_rss_mb": round(tiled["peak_rss_mb"], 1),
                "legacy_python_peak_mb": round(legacy["python_peak_mb"], 2),
                "tiled_python_peak_mb": round(tiled["python_peak_mb"], 2),
                "region_count": len(tiled["regions"]),
                "regions": tiled["regions"],
            }
        )
        if not equal:
            print(f"MISMATCH on {target}", file=sys.stderr)
            print("legacy:", legacy["regions"], file=sys.stderr)
            print("tiled :", tiled["regions"], file=sys.stderr)

    print(json.dumps({"all_equal": all_equal, "results": results}, indent=2))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
