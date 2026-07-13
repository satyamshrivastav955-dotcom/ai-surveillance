"""Headless throughput bench for the Phase 1 detector+tracker path.

Runs the *same* code path as main_loop (capture -> shared YOLO -> ByteTrack)
but without cv2.imshow, so it can run in a windowless shell. Use this to
verify the >=20 FPS @ 640x640 success criterion when you can't watch the
live display.

    python benchmarks/fps_bench.py --seconds 10
    python benchmarks/fps_bench.py --source synthetic --fps 60
    python benchmarks/fps_bench.py --source file --path tests/clip.mp4 --seconds 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_pipeline_config
from core.detector import Detector
from core.tracker import Tracker
from core.video_source import build_source


def _vram() -> tuple[int, int]:
    import torch
    return (
        int(torch.cuda.memory_allocated() / (1024 * 1024)),
        int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--source", choices=["webcam", "file", "synthetic"], default="webcam")
    p.add_argument("--path", default=None)
    p.add_argument("--fps", type=float, default=30.0,
                   help="synthetic source target FPS (ignored for webcam/file)")
    p.add_argument("--max-frames", type=int, default=2000)
    args = p.parse_args()

    cfg = load_pipeline_config()
    src_cfg = cfg["source"]
    src_cfg["type"] = args.source
    if args.source == "synthetic":
        src_cfg["fps"] = args.fps
        src_cfg.pop("path", None)
    elif args.path:
        src_cfg["path"] = args.path
    else:
        src_cfg.pop("path", None)

    detector = Detector()
    tracker = Tracker(detector)
    import torch
    torch.cuda.reset_peak_memory_stats()

    source = build_source(src_cfg)
    source.open()
    if not source.isOpened():
        raise RuntimeError("source failed to open")

    src_label = (f"synthetic@{args.fps:.0f}fps" if args.source == "synthetic"
                 else (args.path or args.source))
    print(f"bench: source={src_label}  seconds={args.seconds:.1f}  "
          f"imgsz={detector.imgsz} half={detector.half} device={detector.device}")

    # read one real frame to size warmup against actual input
    ok, frame = source.read()
    if not ok or frame is None:
        # fall back to a synthetic frame if camera read fails
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print("WARN: first read failed — using synthetic frame")
    _ = tracker.update(frame)  # warmup

    n = 0
    t0 = time.perf_counter()
    deadline = t0 + args.seconds
    last_print = t0
    while time.perf_counter() < deadline and n < args.max_frames:
        ok, frame = source.read()
        if not ok or frame is None:
            if args.source == "webcam":
                continue
            break
        tracks = tracker.update(frame)
        n += 1
        now = time.perf_counter()
        if now - last_print >= 2.0:
            print(f"  ... {n} frames, {n / (now - t0):.1f} fps, "
                  f"{len(tracks)} tracks")
            last_print = now
    elapsed = time.perf_counter() - t0
    source.release()

    alloc, peak = _vram()
    print("-" * 50)
    print(f"frames              {n}")
    print(f"elapsed             {elapsed:.2f}s")
    print(f"throughput          {n / elapsed:.1f} FPS")
    print(f"vram alloc / peak   {alloc} / {peak} MB")
    print(f"imgsz / half        {detector.imgsz} / {detector.half}")
    print(f"target >=20 FPS      {'OK' if n / elapsed >= 20 else 'BELOW'}")
    print(f"vram <2GB           {'OK' if peak < 2048 else 'OVER'}")


if __name__ == "__main__":
    main()