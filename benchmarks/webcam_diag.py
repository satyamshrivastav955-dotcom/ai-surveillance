"""Webcam capture bottleneck diagnostic.

Three checks:
  (1) print cap.get(cv2.CAP_PROP_FPS) reported by the driver
  (2) try both backends (default MSMF + CAP_DSHOW) on Windows
  (3) raw cap.read() loop for N seconds with NO inference — isolates
      the capture-only FPS from any inference overhead.

  python benchmarks/webcam_diag.py --seconds 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _backend_name(flag: int) -> str:
    return {cv2.CAP_DSHOW: "CAP_DSHOW", cv2.CAP_MSMF: "CAP_MSMF", -1: "DEFAULT"}.get(flag, str(flag))


def _probe(index: int, backend: int, seconds: float) -> dict:
    """Open webcam and report driver-claimed FPS + measured read-only FPS."""
    cap = cv2.VideoCapture(index) if backend < 0 else cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"  [{_backend_name(backend):9}] FAILED to open")
        return {}

    # try to force a reasonable resolution/prop before reading the claimed fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    claimed_fps = cap.get(cv2.CAP_PROP_FPS)
    backend_actual = cap.get(cv2.CAP_PROP_BACKEND)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc_s = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)]) if fourcc else "?"

    print(f"  [{_backend_name(backend):9}] opened  backend_id={backend_actual} "
          f"claimed_fps={claimed_fps}  res={w}x{h}  fourcc={fourcc_s!r}")

    # --- raw read-only timing, no inference whatsoever ---
    # warmup: discard first few frames (auto-exposure / focus settling)
    for _ in range(10):
        cap.read()
    n = 0
    last_pts = []
    t0 = time.perf_counter()
    deadline = t0 + seconds
    per_read_ns = []
    while time.perf_counter() < deadline:
        ts = time.perf_counter_ns()
        ok, frame = cap.read()
        rdt = time.perf_counter_ns() - ts
        if not ok or frame is None:
            continue
        per_read_ns.append(rdt)
        if n < 30:
            # cv2 timestamps (may be 0 on some drivers)
            last_pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        n += 1
    elapsed = time.perf_counter() - t0
    cap.release()

    fps = n / elapsed if elapsed > 0 else 0
    import statistics
    if per_read_ns:
        mean_ms = (sum(per_read_ns) / len(per_read_ns)) / 1e6
        p50 = statistics.median(per_read_ns) / 1e6
        p99 = sorted(per_read_ns)[int(len(per_read_ns) * 0.99) - 1] / 1e6
    else:
        mean_ms = p50 = p99 = 0
    print(f"  [{_backend_name(backend):9}] raw read-only: {n} frames / {elapsed:.2f}s "
          f"= {fps:.2f} FPS   read-latency ms: mean={mean_ms:.2f} p50={p50:.2f} p99={p99:.2f}")
    if last_pts and any(last_pts):
        deltas = [b - a for a, b in zip(last_pts[:-1], last_pts[1:]) if (b - a) > 0]
        if deltas:
            print(f"  [{_backend_name(backend):9}] driver-reported inter-frame "
                  f"interval (first 29 frames): mean={sum(deltas) / len(deltas):.2f} ms")
    print()
    return {"backend": _backend_name(backend), "fps": fps, "mean_read_ms": mean_ms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()

    print(f"=== webcam capture diagnostic  index={args.index} duration={args.seconds}s ===")
    print(f"cv2 build: {cv2.getBuildInformation().splitlines()[0]}")
    print(f"opencv-python: {cv2.__version__}")
    print()

    # The WebcamSource in core/video_source.py uses CAP_DSHOW explicitly.
    # Probe both the default backend AND DSHOW so we can compare them.
    backends = [
        ("DEFAULT (auto)", -1),
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
    ]
    results = []
    for label, flag in backends:
        print(f"--- probing {label} (flag={flag}) ---")
        r = _probe(args.index, flag, args.seconds)
        if r:
            r["label"] = label
            results.append(r)
    print("=== summary ===")
    for r in results:
        print(f"  {r['label']:18} read_fps={r['fps']:.2f} mean_read={r['mean_read_ms']:.2f}ms")


if __name__ == "__main__":
    main()