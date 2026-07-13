"""Headless throughput bench for the Phase 2 detector+tracker+[pose] path.

Runs the *same* hot-path code as main_loop (capture -> shared YOLO -> ByteTrack
-> optional pose on person crops -> optional fall state machine) but without
cv2.imshow, so it can run in a windowless shell. Use this to verify the
FPS/VRAM success criteria when you can't watch the live display.

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

from core.config import load_pipeline_config, load_models_config
from core.detector import Detector
from core.tracker import Tracker
from core.video_source import build_source
from pipeline.frame_router import FrameRouter


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
    p.add_argument("--no-pose", action="store_true",
                   help="skip the pose model even if enabled in config (A/B compare)")
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

    # Phase 2: optionally load the pose model + fall detector
    features = cfg.get("features", {})
    router = FrameRouter(cfg.get("router", {}))
    use_pose = features.get("pose", False) and router.is_enabled("pose") and not args.no_pose
    pose_every = router.every("pose") if use_pose else 1
    pose_est = None
    fall_det = None
    if use_pose:
        from core.pose import PoseEstimator
        from core.state_machine import FallDetector
        pose_est = PoseEstimator()
        fall_det = FallDetector(load_models_config().get("fall", {}))

    import torch
    torch.cuda.reset_peak_memory_stats()

    source = build_source(src_cfg)
    source.open()
    if not source.isOpened():
        raise RuntimeError("source failed to open")

    src_label = (f"synthetic@{args.fps:.0f}fps" if args.source == "synthetic"
                 else (args.path or args.source))
    print(f"bench: source={src_label}  seconds={args.seconds:.1f}  "
          f"imgsz={detector.imgsz} half={detector.half} device={detector.device} "
          f"pose={'on (every='+str(pose_every)+')' if pose_est else 'off'}")

    # read one real frame to size warmup against actual input
    ok, frame = source.read()
    if not ok or frame is None:
        # fall back to a synthetic frame if camera read fails
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print("WARN: first read failed — using synthetic frame")
    tracks = tracker.update(frame)  # warmup
    if pose_est is not None:
        persons = [t for t in tracks if t.cls == 0]
        if persons:
            pose_est.estimate_crops(frame, persons)
            torch.cuda.synchronize()

    n = 0
    n_poses = 0
    n_falls = 0
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
        if pose_est is not None and router.should_run("pose", n):
            persons = [t for t in tracks if t.cls == 0]
            if persons:
                poses = pose_est.estimate_crops(frame, persons)
                n_poses += len(poses)
                if fall_det is not None:
                    events = fall_det.update(poses, frame_idx=n, t=time.perf_counter())
                    n_falls += len(events)
        n += 1
        now = time.perf_counter()
        if now - last_print >= 2.0:
            extra = f" p:{n_poses}"
            if pose_est is not None:
                extra += f" falls:{n_falls}"
            print(f"  ... {n} frames, {n / (now - t0):.1f} fps, "
                  f"{len(tracks)} tracks{extra}")
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
    if pose_est is not None:
        print(f"pose imgsz / half   {pose_est.imgsz} / {pose_est.half}")
        print(f"poses (running tot) {n_poses}")
        print(f"fall events         {n_falls}")
    print(f"target >=20 FPS      {'OK' if n / elapsed >= 20 else 'BELOW'}")
    print(f"vram <2GB           {'OK' if peak < 2048 else 'OVER'}")


if __name__ == "__main__":
    main()