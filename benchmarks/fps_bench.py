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

    # Phase 3: optionally load ReID + face + identity fusion
    models_cfg = load_models_config()
    use_reid = features.get("reid", False) and router.is_enabled("reid") and not args.no_pose
    use_face = features.get("face", False) and router.is_enabled("face") and not args.no_pose
    reid_every = router.every("reid") if use_reid else 1
    face_every = router.every("face") if use_face else 1
    reid_mgr = None
    face_rec = None
    identity_mgr = None
    if use_reid:
        from core.reid import ReIDExtractor, ReIDManager, ReIDIndex
        reid_ext = ReIDExtractor(models_cfg)
        reid_idx = ReIDIndex(models_cfg)
        reid_mgr = ReIDManager(reid_ext, reid_idx, models_cfg)
    if use_face:
        from core.face import FaceRecognizer
        face_rec = FaceRecognizer(models_cfg)
        idx_path = models_cfg.get("face", {}).get("index_path", "models/face_index")
        face_rec.load_index(idx_path)
    if use_reid or use_face:
        from core.identity import IdentityManager
        identity_mgr = IdentityManager()

    # Phase 4: fire/smoke, smoking, phone, gathering, violence
    fire_smoke_det = None
    smoking_det = None
    phone_det = None
    gathering_det = None
    violence_det = None
    if features.get("fire_smoke") and router.is_enabled("fire_smoke") and not args.no_pose:
        from core.events import FireSmokeDetector
        fire_smoke_det = FireSmokeDetector(models_cfg.get("fire_smoke", {}))
    if features.get("smoking") and router.is_enabled("smoking") and not args.no_pose:
        from core.events import SmokingDetector
        smoking_det = SmokingDetector(models_cfg.get("smoking", {}))
    if features.get("phone") and router.is_enabled("phone") and not args.no_pose:
        from core.events import PhoneWatcherDetector
        phone_det = PhoneWatcherDetector(models_cfg.get("phone", {}),
                                         detector_model=detector.model)
    if features.get("gathering") and router.is_enabled("gathering") and not args.no_pose:
        from core.events import GatheringDetector
        gathering_det = GatheringDetector(models_cfg.get("gathering", {}))
    if features.get("violence") and router.is_enabled("violence") and not args.no_pose:
        from core.events import ViolenceDetector
        violence_det = ViolenceDetector(models_cfg.get("violence", {}))

    import torch
    torch.cuda.reset_peak_memory_stats()

    source = build_source(src_cfg)
    source.open()
    if not source.isOpened():
        raise RuntimeError("source failed to open")

    src_label = (f"synthetic@{args.fps:.0f}fps" if args.source == "synthetic"
                 else (args.path or args.source))
    extras = []
    if pose_est: extras.append(f"pose:every={pose_every}")
    if reid_mgr: extras.append(f"reid:every={reid_every}")
    if face_rec: extras.append(f"face:every={face_every}")
    if fire_smoke_det: extras.append(f"fire_smoke:every={router.every('fire_smoke')}")
    if phone_det: extras.append(f"phone:every={router.every('phone')}")
    if gathering_det: extras.append(f"gathering:every={router.every('gathering')}")
    if violence_det: extras.append(f"violence:every={router.every('violence')}")
    print(f"bench: source={src_label}  seconds={args.seconds:.1f}  "
          f"imgsz={detector.imgsz} half={detector.half} device={detector.device}  "
          f"{' '.join(extras) if extras else 'features=off'}")

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
    n_relinks = 0
    n_face_matches = 0
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
        if reid_mgr is not None and router.should_run("reid", n):
            relinks = reid_mgr.on_tracks_updated(frame, tracks)
            n_relinks += sum(1 for r in relinks if r.matched_track_id is not None)
        if face_rec is not None and router.should_run("face", n):
            for tr in tracks:
                if getattr(tr, "cls", -1) != 0 or getattr(tr, "track_id", -1) < 0:
                    continue
                x1, y1, x2, y2 = tr.xyxy
                fx1 = max(0, int(x1)); fy1 = max(0, int(y1))
                fx2 = min(frame.shape[1], int(x2)); fy2 = min(frame.shape[0], int(y2))
                if fx2 - fx1 < 32 or fy2 - fy1 < 32:
                    continue
                crop = frame[fy1:fy2, fx1:fx2]
                fms = face_rec.process_person_crop(crop, (fx1, fy1), tr.track_id)
                n_face_matches += sum(1 for fm in fms if fm.name is not None)
        # Phase 4 detectors
        if fire_smoke_det is not None and router.should_run("fire_smoke", n):
            fire_smoke_det.detect(frame, n)
        if smoking_det is not None and router.should_run("smoking", n):
            smoking_det.detect(frame, tracks, n)
        if phone_det is not None and router.should_run("phone", n):
            phone_det.detect(frame, tracks, [], n)
        if gathering_det is not None and router.should_run("gathering", n):
            gathering_det.detect(tracks, n, t=time.perf_counter())
        if violence_det is not None and router.should_run("violence", n):
            violence_det.detect(tracks, n, t=time.perf_counter())
        n += 1
        now = time.perf_counter()
        if now - last_print >= 2.0:
            extra = f" p:{n_poses}"
            if pose_est is not None:
                extra += f" falls:{n_falls}"
            if reid_mgr is not None:
                extra += f" relinks:{n_relinks}"
            if face_rec is not None:
                extra += f" faces:{n_face_matches}"
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
    if reid_mgr is not None:
        print(f"reid relinks        {n_relinks}")
    if face_rec is not None:
        print(f"face matches        {n_face_matches}")
    print(f"target >=20 FPS      {'OK' if n / elapsed >= 20 else 'BELOW'}")
    print(f"vram <5GB           {'OK' if peak < 5120 else 'OVER'}")


if __name__ == "__main__":
    main()