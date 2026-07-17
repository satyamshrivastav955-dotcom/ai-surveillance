"""Phase 2 main loop: capture -> detect -> track -> [pose -> fall] -> display.

Phase 1's pipeline stays intact (single shared YOLO detector, ByteTrack on
top). Phase 2 adds the *optional* pose stage, gated by FrameRouter and
toggleable from configs/pipeline.yaml — pose runs on person crops only and
feeds the rule-based fall detector. Phases 3+ will plug in the same way.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.config import load_pipeline_config, load_models_config
from core.detector import Detector
from core.tracker import Tracker
from core.video_source import VideoSource, build_source
from pipeline.frame_router import FrameRouter

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# distinct colors per track id for visual debugging
_TRACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0),
]


def _color_for(tid: int) -> tuple[int, int, int]:
    return _TRACK_COLORS[tid % len(_TRACK_COLORS)]


def _draw_tracks(frame: np.ndarray, tracks) -> None:
    for t in tracks:
        x1, y1, x2, y2 = t.xyxy
        c = _color_for(t.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f"id{t.track_id} {COCO_NAMES[t.cls] if 0 <= t.cls < len(COCO_NAMES) else t.cls} {t.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, label, (x1 + 2, max(th, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# COCO-pose keypoint skeleton pairs for visualization (indices, see core/pose.py)
_POSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # face
    (5, 6), (5, 11), (6, 12), (11, 12),  # torso
    (5, 7), (7, 9), (6, 8), (8, 10),      # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]
_KPT_COLOR = (0, 255, 255)
_SKELETON_COLOR = (255, 255, 0)


def _draw_pose(frame: np.ndarray, poses) -> None:
    """Draw keypoints + skeleton for one frame's poses (already remapped to full-frame coords)."""
    for p in poses:
        k = p.keypoints
        # lines first so dots land on top
        for a, b in _POSE_SKELETON:
            xa, ya, ca = k[a]
            xb, yb, cb = k[b]
            if ca > 0.3 and cb > 0.3:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                         _SKELETON_COLOR, 1)
        for x, y, c in k:
            if c > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, _KPT_COLOR, -1)


def _draw_falls(frame: np.ndarray, fall_events) -> None:
    """Big red FALL label per event so it's visible on the live display."""
    for ev in fall_events:
        cx = 50
        cy = 60 + 30 * ev.track_id % 5
        cv2.rectangle(frame, (cx, cy - 20), (cx + 240, cy + 5), (0, 0, 128), -1)
        cv2.putText(frame, f"FALL id{ev.track_id} f{ev.frame_idx}",
                    (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# Phase 4 event bbox colors (BGR)
_FIRE_COLOR  = (0, 80, 255)    # orange-red
_SMOKE_COLOR = (200, 200, 200)  # light gray
_PHONE_COLOR = (255, 0, 255)   # magenta


def _draw_phase4_events(frame: np.ndarray, phase4_events: list) -> None:
    """Draw bounding boxes + labels for FIRE, SMOKE, and PHONE events on the HUD.

    FIRE  -> orange/red box with 'FIRE' label
    SMOKE -> light gray box with 'SMOKE' label
    PHONE -> magenta box with 'PHONE id{track_id}' label

    Other event types (GATHERING, VIOLENCE, SMOKING) don't have per-pixel bboxes
    so they are skipped here — their labels appear in the HUD text bar instead.
    """
    for ev in phase4_events:
        et = ev.event_type
        if et == "FIRE":
            bbox = ev.details.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), _FIRE_COLOR, 3)
                (tw, th), _ = cv2.getTextSize("FIRE", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), _FIRE_COLOR, -1)
                cv2.putText(frame, "FIRE", (x1 + 3, max(th, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif et == "SMOKE":
            bbox = ev.details.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), _SMOKE_COLOR, 2)
                (tw, th), _ = cv2.getTextSize("SMOKE", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), _SMOKE_COLOR, -1)
                cv2.putText(frame, "SMOKE", (x1 + 3, max(th, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        elif et == "PHONE":
            bbox = ev.details.get("phone_bbox")
            tid  = ev.details.get("track_id", "?")
            if bbox:
                x1, y1, x2, y2 = bbox
                label = f"PHONE id{tid}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), _PHONE_COLOR, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), _PHONE_COLOR, -1)
                cv2.putText(frame, label, (x1 + 3, max(th, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _vram_mb() -> int:
    try:
        import torch
        return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        return -1


def _vram_reserved_mb() -> int:
    try:
        import torch
        return int(torch.cuda.memory_reserved() / (1024 * 1024))
    except Exception:
        return -1


def _open_vram_log(path: str) -> tuple[Any, Any] | None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["t_iso", "frame_idx", "fps", "vram_alloc_mb", "vram_reserved_mb"])
    f.flush()
    return w, f


def run(config_path: str | None = None) -> None:
    cfg = load_pipeline_config() if config_path is None else _load_extra(config_path)
    src_cfg = cfg["source"]
    disp = cfg.get("display", {})
    perf = cfg.get("perf", {})

    router = FrameRouter(cfg.get("router", {}))

    # Motion prefilter — skip heavy stages on static frames
    motion_cfg = cfg.get("motion_filter", {})
    motion_filter = None
    if motion_cfg.get("enabled", True):
        from core.motion_filter import MotionPrefilter
        motion_filter = MotionPrefilter(
            threshold=motion_cfg.get("threshold", 0.01),
            min_changed_pixels=motion_cfg.get("min_changed_pixels", 1000)
        )
        print("[motion] prefilter enabled")

    # --- models -----------------------------------------------------------
    detector = Detector()
    tracker = Tracker(detector)

    # Phase 2: pose + fall detector. Only constructed if enabled in features
    # AND the FrameRouter stage is enabled — defensive double-check so a stale
    # config can't load a heavy model then never run it.
    features = cfg.get("features", {})
    pose_est = None
    fall_det = None
    if features.get("pose") and router.is_enabled("pose"):
        from core.pose import PoseEstimator
        from core.state_machine import FallDetector
        pose_est = PoseEstimator()
        fall_cfg = (load_pipeline_config() if config_path is None else _load_extra(config_path))
        # fall thresholds live in models.yaml, not pipeline.yaml
        from core.config import load_models_config
        fall_det = FallDetector(load_models_config().get("fall", {}))
        print(f"[phase2] pose enabled  imgsz={pose_est.imgsz} half={pose_est.half}")
    if features.get("fall_detection") and fall_det is None:
        # fall_detection feature without pose is meaningless — warn loudly
        from core.state_machine import FallDetector
        from core.config import load_models_config
        fall_det = FallDetector(load_models_config().get("fall", {}))
        print("[phase2] WARN fall_detection enabled but pose disabled — "
              "FallDetector will never receive poses.")

    # Phase 3: ReID + face recognition + identity fusion.
    # All gated by FrameRouter + features config (constraint #6).
    models_cfg = load_models_config()
    reid_mgr = None
    face_rec = None
    identity_mgr = None
    if features.get("reid") and router.is_enabled("reid"):
        from core.reid import ReIDExtractor, ReIDManager, ReIDIndex
        reid_ext = ReIDExtractor(models_cfg)
        reid_idx = ReIDIndex(models_cfg)
        reid_mgr = ReIDManager(reid_ext, reid_idx, models_cfg)
        print(f"[phase3] reid enabled  model={models_cfg.get('reid', {}).get('model', 'resnet18')} "
              f"dim={reid_ext.input_size} every={router.every('reid')}")
    if features.get("face") and router.is_enabled("face"):
        from core.face import FaceRecognizer
        face_rec = FaceRecognizer(models_cfg)
        # load the persistent face index if it exists
        idx_path = models_cfg.get("face", {}).get("index_path", "models/face_index")
        face_rec.load_index(idx_path)
        print(f"[phase3] face enabled  pack={models_cfg.get('face', {}).get('model_pack', 'buffalo_s')} "
              f"enrolled={len(face_rec._labels)} every={router.every('face')}")
    if features.get("identity_fusion") and (reid_mgr is not None or face_rec is not None):
        from core.identity import IdentityManager
        identity_mgr = IdentityManager()
        print("[phase3] identity fusion enabled")

    # Phase 4: fire/smoke, smoking, phone, gathering, violence, object_left.
    # All gated by FrameRouter + features config (constraint #6).
    fire_smoke_det = None
    smoking_det = None
    phone_det = None
    gathering_det = None
    violence_det = None
    object_left_det = None
    if features.get("fire_smoke") and router.is_enabled("fire_smoke"):
        from core.events import FireSmokeDetector
        fire_smoke_det = FireSmokeDetector(models_cfg.get("fire_smoke", {}))
        print(f"[phase4] fire_smoke enabled  every={router.every('fire_smoke')}  "
              f"method={'hsv_heuristic' if models_cfg.get('fire_smoke', {}).get('weights') is None else 'yolo'}")
    if features.get("smoking") and router.is_enabled("smoking"):
        from core.events import SmokingDetector
        smoking_det = SmokingDetector(models_cfg.get("smoking", {}))
        print(f"[phase4] smoking enabled  every={router.every('smoking')}  "
              f"method={'glow_heuristic' if models_cfg.get('smoking', {}).get('weights') is None else 'yolo'}")
    if features.get("phone") and router.is_enabled("phone"):
        from core.events import PhoneWatcherDetector
        # PhoneWatcherDetector always loads its own independent YOLOv8n instance
        # (COCO class 67). detector_model kwarg is accepted but always ignored.
        phone_det = PhoneWatcherDetector(models_cfg.get("phone", {}))
        print(f"[phase4] phone enabled  every={router.every('phone')}  imgsz={phone_det.imgsz}  "
              f"conf={phone_det.conf}  (independent YOLO instance, class=67)")
    if features.get("gathering") and router.is_enabled("gathering"):
        from core.events import GatheringDetector
        gathering_det = GatheringDetector(models_cfg.get("gathering", {}))
        print(f"[phase4] gathering enabled  every={router.every('gathering')}  "
              f"min_people={gathering_det.min_people}")
    if features.get("violence") and router.is_enabled("violence"):
        from core.events import ViolenceDetector
        violence_det = ViolenceDetector(models_cfg.get("violence", {}))
        print(f"[phase4] violence enabled  every={router.every('violence')}  "
              f"method=heuristic_placeholder")
    if features.get("object_left") and router.is_enabled("object_left"):
        from core.events import ObjectLeftDetector
        object_left_det = ObjectLeftDetector(models_cfg.get("object_left", {}))
        print(f"[phase4] object_left enabled  every={router.every('object_left')}  "
              f"min_stationary={object_left_det.min_stationary_s}s")

    # --- source -----------------------------------------------------------
    source: VideoSource = build_source(src_cfg)
    source.open()
    if not source.isOpened():
        raise RuntimeError("Video source failed to open")

    # --- event logging (Phase 5 foundation) ---
    event_logger = None
    if features.get("event_logging", True):
        from core.event_logger import EventLogger
        event_logger = EventLogger(
            db_path=perf.get("event_db_path", "data/events.db"),
            keyframes_dir=perf.get("keyframes_dir", "data/keyframes")
        )
        print("[phase5] event logging enabled")

    # --- logging ----------------------------------------------------------
    vram_writer = None
    vram_file = None
    if perf.get("vram_log_path"):
        opened = _open_vram_log(perf["vram_log_path"])
        if opened:
            vram_writer, vram_file = opened
    vram_log_every = float(perf.get("vram_log_every_s", 5))
    vram_warn_gb = float(perf.get("vram_warn_above_gb", 5))
    fps_warn = float(perf.get("fps_warn_below", 20))

    win = disp.get("window", "ai-surveillance")
    resize_w = disp.get("resize_w")
    show_fps = disp.get("show_fps", True)
    show_vram = disp.get("show_vram", True)

    frame_idx = 0
    fps = 0.0
    last_t = time.perf_counter()
    fps_window_t = last_t
    fps_window_n = 0
    last_vram_log = last_t

    print(f"[phase4] router={router}  source={src_cfg.get('type')}  "
          f"imgsz={detector.imgsz} half={detector.half} device={detector.device}")

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                # file EOF with loop=False or camera gone — bail
                print("[phase4] source ended (no frame).")
                break

            # --- pipeline stages: all gated by the FrameRouter --------------
            # Phase 1: detect + track every frame. track() internally runs the
            # shared YOLO pass; we don't call detect() separately to avoid a
            # double pass (constraint: "One shared detector").
            tracks = []
            if router.should_run("track", frame_idx):
                tracks = tracker.update(frame)
            elif router.should_run("detect", frame_idx):
                tracks = [  # reuse Track shape so draw code is uniform
                    __import__("core.tracker", fromlist=["Track"]).Track(
                        -1, d.cls, d.conf, d.xyxy)
                    for d in detector.detect(frame)
                ]

            # --- Motion prefilter: skip heavy stages on static frames ---
            has_motion = True  # Default to True (process everything)
            if motion_filter is not None:
                has_motion = motion_filter.has_motion(frame)

            # --- Phase 2: pose on person crops, then fall state machine ---
            poses = []
            fall_events = []
            if has_motion and pose_est is not None and router.should_run("pose", frame_idx):
                # Run pose only on person-class tracks (COCO cls == 0). If the
                # detector was configured with classes=null and the scene has
                # /no/ persons, this is empty — pose simply doesn't fire that
                # frame, which is the VRAM/compute win we want.
                person_tracks = [t for t in tracks if getattr(t, "cls", -1) == 0]
                poses = pose_est.estimate_crops(frame, person_tracks)
                if fall_det is not None:
                    fall_events = fall_det.update(
                        poses, frame_idx=frame_idx, t=time.perf_counter())
                    for ev in fall_events:
                        print(f"[phase2] FALL id={ev.track_id} "
                              f"frame={ev.frame_idx} aspect={ev.aspect_now:.2f}")

            # --- Phase 3: ReID + face recognition + identity fusion ---
            identity_events = []
            if has_motion and reid_mgr is not None and router.should_run("reid", frame_idx):
                relinks = reid_mgr.on_tracks_updated(frame, tracks)
                for m in relinks:
                    if m.matched_track_id is not None:
                        print(f"[phase3] REID re-link: track {m.new_track_id} "
                              f"-> lost track {m.matched_track_id} "
                              f"(sim={m.similarity:.2f})")
                    if identity_mgr is not None:
                        ev = identity_mgr.on_reid_relink(m, frame_idx=frame_idx)
                        if ev:
                            identity_events.append(ev)
                            print(f"[phase3] IDENTITY: track {ev.track_id} "
                                  f"= '{ev.label}' (via {ev.source})")

            if has_motion and face_rec is not None and router.should_run("face", frame_idx):
                for tr in tracks:
                    if getattr(tr, "cls", -1) != 0:
                        continue
                    tid = getattr(tr, "track_id", -1)
                    if tid < 0:
                        continue
                    x1, y1, x2, y2 = tr.xyxy
                    fx1 = max(0, int(x1)); fy1 = max(0, int(y1))
                    fx2 = min(frame.shape[1], int(x2)); fy2 = min(frame.shape[0], int(y2))
                    if fx2 - fx1 < 32 or fy2 - fy1 < 32:
                        continue
                    crop = frame[fy1:fy2, fx1:fx2]
                    matches = face_rec.process_person_crop(crop, (fx1, fy1), tid)
                    for fm in matches:
                        if fm.name is not None:
                            print(f"[phase3] FACE: track {tid} = '{fm.name}' "
                                  f"(sim={fm.similarity:.2f})")
                        if identity_mgr is not None:
                            ev = identity_mgr.on_face_match(fm, frame_idx=frame_idx)
                            if ev:
                                identity_events.append(ev)
                                # propagate the face-confirmed identity to the ReID index
                                if reid_mgr is not None:
                                    reid_mgr.set_label(tid, ev.label)
                                print(f"[phase3] IDENTITY: track {ev.track_id} "
                                      f"= '{ev.label}' (via {ev.source})")

            # --- Phase 4: fire/smoke, smoking, phone, gathering, violence, object_left ---
            phase4_events: list = []
            # Fire/smoke events can happen in static scenes (e.g., monitoring),
            # so they're not gated by motion. Others are gated.
            if fire_smoke_det is not None and router.should_run("fire_smoke", frame_idx):
                phase4_events.extend(fire_smoke_det.detect(frame, frame_idx))
            if has_motion and smoking_det is not None and router.should_run("smoking", frame_idx):
                phase4_events.extend(smoking_det.detect(frame, tracks, frame_idx))
            if has_motion and phone_det is not None and router.should_run("phone", frame_idx):
                phase4_events.extend(phone_det.detect(frame, tracks, poses, frame_idx))
            if has_motion and gathering_det is not None and router.should_run("gathering", frame_idx):
                phase4_events.extend(gathering_det.detect(tracks, frame_idx, t=time.perf_counter()))
            if has_motion and violence_det is not None and router.should_run("violence", frame_idx):
                phase4_events.extend(violence_det.detect(tracks, frame_idx, t=time.perf_counter()))
            if has_motion and object_left_det is not None and router.should_run("object_left", frame_idx):
                phase4_events.extend(object_left_det.detect(tracks, frame_idx, t=time.perf_counter()))
            for ev in phase4_events:
                print(f"[phase4] {ev.event_type} frame={ev.frame_idx} {ev.details}")

            # --- event logging (Phase 5 foundation) ---
            if event_logger is not None:
                for ev in phase4_events:
                    event_logger.log_event(ev, frame, frame_idx)
                for ev in fall_events:
                    event_logger.log_event(ev, frame, frame_idx)
                for ev in identity_events:
                    event_logger.log_event(ev, frame, frame_idx)

            # --- display ----------------------------------------------------
            if disp.get("enabled", True):
                vis = frame
                # draw tracks with identity labels if available
                for t in tracks:
                    x1, y1, x2, y2 = t.xyxy
                    c = _color_for(t.track_id)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
                    # build label: track ID + class + identity if known
                    label_parts = [f"id{t.track_id}"]
                    if 0 <= t.cls < len(COCO_NAMES):
                        label_parts.append(COCO_NAMES[t.cls])
                    if identity_mgr is not None:
                        lbl = identity_mgr.get_label(t.track_id)
                        if lbl:
                            label_parts.append(lbl)
                    label_parts.append(f"{t.conf:.2f}")
                    label = " ".join(label_parts)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), c, -1)
                    cv2.putText(vis, label, (x1 + 2, max(th, y1 - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if poses:
                    _draw_pose(vis, poses)
                if fall_events:
                    _draw_falls(vis, fall_events)
                if phase4_events:
                    _draw_phase4_events(vis, phase4_events)
                if resize_w and vis.shape[1] != resize_w:
                    scale = resize_w / vis.shape[1]
                    vis = cv2.resize(vis, (resize_w, int(vis.shape[0] * scale)))
                hud = []
                if show_fps:
                    hud.append(f"FPS:{fps:5.1f}")
                if show_vram:
                    hud.append(f"VRAM:{_vram_mb()}MB/{_vram_reserved_mb()}MB")
                hud.append(f"f:{frame_idx} n:{len(tracks)}")
                if pose_est is not None:
                    hud.append(f"p:{len(poses)}")
                if fall_events:
                    hud.append("FALL!")
                if identity_events:
                    hud.append("ID!")
                if phase4_events:
                    for ev in phase4_events:
                        hud.append(f"{ev.event_type}!")
                cv2.putText(vis, " | ".join(hud), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    print("[phase4] quit requested.")
                    break

            # --- perf bookkeeping ------------------------------------------
            now = time.perf_counter()
            fps_window_n += 1
            if now - fps_window_t >= 0.5:
                fps = fps_window_n / (now - fps_window_t)
                fps_window_t = now
                fps_window_n = 0
                if fps < fps_warn:
                    print(f"[phase4] WARN fps={fps:.1f} below target {fps_warn}")

            if vram_writer is not None and (now - last_vram_log) >= vram_log_every:
                alloc = _vram_mb()
                vram_writer.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S"), frame_idx, f"{fps:.2f}",
                    alloc, _vram_reserved_mb(),
                ])
                # csv.writer has no flush(); flush the underlying file handle so
                # a crash / SIGINT mid-run doesn't lose buffered samples.
                vram_file.flush()
                last_vram_log = now
                if alloc / 1024.0 > vram_warn_gb:
                    print(f"[phase4] WARN VRAM {alloc}MB exceeds {vram_warn_gb}GB budget")

            frame_idx += 1
    except KeyboardInterrupt:
        print("[phase4] interrupted.")
    finally:
        source.release()
        if disp.get("enabled", True):
            cv2.destroyAllWindows()
        if vram_file is not None:
            vram_file.close()
        print(f"[phase4] done. frames={frame_idx} final_fps={fps:.1f} "
              f"vram_alloc={_vram_mb()}MB reserved={_vram_reserved_mb()}MB")


def _load_extra(path: str):
    from core.config import load_yaml
    return load_yaml(path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="path to a pipeline yaml (defaults to configs/pipeline.yaml)")
    args = p.parse_args()
    run(args.config)
