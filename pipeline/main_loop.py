"""Phase 2 main loop: capture -> detect -> track -> [pose -> fall] -> display.

Phase 1's pipeline stays intact (single shared YOLO detector, ByteTrack on
top). Phase 2 adds the *optional* pose stage, gated by FrameRouter and
toggleable from configs/pipeline.yaml — pose runs on person crops only and
feeds the rule-based fall detector. Phases 3+ will plug in the same way.
"""
from __future__ import annotations

import csv
import threading
import time
import traceback
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
    from core.config import load_models_config  # always needed; don't import conditionally
    cfg = load_pipeline_config() if config_path is None else _load_extra(config_path)
    src_cfg = cfg["source"]
    disp = cfg.get("display", {})
    perf = cfg.get("perf", {})

    # ONE authoritative frame rate, read once here and passed explicitly to
    # every module that converts between seconds and frames. No module derives
    # or assumes its own.
    source_fps = float(src_cfg["fps"]) if src_cfg.get("fps") else None
    if source_fps is None:
        raise KeyError(
            "[pipeline] source.fps is missing from pipeline.yaml. Every time-based "
            "detection window (fall/violence sustain, fight buffers, object-left "
            "stationarity, VLM escalation context) is sized from it. Refusing to "
            "assume 30 fps."
        )
    print(f"[pipeline] source.fps={source_fps} (authoritative; passed to all "
          f"time-windowed modules)")

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
    pose_smoother = None
    if features.get("pose") and router.is_enabled("pose"):
        from core.pose import PoseEstimator
        from core.state_machine import FallDetector
        from core.pose_smoother import PoseSmoother
        pose_est = PoseEstimator()
        fall_cfg = (load_pipeline_config() if config_path is None else _load_extra(config_path))
        # fall thresholds live in models.yaml, not pipeline.yaml
        from core.config import load_models_config
        fall_det = FallDetector(load_models_config().get("fall", {}))
        pose_smoother = PoseSmoother(load_models_config().get("pose_smoother", {}))
        print(f"[pose] pose enabled  imgsz={pose_est.imgsz} half={pose_est.half}")
        print(f"[a] pose_smoother enabled  min_cutoff={pose_smoother._min_cutoff}  "
              f"beta={pose_smoother._beta}  gate_conf={pose_smoother._gate_conf}")
    if features.get("fall_detection") and fall_det is None:
        # fall_detection feature without pose is meaningless — warn loudly
        from core.state_machine import FallDetector
        from core.config import load_models_config
        fall_det = FallDetector(load_models_config().get("fall", {}))
        print("[pose] WARN fall_detection enabled but pose disabled — "
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
        print(f"[identity] reid enabled  model={models_cfg.get('reid', {}).get('model', 'resnet18')} "
              f"dim={reid_ext.input_size} every={router.every('reid')}")
    if features.get("face") and router.is_enabled("face"):
        from core.face import FaceRecognizer
        face_rec = FaceRecognizer(models_cfg)
        # load the persistent face index if it exists
        idx_path = models_cfg.get("face", {}).get("index_path", "models/face_index")
        face_rec.load_index(idx_path)
        print(f"[identity] face enabled  pack={models_cfg.get('face', {}).get('model_pack', 'buffalo_s')} "
              f"enrolled={len(face_rec._labels)} every={router.every('face')}")
    if features.get("identity_fusion") and (reid_mgr is not None or face_rec is not None):
        from core.identity import IdentityManager
        identity_mgr = IdentityManager()
        print("[identity] identity fusion enabled")

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
        print(f"[pipeline] fire_smoke enabled  every={router.every('fire_smoke')}  "
              f"method={'hsv_heuristic' if models_cfg.get('fire_smoke', {}).get('weights') is None else 'yolo'}")
    if features.get("smoking") and router.is_enabled("smoking"):
        from core.events import SmokingDetector
        smoking_det = SmokingDetector(models_cfg.get("smoking", {}))
        print(f"[pipeline] smoking enabled  every={router.every('smoking')}  "
              f"method={'glow_heuristic' if models_cfg.get('smoking', {}).get('weights') is None else 'yolo'}")
    if features.get("phone") and router.is_enabled("phone"):
        from core.events import PhoneWatcherDetector
        # PhoneWatcherDetector always loads its own independent YOLOv8n instance
        # (COCO class 67). detector_model kwarg is accepted but always ignored.
        phone_det = PhoneWatcherDetector(models_cfg.get("phone", {}))
        print(f"[pipeline] phone enabled  every={router.every('phone')}  imgsz={phone_det.imgsz}  "
              f"conf={phone_det.conf}  (independent YOLO instance, class=67)")
    if features.get("gathering") and router.is_enabled("gathering"):
        from core.events import GatheringDetector
        gathering_det = GatheringDetector(models_cfg.get("gathering", {}))
        print(f"[pipeline] gathering enabled  every={router.every('gathering')}  "
              f"min_people={gathering_det.min_people}")
    if features.get("violence") and router.is_enabled("violence"):
        from core.events import ViolenceDetector
        violence_det = ViolenceDetector(models_cfg.get("violence", {}))
        print(f"[pipeline] violence enabled  every={router.every('violence')}  "
              f"method={violence_det.method}")

    # Phase 5D: skeleton-based fight detector
    fight_det = None
    clip_writer = None
    if features.get("fight") and router.is_enabled("fight"):
        from core.fight_detector import FightDetector
        fight_det = FightDetector(models_cfg.get("fight", {}), fps=source_fps)
        print(f"[d] fight_detector enabled  every={router.every('fight')}  "
              f"proximity_px={fight_det._proximity_px}")
    clip_cfg = cfg.get("clip_writer", {})
    if clip_cfg.get("enabled", True) and fight_det is not None:
        from core.clip_writer import ClipWriter
        clip_writer = ClipWriter(clip_cfg, fps=source_fps)
        print(f"[d] clip_writer enabled  buffer_s={clip_writer._buf_s}s  "
              f"dir={clip_writer._clips_dir}")

    if features.get("object_left") and router.is_enabled("object_left"):
        from core.events import ObjectLeftDetector
        object_left_det = ObjectLeftDetector(models_cfg.get("object_left", {}), fps=source_fps)
        print(f"[pipeline] object_left enabled  every={router.every('object_left')}  "
              f"min_stationary={object_left_det.min_stationary_s}s")

    # Phase 5B: Pedestrian Attribute Recognition (PAR)
    par_det = None
    par_agg = None
    if features.get("par") and router.is_enabled("par"):
        from core.par import PARDetector
        from core.par_aggregator import PARAggregator
        par_det = PARDetector(models_cfg.get("par", {}))
        par_agg = PARAggregator(window_size=models_cfg.get("par", {}).get("window_frames", 15))
        print(f"[b] par enabled  every={router.every('par')}")

    # Phase 6: VLM Layer — escalated inference only (no ambient pass).
    # process_frame() returns None on non-escalated frames (honest about not
    # running inference) and VLMOutput on escalated frames.
    vlm_integration = None
    # Background VLM thread — VLM inference takes 3-8s; running on main thread
    # drops FPS to ~0 for that duration.  We submit frames to a daemon thread and
    # read results back each frame — zero FPS cost when VLM is not running.
    _vlm_thread: threading.Thread | None = None
    _vlm_result: list[Any] = []          # [VLMOutput, frame] when done, [] while running
    _vlm_lock = threading.Lock()
    _vlm_last_output: list[Any] = []     # [most_recent_output] or []
    _vlm_scene_text: list[str] = [""]   # [human-readable scene description] — list for mutability

    if features.get("vlm", False):
        try:
            from vlm import VLMIntegration
            vlm_integration = VLMIntegration()
            # Set capture resolution so the warmup pass exercises the true
            # frame size, not a hardcoded default.
            _sw, _sh = src_cfg.get("width"), src_cfg.get("height")
            if _sw and _sh:
                vlm_integration.set_frame_size(int(_sw), int(_sh))
            # Pass the authoritative events.db path (single source of truth
            # from pipeline.yaml) so VLM events land in the same DB.
            _db_path = perf.get("event_db_path", "data/events.db")
            vlm_ok = vlm_integration.initialize(event_db_path=_db_path)
            if vlm_ok:
                print(
                    f"[vlm] VLM layer enabled  mode=escalated_only  "
                    f"model={vlm_integration.vlm_manager.config.model_name}  "
                    f"vlm_events_db={_db_path}"
                )
            else:
                print("[vlm] WARN: VLM initialization failed — running without VLM layer")
                vlm_integration = None
        except Exception as e:
            import traceback
            print(f"[vlm] WARN: VLM not available: {e}")
            traceback.print_exc()
            vlm_integration = None

    # --- Bitchat mesh integration ------------------------------------------
    bitchat_client = None
    try:
        from core.bitchat import build_client_from_config
        bitchat_client = build_client_from_config(cfg)
    except Exception as _bc_e:
        print(f"[bitchat] WARN: could not initialise Bitchat client: {_bc_e}")

    _bc_send_scene     = cfg.get("bitchat", {}).get("send_scene", True)

    source: VideoSource = build_source(src_cfg)
    source.open()
    if not source.isOpened():
        raise RuntimeError("Video source failed to open")

    # --- event logging (Phase 5 foundation — SQLite per-event) ---
    event_logger = None
    if features.get("event_logging", True):
        from core.event_logger import EventLogger
        event_logger = EventLogger(
            db_path=perf.get("event_db_path", "data/events.db"),
            keyframes_dir=perf.get("keyframes_dir", "data/keyframes")
        )
        print("[] event logging enabled (SQLite)")

    # --- event buffer (Phase 5F — windowed JSON flush) ---
    event_buffer = None
    if features.get("event_buffer", True):
        from core.event_buffer import EventBuffer
        out_cfg = cfg.get("output", {})
        event_buffer = EventBuffer(out_cfg, fps=source_fps)
        print(f"[f] event_buffer enabled  flush={event_buffer._flush_interval}s  "
              f"dir={event_buffer._json_dir}")

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
    max_frames = int(perf.get("max_frames", 0))   # 0 = unlimited

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

    print(f"[pipeline] router={router}  source={src_cfg.get('type')}  "
          f"imgsz={detector.imgsz} half={detector.half} device={detector.device}")

    # --- per-stage failure isolation ------------------------------------
    # A single stage raising must not kill the whole run. Every failure is
    # printed loudly with the stage name and traceback; after MAX_STAGE_FAILURES
    # the stage is disabled for the rest of the run so the log isn't flooded
    # with the same traceback every frame. A disabled stage means its output is
    # MISSING, not clean — that is stated explicitly at disable time and again
    # in the shutdown summary, so a partial run is never mistaken for a good one.
    # Model-load errors still abort at startup; this covers per-frame execution only.
    MAX_STAGE_FAILURES = 3
    _stage_failures: dict[str, int] = {}
    _disabled_stages: set[str] = set()

    def stage(name, fn, *args, default=None, **kwargs):
        """Run one per-frame pipeline stage with failure isolation."""
        if name in _disabled_stages:
            return default
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            n = _stage_failures.get(name, 0) + 1
            _stage_failures[name] = n
            print(f"[pipeline] STAGE '{name}' FAILED "
                  f"({n}/{MAX_STAGE_FAILURES}) at frame {frame_idx}: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
            if n >= MAX_STAGE_FAILURES:
                _disabled_stages.add(name)
                print(f"[pipeline] *** STAGE '{name}' DISABLED after {n} failures. "
                      f"Its detections are ABSENT for the rest of this run. "
                      f"Results from here on are INCOMPLETE, not clean. ***")
            return default

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                # file EOF with loop=False or camera gone — bail
                print("[pipeline] source ended (no frame).")
                break

            # --- pipeline stages: all gated by the FrameRouter --------------
            # Phase 1: detect + track every frame. track() internally runs the
            # shared YOLO pass; we don't call detect() separately to avoid a
            # double pass (constraint: "One shared detector").
            tracks = []
            if router.should_run("track", frame_idx):
                tracks = stage("track", tracker.update, frame, default=[])
            elif router.should_run("detect", frame_idx):
                tracks = stage("detect", lambda: [  # reuse Track shape so draw code is uniform
                    __import__("core.tracker", fromlist=["Track"]).Track(
                        -1, d.cls, d.conf, d.xyxy)
                    for d in detector.detect(frame)
                ], default=[])

            # --- Motion prefilter: skip heavy stages on static frames ---
            has_motion = True  # Default to True (process everything)
            if motion_filter is not None:
                has_motion = stage("motion", motion_filter.has_motion, frame, default=True)

            # --- Phase 2: pose on person crops, then fall state machine ---
            poses = []
            fall_events = []
            if has_motion and pose_est is not None and router.should_run("pose", frame_idx):
                # Run pose only on person-class tracks (COCO cls == 0). If the
                # detector was configured with classes=null and the scene has
                # /no/ persons, this is empty — pose simply doesn't fire that
                # frame, which is the VRAM/compute win we want.
                def _pose_stage():
                    person_tracks = [t for t in tracks if getattr(t, "cls", -1) == 0]
                    raw_poses = pose_est.estimate_crops(frame, person_tracks)
                    # Phase 5A: smooth keypoints before feeding downstream
                    if pose_smoother is not None:
                        return pose_smoother.update(raw_poses, ema_tracks=person_tracks)
                    return raw_poses

                poses = stage("pose", _pose_stage, default=[])
                if fall_det is not None:
                    def _fall_stage():
                        evs = fall_det.update(
                            poses, frame_idx=frame_idx, t=time.perf_counter())
                        for ev in evs:
                            print(f"[pose] FALL id={ev.track_id} "
                                  f"frame={ev.frame_idx} aspect={ev.aspect_now:.2f}")
                        return evs

                    fall_events = stage("fall", _fall_stage, default=[])

            # --- Phase 3: ReID + face recognition + identity fusion ---
            identity_events = []
            if has_motion and reid_mgr is not None and router.should_run("reid", frame_idx):
                def _reid_stage():
                    evs = []
                    relinks = reid_mgr.on_tracks_updated(frame, tracks)
                    for m in relinks:
                        if m.matched_track_id is not None:
                            print(f"[identity] REID re-link: track {m.new_track_id} "
                                  f"-> lost track {m.matched_track_id} "
                                  f"(sim={m.similarity:.2f})")
                        if identity_mgr is not None:
                            ev = identity_mgr.on_reid_relink(m, frame_idx=frame_idx)
                            if ev:
                                evs.append(ev)
                                print(f"[identity] IDENTITY: track {ev.track_id} "
                                      f"= '{ev.label}' (via {ev.source})")
                    return evs

                identity_events.extend(stage("reid", _reid_stage, default=[]))

                # Phase 5.2: Periodic rebuild_index to prevent unbounded FAISS growth.
                # IndexFlatIP has no delete; without this the index accumulates stale
                # lost-track embeddings forever. Every 300 frames ≈ 10s at 30fps.
                if frame_idx > 0 and frame_idx % 300 == 0:
                    try:
                        before = reid_mgr.index.index.ntotal
                        reid_mgr.index.rebuild_index()
                        after = reid_mgr.index.index.ntotal
                        if before != after:
                            print(f"[reid] rebuild_index: {before} -> {after} entries")
                    except Exception as _e:
                        print(f"[reid] WARN: rebuild_index failed: {_e}")


            if has_motion and face_rec is not None and router.should_run("face", frame_idx):
                def _face_stage():
                    evs = []
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
                                print(f"[identity] FACE: track {tid} = '{fm.name}' "
                                      f"(sim={fm.similarity:.2f})")
                            if identity_mgr is not None:
                                ev = identity_mgr.on_face_match(fm, frame_idx=frame_idx)
                                if ev:
                                    evs.append(ev)
                                    # propagate the face-confirmed identity to the ReID index
                                    if reid_mgr is not None:
                                        reid_mgr.set_label(tid, ev.label)
                                    print(f"[identity] IDENTITY: track {ev.track_id} "
                                          f"= '{ev.label}' (via {ev.source})")
                    return evs

                identity_events.extend(stage("face", _face_stage, default=[]))

            # --- Phase 4: fire/smoke, smoking, phone, gathering, violence, object_left ---
            phase4_events: list = []
            # Fire/smoke events can happen in static scenes (e.g., monitoring),
            # so they're not gated by motion. Others are gated.
            if fire_smoke_det is not None and router.should_run("fire_smoke", frame_idx):
                phase4_events.extend(stage("fire_smoke", fire_smoke_det.detect,
                                           frame, frame_idx, default=[]))
            if has_motion and smoking_det is not None and router.should_run("smoking", frame_idx):
                phase4_events.extend(stage("smoking", smoking_det.detect,
                                           frame, tracks, poses, frame_idx, default=[]))
            if has_motion and phone_det is not None and router.should_run("phone", frame_idx):
                phase4_events.extend(stage("phone", phone_det.detect,
                                           frame, tracks, poses, frame_idx, default=[]))
            if has_motion and gathering_det is not None and router.should_run("gathering", frame_idx):
                phase4_events.extend(stage("gathering", gathering_det.detect,
                                           tracks, frame_idx, t=time.perf_counter(), default=[]))
            if has_motion and violence_det is not None and router.should_run("violence", frame_idx):
                phase4_events.extend(stage("violence", violence_det.detect,
                                           tracks, frame_idx, t=time.perf_counter(),
                                           frame=frame, default=[]))
            if has_motion and object_left_det is not None and router.should_run("object_left", frame_idx):
                phase4_events.extend(stage("object_left", object_left_det.detect,
                                           tracks, frame_idx, t=time.perf_counter(), default=[]))
            for ev in phase4_events:
                print(f"[pipeline] {ev.event_type} frame={ev.frame_idx} {ev.details}")

            # --- Phase 5D: skeleton-based fight detection ---
            fight_events: list = []
            if has_motion and fight_det is not None and router.should_run("fight", frame_idx):
                def _fight_stage():
                    evs = fight_det.detect(
                        smooth_poses=poses, tracks=tracks,
                        frame_idx=frame_idx, t=time.perf_counter()
                    )
                    for ev in evs:
                        if clip_writer is not None:
                            clip_path = clip_writer.flush_for_event(ev)
                            if clip_path:
                                print(f"[d] FIGHT clip saved: {clip_path}")
                        print(f"[d] FIGHT tracks={ev.track_ids} "
                              f"conf={ev.confidence:.2f} frame={ev.frame_idx}")
                    return evs

                fight_events = stage("fight", _fight_stage, default=[])

            # --- Phase 5D: ClipWriter ring-buffer push (every frame) ---
            if clip_writer is not None:
                clip_writer.push(frame)

            # --- Phase 5B: Pedestrian Attribute Recognition (PAR) ---
            if has_motion and par_det is not None and router.should_run("par", frame_idx):
                def _par_stage():
                    active_tids = set()
                    for tr in tracks:
                        if getattr(tr, "cls", -1) != 0:
                            continue
                        tid = getattr(tr, "track_id", -1)
                        if tid < 0:
                            continue
                        active_tids.add(tid)

                        x1, y1, x2, y2 = tr.xyxy
                        w = x2 - x1
                        h = y2 - y1
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        # Scale by 1.3 for loose crop context
                        nw = int(w * 1.3)
                        nh = int(h * 1.3)
                        nx1 = max(0, int(cx - nw / 2))
                        ny1 = max(0, int(cy - nh / 2))
                        nx2 = min(frame.shape[1], int(cx + nw / 2))
                        ny2 = min(frame.shape[0], int(cy + nh / 2))

                        if nx2 - nx1 >= 16 and ny2 - ny1 >= 16:
                            crop = frame[ny1:ny2, nx1:nx2]
                            raw_attrs = par_det.process_crop(crop, tid)
                            if par_agg is not None:
                                par_agg.update(tid, raw_attrs)

                    # Prune old tracks from the aggregator
                    if par_agg is not None:
                        par_agg.prune(active_tids)

                stage("par", _par_stage)

            # --- Phase 6: VLM escalated inference (background thread) ---
            # Check if a previous VLM thread has finished and collect its result.
            vlm_output = None
            vlm_frame = None
            if vlm_integration is not None:
                with _vlm_lock:
                    if _vlm_result:
                        vlm_output, vlm_frame = _vlm_result.pop(0)
                        if len(_vlm_last_output):
                            _vlm_last_output[0] = vlm_output
                        else:
                            _vlm_last_output.append(vlm_output)
                        if vlm_output is not None:
                            ents  = vlm_output.detected_entities or []
                            scene = getattr(vlm_output, "scene_description", "")
                            
                            if vlm_output.is_escalated:
                                print(
                                    f"[vlm] ESCALATED frame={vlm_output.frame_idx}  "
                                    f"reason={vlm_output.reasoning!r}  "
                                    f"entities={len(ents)}  conf={vlm_output.confidence:.2f}"
                                )
                                if scene:
                                    print(f"[vlm] \U0001f4f8 SCENE: {scene}")
                                if ents:
                                    for e in ents:
                                        print(
                                            f"  \u2192 [{e.get('type','?').upper()}] "
                                            f"conf={e.get('confidence',0):.0%}  "
                                            f"{e.get('description','')}"
                                        )
                            
                            # Use real scene description for the on-screen overlay.
                            if scene:
                                _vlm_scene_text[0] = scene
                            elif ents:
                                _vlm_scene_text[0] = "; ".join(
                                    f"{e.get('type','?')}: {e.get('description','')}"
                                    for e in ents
                                )
                            else:
                                _vlm_scene_text[0] = "VLM: nothing detected"
                                if vlm_output.is_escalated:
                                    print("  \u2192 (no entities detected by VLM)")
                            
                            # Send to BitChat immediately with the frame that was analyzed
                            if bitchat_client is not None and _bc_send_scene:
                                if scene:
                                    print(f"[bitchat] VLM scene ready: {scene[:60]}...")
                                    if vlm_frame is not None:
                                        bitchat_client.send_scene(scene, vlm_frame)
                                        print(f"[bitchat] Sent VLM scene to BitChat with frame")
                                    else:
                                        bitchat_client.send_scene(scene, frame)
                                        print(f"[bitchat] Sent VLM scene to BitChat (fallback frame)")
                                else:
                                    print(f"[bitchat] No VLM scene to send")


                # Submit a new VLM inference in background if the router says to
                # AND no thread is currently running.
                if (router.should_run("vlm_escalated", frame_idx)
                        and (_vlm_thread is None or not _vlm_thread.is_alive())):
                    all_events_this_frame = (
                        phase4_events + fall_events + identity_events + fight_events
                    )
                    _frame_snap = frame.copy()   # copy so thread doesn't see a later frame
                    _fidx_snap  = frame_idx

                    def _vlm_worker(f=_frame_snap, fi=_fidx_snap, evts=all_events_this_frame):
                        try:
                            out = vlm_integration.process_frame(f, fi, evts, tracks)
                            with _vlm_lock:
                                _vlm_result.append((out, f))
                        except Exception as _e:
                            print(f"[vlm] background thread error: {_e}")
                            with _vlm_lock:
                                _vlm_result.append((None, None))

                    _vlm_thread = threading.Thread(target=_vlm_worker, daemon=True)
                    _vlm_thread.start()

            # --- event logging (Phase 5 foundation — SQLite per-event) ---
            if event_logger is not None:
                for ev in phase4_events:
                    event_logger.log_event(ev, frame, frame_idx)
                for ev in fall_events:
                    event_logger.log_event(ev, frame, frame_idx)
                for ev in identity_events:
                    event_logger.log_event(ev, frame, frame_idx)
                for ev in fight_events:
                    event_logger.log_event(ev, frame, frame_idx)

            # --- Bitchat mesh alerts -----------------------------------------
            # No detector event logs are sent (the user only wants the image +
            # VLM description, handled in the VLM result block above).  This
            # also keeps the phone's 5s API rate window free so the image and
            # description are not silently dropped by Bitchat.

            # --- Phase 5F: EventBuffer — aggregate + windowed JSON flush ---
            if event_buffer is not None:
                # Update per-track state for all visible person tracks
                for tr in tracks:
                    if getattr(tr, "cls", -1) != 0:
                        continue
                    tid = getattr(tr, "track_id", -1)
                    if tid < 0:
                        continue
                    # Get pose_quality for this track if we have smoother
                    pq = "missing"
                    if pose_smoother is not None:
                        pq = pose_smoother.get_quality(tid)

                    # Look up aggregated attributes for this track
                    attrs = {}
                    if par_agg is not None:
                        attrs = par_agg.get_attributes(tid)

                    event_buffer.update_track(
                        track_id=tid, bbox=tr.xyxy,
                        pose_quality=pq, attributes=attrs,
                        frame_idx=frame_idx
                    )
                # Append all events from this frame
                for ev in phase4_events:
                    event_buffer.append(ev, frame_idx=frame_idx)
                for ev in fall_events:
                    event_buffer.append(ev, frame_idx=frame_idx)
                for ev in identity_events:
                    event_buffer.append(ev, frame_idx=frame_idx)
                for ev in fight_events:
                    event_buffer.append(ev, frame_idx=frame_idx)
                # Timed JSON flush
                flushed = event_buffer.maybe_flush(frame_idx=frame_idx)
                if flushed is not None:
                    n_tracks = len(flushed.get("tracks", []))
                    n_scene  = len(flushed.get("scene_events", []))
                    print(f"[f] JSON flushed: {n_tracks} tracks, "
                          f"{n_scene} scene events")

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
                if fight_events:
                    for ev in fight_events:
                        hud.append(f"FIGHT({ev.track_ids[0]},{ev.track_ids[1]})!")
                # VLM status in HUD (Phase 6)
                if vlm_integration is not None:
                    vlm_stats = vlm_integration.get_stats()
                    is_running = _vlm_thread is not None and _vlm_thread.is_alive()
                    if is_running:
                        hud.append("VLM:thinking...")
                    elif vlm_output and vlm_output.is_escalated:
                        hud.append("VLM!")
                    # Draw VLM scene description at bottom of frame
                    if _vlm_scene_text[0]:
                        scene_lines = [_vlm_scene_text[0][i:i+80]
                                       for i in range(0, min(len(_vlm_scene_text[0]), 240), 80)]
                        frame_h = vis.shape[0]
                        for li, line in enumerate(scene_lines):
                            y_pos = frame_h - 12 - (len(scene_lines) - 1 - li) * 18
                            cv2.rectangle(vis,
                                          (0, y_pos - 14),
                                          (min(len(line) * 9 + 6, vis.shape[1]), y_pos + 4),
                                          (0, 0, 0), -1)
                            cv2.putText(vis, line, (4, y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)
                    vram_vlm = vlm_stats.get("vlm", {}).get("vram_mb", 0)
                    if vram_vlm > 0:
                        hud.append(f"VLM:{vram_vlm}MB")
                # PAR attributes on track labels (Phase 5B visual debug)
                if par_agg is not None:
                    for t in tracks:
                        if getattr(t, "cls", -1) != 0:
                            continue
                        tid = getattr(t, "track_id", -1)
                        if tid < 0:
                            continue
                        attrs = par_agg.get_attributes(tid)
                        if attrs:
                            uc = attrs.get("upper_color", "")
                            lc = attrs.get("lower_color", "")
                            g  = attrs.get("gender", "")[:1].upper()
                            par_label = f"{g} {uc}/{lc}"
                            x1, y1, x2, y2 = t.xyxy
                            cv2.putText(vis, par_label, (x1, y2 + 14),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1)
                cv2.putText(vis, " | ".join(hud), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    print("[pipeline] quit requested.")
                    break

            # --- perf bookkeeping ------------------------------------------
            now = time.perf_counter()
            fps_window_n += 1
            if now - fps_window_t >= 0.5:
                fps = fps_window_n / (now - fps_window_t)
                fps_window_t = now
                fps_window_n = 0
                if fps < fps_warn:
                    print(f"[pipeline] WARN fps={fps:.1f} below target {fps_warn}")

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
                    print(f"[pipeline] WARN VRAM {alloc}MB exceeds {vram_warn_gb}GB budget")

            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                print(f"[pipeline] max_frames={max_frames} reached, stopping.")
                break
    except KeyboardInterrupt:
        print("[pipeline] interrupted.")
    finally:
        source.release()
        
        if disp.get("enabled", True):
            cv2.destroyAllWindows()
        
        if vram_file is not None:
            vram_file.close()
        
        if event_logger is not None:
            event_logger.close()
        
        if face_rec is not None:
            idx_path = models_cfg.get("face", {}).get("index_path", "models/face_index")
            try:
                face_rec.save_index(idx_path)
            except Exception as e:
                print(f"[face] WARNING: Failed to save face index: {e}")
        
        if vlm_integration is not None:
            vlm_stats = vlm_integration.get_stats()
            print(
                f"[vlm] Stats: {vlm_integration._frame_count} frames processed, "
                f"{vlm_stats.get('vlm', {}).get('escalation_count', 0)} escalations"
            )
            vlm_integration.close()
        
        if reid_mgr is not None and hasattr(reid_mgr, 'index'):
            try:
                reid_mgr.index.rebuild_index()
            except Exception as e:
                print(f"[reid] WARNING: Failed to rebuild index: {e}")
        
        if _stage_failures:
            print("[pipeline] *** THIS RUN WAS DEGRADED — stage failures: ***")
            for _name, _n in sorted(_stage_failures.items()):
                _state = "DISABLED" if _name in _disabled_stages else "recovered"
                print(f"[pipeline]     {_name}: {_n} failure(s) -> {_state}")
            if _disabled_stages:
                print(f"[pipeline] *** Output is INCOMPLETE. Disabled stages produced "
                      f"NO detections: {sorted(_disabled_stages)} ***")

        print(f"[pipeline] done. frames={frame_idx} final_fps={fps:.1f} "
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
