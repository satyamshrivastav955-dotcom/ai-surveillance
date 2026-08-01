"""Event detectors — fire/smoke, smoking, phone-watching, gathering, violence, object-left.

All detectors emit generic Event objects (VLM-agnostic):
  1. FireSmokeDetector     — YOLO fine-tuned on D-Fire for fire/smoke detection
  2. SmokingDetector       — YOLO fine-tuned for cigarette/vape detection
  3. PhoneWatcherDetector  — YOLO class 67 (cell phone) + head-pose heuristic
  4. GatheringDetector     — Fixed-radius clustering on track centroids
  5. ViolenceDetector      — Rule-based: bbox overlap + rapid motion (placeholder)
  6. ObjectLeftDetector    — Track stationary non-person objects (bags, backpacks)

All events inherit from BaseEvent for consistent interface.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BaseEvent:
    """Base class for all events. Provides consistent interface."""
    event_type: str
    t_iso: str
    frame_idx: int
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event_type,
            "t_iso": self.t_iso,
            "frame_idx": self.frame_idx,
            **self.details,
        }

    @property
    def confidence(self) -> float:
        return self.details.get("confidence", 0.5)


@dataclass  
class Event(BaseEvent):
    """Generic event from any detector."""
    pass


# =============================================================================
# 1. Fire/Smoke Detection — color heuristic placeholder
# =============================================================================

class FireSmokeDetector:
    """Detects fire and smoke using YOLO when trained weights are available.

    YOLO MODE (weights path set in config):
      Uses YOLOv8n fine-tuned on D-Fire dataset for accurate fire/smoke detection.
      Classes: 0=smoke, 1=fire (rabahdev/fire-smoke-yolov8n from HuggingFace).

    HSV FALLBACK (weights=null):
      Uses HSV color thresholding as a documented placeholder.
      CONFIRMED UNRELIABLE via live testing (see PROGRESS.md) — false-positives
      AND false-negatives both observed. Not recommended for production.

    Config (`fire_smoke` in models.yaml):
      weights: path to fine-tuned .pt, or null for HSV fallback
      fire_hsv_low/high: HSV bounds for fire color (fallback only)
      smoke_hsv_low/high: HSV bounds for smoke color (fallback only)
      fire_min_pixel_ratio: fraction of frame matching fire color (default 0.01)
      smoke_min_pixel_ratio: fraction of frame matching smoke color (default 0.60)
      fire_min_duration: consecutive fire detections required (default 2)
      conf: YOLO confidence threshold (default 0.25)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("fire_smoke")
        f = self.cfg
        
        # YOLO model path
        self.weights = f.get("weights")
        self.model = None
        self.conf = float(f.get("conf", 0.45))
        self.imgsz = int(f.get("imgsz", 640))
        self._use_yolo = False

        # Multi-frame confirmation: require detection in >= min_consecutive_frames
        # of the last 5 relevant frames before emitting an event. Kills single-frame
        # false positives from bright backgrounds, reflections, and windows.
        self._min_consecutive = int(f.get("min_consecutive_frames", 2))
        from collections import deque
        self._fire_window:  deque = deque(maxlen=5)   # True/False per relevant frame
        self._smoke_window: deque = deque(maxlen=5)
        
        # Try loading YOLO model if weights path provided
        if self.weights is not None:
            try:
                from pathlib import Path
                from ultralytics import YOLO
                weights_path = Path(self.weights)
                if weights_path.exists():
                    self.model = YOLO(str(weights_path))
                    self.model.fuse()
                    # Warmup
                    dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                    self.model.predict(dummy, imgsz=self.imgsz, device=0, half=True,
                                       conf=self.conf, verbose=False)
                    self._use_yolo = True
                    print(f"[phase4] fire_smoke: loaded YOLO model from {self.weights}  "
                          f"classes=[smoke, fire]  conf={self.conf}  "
                          f"min_consecutive={self._min_consecutive}")
                else:
                    print(f"[phase4] WARN: fire_smoke.weights={self.weights} not found, "
                          f"falling back to HSV heuristic")
            except Exception as e:
                print(f"[phase4] WARN: failed to load fire/smoke YOLO model: {e}")
                print(f"[phase4] WARN: falling back to HSV heuristic")
        
        # HSV fallback params (used only when YOLO not available)
        self.fire_low = np.array(f.get("fire_hsv_low", [0, 100, 150]), dtype=np.uint8)
        self.fire_high = np.array(f.get("fire_hsv_high", [25, 255, 255]), dtype=np.uint8)
        self.smoke_low = np.array(f.get("smoke_hsv_low", [0, 0, 140]), dtype=np.uint8)
        self.smoke_high = np.array(f.get("smoke_hsv_high", [180, 30, 220]), dtype=np.uint8)
        self.fire_min_ratio = float(f.get("fire_min_pixel_ratio", 0.01))
        self.smoke_min_ratio = float(f.get("smoke_min_pixel_ratio", 0.60))
        self.fire_min_duration = int(f.get("fire_min_duration", 2))
        # duration tracking (HSV fallback)
        self._fire_consecutive = 0
        self._fire_last_bbox: tuple | None = None

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Event]:
        events: list[Event] = []
        
        if self._use_yolo and self.model is not None:
            return self._detect_yolo(frame, frame_idx)
        else:
            return self._detect_hsv(frame, frame_idx)
    
    def _detect_yolo(self, frame: np.ndarray, frame_idx: int) -> list[Event]:
        """Fire/smoke detection using YOLOv8n fine-tuned on D-Fire.

        Multi-frame confirmation: raw YOLO candidates are collected every call.
        A FIRE or SMOKE event is only emitted once the rolling 5-frame window
        contains >= min_consecutive_frames detections of that class. This
        eliminates single-frame false positives from bright backgrounds.
        """
        events: list[Event] = []
        res = self.model.predict(frame, imgsz=self.imgsz, device=0, half=True,
                                 conf=self.conf, verbose=False)[0]

        # Collect raw candidates this frame
        raw_fire:  list[tuple] = []   # (x1,y1,x2,y2, conf)
        raw_smoke: list[tuple] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy   = res.boxes.xyxy.cpu().numpy().astype(int)
            confs  = res.boxes.conf.cpu().numpy()
            clsids = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, clsid in zip(xyxy, confs, clsids):
                if clsid == 0:
                    raw_smoke.append((int(x1), int(y1), int(x2), int(y2), float(c)))
                else:
                    raw_fire.append((int(x1), int(y1), int(x2), int(y2), float(c)))

        # Update rolling detection windows
        self._smoke_window.append(len(raw_smoke) > 0)
        self._fire_window.append(len(raw_fire) > 0)

        # Emit events only when window has enough confirmations
        if sum(self._smoke_window) >= self._min_consecutive and raw_smoke:
            # Use highest-confidence detection as the representative bbox
            best = max(raw_smoke, key=lambda t: t[4])
            x1, y1, x2, y2, c = best
            events.append(Event(
                event_type="SMOKE",
                t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                frame_idx=frame_idx,
                details={"bbox": (x1, y1, x2, y2),
                         "confidence": round(c, 3),
                         "method": "yolov8n_dfire",
                         "confirmed_frames": int(sum(self._smoke_window))},
            ))
        if sum(self._fire_window) >= self._min_consecutive and raw_fire:
            best = max(raw_fire, key=lambda t: t[4])
            x1, y1, x2, y2, c = best
            events.append(Event(
                event_type="FIRE",
                t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                frame_idx=frame_idx,
                details={"bbox": (x1, y1, x2, y2),
                         "confidence": round(c, 3),
                         "method": "yolov8n_dfire",
                         "confirmed_frames": int(sum(self._fire_window))},
            ))
        return events
    
    def _detect_hsv(self, frame: np.ndarray, frame_idx: int) -> list[Event]:
        """Fire/smoke detection using HSV color thresholding (fallback)."""
        import cv2
        events: list[Event] = []
        h, w = frame.shape[:2]
        total_pixels = h * w
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # --- fire mask ---
        fire_mask = cv2.inRange(hsv, self.fire_low, self.fire_high)
        fire_ratio = float(cv2.countNonZero(fire_mask)) / total_pixels
        if fire_ratio >= self.fire_min_ratio:
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest)
                self._fire_consecutive += 1
                self._fire_last_bbox = (int(x), int(y), int(x+bw), int(y+bh))
                # only fire after min_duration consecutive detections
                if self._fire_consecutive >= self.fire_min_duration:
                    events.append(Event(
                        event_type="FIRE",
                        t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        frame_idx=frame_idx,
                        details={"bbox": self._fire_last_bbox,
                                 "pixel_ratio": round(fire_ratio, 4),
                                 "consecutive": self._fire_consecutive,
                                 "method": "hsv_heuristic_fallback"},
                    ))
        else:
            # reset consecutive counter when fire is not seen
            self._fire_consecutive = 0
            self._fire_last_bbox = None
        # --- smoke mask (tightened) ---
        smoke_mask = cv2.inRange(hsv, self.smoke_low, self.smoke_high)
        smoke_ratio = float(cv2.countNonZero(smoke_mask)) / total_pixels
        if smoke_ratio >= self.smoke_min_ratio:
            contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest)
                events.append(Event(
                    event_type="SMOKE",
                    t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    frame_idx=frame_idx,
                    details={"bbox": (int(x), int(y), int(x+bw), int(y+bh)),
                             "pixel_ratio": round(smoke_ratio, 4),
                             "method": "hsv_heuristic_fallback"},
                ))
        return events


# =============================================================================
# 2. Smoking Detection — placeholder heuristic
# =============================================================================

class SmokingDetector:
    """Detects smoking using YOLO when trained weights are available.

    YOLO MODE (weights path set in config):
      Uses YOLOv8n fine-tuned for cigarette/vape detection.
      Classes: Smoke (0), Person (1), Cigarette (2), Vape (3)
      Model source: cadilak/smoking-detection-yolov8 (HuggingFace)

    HSV FALLBACK (weights=null):
      Uses glow heuristic near face/hand region.
      ROUGH PLACEHOLDER — bright reflections can trigger false positives.

    Config (`smoking` in models.yaml):
      weights: path to fine-tuned .pt, or null for HSV fallback
      conf: YOLO confidence threshold (default 0.25)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("smoking")
        
        # YOLO model path
        self.weights = self.cfg.get("weights")
        self.model = None
        self.conf = float(self.cfg.get("conf", 0.25))
        self.imgsz = int(self.cfg.get("imgsz", 320))
        self._use_yolo = False
        self._using_hsv_fallback = False
        
        # Try loading YOLO model if weights path provided
        if self.weights is not None:
            try:
                from pathlib import Path
                from ultralytics import YOLO
                weights_path = Path(self.weights)
                if weights_path.exists():
                    self.model = YOLO(str(weights_path))
                    self.model.fuse()
                    # Warmup
                    dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                    self.model.predict(dummy, imgsz=self.imgsz, device=0, half=True,
                                       conf=self.conf, verbose=False)
                    self._use_yolo = True
                    print(f"[phase4] smoking: loaded YOLO model from {self.weights}  "
                          f"classes=[Cigarette, Vape]  conf={self.conf}")
                else:
                    self._using_hsv_fallback = True
                    print(
                        f"[phase4] *** WARNING: smoking.weights={self.weights} not found. "
                        "SmokingDetector is operating in HSV-glow FALLBACK mode. "
                        "Output is DEGRADED: bright reflections cause false positives, "
                        "actual cigarettes in indirect light will be missed. "
                        "This is a rough placeholder, not a production detector. ***"
                    )
            except Exception as e:
                self._using_hsv_fallback = True
                print(
                    f"[phase4] *** WARNING: failed to load smoking YOLO model: {e}. "
                    "SmokingDetector is operating in HSV-glow FALLBACK mode. "
                    "Output is DEGRADED — see above for root cause. ***"
                )
        else:
            self._using_hsv_fallback = True
            print(
                "[phase4] *** WARNING: smoking.weights is null in models.yaml. "
                "SmokingDetector is operating in HSV-glow FALLBACK mode. "
                "Output is DEGRADED: bright reflections cause false positives, "
                "actual cigarettes in indirect light will be missed. ***"
            )
        
        # HSV fallback params
        self.min_object_area = int(self.cfg.get("min_object_area", 20))
        self.glow_hsv_low = np.array(self.cfg.get("glow_hsv_low", [0, 100, 200]), dtype=np.uint8)
        self.glow_hsv_high = np.array(self.cfg.get("glow_hsv_high", [20, 255, 255]), dtype=np.uint8)

        # Gesture tracking: track_id -> {"start_time": float, "wrist_history": list, "last_seen": float}
        self._gesture_history: dict[int, dict] = {}
        # Oscillation detection params (Phase 5E spec: "repeated small motions")
        self._min_oscillations = int(self.cfg.get("min_oscillations", 0))
        self._oscillation_window = int(self.cfg.get("oscillation_window_frames", 30))

    def detect(self, frame: np.ndarray, tracks: list, poses: list | int | None = None,
               frame_idx: int = 0, t: float | None = None) -> list[Event]:
        if isinstance(poses, int):
            frame_idx = poses
            poses = []
        if poses is None:
            poses = []
        if t is None:
            t = time.perf_counter()
        events: list[Event] = []

        pose_map = {p.track_id: p for p in poses if p.track_id >= 0}
        active_ids = set()

        for tr in tracks:
            if getattr(tr, "cls", -1) != 0:
                continue
            tid = getattr(tr, "track_id", -1)
            if tid < 0:
                continue
            active_ids.add(tid)

            tx1, ty1, tx2, ty2 = tr.xyxy
            H = ty2 - ty1
            if H <= 0:
                continue

            pose = pose_map.get(tid)
            if pose is None:
                self._gesture_history.pop(tid, None)
                continue

            kpts = pose.keypoints  # (17, 3)
            if len(kpts) < 11:
                self._gesture_history.pop(tid, None)
                continue

            nose = kpts[0]
            l_wrist = kpts[9]
            r_wrist = kpts[10]

            # Stage 1: check gesture proximity (<0.15 * person height)
            stage1_active = False
            chosen_wrist = None
            if nose[2] >= 0.3:
                # Check left wrist
                if l_wrist[2] >= 0.3:
                    d_left = np.sqrt((l_wrist[0] - nose[0])**2 + (l_wrist[1] - nose[1])**2)
                    if d_left < 0.15 * H:
                        stage1_active = True
                        chosen_wrist = l_wrist

                # Check right wrist
                if r_wrist[2] >= 0.3:
                    d_right = np.sqrt((r_wrist[0] - nose[0])**2 + (r_wrist[1] - nose[1])**2)
                    if d_right < 0.15 * H:
                        if chosen_wrist is not None:
                            d_prev = np.sqrt((chosen_wrist[0] - nose[0])**2 + (chosen_wrist[1] - nose[1])**2)
                            if d_right < d_prev:
                                chosen_wrist = r_wrist
                        else:
                            stage1_active = True
                            chosen_wrist = r_wrist

            if stage1_active and chosen_wrist is not None:
                h = self._gesture_history.setdefault(tid, {"start_time": t, "wrist_history": [], "last_seen": t})
                h["last_seen"] = t
                h["wrist_history"].append((float(chosen_wrist[0]), float(chosen_wrist[1])))
                if len(h["wrist_history"]) > 90:
                    h["wrist_history"].pop(0)

                duration = t - h["start_time"]
                if duration >= 2.0:
                    # Phase 5E: oscillation gate — require rhythmic raise/lower
                    # pattern before invoking YOLO confirmation. Genuine smoking
                    # has a rhythmic hand rise-pause-lower cycle; static hand-near-face
                    # (phone call, chin rest, scratching) does not.
                    wrist_ys = [wy for (_, wy) in h["wrist_history"]]
                    osc_window = wrist_ys[-self._oscillation_window:] if len(wrist_ys) > self._oscillation_window else wrist_ys
                    oscillations = 0
                    if len(osc_window) >= 4:
                        dy = [osc_window[k+1] - osc_window[k] for k in range(len(osc_window) - 1)]
                        for k in range(1, len(dy)):
                            if (dy[k] > 0 and dy[k-1] < 0) or (dy[k] < 0 and dy[k-1] > 0):
                                oscillations += 1

                    if oscillations < self._min_oscillations:
                        # Not enough oscillation — skip YOLO confirmation this frame
                        continue

                    cx, cy = chosen_wrist[0], chosen_wrist[1]
                    size = int(0.25 * H)
                    hx1 = max(0, int(cx - size // 2))
                    hy1 = max(0, int(cy - size // 2))
                    hx2 = min(frame.shape[1], int(cx + size // 2))
                    hy2 = min(frame.shape[0], int(cy + size // 2))

                    if hx2 - hx1 >= 16 and hy2 - hy1 >= 16:
                        hand_crop = frame[hy1:hy2, hx1:hx2]
                        confirmed = False
                        conf_val = 0.0
                        class_name = "unknown"
                        bbox_crop = None

                        if self._use_yolo and self.model is not None:
                            res = self.model.predict(hand_crop, imgsz=self.imgsz, device=0, half=True,
                                                     conf=self.conf, verbose=False)[0]
                            if res.boxes is not None and len(res.boxes) > 0:
                                crop_confs = res.boxes.conf.cpu().numpy()
                                crop_clsids = res.boxes.cls.cpu().numpy().astype(int)
                                crop_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                                for box, c, clsid in zip(crop_xyxy, crop_confs, crop_clsids):
                                    if clsid in [2, 3]:
                                        confirmed = True
                                        conf_val = float(c)
                                        class_name = "cigarette" if clsid == 2 else "vape"
                                        bbox_crop = (int(box[0] + hx1), int(box[1] + hy1),
                                                     int(box[2] + hx1), int(box[3] + hy1))
                                        break
                        else:
                            import cv2
                            hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
                            glow_mask = cv2.inRange(hsv, self.glow_hsv_low, self.glow_hsv_high)
                            glow_area = cv2.countNonZero(glow_mask)
                            if glow_area >= self.min_object_area:
                                confirmed = True
                                conf_val = 0.5
                                class_name = "glow"
                                bbox_crop = (hx1, hy1, hx2, hy2)

                        if confirmed:
                            events.append(Event(
                                event_type="SMOKING",
                                t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                frame_idx=frame_idx,
                                details={
                                    "track_id": tid,
                                    "bbox": bbox_crop,
                                    "confidence": round(conf_val, 3),
                                    "class": class_name,
                                    "method": "pose_gesture_primary_yolo" if self._use_yolo else "pose_gesture_primary_hsv",
                                    "duration_s": round(duration, 2),
                                    "oscillations": oscillations
                                }
                            ))
                            # Reset start time to implement cooldown
                            h["start_time"] = t
            else:
                self._gesture_history.pop(tid, None)

        # Prune stale history
        for tid in list(self._gesture_history.keys()):
            if tid not in active_ids or (t - self._gesture_history[tid]["last_seen"] > 10.0):
                self._gesture_history.pop(tid, None)

        return events


# =============================================================================
# 3. Phone-Watching Detection — YOLO cell-phone detection + head-pose heuristic
# =============================================================================

class PhoneWatcherDetector:
    """Detects phone-watching behavior.

    Detection logic:
      (a) A cell phone (COCO class 67) is detected anywhere in frame OR near
          a tracked person's extended bounding box (generous proximity check,
          not strict overlap — phone may be held at side or partially off-body).
      (b) The person's head is tilted down — inferred from pose keypoints:
          nose y > shoulder midpoint y means the head is below the shoulder
          line, suggesting looking down at a phone. Defaults to True (fires
          anyway) when pose keypoints are unavailable or low-confidence.

    Uses a SEPARATE, independent YOLOv8n instance dedicated to phone detection
    (COCO class 67). This is intentionally NOT shared with the tracker's
    detector — sharing caused conflicts because the tracker uses persist=True
    with classes=[0] (persons), and calling predict() with classes=[67] on the
    same model corrupted the tracker state. The second YOLO adds ~6MB VRAM
    (same yolov8n weights, separate model instance) and runs at low cadence
    (every 10 frames) so the compute cost is minimal.

    Debug: set env PHONE_DEBUG=1 to print raw top-N detection confidences/classes
    seen on every call, even below the trigger threshold. Fastest way to tell if
    the model is "almost detecting but below threshold" vs "not classifying at all".
    """

    def __init__(self, cfg: dict[str, Any] | None = None,
                 detector_model=None):
        # detector_model param is accepted for backward compat but ALWAYS IGNORED.
        # PhoneWatcherDetector loads its own independent YOLO to avoid the sharing
        # conflict where predict(classes=[67]) on the tracker model corrupts
        # ByteTrack state (tracker uses persist=True with classes=[0]).
        self.cfg = cfg if cfg is not None else _load_cfg("phone")
        self.imgsz = int(self.cfg.get("imgsz", 480))
        # Lower default conf to 0.15 — a phone held at an angle, partially
        # gripped, or at typical desk distance may not reach 0.3+. The shared
        # detector uses 0.35 for persons (strong signal); phones are harder.
        self.conf = float(self.cfg.get("conf", 0.15))
        self.device = 0
        # Hysteresis: require confirm_frames consecutive detections before
        # emitting. Then hold the event alive for hold_frames frames after the
        # last detection. This converts intermittent flickers (detect / miss /
        # detect) into a sustained, stable PHONE alert.
        self._confirm_frames = int(self.cfg.get("confirm_frames", 3))
        self._hold_frames    = int(self.cfg.get("hold_frames", 15))
        # per-track state: {track_id: {"consec": int, "hold": int, "last_bbox": tuple, "last_conf": float}}
        self._track_state: dict[int, dict] = {}
        import os
        self._debug = os.environ.get("PHONE_DEBUG", "0") == "1"
        from ultralytics import YOLO
        weights = self.cfg.get("weights", "models/yolov8n.pt")
        self.model = YOLO(weights)
        self.model.fuse()
        self._owns_model = True
        # warmup with a phone-class-only detection
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.imgsz, device=0, half=True,
                           conf=0.01, classes=[67], verbose=False)
        self._nose_idx = 0    # COCO-pose keypoint index for nose
        self._l_shoulder_idx = 5
        self._r_shoulder_idx = 6
        print(f"[phase4] phone detector: own independent YOLO instance  "
              f"weights={weights}  imgsz={self.imgsz}  conf={self.conf}  "
              f"class=67(cell_phone)  confirm={self._confirm_frames}  "
              f"hold={self._hold_frames}  PHONE_DEBUG={self._debug}")

    def detect(self, frame: np.ndarray, tracks: list, poses: list,
               frame_idx: int) -> list[Event]:
        """Detect phone-watching with confirm+hold hysteresis.

        For each tracked person:
          1. Run YOLO phone detection on the full frame.
          2. Check proximity of any detected phone to the person's expanded bbox.
          3. Check head-pose (looking down) from pose keypoints.
          4. If all checks pass, increment the per-track confirm counter.
          5. Emit a PHONE event only once confirm_frames consecutive detections
             have accumulated (kills single-frame noise).
          6. After a confirmed detection, hold the event alive for hold_frames
             frames even when YOLO misses — this removes the flicker where the
             phone is visible but YOLO confidence dips for 1-2 frames.
        """
        import os
        events: list[Event] = []
        # Run phone detection on the full frame at low res.
        raw_conf_floor = 0.01 if self._debug else self.conf
        res = self.model.predict(frame, imgsz=self.imgsz, device=0, half=True,
                                 conf=raw_conf_floor, classes=[67], verbose=False)[0]

        # --- PHONE_DEBUG: dump all raw candidates regardless of threshold ---
        if self._debug and res.boxes is not None and len(res.boxes) > 0:
            raw_confs = res.boxes.conf.cpu().numpy()
            raw_cls = res.boxes.cls.cpu().numpy().astype(int)
            raw_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            print(f"[PHONE_DEBUG] frame={frame_idx}  raw_detections={len(raw_confs)}")
            for i, (rc, rcl, rbbox) in enumerate(zip(raw_confs, raw_cls, raw_xyxy)):
                above = "ABOVE" if rc >= self.conf else "below"
                print(f"  [{i}] cls={rcl}(cell_phone)  conf={rc:.3f}  "
                      f"{above}_threshold({self.conf})  bbox={tuple(rbbox)}")
        elif self._debug:
            print(f"[PHONE_DEBUG] frame={frame_idx}  no detections at all (conf_floor=0.01, class=67)")

        # Apply self.conf threshold to build the final phone_boxes list
        phone_boxes: list[tuple[tuple, float]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy  = res.boxes.xyxy.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for (bx1, by1, bx2, by2), c in zip(xyxy, confs):
                if c >= self.conf:
                    phone_boxes.append(((int(bx1), int(by1), int(bx2), int(by2)), float(c)))

        # Build pose map: track_id -> keypoints
        pose_map: dict[int, Any] = {}
        for p in poses:
            if p.track_id >= 0:
                pose_map[p.track_id] = p.keypoints

        fh, fw = frame.shape[:2]

        # Determine which track IDs are still active this frame
        active_tids: set[int] = set()
        for tr in tracks:
            if getattr(tr, "cls", -1) != 0:
                continue
            tid = getattr(tr, "track_id", -1)
            if tid >= 0:
                active_tids.add(tid)

        for tr in tracks:
            if getattr(tr, "cls", -1) != 0:
                continue
            tid = getattr(tr, "track_id", -1)
            if tid < 0:
                continue

            # Initialise per-track state
            st = self._track_state.setdefault(tid, {
                "consec": 0, "hold": 0, "last_bbox": None, "last_conf": 0.0
            })

            tx1, ty1, tx2, ty2 = tr.xyxy
            # Expand person bbox by 50% on each side to catch phones held at side
            pad_x = int((tx2 - tx1) * 0.5)
            pad_y = int((ty2 - ty1) * 0.5)
            ex1 = max(0,  tx1 - pad_x)
            ey1 = max(0,  ty1 - pad_y)
            ex2 = min(fw, tx2 + pad_x)
            ey2 = min(fh, ty2 + pad_y)

            # Check if any phone box is near this person AND they are looking down
            detected_this_frame = False
            for (pb, pc) in phone_boxes:
                px1, py1, px2, py2 = pb
                ox1 = max(ex1, px1); oy1 = max(ey1, py1)
                ox2 = min(ex2, px2); oy2 = min(ey2, py2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue  # phone not near this person

                # Head-pose check
                looking_down = True
                kpts = pose_map.get(tid)
                if kpts is not None:
                    nose  = kpts[self._nose_idx]
                    l_sh  = kpts[self._l_shoulder_idx]
                    r_sh  = kpts[self._r_shoulder_idx]
                    sh_ys = []
                    if l_sh[2] > 0.3:
                        sh_ys.append(l_sh[1])
                    if r_sh[2] > 0.3:
                        sh_ys.append(r_sh[1])
                    if nose[2] > 0.3 and len(sh_ys) >= 1:
                        shoulder_mid_y = sum(sh_ys) / len(sh_ys)
                        looking_down = nose[1] > shoulder_mid_y

                if looking_down:
                    detected_this_frame = True
                    st["last_bbox"] = pb
                    st["last_conf"] = pc
                    break  # one phone per person per frame

            # Update confirm / hold counters
            if detected_this_frame:
                st["consec"] += 1
                st["hold"] = self._hold_frames   # reset hold timer
            else:
                st["consec"] = 0                  # break the confirm streak
                st["hold"]   = max(0, st["hold"] - 1)

            # Emit event when confirmed OR within hold window
            should_fire = (
                (detected_this_frame and st["consec"] >= self._confirm_frames)
                or (not detected_this_frame and st["hold"] > 0 and st["last_bbox"] is not None)
            )
            if should_fire:
                events.append(Event(
                    event_type="PHONE",
                    t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    frame_idx=frame_idx,
                    details={"track_id": tid,
                             "phone_bbox": st["last_bbox"],
                             "phone_conf": round(st["last_conf"], 3),
                             "looking_down": True,
                             "confirm_count": st["consec"],
                             "hold_remaining": st["hold"]},
                ))

        # Prune state for tracks no longer active
        for tid in list(self._track_state.keys()):
            if tid not in active_tids:
                del self._track_state[tid]

        return events


# =============================================================================
# 4. Personnel Gathering Detection — DBSCAN clustering on track centroids
# =============================================================================

def _point_in_polygon(x: float, y: float, polygon: list[list[float]] | None) -> bool:
    if polygon is None or not polygon:
        return True
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class GatheringDetector:
    """Detects personnel gathering: N+ people within a radius.

    No model needed — clusters tracked person centroids using a fixed-radius
    grouping (simpler than DBSCAN, no sklearn dependency). Fires when a cluster
    of >= min_people persons exists within radius_pixels and inside an ROI polygon,
    sustained for >= sustained_s seconds continuously.

    Config (`gathering` in models.yaml):
      min_people: minimum cluster size to trigger (default 3)
      radius_pixels: max distance between any two people in a cluster (default 150)
      cooldown_s: suppress re-trigger for the same cluster (default 10.0)
      sustained_s: seconds the count must exceed threshold before firing (default 2.0)
      sustained_frames: DEPRECATED — converted to sustained_s assuming 30fps if sustained_s not set
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("gathering")
        self.min_people = int(self.cfg.get("min_people", 3))
        self.radius_pixels = float(self.cfg.get("radius_pixels", 150))
        self.cooldown_s = float(self.cfg.get("cooldown_s", 10.0))
        # Wall-clock sustained duration (spec §5: "trigger if count > N sustained for > T seconds")
        if "sustained_s" in self.cfg:
            self.sustained_s = float(self.cfg["sustained_s"])
        elif "sustained_frames" in self.cfg:
            # Legacy frame-count config: convert with the pipeline's real capture
            # rate, not an assumed 30 fps.
            from core.config import load_fps
            self.sustained_s = int(self.cfg["sustained_frames"]) / load_fps()
        else:
            self.sustained_s = 0.0

        # Wall-clock timestamp when threshold was first continuously exceeded per ROI
        self._sustained_since: dict[str, float | None] = {}
        self._last_fire_times: dict[str, float] = {}

        # Load ROIs config from configs/roi.yaml
        from core.config import load_pipeline_config, load_yaml
        try:
            p_cfg = load_pipeline_config()
            self.camera_id = p_cfg.get("output", {}).get("camera_id", "cam_01")
        except Exception:
            self.camera_id = "cam_01"

        self.rois = []
        try:
            roi_data = load_yaml("roi.yaml")
            self.rois = roi_data.get("cameras", {}).get(self.camera_id, {}).get("rois", [])
        except Exception as e:
            print(f"[GatheringDetector] Warning: failed to load roi.yaml: {e}")

        # Fallback if no ROIs defined
        if not self.rois:
            self.rois = [{"id": "default", "polygon": None}]

    def detect(self, tracks: list, frame_idx: int,
               t: float | None = None) -> list[Event]:
        if t is None:
            t = time.perf_counter()
        events: list[Event] = []

        # collect all person centroids
        all_centroids: list[tuple[int, int, int]] = []   # (cx, cy, track_id)
        for tr in tracks:
            if getattr(tr, "cls", -1) != 0:
                continue
            tid = getattr(tr, "track_id", -1)
            if tid < 0:
                continue
            x1, y1, x2, y2 = tr.xyxy
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            all_centroids.append((cx, cy, tid))

        # Check each ROI
        for roi in self.rois:
            roi_id = roi.get("id", "default")
            polygon = roi.get("polygon", None)

            # Filter centroids in this ROI
            centroids = []
            for cx, cy, tid in all_centroids:
                if _point_in_polygon(cx, cy, polygon):
                    centroids.append((cx, cy, tid))

            if len(centroids) < self.min_people:
                self._sustained_since[roi_id] = None
                continue

            # fixed-radius clustering within the ROI
            used = set()
            clusters: list[list[tuple[int, int, int]]] = []
            for i, (cx, cy, tid) in enumerate(centroids):
                if i in used:
                    continue
                cluster = [(cx, cy, tid)]
                used.add(i)
                for j, (ox, oy, otid) in enumerate(centroids):
                    if j in used:
                        continue
                    # check distance to any member of the cluster
                    for (mcx, mcy, _) in cluster:
                        if np.sqrt((ox - mcx)**2 + (oy - mcy)**2) <= self.radius_pixels:
                            cluster.append((ox, oy, otid))
                            used.add(j)
                            break
                clusters.append(cluster)

            # Find valid clusters meeting threshold
            valid_clusters = [c for c in clusters if len(c) >= self.min_people]

            if not valid_clusters:
                self._sustained_since[roi_id] = None
                continue

            # Mark when threshold was first continuously exceeded
            if self._sustained_since.get(roi_id) is None:
                self._sustained_since[roi_id] = t

            sustained_duration = t - self._sustained_since[roi_id]
            if sustained_duration >= self.sustained_s:
                last_fire_t = self._last_fire_times.get(roi_id, 0.0)
                if t - last_fire_t >= self.cooldown_s:
                    best_cluster = max(valid_clusters, key=len)
                    tids = [c[2] for c in best_cluster]
                    events.append(Event(
                        event_type="GATHERING",
                        t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        frame_idx=frame_idx,
                        details={"count": len(best_cluster),
                                 "track_ids": tids,
                                 "radius_px": int(self.radius_pixels),
                                 "roi_id": roi_id,
                                 "duration_seconds": round(sustained_duration, 2)},
                    ))
                    self._last_fire_times[roi_id] = t
        return events


# =============================================================================
# 5. Violence/Fighting Detection — ResNet50-based binary classifier
# =============================================================================

def _build_resnet50_violence_model(weights_path: str | None, device: str, half: bool):
    """Build a ResNet50 binary violence classifier from a trained checkpoint.

    REQUIRES a fine-tuned checkpoint.  When ``weights_path`` is None this
    function raises ``RuntimeError`` rather than building a randomly-initialised
    head, because:

    - The ResNet50 backbone was trained on ImageNet (1000 generic classes).
    - Replacing its FC with a randomly-initialised Linear(2048, 2) and running
      softmax(logits) produces random violence/no_violence scores on every call.
    - Those random scores fired VIOLENCE events and printed
      '[violence/resnet50] model ready' as if a real model was loaded.
      This is fabricated output.

    To use fine-tuned weights (e.g., trained on UCF-Crime / RWF-2000):
      1. Obtain or train a checkpoint (Linear(2048, 2) head, softmax output).
      2. Set ``violence.weights`` in configs/models.yaml to the .pt path.
      3. This function will load the checkpoint and return a ready model.

    Returns:
        (model, transform, device_obj) -- ready for .eval() inference.
        Raises RuntimeError if weights_path is None or file not found.
        Returns None only if torchvision is unavailable.
    """
    if weights_path is None:
        raise RuntimeError(
            "[violence/resnet50] REFUSING TO LOAD: violence.weights is null in "
            "configs/models.yaml.  ViolenceDetector requires a checkpoint trained on "
            "a violence-detection dataset (e.g. UCF-Crime, RWF-2000).  An ImageNet "
            "ResNet50 with a randomly initialised Linear(2048, 2) head produces random "
            "violence/no_violence scores on every call -- this is fabricated output, "
            "not a detection.  Set features.violence: false in configs/pipeline.yaml "
            "(already done) or supply a real checkpoint.  The skeleton-based "
            "FightDetector (features.fight: true) remains enabled and covers "
            "violent-interaction detection with real kinematic signal."
        )

    try:
        import torch
        import torchvision.models as tv_models
        import torchvision.transforms as T
    except ImportError:
        return None

    # --- backbone ---
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = __import__("torch.nn", fromlist=["Linear"]).Linear(in_features, 2)

    # --- load fine-tuned weights (weights_path is guaranteed non-None here) ---
    from pathlib import Path
    p = Path(weights_path)
    if not p.exists():
        raise RuntimeError(
            f"[violence/resnet50] weights file not found: {p}.  "
            "Cannot build violence classifier without a trained checkpoint."
        )
    state = torch.load(str(p), map_location="cpu")
    # accept both raw state_dict and {"model": state_dict} checkpoints
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[violence/resnet50] WARN missing keys: {missing[:5]}")
    if unexpected:
        print(f"[violence/resnet50] WARN unexpected keys: {unexpected[:5]}")
    print(f"[violence/resnet50] loaded fine-tuned weights: {p}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    if half and dev.type == "cuda":
        model = model.half()
    model.eval()

    # --- standard ImageNet preprocessing (224x224, normalised) ---
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, dev


class ViolenceDetector:
    """ResNet50-based binary violence classifier.

    Classifies each incoming frame as 'violence' or 'no_violence' using a
    ResNet50 backbone (ImageNet-pretrained, optional fine-tuned head) with a
    rolling confidence window to suppress single-frame false positives.

    Architecture
    ------------
    - Backbone : torchvision ResNet50 (IMAGENET1K_V2 pretrained)
    - Head     : Linear(2048 → 2) binary classifier
    - Input    : full frame resized to 224×224, ImageNet normalised
    - Output   : softmax(class=1) confidence pushed into a deque of length
                 `window_frames`; fires VIOLENCE when
                 mean(deque) >= conf_threshold for >= min_sustained_frames
                 of the last window.

    Advantages over the old heuristic
    ----------------------------------
    Research on UCF-Crime and RWF-2000 shows ResNet50 reaches notably higher
    precision and recall on complex violent scenes (close grappling, weapon
    use, crowd fights) compared to proximity/motion heuristics that confuse
    hugs, handshakes, and fast walking with violence.

    Fallback
    --------
    If `torch` / `torchvision` are not installed, the detector transparently
    falls back to the old IoU+motion heuristic with a warning at startup.

    Config keys (`violence:` in models.yaml)
    ----------------------------------------
    weights          : path to fine-tuned .pt checkpoint (null = ImageNet only)
    device           : "cuda:0" | "cpu"
    half             : FP16 inference on CUDA (saves ~50 MB VRAM)
    imgsz            : input size (default 224 — ResNet50 canonical)
    conf_threshold   : mean window confidence to trigger VIOLENCE (default 0.65)
    window_frames    : rolling window length (default 10)
    min_sustained_frames : frames in window that must exceed threshold (default 4)
    cooldown_s       : suppress re-trigger after firing (default 10.0)
    # Legacy heuristic fallback keys (used only when torch unavailable):
    iou_threshold, motion_threshold, window_s
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("violence")

        # --- ResNet50 inference config ---
        self._conf_threshold    = float(self.cfg.get("conf_threshold",    0.65))
        self._window_frames     = int(self.cfg.get("window_frames",       10))
        self._min_sustained     = int(self.cfg.get("min_sustained_frames", 4))
        self.cooldown_s         = float(self.cfg.get("cooldown_s",         10.0))

        # --- try to build the deep model ---
        result = _build_resnet50_violence_model(
            weights_path=self.cfg.get("weights"),
            device=str(self.cfg.get("device", "cuda:0")),
            half=bool(self.cfg.get("half", True)),
        )

        if result is not None:
            self._model, self._transform, self._device = result
            self.method = "resnet50"
            print(
                f"[violence/resnet50] model ready  "
                f"device={self._device}  "
                f"conf_threshold={self._conf_threshold}  "
                f"window={self._window_frames}  "
                f"min_sustained={self._min_sustained}"
            )
        else:
            # --- fallback to old heuristic ---
            self._model = None
            self.method = "heuristic_fallback"
            print(
                "[violence/resnet50] WARN torch/torchvision unavailable — "
                "falling back to IoU+motion heuristic"
            )

        # --- rolling confidence deque (resnet50 path) ---
        from collections import deque
        self._conf_window: "deque[float]" = deque(maxlen=self._window_frames)
        self._last_fire_t: float = 0.0

        # --- heuristic state (fallback path) ---
        self.iou_threshold    = float(self.cfg.get("iou_threshold",   0.3))
        self.motion_threshold = float(self.cfg.get("motion_threshold", 40.0))
        self.window_s         = float(self.cfg.get("window_s",         1.5))
        self._pair_state: dict[tuple[int, int], dict] = {}

    # ------------------------------------------------------------------
    # Public API (called from main_loop.py)
    # ------------------------------------------------------------------

    def detect(
        self,
        tracks: list,
        frame_idx: int,
        t: float | None = None,
        frame: "np.ndarray | None" = None,
    ) -> list[Event]:
        """Run violence detection on the current frame.

        Args:
            tracks    : active tracker outputs (used by heuristic fallback and
                        to gate inference on frames with ≥1 person)
            frame_idx : current frame counter
            t         : wall-clock time (perf_counter)
            frame     : raw BGR frame from the camera (required for ResNet50 path)
        """
        if t is None:
            t = time.perf_counter()

        if self._model is not None and frame is not None:
            return self._detect_resnet50(tracks, frame, frame_idx, t)
        else:
            return self._detect_heuristic(tracks, frame_idx, t)

    # ------------------------------------------------------------------
    # ResNet50 inference path
    # ------------------------------------------------------------------

    def _detect_resnet50(
        self,
        tracks: list,
        frame: "np.ndarray",
        frame_idx: int,
        t: float,
    ) -> list[Event]:
        """Run ResNet50 classifier; accumulate confidence; fire when sustained."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return []

        # Count visible persons — skip inference if scene is empty
        n_persons = sum(1 for tr in tracks if getattr(tr, "cls", -1) == 0)

        # Push 0.0 confidence (no_violence) for empty scenes to keep the
        # rolling window honest during calm periods
        if n_persons == 0:
            self._conf_window.append(0.0)
            return []

        # --- preprocess: BGR numpy -> RGB -> tensor ---
        frame_rgb = frame[:, :, ::-1].copy()  # BGR → RGB
        inp = self._transform(frame_rgb).unsqueeze(0)  # (1, 3, 224, 224)
        inp = inp.to(self._device)
        if next(self._model.parameters()).dtype == __import__("torch").float16:
            inp = inp.half()

        with __import__("torch").no_grad():
            logits = self._model(inp)                 # (1, 2)
            probs  = F.softmax(logits, dim=1)         # (1, 2)
            violence_conf = float(probs[0, 1].cpu())  # index 1 = violence class

        self._conf_window.append(violence_conf)

        # --- rolling window decision ---
        if len(self._conf_window) < self._window_frames:
            return []

        sustained = sum(1 for c in self._conf_window if c >= self._conf_threshold)
        if sustained < self._min_sustained:
            return []

        if t - self._last_fire_t < self.cooldown_s:
            return []

        mean_conf = sum(self._conf_window) / max(len(self._conf_window), 1)
        self._last_fire_t = t
        self._conf_window.clear()

        return [Event(
            event_type="VIOLENCE",
            t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
            frame_idx=frame_idx,
            details={
                "method":           "resnet50",
                "confidence":       round(violence_conf, 3),
                "window_mean_conf": round(mean_conf, 3),
                "sustained_frames": sustained,
                "n_persons":        n_persons,
            },
        )]

    # ------------------------------------------------------------------
    # Legacy heuristic fallback path (IoU + centroid motion)
    # ------------------------------------------------------------------

    def _detect_heuristic(
        self,
        tracks: list,
        frame_idx: int,
        t: float,
    ) -> list[Event]:
        """Original IoU+motion heuristic — kept as torch-free fallback.

        Fires when two tracked persons have significant bbox overlap (IoU ≥
        iou_threshold) *and* rapid relative centroid motion sustained over
        window_s seconds.  Known to produce false positives on hugs/handshakes
        — use ResNet50 path in production.
        """
        events: list[Event] = []

        persons: list[tuple[int, tuple, tuple]] = []
        for tr in tracks:
            if getattr(tr, "cls", -1) != 0:
                continue
            tid = getattr(tr, "track_id", -1)
            if tid < 0:
                continue
            x1, y1, x2, y2 = tr.xyxy
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            persons.append((tid, (x1, y1, x2, y2), (cx, cy)))

        if len(persons) < 2:
            self._pair_state.clear()
            return events

        active_pairs: set[tuple[int, int]] = set()
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                tid_a, box_a, cen_a = persons[i]
                tid_b, box_b, cen_b = persons[j]
                pair_key = (min(tid_a, tid_b), max(tid_a, tid_b))
                active_pairs.add(pair_key)

                iou = _compute_iou(box_a, box_b)
                if iou < self.iou_threshold:
                    if pair_key in self._pair_state:
                        self._pair_state[pair_key]["motion_active"] = False
                        self._pair_state[pair_key]["contact_since"] = t
                    continue

                st = self._pair_state.setdefault(pair_key, {
                    "contact_since": t,
                    "last_cen_a": cen_a,
                    "last_cen_b": cen_b,
                    "motion_active": False,
                })
                rel_motion = np.sqrt(
                    (cen_a[0] - st["last_cen_a"][0])**2 + (cen_a[1] - st["last_cen_a"][1])**2
                ) + np.sqrt(
                    (cen_b[0] - st["last_cen_b"][0])**2 + (cen_b[1] - st["last_cen_b"][1])**2
                )
                st["last_cen_a"] = cen_a
                st["last_cen_b"] = cen_b
                if rel_motion >= self.motion_threshold:
                    st["motion_active"] = True
                else:
                    st["motion_active"] = False

                if st["motion_active"] and (t - st["contact_since"]) >= self.window_s:
                    if t - self._last_fire_t >= self.cooldown_s:
                        events.append(Event(
                            event_type="VIOLENCE",
                            t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            frame_idx=frame_idx,
                            details={
                                "pair":       list(pair_key),
                                "iou":        round(iou, 3),
                                "rel_motion": round(rel_motion, 1),
                                "duration_s": round(t - st["contact_since"], 2),
                                "method":     "heuristic_fallback",
                            },
                        ))
                        self._last_fire_t = t
                    st["contact_since"] = t
                    st["motion_active"] = False

        for pk in list(self._pair_state.keys()):
            if pk not in active_pairs:
                del self._pair_state[pk]

        return events


# =============================================================================
# 6. Object-Left-Behind Detection — track stationary non-person objects
# =============================================================================

class ObjectLeftDetector:
    """Detects objects left behind (bags, backpacks, suitcases, etc).

    Tracks non-person objects that remain stationary (>30s default) at roughly
    the same position. Uses existing tracker infrastructure — no new model needed.

    COCO classes monitored:
      - 24: backpack
      - 26: handbag
      - 28: suitcase
      - 39: bottle
      - 56-61: chair, couch, potted plant, bed, dining table, toilet (optional)

    Config (`object_left` in models.yaml):
      min_stationary_s: seconds an object must be stationary (default 30.0)
      position_variance_threshold: max variance in position to count as stationary (default 100)
      cooldown_s: suppress re-trigger for same object (default 60.0)
    """

    COCO_OBJECT_CLASSES = [24, 26, 28, 39, 56, 57, 58, 59, 60, 61]

    def __init__(self, cfg: dict[str, Any] | None = None, fps: float | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("object_left")
        self.min_stationary_s = float(self.cfg.get("min_stationary_s", 30.0))
        self.position_variance_threshold = float(self.cfg.get("position_variance_threshold", 100.0))
        self.cooldown_s = float(self.cfg.get("cooldown_s", 60.0))
        # Sample-count windows below are seconds converted at the real capture
        # rate. Caller may pass fps explicitly; otherwise read the one
        # authoritative value from pipeline.yaml. Never assumed to be 30.
        if fps is None:
            from core.config import load_fps
            fps = load_fps()
        self.fps = float(fps)
        # Need at least this much history before a stationarity verdict is valid.
        self._min_samples = max(2, int(round(self.fps)))   # 1 second of data
        # track_id -> [(cx, cy, t), ...]
        self._object_history: dict[int, list[tuple[float, float, float]]] = {}
        self._last_fire_t: dict[int, float] = {}

    def detect(self, tracks: list, frame_idx: int,
               t: float | None = None) -> list[Event]:
        if t is None:
            t = time.perf_counter()
        events: list[Event] = []

        current_ids = set()
        for tr in tracks:
            tid = getattr(tr, "track_id", -1)
            if tid < 0:
                continue
            cls = getattr(tr, "cls", -1)
            if cls == 0:  # skip persons
                continue
            if cls not in self.COCO_OBJECT_CLASSES:
                continue

            current_ids.add(tid)
            x1, y1, x2, y2 = tr.xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            history = self._object_history.setdefault(tid, [])
            history.append((cx, cy, t))

            # Keep min_stationary_s of samples (+ headroom), sized at real FPS.
            max_samples = int(self.min_stationary_s * self.fps + 100)
            if len(history) > max_samples:
                history = history[-max_samples:]
                self._object_history[tid] = history

            # Check if object has been stationary long enough
            if len(history) < self._min_samples:  # need at least 1 second of data
                continue

            # Check cooldown
            if tid in self._last_fire_t and (t - self._last_fire_t[tid]) < self.cooldown_s:
                continue

            # Compute position variance over recent history
            recent = history[-int(self.min_stationary_s * self.fps + 1):]
            if len(recent) < self._min_samples:
                continue

            positions = np.array(recent)
            variance = np.var(positions[:, :2], axis=0)
            max_variance = np.max(variance)

            if max_variance < self.position_variance_threshold:
                duration = recent[-1][2] - recent[0][2]
                if duration >= self.min_stationary_s:
                    COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                  "train", "truck", "boat", "traffic light", "fire hydrant",
                                  "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                  "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                  "baseball glove", "skateboard", "surfboard", "tennis racket",
                                  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                  "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                  "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                  "potted plant", "bed", "dining table", "toilet", "tv", "laptop"]
                    class_name = COCO_NAMES[cls] if 0 <= cls < len(COCO_NAMES) else f"class_{cls}"
                    events.append(Event(
                        event_type="OBJECT_LEFT",
                        t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        frame_idx=frame_idx,
                        details={"track_id": tid,
                                 "object_class": class_name,
                                 "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                 "stationary_duration_s": round(duration, 1),
                                 "position_variance": round(float(max_variance), 2),
                                 "method": "position_history_tracking"},
                    ))
                    self._last_fire_t[tid] = t

        # Clean up history for objects no longer tracked
        for tid in list(self._object_history.keys()):
            if tid not in current_ids:
                del self._object_history[tid]
            if tid in self._last_fire_t and (t - self._last_fire_t[tid]) > self.cooldown_s * 2:
                del self._last_fire_t[tid]

        return events


# =============================================================================
# Helpers
# =============================================================================

def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """IoU of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _load_cfg(section: str) -> dict[str, Any]:
    from core.config import load_models_config
    return load_models_config().get(section, {})