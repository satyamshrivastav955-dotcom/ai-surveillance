"""Phase 4 event detectors — fire/smoke, smoking, phone-watching, gathering, violence, object-left.

Six features, all gated through the FrameRouter, all VLM-agnostic (constraint #5):

  1. FireSmokeDetector     — YOLO fine-tuned on D-Fire for fire/smoke detection
  2. SmokingDetector       — YOLO fine-tuned for cigarette/vape detection
  3. PhoneWatcherDetector  — YOLO class 67 (cell phone) + head-pose heuristic
  4. GatheringDetector     — Fixed-radius clustering on track centroids
  5. ViolenceDetector      — Rule-based: bbox overlap + rapid motion (placeholder)
  6. ObjectLeftDetector    — Track stationary non-person objects (bags, backpacks)

All detectors emit generic Event dicts (VLM-agnostic) that the Phase 5 event
bus will consume.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Event:
    """Generic event from any Phase 4 detector."""
    event_type: str         # "FIRE" | "SMOKE" | "SMOKING" | "PHONE" | "GATHERING" | "VIOLENCE" | "OBJECT_LEFT"
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
                    print(f"[phase4] WARN: smoking.weights={self.weights} not found, "
                          f"falling back to HSV heuristic")
            except Exception as e:
                print(f"[phase4] WARN: failed to load smoking YOLO model: {e}")
        
        # HSV fallback params
        self.min_object_area = int(self.cfg.get("min_object_area", 20))
        self.glow_hsv_low = np.array(self.cfg.get("glow_hsv_low", [0, 100, 200]), dtype=np.uint8)
        self.glow_hsv_high = np.array(self.cfg.get("glow_hsv_high", [20, 255, 255]), dtype=np.uint8)

    def detect(self, frame: np.ndarray, tracks: list, frame_idx: int) -> list[Event]:
        if self._use_yolo and self.model is not None:
            return self._detect_yolo(frame, tracks, frame_idx)
        else:
            return self._detect_hsv(frame, tracks, frame_idx)
    
    def _detect_yolo(self, frame: np.ndarray, tracks: list, frame_idx: int) -> list[Event]:
        """Smoking detection using YOLO for cigarette/vape detection."""
        events: list[Event] = []
        res = self.model.predict(frame, imgsz=self.imgsz, device=0, half=True,
                                 conf=self.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return events
        
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        clsids = res.boxes.cls.cpu().numpy().astype(int)
        
        # Filter for Cigarette (2) or Vape (3) classes
        for (x1, y1, x2, y2), conf, clsid in zip(xyxy, confs, clsids):
            if clsid not in [2, 3]:  # Cigarette or Vape only
                continue
            # Find which track this cigarette is near
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            best_tid = -1
            best_dist = float('inf')
            for tr in tracks:
                if getattr(tr, "cls", -1) != 0:
                    continue
                tid = getattr(tr, "track_id", -1)
                if tid < 0:
                    continue
                tx1, ty1, tx2, ty2 = tr.xyxy
                tcx = (tx1 + tx2) / 2
                tcy = (ty1 + ty2) / 2
                dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            
            events.append(Event(
                event_type="SMOKING",
                t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                frame_idx=frame_idx,
                details={"track_id": best_tid,
                         "bbox": (int(x1), int(y1), int(x2), int(y2)),
                         "confidence": round(float(conf), 3),
                         "class": "cigarette" if clsid == 2 else "vape",
                         "method": "yolov8n_smoking"},
            ))
        return events
    
    def _detect_hsv(self, frame: np.ndarray, tracks: list, frame_idx: int) -> list[Event]:
        """Smoking detection using HSV glow heuristic (fallback)."""
        import cv2
        events: list[Event] = []
        for tr in tracks:
            if getattr(tr, "cls", -1) != 0:
                continue
            tid = getattr(tr, "track_id", -1)
            if tid < 0:
                continue
            x1, y1, x2, y2 = tr.xyxy
            # focus on the upper portion of the person (face/hand area)
            upper_h = int((y2 - y1) * 0.4)
            fx1 = max(0, int(x1)); fy1 = max(0, int(y1))
            fx2 = min(frame.shape[1], int(x2)); fy2 = min(frame.shape[0], int(y1 + upper_h))
            if fx2 - fx1 < 20 or fy2 - fy1 < 20:
                continue
            crop = frame[fy1:fy2, fx1:fx2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            glow_mask = cv2.inRange(hsv, self.glow_hsv_low, self.glow_hsv_high)
            glow_area = cv2.countNonZero(glow_mask)
            if glow_area >= self.min_object_area:
                events.append(Event(
                    event_type="SMOKING",
                    t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    frame_idx=frame_idx,
                    details={"track_id": tid,
                             "glow_area": int(glow_area),
                             "method": "glow_heuristic_fallback"},
                ))
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

class GatheringDetector:
    """Detects personnel gathering: N+ people within a radius.

    No model needed — clusters tracked person centroids using a fixed-radius
    grouping (simpler than DBSCAN, no sklearn dependency). Fires when a cluster
    of >= min_people persons exists within radius_pixels.

    Config (`gathering` in models.yaml):
      min_people: minimum cluster size to trigger (default 3)
      radius_pixels: max distance between any two people in a cluster (default 150)
      cooldown_s: suppress re-trigger for the same cluster (default 10.0)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("gathering")
        self.min_people = int(self.cfg.get("min_people", 3))
        self.radius_pixels = float(self.cfg.get("radius_pixels", 150))
        self.cooldown_s = float(self.cfg.get("cooldown_s", 10.0))
        self._last_fire_t: float = 0.0

    def detect(self, tracks: list, frame_idx: int,
               t: float | None = None) -> list[Event]:
        if t is None:
            t = time.perf_counter()
        events: list[Event] = []
        # collect person centroids
        centroids: list[tuple[int, int, int]] = []   # (cx, cy, track_id)
        for tr in tracks:
            if getattr(tr, "cls", -1) != 0:
                continue
            tid = getattr(tr, "track_id", -1)
            if tid < 0:
                continue
            x1, y1, x2, y2 = tr.xyxy
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centroids.append((cx, cy, tid))
        if len(centroids) < self.min_people:
            return events

        # fixed-radius clustering: greedily group nearby centroids
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

        # check for clusters that meet the threshold
        if t - self._last_fire_t < self.cooldown_s:
            return events
        for cluster in clusters:
            if len(cluster) >= self.min_people:
                tids = [c[2] for c in cluster]
                events.append(Event(
                    event_type="GATHERING",
                    t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    frame_idx=frame_idx,
                    details={"count": len(cluster),
                             "track_ids": tids,
                             "radius_px": int(self.radius_pixels)},
                ))
                self._last_fire_t = t
                break   # one gathering event per frame
        return events


# =============================================================================
# 5. Violence/Fighting Detection — rule-based heuristic placeholder
# =============================================================================

class ViolenceDetector:
    """Rule-based violence/fighting detector.

    PRODUCTION TODO: Replace with a lightweight temporal action classifier
    (MoViNet-A0 per the spec). The current heuristic uses two signals:
      (a) Two tracked persons have significantly overlapping bboxes (IoU > threshold)
      (b) Rapid relative centroid motion between the two persons, sustained
          over a minimum duration window

    WEAK PLACEHOLDER — even after tightening based on live false-positive
    testing, this heuristic cannot distinguish fighting from handshakes, hugs,
    or normal close interaction. The VLM layer (Phase 6) is expected to
    disambiguate. Real violence detection needs a proper temporal action model.

    Live-testing changes:
      - IoU threshold raised from 0.1 -> 0.3 (was triggering on incidental
        overlap between people standing near each other)
      - Motion threshold raised from 15.0 -> 40.0 px (was triggering on
        normal talking/gesturing movement)
      - Min duration raised from 1.0s -> 1.5s (was firing on brief 3-frame
        proximity windows)
      - Motion must be sustained (not a single-frame spike): motion_active
        resets to False if any subsequent frame has low motion

    Config (`violence` in models.yaml):
      iou_threshold: bbox overlap to count as "close contact" (default 0.3)
      motion_threshold: relative centroid speed in px/frame (default 40.0)
      window_s: sustained contact+motion duration to trigger (default 1.5)
      cooldown_s: suppress re-trigger (default 10.0)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("violence")
        self.iou_threshold = float(self.cfg.get("iou_threshold", 0.3))
        self.motion_threshold = float(self.cfg.get("motion_threshold", 40.0))
        self.window_s = float(self.cfg.get("window_s", 1.5))
        self.cooldown_s = float(self.cfg.get("cooldown_s", 10.0))
        # per-pair state
        self._pair_state: dict[tuple[int, int], dict] = {}
        self._last_fire_t: float = 0.0

    def detect(self, tracks: list, frame_idx: int,
               t: float | None = None) -> list[Event]:
        if t is None:
            t = time.perf_counter()
        events: list[Event] = []

        # collect person bboxes + centroids
        persons: list[tuple[int, tuple, tuple]] = []   # (track_id, xyxy, centroid)
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

        # check all pairs
        active_pairs: set[tuple[int, int]] = set()
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                tid_a, box_a, cen_a = persons[i]
                tid_b, box_b, cen_b = persons[j]
                pair_key = (min(tid_a, tid_b), max(tid_a, tid_b))
                active_pairs.add(pair_key)

                # signal (a): significant bbox overlap
                iou = _compute_iou(box_a, box_b)
                if iou < self.iou_threshold:
                    # not overlapping enough — reset this pair's motion state
                    if pair_key in self._pair_state:
                        self._pair_state[pair_key]["motion_active"] = False
                        self._pair_state[pair_key]["contact_since"] = t
                    continue

                # signal (b): rapid relative motion (sustained)
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
                # motion must be SUSTAINED — a single slow frame resets motion_active
                if rel_motion >= self.motion_threshold:
                    st["motion_active"] = True
                else:
                    st["motion_active"] = False

                # check if both signals have been sustained for the full window
                if st["motion_active"] and (t - st["contact_since"]) >= self.window_s:
                    if t - self._last_fire_t >= self.cooldown_s:
                        events.append(Event(
                            event_type="VIOLENCE",
                            t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            frame_idx=frame_idx,
                            details={"pair": list(pair_key),
                                     "iou": round(iou, 3),
                                     "rel_motion": round(rel_motion, 1),
                                     "duration_s": round(t - st["contact_since"], 2),
                                     "method": "heuristic_placeholder_tightened"},
                        ))
                        self._last_fire_t = t
                    st["contact_since"] = t   # reset to avoid immediate re-fire
                    st["motion_active"] = False

        # expire pair state for pairs no longer active
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

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("object_left")
        self.min_stationary_s = float(self.cfg.get("min_stationary_s", 30.0))
        self.position_variance_threshold = float(self.cfg.get("position_variance_threshold", 100.0))
        self.cooldown_s = float(self.cfg.get("cooldown_s", 60.0))
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

            # Keep last N samples (30s at 30fps = 900 samples max)
            max_samples = int(self.min_stationary_s * 30 + 100)
            if len(history) > max_samples:
                history = history[-max_samples:]
                self._object_history[tid] = history

            # Check if object has been stationary long enough
            if len(history) < 30:  # need at least 1 second of data
                continue

            # Check cooldown
            if tid in self._last_fire_t and (t - self._last_fire_t[tid]) < self.cooldown_s:
                continue

            # Compute position variance over recent history
            recent = history[-int(self.min_stationary_s * 30 + 1):]
            if len(recent) < 30:
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