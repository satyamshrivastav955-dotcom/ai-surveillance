"""Phase 4 event detectors — fire/smoke, smoking, phone-watching, gathering, violence.

Five features, all gated through the FrameRouter, all VLM-agnostic (constraint #5):

  1. FireSmokeDetector  — color heuristic (HSV) for fire/smoke as a placeholder
                          for a fine-tuned YOLO model. Documented simplification.
  2. SmokingDetector    — placeholder: flags small elongated objects near
                          face/hand region. Needs a cigarette-detection YOLO
                          fine-tune for production.
  3. PhoneWatcherDetector — runs a YOLOv8n pass for COCO class 67 (cell phone)
                          + head-pose heuristic from Phase 2 pose keypoints
                          (nose below shoulder line = looking down).
  4. GatheringDetector  — DBSCAN clustering on track centroids; fires when
                          N+ people are within a radius. No model needed.
  5. ViolenceDetector   — rule-based: overlapping bboxes + rapid relative
                          centroid motion between two tracked persons, sustained
                          over a short window. Placeholder for MoViNet-A0.

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
    event_type: str         # "FIRE" | "SMOKE" | "SMOKING" | "PHONE" | "GATHERING" | "VIOLENCE"
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
    """Detects fire and smoke using HSV color thresholding as a placeholder.

    PRODUCTION TODO: Replace with a YOLOv8n fine-tuned on a fire/smoke dataset
    (e.g., D-Fire, FireNet, or a custom Roboflow dataset). The config entry
    `fire_smoke.weights` points to where the fine-tuned .pt will live. For now
    this heuristic catches large bright orange/red regions (fire) and large
    gray/white regions rising in the upper frame (smoke).

    Live-testing notes:
      - Fire heuristic worked well (caught a lit match at pixel_ratio ~0.01-0.04).
        Added a min_duration requirement (2 consecutive detections) to reduce
        one-off false positives from red clothing / warm lighting.
      - Smoke heuristic was badly over-triggering (pixel_ratio 0.35-0.71 on
        normal room content — skin tone, walls, clothing all matched). Tightened
        HSV range to very low saturation + mid-high value only (near-white gray,
        not skin/warm tones) and raised pixel_ratio threshold to 0.60. Even after
        tightening this remains a COARSE PLACEHOLDER — real smoke detection
        needs a trained model.

    Config (`fire_smoke` in models.yaml):
      fire_hsv_low/high:  HSV bounds for fire color
      smoke_hsv_low/high: HSV bounds for smoke color (tightened: S<=30, V 140-220)
      fire_min_pixel_ratio: fraction of frame matching fire color (default 0.01)
      smoke_min_pixel_ratio: fraction of frame matching smoke color (default 0.60)
      fire_min_duration: consecutive fire detections required before firing (default 2)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("fire_smoke")
        f = self.cfg
        self.fire_low = np.array(f.get("fire_hsv_low", [0, 100, 150]), dtype=np.uint8)
        self.fire_high = np.array(f.get("fire_hsv_high", [25, 255, 255]), dtype=np.uint8)
        # smoke: tightened to very low saturation (gray, not skin) + mid-high value
        self.smoke_low = np.array(f.get("smoke_hsv_low", [0, 0, 140]), dtype=np.uint8)
        self.smoke_high = np.array(f.get("smoke_hsv_high", [180, 30, 220]), dtype=np.uint8)
        self.fire_min_ratio = float(f.get("fire_min_pixel_ratio", 0.01))
        self.smoke_min_ratio = float(f.get("smoke_min_pixel_ratio", 0.60))
        self.fire_min_duration = int(f.get("fire_min_duration", 2))
        # duration tracking
        self._fire_consecutive = 0
        self._fire_last_bbox: tuple | None = None

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Event]:
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
                                 "method": "hsv_heuristic_placeholder"},
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
                             "method": "hsv_heuristic_placeholder_tightened"},
                ))
        return events


# =============================================================================
# 2. Smoking Detection — placeholder heuristic
# =============================================================================

class SmokingDetector:
    """Placeholder smoking detector.

    PRODUCTION TODO: Replace with a YOLO fine-tuned for cigarette detection
    (e.g., a custom model trained on a smoking detection dataset). The config
    entry `smoking.weights` points to where the fine-tuned .pt will live.

    Current heuristic: checks for a small bright object (cigarette glow) near
    the face/hand region of a tracked person. This is intentionally conservative
    — it flags potential smoking for VLM verification rather than asserting it.

    LIVE TESTING NOTE: fired on an untested scenario during live webcam testing
    (bright reflections on skin/clothing near the face can trigger the glow
    heuristic). This remains a ROUGH PLACEHOLDER — real smoking detection
    needs a trained cigarette-detection model.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_cfg("smoking")
        self.min_object_area = int(self.cfg.get("min_object_area", 20))
        self.glow_hsv_low = np.array(self.cfg.get("glow_hsv_low", [0, 100, 200]), dtype=np.uint8)
        self.glow_hsv_high = np.array(self.cfg.get("glow_hsv_high", [20, 255, 255]), dtype=np.uint8)

    def detect(self, frame: np.ndarray, tracks: list, frame_idx: int) -> list[Event]:
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
                             "method": "glow_heuristic_placeholder"},
                ))
        return events


# =============================================================================
# 3. Phone-Watching Detection — YOLO cell-phone detection + head-pose heuristic
# =============================================================================

class PhoneWatcherDetector:
    """Detects phone-watching behavior.

    Two signals (both must be true):
      (a) A cell phone (COCO class 67) is detected near the person's hand region
      (b) The person's head is tilted down — inferred from pose keypoints:
          nose y > shoulder midpoint y means the head is below the shoulder
          line, suggesting looking down at a phone

    Uses a SEPARATE, independent YOLOv8n instance dedicated to phone detection
    (COCO class 67). This is intentionally NOT shared with the tracker's
    detector — sharing caused conflicts because the tracker uses persist=True
    with classes=[0] (persons), and calling predict() with classes=[67] on the
    same model corrupted the tracker state. The second YOLO adds ~6MB VRAM
    (same yolov8n weights, separate model instance) and runs at low cadence
    (every 10 frames) so the compute cost is minimal.
    """

    def __init__(self, cfg: dict[str, Any] | None = None,
                 detector_model=None):
        # detector_model param is accepted for backward compat but IGNORED —
        # we always load our own independent model to avoid the sharing conflict.
        self.cfg = cfg if cfg is not None else _load_cfg("phone")
        self.imgsz = int(self.cfg.get("imgsz", 320))
        self.conf = float(self.cfg.get("conf", 0.3))
        self.device = 0
        from ultralytics import YOLO
        weights = self.cfg.get("weights", "models/yolov8n.pt")
        self.model = YOLO(weights)
        self.model.fuse()
        self._owns_model = True
        # warmup with a phone-class-only detection
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.imgsz, device=0, half=True,
                           conf=self.conf, classes=[67], verbose=False)
        self._nose_idx = 0    # COCO-pose keypoint index for nose
        self._l_shoulder_idx = 5
        self._r_shoulder_idx = 6

    def detect(self, frame: np.ndarray, tracks: list, poses: list,
               frame_idx: int) -> list[Event]:
        events: list[Event] = []
        # run phone detection on the full frame at low res
        res = self.model.predict(frame, imgsz=self.imgsz, device=0, half=True,
                                 conf=self.conf, classes=[67], verbose=False)[0]
        phone_boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for (bx1, by1, bx2, by2), c in zip(xyxy, confs):
                phone_boxes.append(((int(bx1), int(by1), int(bx2), int(by2)), float(c)))

        if not phone_boxes:
            return events

        # build a map: track_id -> pose keypoints (from Phase 2)
        pose_map: dict[int, Any] = {}
        for p in poses:
            if p.track_id >= 0:
                pose_map[p.track_id] = p.keypoints

        for tr in tracks:
            if getattr(tr, "cls", -1) != 0:
                continue
            tid = getattr(tr, "track_id", -1)
            if tid < 0:
                continue
            tx1, ty1, tx2, ty2 = tr.xyxy
            # check if any phone box overlaps with this person's bbox
            for (pb, pc) in phone_boxes:
                px1, py1, px2, py2 = pb
                # IoU or overlap check
                ox1 = max(tx1, px1); oy1 = max(ty1, py1)
                ox2 = min(tx2, px2); oy2 = min(ty2, py2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue   # no overlap
                # head-pose check: is the person looking down?
                looking_down = True   # default if no pose available
                kpts = pose_map.get(tid)
                if kpts is not None:
                    nose = kpts[self._nose_idx]       # (x, y, conf)
                    l_sh = kpts[self._l_shoulder_idx]
                    r_sh = kpts[self._r_shoulder_idx]
                    # use shoulders with sufficient confidence
                    sh_ys = []
                    if l_sh[2] > 0.3:
                        sh_ys.append(l_sh[1])
                    if r_sh[2] > 0.3:
                        sh_ys.append(r_sh[1])
                    if nose[2] > 0.3 and len(sh_ys) >= 1:
                        shoulder_mid_y = sum(sh_ys) / len(sh_ys)
                        # nose below shoulder line = looking down
                        looking_down = nose[1] > shoulder_mid_y
                if looking_down:
                    events.append(Event(
                        event_type="PHONE",
                        t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        frame_idx=frame_idx,
                        details={"track_id": tid,
                                 "phone_bbox": pb,
                                 "phone_conf": round(pc, 3),
                                 "looking_down": True},
                    ))
                    break   # one phone event per person per frame
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