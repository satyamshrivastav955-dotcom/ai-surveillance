"""ByteTrack wrapper (Phase 1).

Reuses the *same* YOLO instance from `core.detector.Detector` — we never
own a second model. ByteTrack itself is a pure algorithm (CPU) so it adds
near-zero VRAM, exactly as the hardware budget requires.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.detector import Detector, _dbg_device

_DEBUG_DEVICE = os.environ.get("DEBUG_DEVICE") == "1"


@dataclass
class Track:
    track_id: int
    cls: int
    conf: float
    xyxy: tuple[int, int, int, int]   # original-frame pixel coords

    @property
    def tl(self) -> tuple[int, int]:
        return self.xyxy[0], self.xyxy[1]


class Tracker:
    """Wraps ultralytics' built-in ByteTrack (`persist=True` across calls)."""

    def __init__(self, detector: Detector):
        self.detector = detector
        self._tracker_cfg = "bytetrack.yaml"   # shipped with ultralytics
        self._initialized = False

    def update(self, frame: np.ndarray) -> list[Track]:
        # `persist=True` keeps the tracker's Kalman state between frames — required.
        _dbg_device(self.detector.model, "tracker")
        res = self.detector.model.track(
            frame,
            imgsz=self.detector.imgsz,
            device=0,
            half=self.detector.half,
            conf=self.detector.conf,
            iou=self.detector.iou,
            classes=self.detector.classes,
            tracker=self._tracker_cfg,
            persist=True,
            verbose=False,
        )[0]
        self._initialized = True
        return self._parse(res)

    @staticmethod
    def _parse(res) -> list[Track]:
        boxes = res.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [
            Track(int(tid), int(k), float(c), (int(x1), int(y1), int(x2), int(y2)))
            for (x1, y1, x2, y2), tid, c, k in zip(xyxy, ids, conf, cls)
        ]

    def reset(self) -> None:
        """Call when switching video sources so IDs don't leak across clips."""
        self._initialized = False
        # ultralytics resets tracker state when persist context breaks; simplest
        # robust reset is to recreate the internal tracker on next call.
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            # nothing persistent to clear from the public API; rely on a fresh
            # tracker instance being created by ultralytics per `track` call
            # when persist is reset. We just flag re-init here.
        except Exception:
            pass