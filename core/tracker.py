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
    """Wraps ultralytics' built-in ByteTrack (`persist=True` across calls).

    When the detector is in `onnx_direct` mode (no ultralytics model), we
    fall back to: detector.detect() -> wrap detections in a minimal ultralytics
    Results-like object -> run ultralytics' BYTETracker.update() directly.
    This keeps the tracking algorithm identical across all detector runtimes.
    """

    def __init__(self, detector: Detector):
        self.detector = detector
        self._tracker_cfg = "bytetrack.yaml"   # shipped with ultralytics
        self._initialized = False
        # for onnx_direct mode: hold a persistent BYTETracker instance
        self._direct_tracker = None
        self._direct_args = None

    def update(self, frame: np.ndarray) -> list[Track]:
        if self.detector._direct is not None:
            return self._update_direct(frame)
        # ultralytics path: detector.model is a YOLO; use its .track() method
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

    def _update_direct(self, frame: np.ndarray) -> list[Track]:
        """Path for onnx_direct detector mode: run BYTETracker.update() directly
        on the detector's plain Detection objects. Reuses ultralytics' tracker
        internals so the algorithm matches the ultralytics path bit-for-bit."""
        dets = self.detector.detect(frame)
        # lazily build the BYTETracker + its args namespace
        if self._direct_tracker is None:
            from ultralytics.cfg import get_cfg
            from ultralytics.trackers.byte_tracker import BYTETracker
            from pathlib import Path
            import ultralytics
            cfg_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / self._tracker_cfg
            self._direct_args = get_cfg(cfg_path)
            self._direct_tracker = BYTETracker(self._direct_args)
        # build a minimal Results-like object for BYTETracker.update()
        # it expects results.boxes.xyxy, results.boxes.conf, results.boxes.cls,
        # results.boxes.xywh — all in original-frame pixel coords
        if not dets:
            results = _MinimalResults.empty()
        else:
            xyxy = np.array([d.xyxy for d in dets], dtype=np.float32)
            conf = np.array([d.conf for d in dets], dtype=np.float32)
            cls = np.array([d.cls for d in dets], dtype=np.float32)
            results = _MinimalResults(xyxy, conf, cls)
        out = self._direct_tracker.update(results, img=frame)
        # output is a numpy array of [x1, y1, x2, y2, id, score, cls, idx]
        tracks = []
        if out is not None and len(out) > 0:
            for row in out:
                # idx (col 7) is the detection index — drop it
                x1, y1, x2, y2, tid, conf, cls = row[:7]
                tracks.append(Track(int(tid), int(cls), float(conf),
                                    (int(x1), int(y1), int(x2), int(y2))))
        self._initialized = True
        return tracks

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
        self._direct_tracker = None   # force rebuild on next update()


class _MinimalResults:
    """Minimal stand-in for ultralytics.Results that BYTETracker.update()
    consumes. The tracker reads `.conf`, `.cls`, `.xywh`, `.xyxy`, and slices
    with boolean masks (`results[mask]` -> another _MinimalResults)."""
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        # xywh needed by BYTETracker — convert xyxy -> xywh
        if len(xyxy) > 0:
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
            w = x2 - x1; h = y2 - y1
            self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __getitem__(self, mask) -> "_MinimalResults":
        # support boolean-array slicing like numpy
        return _MinimalResults(self.xyxy[mask], self.conf[mask], self.cls[mask])

    def __len__(self) -> int:
        return len(self.conf)

    @classmethod
    def empty(cls) -> "_MinimalResults":
        empty_xyxy = np.zeros((0, 4), dtype=np.float32)
        return cls(empty_xyxy, np.zeros((0,), dtype=np.float32),
                   np.zeros((0,), dtype=np.float32))


class _MinimalBoxes:
    def __init__(self, xyxy, xywh, conf, cls):
        self.xyxy = xyxy
        self.xywh = xywh
        self.conf = conf
        self.cls = cls