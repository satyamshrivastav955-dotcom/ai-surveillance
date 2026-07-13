"""Shared YOLO detector wrapper (Phase 1, constraint #1 of Section 6).

There is exactly ONE YOLO instance in the whole process. Future feature
models (fire/smoke, etc.) reuse this same model pass — they never spawn a
second YOLO. Runs FP16 on CUDA to respect the 6GB VRAM budget.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import load_models_config  # also silences ultralytics' half-warning via logging filter

# Set DEBUG_DEVICE=1 to print the model's actual device immediately before
# every inference call. Used to confirm where compute is happening.
_DEBUG_DEVICE = os.environ.get("DEBUG_DEVICE") == "1"


def _dbg_device(model, where: str) -> None:
    if _DEBUG_DEVICE:
        try:
            print(f"[dbg:{where}] self.model.device={model.device} "
                  f"param.device={next(model.model.parameters()).device}", flush=True)
        except Exception as e:
            print(f"[dbg:{where}] <could not read device: {e}>", flush=True)


@dataclass
class Detection:
    xyxy: tuple[int, int, int, int]   # pixel coords in the *original* frame
    conf: float
    cls: int

    @property
    def tl(self) -> tuple[int, int]:
        return self.xyxy[0], self.xyxy[1]

    @property
    def br(self) -> tuple[int, int]:
        return self.xyxy[2], self.xyxy[3]


class Detector:
    """Ultralytics YOLO wrapper. Singleton-by-construction: instantiate once."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        from ultralytics import YOLO  # local import so importing core.detector stays cheap

        self.cfg = cfg if cfg is not None else load_models_config()
        d = self.cfg["detector"]
        self.imgsz = d["imgsz"]
        self.half = d["half"]
        # CRITICAL: pass device=0 (int CUDA index) explicitly on every inference
        # call, NOT just at construction. Ultralytics will otherwise infer the
        # device from the input tensor / model state, which can silently land on
        # CPU even when the fused weights live on cuda:0. Int 0 == "cuda:0".
        self.device = 0
        self.conf = d["conf"]
        self.iou = d["iou"]
        self.classes = d.get("classes")

        self.model = YOLO(d["weights"])
        # Warmup pass allocates FP16 weights + CUDA context so the first real
        # frame isn't hit with a cold start.
        self.model.fuse()
        _dbg_device(self.model, "warmup")
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(
            dummy, imgsz=self.imgsz, device=0, half=self.half,
            conf=self.conf, iou=self.iou, classes=self.classes, verbose=False,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run a single detection pass. Returns boxes in original-frame pixel coords."""
        _dbg_device(self.model, "detect")
        res = self.model.predict(
            frame, imgsz=self.imgsz, device=0, half=self.half,
            conf=self.conf, iou=self.iou, classes=self.classes, verbose=False,
        )[0]
        return self._parse(res)

    def detect_with_raw(self, frame: np.ndarray):
        """Return both Detections and the raw ultralytics Result (tracker uses this)."""
        _dbg_device(self.model, "detect_with_raw")
        res = self.model.predict(
            frame, imgsz=self.imgsz, device=0, half=self.half,
            conf=self.conf, iou=self.iou, classes=self.classes, verbose=False,
        )[0]
        return self._parse(res), res

    @staticmethod
    def _parse(res) -> list[Detection]:
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [
            Detection((int(x1), int(y1), int(x2), int(y2)), float(c), int(k))
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls)
        ]

    def vram_mb(self) -> int:
        """Best-effort VRAM consumed by this model (CUDA only)."""
        try:
            import torch
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            return -1