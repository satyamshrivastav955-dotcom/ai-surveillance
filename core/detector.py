"""Shared YOLO detector wrapper (Phase 1, constraint #1 of Section 6).

There is exactly ONE YOLO instance in the whole process. Future feature
models (fire/smoke, etc.) reuse this same model pass — they never spawn a
second YOLO. Runs FP16 on CUDA to respect the 6GB VRAM budget.

Two runtime paths (constraint #3 — switchable from configs/models.yaml):
  runtime: pytorch     -> raw ultralytics YOLO.predict on CUDA (rollback path)
  runtime: onnx        -> ultralytics YOLO loaded from the .onnx export,
                          using ONNX Runtime CUDA EP (safe A/B compare)
  runtime: onnx_direct -> direct ORT session + hand-written pre/post (fastest,
                          bypasses ultralytics' Python wrapper; used for hot loop)
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


# =============================================================================
# Direct ONNX Runtime path (no ultralytics wrapper)
# =============================================================================

def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Standard NMS on xyxy boxes + per-box scores. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def _letterbox(img: np.ndarray, new_shape: int) -> tuple[np.ndarray, float, float, float, float]:
    """Resize + pad to (new_shape, new_shape). Returns (out, ratio, (dw, dh) pre-split).

    Mirrors ultralytics' AblateLetterbox logic: scale to fit, pad bottom/right,
    color 114. Returns the padded image + the scale + the pad amounts (un-dw, un-dh)
    so callers can remap boxes back to the original frame.
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    pad_w = new_shape - new_unpad[0]
    pad_h = new_shape - new_unpad[1]
    # ultralytics pads bottom/right by default
    if img.shape[:2] != new_unpad[::-1]:
        img = _cv2_resize(img, new_unpad)
    top = left = 0
    out = _cv2_copy_make_border(img, pad_h, pad_w, top, left)
    return out, r, pad_w, pad_h


def _cv2_resize(img, sz):
    import cv2
    return cv2.resize(img, sz, interpolation=cv2.INTER_LINEAR)


def _cv2_copy_make_border(img, pad_h, pad_w, top, left):
    import cv2
    return cv2.copyMakeBorder(img, top, pad_h, left, pad_w, cv2.BORDER_CONSTANT, value=114)


class _OnnxDirectDetector:
    """Direct ORT session for yolov8n.onnx. Bypasses ultralytics' Python wrapper."""

    def __init__(self, onnx_path: str, imgsz: int, conf: float, iou: float,
                 classes: list[int] | None):
        import os
        # ORT CUDA EP needs cudnn64_9.dll; PyTorch ships it in torch/lib. Add
        # that dir to PATH before creating the session so ORT can find cuDNN.
        try:
            import torch as _t
            torch_lib = os.path.join(os.path.dirname(_t.__file__), "lib")
            if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
        import onnxruntime as ort
        # CUDAExecutionProvider on device 0 — same FP16 model runs on the GPU
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )
        # confirm the CUDA EP is actually active (warn loudly if it fell back)
        eps = self.sess.get_providers()
        if "CUDAExecutionProvider" not in eps:
            print(f"[detector] WARN: ORT CUDA EP unavailable, falling back to {eps}. "
                  f"Check cudnn64_9.dll + CUDA 12.x install.")
        self.input_name = self.sess.get_inputs()[0].name
        # ultralytics' detector ONNX expects NCHW float32 in [0,1] (FP16 weights
        # but ORT auto-casts the input to FP16 internally)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = set(classes) if classes is not None else None
        # warmup
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        for _ in range(3):
            self.sess.run(None, {self.input_name: dummy})

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # preprocess: letterbox to imgsz, normalize, HWC->CHW, add batch dim
        padded, ratio, pad_w, pad_h = _letterbox(frame, self.imgsz)
        img = padded.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]   # HWC -> 1CHW
        # inference
        out = self.sess.run(None, {self.input_name: img})[0]
        # yolov8 detect output: (1, 84, 8400) where 84 = 4 (xywh in pixel space of
        # the letterboxed input) + 80 class scores. Transpose -> (8400, 84).
        preds = out[0].T   # (N, 84)
        # first 4 cols = cx, cy, w, h in padded-image pixel coords
        boxes_xywh = preds[:, :4]
        # remaining 80 cols = class scores; take max per row
        cls_scores = preds[:, 4:]
        # filter by conf
        cls_ids = cls_scores.argmax(axis=1)
        max_scores = cls_scores.max(axis=1)
        keep_mask = max_scores >= self.conf
        if self.classes is not None:
            keep_mask &= np.isin(cls_ids, list(self.classes))
        if not keep_mask.any():
            return []
        boxes_xywh = boxes_xywh[keep_mask]
        max_scores = max_scores[keep_mask]
        cls_ids = cls_ids[keep_mask]
        # xywh -> xyxy (in padded coords)
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - w / 2; y1 = cy - h / 2; x2 = cx + w / 2; y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        # NMS per class (do a single NMS — for persons-only it's equivalent)
        keep_idx = _nms_numpy(boxes_xyxy, max_scores, self.iou)
        # remap to original frame coords (undo letterbox scale + pad)
        orig_h, orig_w = frame.shape[:2]
        results = []
        for i in keep_idx:
            # pad_w/pad_h are at the bottom/right of the padded canvas, so the
            # original image lives at the top-left; to remap a box from padded
            # coords back to original: orig = padded / ratio (no subtraction).
            bx1 = boxes_xyxy[i, 0] / ratio
            by1 = boxes_xyxy[i, 1] / ratio
            bx2 = boxes_xyxy[i, 2] / ratio
            by2 = boxes_xyxy[i, 3] / ratio
            bx1 = max(0, int(bx1)); by1 = max(0, int(by1))
            bx2 = min(orig_w - 1, int(bx2)); by2 = min(orig_h - 1, int(by2))
            results.append(Detection((bx1, by1, bx2, by2), float(max_scores[i]), int(cls_ids[i])))
        return results


class Detector:
    """YOLOv8n detector wrapper with runtime-switchable backend.

    `runtime` (from configs/models.yaml):
        pytorch     - raw ultralytics YOLO.predict on CUDA (rollback path)
        onnx        - ultralytics YOLO loaded from the .onnx export (ORT CUDA EP)
        onnx_direct - direct ORT session + hand-written pre/post (fastest)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else load_models_config()
        d = self.cfg["detector"]
        self.runtime = d.get("runtime", "pytorch")
        self.imgsz = d["imgsz"]
        self.half = d["half"]
        self.device = 0          # always GPU index 0
        self.conf = d["conf"]
        self.iou = d["iou"]
        self.classes = d.get("classes")

        if self.runtime == "onnx_direct":
            # bypass ultralytics entirely
            onnx_path = d.get("onnx") or d["weights"].replace(".pt", ".onnx")
            self._direct = _OnnxDirectDetector(
                onnx_path, self.imgsz, self.conf, self.iou, self.classes)
            self.model = None     # no ultralytics model in this mode
        else:
            from ultralytics import YOLO
            if self.runtime == "onnx":
                weights = d.get("onnx") or d["weights"].replace(".pt", ".onnx")
                self.model = YOLO(weights, task="detect")
            else:  # pytorch
                weights = d["weights"]
                self.model = YOLO(weights)
                self.model.fuse()
            _dbg_device(self.model, "warmup")
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                dummy, imgsz=self.imgsz, device=0, half=self.half,
                conf=self.conf, iou=self.iou, classes=self.classes, verbose=False,
            )
            self._direct = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run a single detection pass. Returns boxes in original-frame pixel coords."""
        if self._direct is not None:
            return self._direct.detect(frame)
        _dbg_device(self.model, "detect")
        res = self.model.predict(
            frame, imgsz=self.imgsz, device=0, half=self.half,
            conf=self.conf, iou=self.iou, classes=self.classes, verbose=False,
        )[0]
        return self._parse(res)

    def detect_with_raw(self, frame: np.ndarray):
        """Return both Detections and the raw ultralytics Result (tracker uses this).

        Only valid when runtime != onnx_direct (direct path has no ultralytics Result).
        """
        if self._direct is not None:
            # not supported in direct mode — return None for the raw result
            return self._direct.detect(frame), None
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