"""YOLOv8n-pose wrapper (Phase 2, Section 5).

Runs pose estimation **only on person crops** from the shared detector's
boxes — never the full frame. This is the throughput win: a 1280x720 frame
has at most a few small person crops, so we run pose at 160x160 per crop
instead of 640x640 on the whole frame.

Two runtime paths (constraint #3 — switchable from configs/models.yaml):
  runtime: pytorch     -> raw ultralytics YOLO.predict on CUDA (rollback path)
  runtime: onnx        -> ultralytics YOLO loaded from the .onnx export, ORT CUDA EP
  runtime: onnx_direct -> direct ORT session + hand-written pre/post (fastest,
                          bypasses ultralytics' Python wrapper; used for hot loop)

COCO-pose 17-keypoint layout (order matters — the fall detector indexes by name):
    0  nose
    1  left eye     2  right eye
    3  left ear     4  right ear
    5  left shoulder  6  right shoulder
    7  left elbow    8  right elbow
    9  left wrist   10 right wrist
    11 left hip     12 right hip
    13 left knee    14 right knee
    15 left ankle   16 right ankle
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import load_models_config
from core.detector import _dbg_device, _letterbox, _nms_numpy

_DEBUG_DEVICE = os.environ.get("DEBUG_DEVICE") == "1"

# Indexes the fall detector cares about (see header above for the full list).
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

# Pose ONNX output layout: (1, 56, N) where 56 = 4 (xywh) + 1 (conf) + 17*3 (kpt x,y,conf)
_POSE_BOX_OFFS = 4
_POSE_CONF_OFFS = 4
_POSE_KPT_OFFS = 5
_N_KPTS = 17


@dataclass
class Pose:
    """Pose result for a single person, in original-frame pixel coordinates.

    `keypoints` is shape (17, 3): (x_px, y_px, conf) per COCO keypoint.
    """
    track_id: int                      # from the upstream tracker (or -1 if none)
    xyxy: tuple[int, int, int, int]    # bbox in original-frame coords (from the detector)
    keypoints: np.ndarray              # (17, 3) float32 — x_px, y_px, conf
    conf: float                        # pose model's own person conf


class _OnnxDirectPose:
    """Direct ORT session for yolov8n-pose.onnx. Bypasses ultralytics' wrapper."""

    def __init__(self, onnx_path: str, imgsz: int, conf: float):
        import os
        # same cudnn PATH fix as the detector
        try:
            import torch as _t
            torch_lib = os.path.join(os.path.dirname(_t.__file__), "lib")
            if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )
        eps = self.sess.get_providers()
        if "CUDAExecutionProvider" not in eps:
            print(f"[pose] WARN: ORT CUDA EP unavailable, falling back to {eps}.")
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        # warmup
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        for _ in range(3):
            self.sess.run(None, {self.input_name: dummy})

    def estimate_crop(self, crop: np.ndarray, fx1: int, fy1: int, track_id: int) -> Pose | None:
        """Run pose on a single person crop; return Pose in full-frame coords."""
        # letterbox the crop to imgsz (pad bottom/right with 114)
        padded, ratio, pad_w, pad_h = _letterbox(crop, self.imgsz)
        img = padded.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.input_name: img})[0]
        # output: (1, 56, N) -> transpose to (N, 56)
        preds = out[0].T
        if preds.shape[1] != _POSE_KPT_OFFS + _N_KPTS * 3:
            return None
        # extract person conf (col 4) and find the best detection
        person_conf = preds[:, _POSE_CONF_OFFS]
        if person_conf.max() < self.conf:
            return None
        best = int(person_conf.argmax())
        # box xywh in padded coords -> xyxy in crop coords (no pad subtraction;
        # pad is bottom/right, so dividing by ratio gives crop coords directly)
        cx, cy, w, h = preds[best, :4]
        # keypoints: 17 * (x, y, conf) starting at col 5
        kpt_flat = preds[best, _POSE_KPT_OFFS:]   # (51,)
        kpts = kpt_flat.reshape(_N_KPTS, 3).astype(np.float32)
        # remap keypoints from padded coords to crop coords (divide by ratio)
        kpts[:, 0] /= ratio
        kpts[:, 1] /= ratio
        # remap from crop coords to full-frame coords (add crop origin)
        kpts[:, 0] += fx1
        kpts[:, 1] += fy1
        # we don't strictly need the box (caller already has it from the detector)
        # but build a clean one for the Pose dataclass
        bx1 = max(0, int((cx - w / 2) / ratio + fx1))
        by1 = max(0, int((cy - h / 2) / ratio + fy1))
        bx2 = int((cx + w / 2) / ratio + fx1)
        by2 = int((cy + h / 2) / ratio + fy1)
        return Pose(
            track_id=track_id,
            xyxy=(bx1, by1, bx2, by2),
            keypoints=kpts,
            conf=float(person_conf[best]),
        )


class PoseEstimator:
    """YOLOv8n-pose wrapper with runtime-switchable backend.

    `runtime` (from configs/models.yaml):
        pytorch     - raw ultralytics YOLO.predict on CUDA (rollback path)
        onnx        - ultralytics YOLO loaded from the .onnx export (ORT CUDA EP)
        onnx_direct - direct ORT session + hand-written pre/post (fastest)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else load_models_config()
        p = self.cfg["pose"]
        self.runtime = p.get("runtime", "pytorch")
        self.imgsz = p["imgsz"]
        self.half = p["half"]
        self.device = 0
        self.conf = p["conf"]

        if self.runtime == "onnx_direct":
            onnx_path = p.get("onnx") or p["weights"].replace(".pt", ".onnx")
            self._direct = _OnnxDirectPose(onnx_path, self.imgsz, self.conf)
            self.model = None
        else:
            from ultralytics import YOLO
            if self.runtime == "onnx":
                weights = p.get("onnx") or p["weights"].replace(".pt", ".onnx")
                self.model = YOLO(weights, task="pose")
            else:  # pytorch
                weights = p["weights"]
                self.model = YOLO(weights)
                self.model.fuse()
            _dbg_device(self.model, "pose_warmup")
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                dummy, imgsz=self.imgsz, device=0, half=self.half,
                conf=self.conf, verbose=False,
            )
            self._direct = None

    def estimate_crops(self, frame: np.ndarray, tracks) -> list[Pose]:
        """Run pose on every person crop extracted from `frame` per `tracks`.

        `tracks` is a list of core.tracker.Track (or any object with `.xyxy`
        and `.track_id`).
        """
        if not tracks:
            return []
        results = []
        for t in tracks:
            x1, y1, x2, y2 = t.xyxy
            fx1 = max(0, int(x1)); fy1 = max(0, int(y1))
            fx2 = min(frame.shape[1], int(x2)); fy2 = min(frame.shape[0], int(y2))
            if fx2 - fx1 < 16 or fy2 - fy1 < 16:
                continue
            crop = frame[fy1:fy2, fx1:fx2]
            if self._direct is not None:
                pose = self._direct.estimate_crop(crop, fx1, fy1, getattr(t, "track_id", -1))
                if pose is not None:
                    results.append(pose)
                continue
            # ultralytics path
            _dbg_device(self.model, "pose_infer")
            res = self.model.predict(
                crop, imgsz=self.imgsz, device=0, half=self.half,
                conf=self.conf, verbose=False,
            )[0]
            if res.keypoints is None or len(res.keypoints.xy) == 0:
                continue
            kpts_xy = res.keypoints.xy
            kpts_conf = res.keypoints.conf
            if hasattr(kpts_xy, "__len__") and len(kpts_xy) == 0:
                continue
            best_i = 0
            if len(kpts_xy) > 1:
                mean_confs = [float(c.mean()) for c in kpts_conf]
                best_i = int(int(np.argmax(mean_confs)))
            xy = kpts_xy[best_i].cpu().numpy().astype(np.float32)
            cf = kpts_conf[best_i].cpu().numpy().astype(np.float32)
            xy[:, 0] += fx1
            xy[:, 1] += fy1
            kpts = np.concatenate([xy, cf[:, None]], axis=1)
            person_conf = float(res.boxes.conf[best_i].cpu().numpy()) if res.boxes is not None and len(res.boxes) > 0 else 1.0
            results.append(Pose(
                track_id=getattr(t, "track_id", -1),
                xyxy=(fx1, fy1, fx2, fy2),
                keypoints=kpts,
                conf=person_conf,
            ))
        return results

    def vram_mb(self) -> int:
        try:
            import torch
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            return -1