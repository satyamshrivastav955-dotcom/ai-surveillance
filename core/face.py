"""Face Recognition (Phase 3, Section 5).

Face detection via SCRFD-500M (small variant) + face embedding via
MobileFaceNet (w600k ArcFace-MobileNet variant), both from the insightface
buffalo_s model pack. Runs at a reduced cadence (every 3rd-5th frame on
person crops per Section 4) — faces don't need per-frame updates.

Face embeddings are stored in a separate FAISS index from body embeddings
(per the spec) so the two identity signals can be managed independently.

The insightface buffalo_s pack auto-downloads to ~/.insightface/models/ on
first use. The pack contains:
  det_500m.onnx   — SCRFD-500M face detection (the "small variant")
  w600k_mbf.onnx  — MobileFaceNet recognition (ArcFace-MobileNet, 512-dim)
  1k3d68.onnx     — 3D 68-point landmark (used for alignment)
  2d106det.onnx   — 2D 106-point landmark
  genderage.onnx  — gender + age estimation

CUDA: insightface uses onnxruntime internally. We add torch/lib to PATH
before loading so ORT can find cudnn64_9.dll (same fix as core/detector.py's
ONNX direct path).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import load_models_config


@dataclass
class FaceDetection:
    """A single detected face in a person crop."""
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in person-crop coords
    conf: float
    embedding: np.ndarray | None      # 512-dim L2-normalized, or None if not yet extracted


@dataclass
class FaceMatch:
    """Result of a face recognition attempt."""
    track_id: int
    name: str | None          # matched name, or None if no match above threshold
    similarity: float         # cosine similarity of best match (0..1)
    face_bbox: tuple[int, int, int, int] | None   # face bbox in full-frame coords


class FaceRecognizer:
    """Face detection + embedding + FAISS index for recognition.

    Wraps insightface's FaceAnalysis app (buffalo_s pack). The app handles
    face detection (SCRFD), alignment (landmarks), and embedding extraction
    (MobileFaceNet) in one `get()` call per image.

    Enrollment: `enroll(name, face_crop)` adds a face embedding to the FAISS
    index with a name label. Recognition: `recognize(face_crop)` searches the
    index and returns the best match above threshold.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        # fix cudnn DLL path for ORT CUDA EP (same as core/detector.py)
        try:
            import torch as _t
            torch_lib = os.path.join(os.path.dirname(_t.__file__), "lib")
            if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

        import faiss
        from insightface.app import FaceAnalysis

        self.cfg = cfg if cfg is not None else load_models_config()
        f = self.cfg.get("face", {})
        self.det_size = f.get("det_size", 320)     # SCRFD input size; 320 is fast
        self.match_threshold = float(f.get("match_threshold", 0.4))
        self.dim = int(f.get("dim", 512))           # MobileFaceNet output dim

        # insightface model pack — buffalo_s = SCRFD-500M + MobileFaceNet
        model_name = f.get("model_pack", "buffalo_s")
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))

        # separate FAISS index for face embeddings
        self.index = faiss.IndexFlatIP(self.dim)
        self._labels: list[str] = []     # parallel: name for each index entry
        self._embeddings: list[np.ndarray] = []   # keep copies for reconstruction

    def detect_and_embed(self, person_crop_bgr: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a person crop and extract embeddings for each.

        Returns one FaceDetection per detected face. Usually 0 or 1 for a
        single-person crop.
        """
        faces = self.app.get(person_crop_bgr)
        results = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            emb = f.embedding  # already L2-normalized by insightface? check
            # insightface embeddings are not always L2-normalized; normalize here
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                continue
            emb_norm = (emb / norm).astype(np.float32)
            results.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                conf=float(f.det_score),
                embedding=emb_norm,
            ))
        return results

    def recognize(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Search the face FAISS index for the best match.

        Returns (name, similarity). name is None if no match above threshold.
        """
        if self.index.ntotal == 0:
            return None, 0.0
        D, I = self.index.search(embedding.reshape(1, -1).astype(np.float32), 1)
        best_sim = float(D[0, 0])
        best_idx = int(I[0, 0])
        if best_idx < 0 or best_sim < self.match_threshold:
            return None, best_sim
        return self._labels[best_idx], best_sim

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """Add a face embedding to the FAISS index under `name`."""
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        self._labels.append(name)
        self._embeddings.append(embedding.copy())

    def process_person_crop(self, person_crop_bgr: np.ndarray,
                            crop_origin: tuple[int, int],
                            track_id: int) -> list[FaceMatch]:
        """Full pipeline: detect faces in crop, embed, recognize against index.

        `crop_origin` is (x_offset, y_offset) of the crop in the full frame,
        used to remap face bboxes to full-frame coords.
        Returns one FaceMatch per detected face.
        """
        detections = self.detect_and_embed(person_crop_bgr)
        results = []
        ox, oy = crop_origin
        for det in detections:
            if det.embedding is None:
                continue
            name, sim = self.recognize(det.embedding)
            fx1, fy1, fx2, fy2 = det.bbox
            results.append(FaceMatch(
                track_id=track_id,
                name=name,
                similarity=sim,
                face_bbox=(fx1 + ox, fy1 + oy, fx2 + ox, fy2 + oy),
            ))
        return results

    def save_index(self, path: str) -> None:
        """Save the FAISS index + labels to disk for persistence across runs."""
        import faiss
        import json
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p.with_suffix(".faiss")))
        with open(p.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self._labels, f)

    def load_index(self, path: str) -> None:
        """Load a previously saved FAISS index + labels."""
        import faiss
        import json
        from pathlib import Path
        p = Path(path)
        if p.with_suffix(".faiss").exists():
            self.index = faiss.read_index(str(p.with_suffix(".faiss")))
            with open(p.with_suffix(".json"), "r", encoding="utf-8") as f:
                self._labels = json.load(f)

    def reset(self) -> None:
        """Clear the face index (not the model)."""
        import faiss
        self.index = faiss.IndexFlatIP(self.dim)
        self._labels.clear()
        self._embeddings.clear()

    def vram_mb(self) -> int:
        try:
            import torch
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            return -1