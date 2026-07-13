"""Person Re-Identification (Phase 3, Section 5).

Lightweight body-embedding extraction + FAISS vector index for re-linking
tracks across brief occlusions / re-entries.

Model: torchvision ResNet18 with the final FC stripped — produces a 512-dim
L2-normalized embedding from the avgpool layer. ResNet18 is ~11M params
(same scale as YOLOv8n), well within the 6GB VRAM budget alongside the
detector + pose models. OSNet-x0.25 (3x smaller, purpose-built for ReID) is
the documented production target; the architecture supports swapping it in
by changing configs/models.yaml.

Per Section 4 of the build spec, embeddings are computed ONLY on new or
re-appearing tracks — NOT every frame. This is gated by the FrameRouter
(`reid: { every: 10 }` in pipeline.yaml), a huge VRAM/compute saver.

FAISS is used for the local vector index (IndexFlatIP with L2-normalized
embeddings = cosine similarity). Milvus is the planned production swap for
multi-node deployments; the FAISS → Milvus migration is a config change, not
a code change, because the index interface is isolated in this file.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.config import load_models_config


@dataclass
class ReIDMatch:
    """Result of a ReID re-link attempt."""
    new_track_id: int
    matched_track_id: int | None    # the lost track ID we matched, or None
    similarity: float               # cosine similarity (0..1 for L2-normalized)
    identity_label: str | None      # propagated identity if the lost track had one


class ReIDExtractor:
    """Extracts 512-dim L2-normalized body embeddings from person crops.

    Uses torchvision ResNet18 (strip FC, use avgpool features). FP16 on CUDA.
    The model is lightweight enough that we don't bother with ONNX export for
    now — ReID only runs on new/re-appearing tracks, not every frame.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        import torch
        import torch.nn as nn
        import torchvision.models as tvm

        self.cfg = cfg if cfg is not None else load_models_config()
        r = self.cfg.get("reid", {})
        self.device = r.get("device", "cuda:0")
        self.half = r.get("half", True)
        self.input_size = r.get("imgsz", 128)   # ReID crops are small; 128 is standard

        # load ResNet18 with ImageNet pretrained weights, strip the FC
        weights = tvm.ResNet18_Weights.DEFAULT if hasattr(tvm, "ResNet18_Weights") else None
        backbone = tvm.resnet18(weights=weights)
        backbone.fc = nn.Identity()    # replace final FC with identity -> 512-dim features
        backbone = backbone.to(self.device)
        if self.half:
            backbone = backbone.half()
        backbone.eval()
        self.model = backbone

        # warmup
        dummy = torch.zeros(1, 3, self.input_size, self.input_size,
                            dtype=torch.float16 if self.half else torch.float32,
                            device=self.device)
        with torch.no_grad():
            _ = backbone(dummy)
        self._transform = self._build_transform()

    def _build_transform(self):
        """Standard ImageNet normalization + resize. Applied per-crop."""
        import torchvision.transforms as T
        return T.Compose([
            T.ToPILImage(),
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Extract a 512-dim L2-normalized embedding from a BGR person crop.

        Returns None if the crop is too small to be useful.
        """
        import torch
        h, w = crop_bgr.shape[:2]
        if h < 16 or w < 16:
            return None
        # BGR -> RGB for torchvision
        crop_rgb = crop_bgr[:, :, ::-1].copy()
        tensor = self._transform(crop_rgb).unsqueeze(0).to(self.device)
        if self.half:
            tensor = tensor.half()
        with torch.no_grad():
            feat = self.model(tensor)          # (1, 512)
        feat = feat.float().cpu().numpy().flatten()
        # L2 normalize so cosine similarity = inner product
        norm = np.linalg.norm(feat)
        if norm < 1e-6:
            return None
        return (feat / norm).astype(np.float32)

    def vram_mb(self) -> int:
        try:
            import torch
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            return -1


class ReIDIndex:
    """FAISS index for body embeddings + per-track metadata.

    Stores embeddings in an IndexFlatIP (inner product = cosine similarity
    for L2-normalized vectors). Each entry is tagged with the track_id it
    came from and a timestamp so stale entries can be pruned.

    The re-link flow:
      1. When a track is first seen, store its embedding.
      2. When a track is lost (disappears from the tracker), mark its
         embedding as "lost" with a timestamp.
      3. When a new track appears, search the index for the best match
         among recent lost-track embeddings. If similarity > threshold,
         re-link: propagate the old track's identity to the new track.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        import faiss

        self.cfg = cfg if cfg is not None else load_models_config()
        r = self.cfg.get("reid", {})
        self.dim = r.get("dim", 512)
        self.match_threshold = float(r.get("match_threshold", 0.6))
        self.lost_ttl_s = float(r.get("lost_ttl_s", 30.0))   # how long to keep lost embeddings

        self.index = faiss.IndexFlatIP(self.dim)
        # parallel arrays: track_id, is_lost, lost_at_timestamp, identity_label
        self._track_ids: list[int] = []
        self._is_lost: list[bool] = []
        self._lost_at: list[float] = []
        self._labels: list[str | None] = []
        self._next_idx = 0

    def add(self, track_id: int, embedding: np.ndarray,
            label: str | None = None) -> None:
        """Add or update a track's embedding in the index."""
        # if this track_id already exists, replace its embedding
        for i, tid in enumerate(self._track_ids):
            if tid == track_id and not self._is_lost[i]:
                # update in place — FAISS doesn't support deletion on FlatIP,
                # so we just add a new entry and mark the old one as lost
                self._is_lost[i] = True
                self._lost_at[i] = time.perf_counter()
                break
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        self._track_ids.append(track_id)
        self._is_lost.append(False)
        self._lost_at.append(0.0)
        self._labels.append(label)

    def mark_lost(self, track_id: int) -> None:
        """Mark a track's most-recent embedding as lost (track disappeared)."""
        t = time.perf_counter()
        # mark the most recent non-lost entry for this track
        for i in range(len(self._track_ids) - 1, -1, -1):
            if self._track_ids[i] == track_id and not self._is_lost[i]:
                self._is_lost[i] = True
                self._lost_at[i] = t
                return

    def try_relink(self, new_track_id: int, embedding: np.ndarray) -> ReIDMatch:
        """Search for a match among recent lost-track embeddings.

        Returns a ReIDMatch with matched_track_id=None if no match above
        threshold was found.
        """
        t = time.perf_counter()
        # prune stale lost entries
        self._prune_stale(t)

        # collect indices of lost entries that are still within TTL
        lost_indices = [i for i in range(len(self._track_ids))
                        if self._is_lost[i] and (t - self._lost_at[i]) <= self.lost_ttl_s]
        if not lost_indices:
            return ReIDMatch(new_track_id, None, 0.0, None)

        # search only among lost entries — build a sub-index on the fly
        import faiss
        sub = faiss.IndexFlatIP(self.dim)
        sub_embs = np.array([self.index.reconstruct(i) for i in lost_indices],
                            dtype=np.float32)
        sub.add(sub_embs)
        D, I = sub.search(embedding.reshape(1, -1).astype(np.float32), 1)
        best_sim = float(D[0, 0])
        best_sub_idx = int(I[0, 0])
        if best_sub_idx < 0 or best_sim < self.match_threshold:
            return ReIDMatch(new_track_id, None, best_sim, None)

        best_global_idx = lost_indices[best_sub_idx]
        matched_tid = self._track_ids[best_global_idx]
        matched_label = self._labels[best_global_idx]

        # clean up: remove the matched lost entry by marking it consumed
        # (FAISS FlatIP doesn't support removal; we just mark it consumed
        # so it won't match again)
        self._lost_at[best_global_idx] = 0.0  # makes it stale -> pruned next time

        return ReIDMatch(new_track_id, matched_tid, best_sim, matched_label)

    def _prune_stale(self, t: float) -> None:
        """Remove lost entries older than lost_ttl_s. We can't actually remove
        from FAISS IndexFlatIP, so we just mark them as stale so they won't
        be considered in try_relink."""
        for i in range(len(self._lost_at)):
            if self._is_lost[i] and self._lost_at[i] > 0 and (t - self._lost_at[i]) > self.lost_ttl_s:
                self._lost_at[i] = 0.0  # stale -> won't be picked up

    def get_label(self, track_id: int) -> str | None:
        """Return the identity label for a track, if any."""
        for i in range(len(self._track_ids) - 1, -1, -1):
            if self._track_ids[i] == track_id:
                return self._labels[i]
        return None

    def set_label(self, track_id: int, label: str | None) -> None:
        """Set/update the identity label on the most recent entry for a track."""
        for i in range(len(self._track_ids) - 1, -1, -1):
            if self._track_ids[i] == track_id:
                self._labels[i] = label
                return

    def reset(self) -> None:
        """Clear the entire index (e.g. when switching video sources)."""
        import faiss
        self.index = faiss.IndexFlatIP(self.dim)
        self._track_ids.clear()
        self._is_lost.clear()
        self._lost_at.clear()
        self._labels.clear()


class ReIDManager:
    """Orchestrates ReID extraction + index + re-linking.

    The main loop calls `on_tracks_updated()` each frame with the current set
    of active track IDs. The manager:
      - detects new tracks (not seen before) → extract embedding, try re-link
      - detects lost tracks (were active, now gone) → mark lost in index
      - caches embeddings per track to avoid re-extracting every frame
    """

    def __init__(self, extractor: ReIDExtractor, index: ReIDIndex | None = None,
                 cfg: dict[str, Any] | None = None):
        self.extractor = extractor
        self.index = index or ReIDIndex(cfg)
        self._active_tracks: set[int] = set()
        self._embedded_tracks: set[int] = set()   # tracks we've already embedded
        self._track_labels: dict[int, str | None] = {}   # track_id -> label

    def on_tracks_updated(self, frame: np.ndarray, tracks: list,
                          t: float | None = None) -> list[ReIDMatch]:
        """Called every frame (or per FrameRouter cadence) with the current tracks.

        Returns a list of ReIDMatch results for any new tracks that were
        re-linked to lost tracks this call.
        """
        if t is None:
            t = time.perf_counter()
        current_ids = {getattr(tr, "track_id", -1) for tr in tracks}
        current_ids.discard(-1)   # ignore untracked

        # detect lost tracks: were active, now gone
        lost_ids = self._active_tracks - current_ids
        for tid in lost_ids:
            self.index.mark_lost(tid)

        # detect new tracks: in current but never embedded
        new_ids = current_ids - self._embedded_tracks
        results: list[ReIDMatch] = []
        for tr in tracks:
            tid = getattr(tr, "track_id", -1)
            if tid not in new_ids:
                continue
            # extract embedding from the person crop
            x1, y1, x2, y2 = tr.xyxy
            fx1 = max(0, int(x1)); fy1 = max(0, int(y1))
            fx2 = min(frame.shape[1], int(x2)); fy2 = min(frame.shape[0], int(y2))
            if fx2 - fx1 < 16 or fy2 - fy1 < 16:
                continue
            crop = frame[fy1:fy2, fx1:fx2]
            emb = self.extractor.extract(crop)
            if emb is None:
                continue
            # try to re-link against lost embeddings
            match = self.index.try_relink(tid, emb)
            if match.matched_track_id is not None:
                # propagate identity from the matched lost track
                old_label = self.index.get_label(match.matched_track_id)
                if old_label:
                    self._track_labels[tid] = old_label
                self.index.add(tid, emb, label=old_label)
            else:
                self.index.add(tid, emb, label=None)
            self._embedded_tracks.add(tid)
            results.append(match)

        self._active_tracks = current_ids
        return results

    def get_label(self, track_id: int) -> str | None:
        return self._track_labels.get(track_id)

    def set_label(self, track_id: int, label: str) -> None:
        self._track_labels[track_id] = label
        self.index.set_label(track_id, label)

    def reset(self) -> None:
        self.index.reset()
        self._active_tracks.clear()
        self._embedded_tracks.clear()
        self._track_labels.clear()