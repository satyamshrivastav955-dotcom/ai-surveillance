"""Person Re-Identification (Phase 3 / Phase 5.1).

Lightweight body-embedding extraction + FAISS vector index for re-linking
tracks across brief occlusions / re-entries.

Backbone (Phase 5.1 upgrade):
  OSNet-x0.25 (torchreid) — 2.2M params, purpose-built for ReID.
  Trained on Market-1501 + DukeMTMC.  Same-person cosine sim ≥ 0.95;
  different-person sim ≈ 0.3–0.5.  Comfortable threshold at 0.65.

  Fallback: ImageNet ResNet18 (strip FC) if OSNet unavailable.
  Same-person sim ≈ 0.99; different-person sim ≈ 0.94 — razor's edge.

Both backbones produce 512-dim L2-normalized embeddings compatible with the
existing FAISS IndexFlatIP.  Switching backbones requires resetting the index
(embeddings from different backbones are not comparable).

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


def _build_osnet(device: str, half: bool) -> tuple[Any, int]:
    """Build OSNet-x0.25 with Market-1501 + DukeMTMC ReID-trained weights.

    Loads the model file directly via importlib to bypass torchreid's top-level
    __init__ (which eagerly imports the training engine and requires tensorboard).

    Weight loading priority:
      1. ~/.cache/torch/checkpoints/osnet_x0_25_market_duke.pth  (ReID trained)
      2. torchreid's pretrained_urls Google Drive download via init_pretrained_weights
      3. ImageNet pretrained (pretrained=True) — degraded; no person-discriminative
         signal (same limitation as ResNet18, different-person sim ≈ 0.94)

    ReID-trained checkpoint: same-person sim ≥ 0.95, different-person sim ≈ 0.3–0.5.
    Allows a comfortable match_threshold of 0.65 with a ~0.3 margin on each side.

    Returns (model, feature_dim=512).
    """
    import importlib.util
    import pathlib
    import torch

    # Locate osnet.py in the installed torchreid package (bypasses engine import)
    try:
        import torchreid as _tr_pkg
        _tr_root = pathlib.Path(_tr_pkg.__file__).parent
    except Exception:
        import sysconfig
        _tr_root = pathlib.Path(sysconfig.get_path("purelib")) / "torchreid"

    osnet_path = _tr_root / "reid" / "models" / "osnet.py"
    if not osnet_path.exists():
        raise FileNotFoundError(f"[reid] osnet.py not found at {osnet_path}")

    spec = importlib.util.spec_from_file_location("_osnet_mod", osnet_path)
    osnet_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(osnet_mod)

    # Build model without any pretrained weights first (we'll load manually)
    model = osnet_mod.osnet_x0_25(num_classes=1, pretrained=False)

    # Try to load the ReID-trained checkpoint from cache
    reid_ckpt = pathlib.Path.home() / ".cache" / "torch" / "checkpoints" / "osnet_x0_25_market_duke.pth"
    if reid_ckpt.exists():
        state = torch.load(str(reid_ckpt), map_location="cpu", weights_only=False)
        # torchreid checkpoints may be wrapped in {'state_dict': ...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # strip 'module.' prefix added by DataParallel if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # strip the classifier head — it was trained for N-way ID classification
        # (e.g. 751 Market-1501 IDs) and is NEVER used in feature extraction.
        # Removing it prevents size mismatch when model is built with num_classes=1.
        state = {k: v for k, v in state.items() if not k.startswith("classifier")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[reid] Loaded ReID checkpoint: {reid_ckpt.name}  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    else:
        # Fall back to torchreid's pretrained_urls download (Google Drive)
        print(f"[reid] ReID checkpoint not found at {reid_ckpt}, "
              f"falling back to torchreid pretrained download ...")
        osnet_mod.init_pretrained_weights(model, "osnet_x0_25")

    model.eval()
    model = model.to(device)
    if half:
        model = model.half()
    return model, model.feature_dim   # feature_dim == 512



def _build_resnet18(device: str, half: bool, input_size: int) -> tuple[Any, int]:
    """Fallback: stripped ResNet18 (ImageNet weights, FC replaced with Identity)."""
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    weights = tvm.ResNet18_Weights.DEFAULT if hasattr(tvm, "ResNet18_Weights") else None
    backbone = tvm.resnet18(weights=weights)
    backbone.fc = nn.Identity()   # strip FC → 512-dim avgpool features
    backbone = backbone.to(device)
    if half:
        backbone = backbone.half()
    backbone.eval()
    return backbone, 512


class ReIDExtractor:
    """Extracts 512-dim L2-normalized body embeddings from person crops.

    Backbone is selected from configs/models.yaml ``reid.model``:

    ``osnet_x0_25`` (default):
        Purpose-built for ReID (trained on Market-1501/DukeMTMC).
        Same-person sim ≥ 0.95, different-person sim ≈ 0.3–0.5.
        Safe match_threshold: 0.65 (wide margin).
        Crop size: 256 × 128 (tall portrait, standard for ReID).

    ``resnet18``:
        ImageNet backbone (general-purpose classifier, not ReID-trained).
        Same-person sim ≈ 0.99, different-person sim ≈ 0.94 (razor's edge).
        Requires match_threshold ≥ 0.95.
        Crop size: imgsz × imgsz (square).

    Both produce 512-dim L2-normalized embeddings — FAISS index is identical.
    Switching backbones requires resetting the index (embeddings incompatible).
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        import torch

        self.cfg = cfg if cfg is not None else load_models_config()
        r = self.cfg.get("reid", {})
        self.device = r.get("device", "cuda:0")
        self.half = r.get("half", True)
        self.input_size = r.get("imgsz", 256)   # used only for ResNet18 fallback

        model_name = r.get("model", "osnet_x0_25").lower()

        if model_name == "osnet_x0_25":
            try:
                self.model, self._feat_dim = _build_osnet(self.device, self.half)
                self._backbone = "osnet_x0_25"
                print(f"[reid] backbone=OSNet-x0.25  dim={self._feat_dim}  "
                      f"device={self.device}  half={self.half}")
            except Exception as e:
                print(f"[reid] WARNING: OSNet-x0.25 load failed ({e}); "
                      f"falling back to ResNet18. Match quality will be degraded.")
                self.model, self._feat_dim = _build_resnet18(
                    self.device, self.half, self.input_size)
                self._backbone = "resnet18_fallback"
                print(f"[reid] backbone=ResNet18 (fallback)  dim={self._feat_dim}")
        else:
            self.model, self._feat_dim = _build_resnet18(
                self.device, self.half, self.input_size)
            self._backbone = "resnet18"
            print(f"[reid] backbone=ResNet18 (dev)  dim={self._feat_dim}  "
                  f"device={self.device}  half={self.half}")

        # Warmup — compile CUDA kernels so the first real crop is fast
        _h = 256 if "osnet" in self._backbone else self.input_size
        _w = 128 if "osnet" in self._backbone else self.input_size
        dummy = torch.zeros(1, 3, _h, _w,
                            dtype=torch.float16 if self.half else torch.float32,
                            device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)

        self._transform = self._build_transform()

    def _build_transform(self):
        """Per-crop preprocessing.

        OSNet standard: 256 × 128 tall portrait crop (height × width).
        ResNet18 fallback: square imgsz × imgsz crop.
        Both use ImageNet mean/std normalization.
        """
        import torchvision.transforms as T
        if "osnet" in self._backbone:
            resize = T.Resize((256, 128))
        else:
            resize = T.Resize((self.input_size, self.input_size))
        return T.Compose([
            T.ToPILImage(),
            resize,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Extract a 512-dim L2-normalized embedding from a BGR person crop.

        Returns None if the crop is too small (< 32 px tall or < 16 px wide).
        """
        import torch
        h, w = crop_bgr.shape[:2]
        if h < 32 or w < 16:
            return None
        crop_rgb = crop_bgr[:, :, ::-1].copy()   # BGR → RGB
        tensor = self._transform(crop_rgb).unsqueeze(0).to(self.device)
        if self.half:
            tensor = tensor.half()
        with torch.no_grad():
            feat = self.model(tensor)   # (1, 512)
        feat = feat.float().cpu().numpy().flatten()
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
        self.match_threshold = float(r.get("match_threshold", 0.95))
        print(f"[reid] match_threshold={self.match_threshold} (from {'config' if 'match_threshold' in r else 'default'})")
        self.lost_ttl_s = float(r.get("lost_ttl_s", 30.0))

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
        for i in range(len(self._lost_at)):
            if self._is_lost[i] and self._lost_at[i] > 0 and (t - self._lost_at[i]) > self.lost_ttl_s:
                self._lost_at[i] = 0.0
    
    def rebuild_index(self) -> None:
        """Rebuild the FAISS index to remove stale entries and free memory.
        
        FAISS IndexFlatIP doesn't support deletion, so we periodically rebuild
        to remove accumulated stale entries.
        """
        import faiss

        active_indices = []
        active_embeddings = []
        
        for i in range(len(self._track_ids)):
            if self._is_lost[i] and self._lost_at[i] == 0.0:
                continue
            if self._is_lost[i] and (time.perf_counter() - self._lost_at[i]) > self.lost_ttl_s:
                continue
            
            active_indices.append(i)
            try:
                emb = self.index.reconstruct(i)
                active_embeddings.append(emb)
            except Exception:
                continue
        
        if not active_embeddings:
            # All entries are stale — replace with a fresh empty index.
            import faiss as _faiss
            self.index = _faiss.IndexFlatIP(self.dim)
            self._track_ids.clear()
            self._is_lost.clear()
            self._lost_at.clear()
            self._labels.clear()
            return

        
        new_index = faiss.IndexFlatIP(self.dim)
        new_index.add(np.array(active_embeddings, dtype=np.float32))
        
        self.index = new_index
        self._track_ids = [self._track_ids[i] for i in active_indices]
        self._is_lost = [self._is_lost[i] for i in active_indices]
        self._lost_at = [self._lost_at[i] for i in active_indices]
        self._labels = [self._labels[i] for i in active_indices]
        
    def get_index_size(self) -> int:
        """Return the number of entries in the index."""
        return self.index.ntotal

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