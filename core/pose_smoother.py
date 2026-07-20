"""Pose smoother — Phase 5A.

Applies three complementary smoothing techniques to remove single-frame skeleton
jitter before the data reaches fight detection and smoking gesture analysis:

  1. EMA bbox stabilisation — smooth the tracker bbox before cropping. A jittery
     crop causes jittery pose even with a perfect pose model.
  2. One-Euro filter (Casiez et al. 2012) per keypoint — low-latency adaptive
     filter that reduces lag on fast motions and noise on slow/static ones.
     Two parameters: min_cutoff (noise floor), beta (speed coefficient).
  3. Confidence-gated hold-last-good-value — when a keypoint's confidence drops
     below gate_conf, the raw detection is ignored and the last high-confidence
     position is held for up to hold_frames before marking the point missing.
  4. Teleport guard — if a keypoint moves > teleport_px in one frame (impossible
     for normal human motion at camera fps), re-initialise the filter at the new
     position rather than letting the filter lag behind an ID switch.

Usage::

    smoother = PoseSmoother(cfg=models_cfg.get("pose_smoother", {}))

    # inside the frame loop, after pose_est.estimate_crops():
    smooth_poses = smoother.update(raw_poses, ema_tracks=tracks)

The returned ``SmoothedPose`` list matches the ``Pose`` list shape so existing
fall detector and display code works without changes — just swap raw poses for
smoothed poses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# COCO 17-keypoint count
_N_KP = 17


# ---------------------------------------------------------------------------
# One-Euro filter (per scalar coordinate)
# ---------------------------------------------------------------------------

class _OneEuro:
    """One-Euro low-pass filter for a single scalar signal.

    Reference: Casiez, Roussel, Vogel (2012). "1€ Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems." CHI '12.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0):
        self._freq = freq
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self._freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self._freq
        edx = self._dx_prev + self._alpha(self._d_cutoff) * (dx - self._dx_prev)
        self._dx_prev = edx
        cutoff = self._min_cutoff + self._beta * abs(edx)
        x_filtered = self._x_prev + self._alpha(cutoff) * (x - self._x_prev)
        self._x_prev = x_filtered
        return x_filtered

    def reset(self, x: float) -> float:
        """Re-initialise at x (used on teleport / ID switch)."""
        self._x_prev = x
        self._dx_prev = 0.0
        return x


# ---------------------------------------------------------------------------
# Per-track, per-keypoint smoother state
# ---------------------------------------------------------------------------

@dataclass
class _KeypointState:
    """Filter state for a single keypoint of a single track."""
    fx: _OneEuro = field(default_factory=_OneEuro)
    fy: _OneEuro = field(default_factory=_OneEuro)
    last_good_xy: tuple[float, float] | None = None
    last_good_conf: float = 0.0
    hold_remaining: int = 0                 # frames left to hold the last good value
    missing: bool = True                    # True until first high-confidence detection


@dataclass
class _TrackState:
    """Smoother state for one track."""
    kp_states: list[_KeypointState] = field(
        default_factory=lambda: [_KeypointState() for _ in range(_N_KP)]
    )
    # EMA-smoothed bbox (x1, y1, x2, y2)
    ema_bbox: tuple[float, float, float, float] | None = None
    pose_quality: str = "missing"           # "stable" | "degraded" | "missing"


# ---------------------------------------------------------------------------
# SmoothedPose — same shape as core.pose.Pose so callers need no changes
# ---------------------------------------------------------------------------

@dataclass
class SmoothedPose:
    """Filtered pose result. Drop-in replacement for core.pose.Pose.

    Attributes
    ----------
    track_id : int
    xyxy     : smoothed bbox in full-frame coords
    keypoints: (17, 3) float32 — filtered x_px, y_px, conf (conf set to 0 when missing)
    conf     : original pose model confidence (unsmoothed)
    pose_quality : "stable" | "degraded" | "missing"
    """
    track_id: int
    xyxy: tuple[int, int, int, int]
    keypoints: np.ndarray   # (17, 3)
    conf: float
    pose_quality: str = "stable"


# ---------------------------------------------------------------------------
# PoseSmoother
# ---------------------------------------------------------------------------

class PoseSmoother:
    """Applies One-Euro + confidence-gating + EMA bbox to raw pose outputs.

    Parameters (from ``pose_smoother`` block in models.yaml):
      min_cutoff   : One-Euro min cutoff frequency (Hz). Higher = more responsive,
                     more noise. Lower = smoother, more lag. Default 1.0.
      beta         : One-Euro speed coefficient. Higher = faster adaptation.
                     Default 0.007.
      gate_conf    : Keypoint confidence below which the raw detection is
                     rejected in favour of hold-last-good. Default 0.30.
      hold_frames  : How many frames to hold the last good position when
                     confidence is low before marking the keypoint missing.
                     Default 6.
      ema_alpha    : EMA smoothing factor for bbox coordinates [0..1].
                     1.0 = no smoothing, 0.0 = frozen. Default 0.6.
      teleport_px  : If a keypoint moves more than this many pixels in one
                     frame, re-initialise the filter (likely an ID switch).
                     Default 100 px.
      freq         : Camera frame rate hint for One-Euro filter. Default 30.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self._min_cutoff  = float(cfg.get("min_cutoff",  1.0))
        self._beta        = float(cfg.get("beta",        0.007))
        self._gate_conf   = float(cfg.get("gate_conf",   0.30))
        self._hold_frames = int(cfg.get("hold_frames",   6))
        self._ema_alpha   = float(cfg.get("ema_alpha",   0.6))
        self._teleport_px = float(cfg.get("teleport_px", 100.0))
        self._freq        = float(cfg.get("freq",        30.0))

        self._tracks: dict[int, _TrackState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, raw_poses, ema_tracks=None) -> list[SmoothedPose]:
        """Smooth a list of raw Pose objects for the current frame.

        Parameters
        ----------
        raw_poses  : list of core.pose.Pose (or any obj with track_id, xyxy,
                     keypoints (17,3), conf).
        ema_tracks : list of tracker Tracks used to EMA-smooth bbox. If None,
                     bbox from the pose object is used.

        Returns
        -------
        list of SmoothedPose, same length as ``raw_poses``.
        """
        # Build a track->raw-bbox map from tracker output for EMA
        tracker_bboxes: dict[int, tuple] = {}
        if ema_tracks is not None:
            for t in ema_tracks:
                tid = getattr(t, "track_id", -1)
                if tid >= 0:
                    tracker_bboxes[tid] = tuple(t.xyxy)

        active_ids: set[int] = set()
        smoothed: list[SmoothedPose] = []

        for pose in raw_poses:
            tid = pose.track_id
            active_ids.add(tid)

            # Initialise per-track state if new track
            if tid not in self._tracks:
                self._tracks[tid] = _TrackState(
                    kp_states=[
                        _KeypointState(
                            fx=_OneEuro(self._freq, self._min_cutoff, self._beta),
                            fy=_OneEuro(self._freq, self._min_cutoff, self._beta),
                        )
                        for _ in range(_N_KP)
                    ]
                )
            ts = self._tracks[tid]

            # 1. EMA-smooth bbox
            raw_bbox = tracker_bboxes.get(tid) or tuple(int(v) for v in pose.xyxy)
            if ts.ema_bbox is None:
                ts.ema_bbox = tuple(float(v) for v in raw_bbox)
            else:
                a = self._ema_alpha
                ts.ema_bbox = tuple(
                    a * r + (1 - a) * e
                    for r, e in zip(raw_bbox, ts.ema_bbox)
                )
            smooth_bbox = tuple(int(v) for v in ts.ema_bbox)

            # 2. Per-keypoint One-Euro + confidence gating
            kp_raw = pose.keypoints  # (17, 3)
            kp_out = np.zeros((_N_KP, 3), dtype=np.float32)
            n_stable = 0
            n_missing = 0

            for i in range(_N_KP):
                x_r, y_r, conf_r = float(kp_raw[i, 0]), float(kp_raw[i, 1]), float(kp_raw[i, 2])
                ks = ts.kp_states[i]

                if conf_r >= self._gate_conf:
                    # Teleport guard: if filter has a previous value and the raw
                    # position jumps too far, re-init rather than lagging badly
                    if ks.last_good_xy is not None:
                        dx = x_r - ks.last_good_xy[0]
                        dy = y_r - ks.last_good_xy[1]
                        if math.hypot(dx, dy) > self._teleport_px:
                            ks.fx.reset(x_r)
                            ks.fy.reset(y_r)

                    x_f = ks.fx.filter(x_r)
                    y_f = ks.fy.filter(y_r)
                    ks.last_good_xy = (x_f, y_f)
                    ks.last_good_conf = conf_r
                    ks.hold_remaining = self._hold_frames
                    ks.missing = False
                    kp_out[i] = [x_f, y_f, conf_r]
                    n_stable += 1
                else:
                    # Low confidence — hold last good or mark missing
                    if ks.hold_remaining > 0 and ks.last_good_xy is not None:
                        ks.hold_remaining -= 1
                        x_f, y_f = ks.last_good_xy
                        # Decayed confidence so downstream knows it's held
                        held_conf = ks.last_good_conf * (ks.hold_remaining / self._hold_frames)
                        kp_out[i] = [x_f, y_f, max(0.0, held_conf)]
                        n_stable += 1
                    else:
                        ks.missing = True
                        ks.hold_remaining = 0
                        kp_out[i] = [0.0, 0.0, 0.0]
                        n_missing += 1

            # 3. Quality label
            if n_missing == 0:
                quality = "stable"
            elif n_stable >= _N_KP // 2:
                quality = "degraded"
            else:
                quality = "missing"
            ts.pose_quality = quality

            smoothed.append(SmoothedPose(
                track_id=tid,
                xyxy=smooth_bbox,
                keypoints=kp_out,
                conf=pose.conf,
                pose_quality=quality,
            ))

        # Prune stale track states (tracks that disappeared this frame)
        stale = [tid for tid in self._tracks if tid not in active_ids]
        for tid in stale:
            del self._tracks[tid]

        return smoothed

    def get_quality(self, track_id: int) -> str:
        """Return the last known pose quality for a track (or 'missing')."""
        ts = self._tracks.get(track_id)
        return ts.pose_quality if ts is not None else "missing"

    def reset_track(self, track_id: int) -> None:
        """Drop all smoother state for a track (call on explicit ID reset)."""
        self._tracks.pop(track_id, None)
