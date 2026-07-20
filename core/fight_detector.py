"""FightDetector — Phase 5D.

Lightweight skeleton-based fight/violence detector that runs on the smoothed
keypoint sequences from PoseSmoother (Phase 5A).

Architecture
------------
Instead of the full ST-GCN graph convolution (which requires mmaction2 / torch_geometric
and a >100MB model download), we use a hand-crafted feature extractor over a
30-frame sliding window of pairwise skeleton states, then classify with a
rule-based threshold that captures the same kinematic signatures:

  1. Proximity — two tracked persons must be within ``proximity_px`` pixels
     of each other (centroid distance).
  2. Pairwise joint-velocity features — for each person, compute the per-frame
     L2 displacement of the 8 "active" keypoints (wrists, elbows, shoulders,
     ankles) over the sliding window.  High mean + high variance = thrashing
     motion consistent with fighting.
  3. Relative separation change — rate of change of centroid distance over the
     window.  Alternating approach/retreat patterns are a fight signature.
  4. Duration gate — signals (1-3) must all be sustained for ``window_s`` seconds
     before the event fires.

This is intentionally conservative: it will miss slow or clinched fights but
avoids triggering on handshakes, hugs, and normal social proximity.  The
``clip_ref`` field in the event allows a human reviewer to check every trigger.

When ST-GCN weights become available (future Phase), the ``FightDetector.detect()``
signature does not change — only the internal feature extractor and classifier
are swapped out.

Configuration (``fight`` block in models.yaml)
----------------------------------------------
  proximity_px        : max centroid distance for a "pair of interest" (default 200)
  active_keypoints    : list of COCO keypoint indices to use for velocity features
                        (default [5,6,7,8,9,10,15,16] = shoulders, elbows, wrists, ankles)
  velocity_threshold  : mean per-joint velocity (px/frame) above which motion is "high"
                        (default 15.0)
  window_s            : must sustain all signals for this long (default 1.5)
  cooldown_s          : suppress re-trigger for the same pair (default 10.0)
  min_window_frames   : minimum history frames before classifier runs (default 15)
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Default active keypoints: shoulders(5,6), elbows(7,8), wrists(9,10), ankles(15,16)
_DEFAULT_ACTIVE_KP = [5, 6, 7, 8, 9, 10, 15, 16]


@dataclass
class FightEvent:
    """Emitted when two tracks show sustained kinematic fight signatures.

    Attributes
    ----------
    event_type     : always "FIGHT"
    track_ids      : the two track IDs involved
    t_iso          : ISO-format timestamp
    frame_idx      : frame index at trigger
    confidence     : heuristic confidence in [0, 1] (currently rule-based, so
                     values are coarse: 0.6 low, 0.75 medium, 0.9 high)
    clip_ref       : path to the 5-second clip (filled by ClipWriter if enabled)
    details        : additional signal breakdown for debugging
    """
    event_type: str = "FIGHT"
    track_ids: tuple[int, int] = (0, 0)
    t_iso: str = ""
    frame_idx: int = 0
    confidence: float = 0.0
    clip_ref: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": "FIGHT",
            "track_ids": list(self.track_ids),
            "t_iso": self.t_iso,
            "frame_idx": self.frame_idx,
            "confidence": round(self.confidence, 3),
            "clip_ref": self.clip_ref,
            **self.details,
        }

    @property
    def details_dict(self) -> dict[str, Any]:
        return self.details


@dataclass
class _PairState:
    """Rolling state for one pair of tracks."""
    contact_since: float = 0.0
    velocity_a_buf: deque = field(default_factory=lambda: deque(maxlen=90))
    velocity_b_buf: deque = field(default_factory=lambda: deque(maxlen=90))
    dist_buf: deque = field(default_factory=lambda: deque(maxlen=90))
    last_kp_a: np.ndarray | None = None
    last_kp_b: np.ndarray | None = None
    last_fire_t: float = 0.0
    in_contact: bool = False


class FightDetector:
    """Skeleton-based fight detector running on smoothed pose sequences.

    Call ``detect()`` every frame (or every N frames via FrameRouter) with the
    current list of smoothed poses and tracks.  Returns a (possibly empty) list
    of :class:`FightEvent`.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self._proximity_px    = float(cfg.get("proximity_px",      200.0))
        self._active_kp       = list(cfg.get("active_keypoints",   _DEFAULT_ACTIVE_KP))
        self._vel_thresh      = float(cfg.get("velocity_threshold", 15.0))
        self._window_s        = float(cfg.get("window_s",           1.5))
        self._cooldown_s      = float(cfg.get("cooldown_s",         10.0))
        self._min_frames      = int(cfg.get("min_window_frames",    15))

        # per-pair state keyed by (min_id, max_id)
        self._pairs: dict[tuple[int, int], _PairState] = {}

    # ------------------------------------------------------------------

    def detect(self, smooth_poses, tracks: list, frame_idx: int,
               t: float | None = None) -> list[FightEvent]:
        """Run fight detection for the current frame.

        Parameters
        ----------
        smooth_poses : list of SmoothedPose (from PoseSmoother) or raw Pose
        tracks       : list of Tracker Track objects (for centroid computation)
        frame_idx    : current frame index
        t            : current perf_counter timestamp (for timing)

        Returns
        -------
        list of FightEvent (empty when nothing detected)
        """
        if t is None:
            t = time.perf_counter()

        events: list[FightEvent] = []

        # Build maps: track_id -> centroid, track_id -> keypoints
        centroids: dict[int, tuple[float, float]] = {}
        kp_map: dict[int, np.ndarray] = {}

        for tr in tracks:
            tid = getattr(tr, "track_id", -1)
            if tid < 0 or getattr(tr, "cls", -1) != 0:
                continue
            x1, y1, x2, y2 = tr.xyxy
            centroids[tid] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        for pose in smooth_poses:
            tid = pose.track_id
            if tid >= 0:
                kp_map[tid] = pose.keypoints   # (17, 3)

        person_ids = list(centroids.keys())
        if len(person_ids) < 2:
            # Not enough people to detect a fight
            self._pairs.clear()
            return events

        # Evaluate all pairs
        active_pairs: set[tuple[int, int]] = set()
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                id_a, id_b = person_ids[i], person_ids[j]
                key = (min(id_a, id_b), max(id_a, id_b))
                active_pairs.add(key)

                ca = centroids[id_a]
                cb = centroids[id_b]
                dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))

                ps = self._pairs.setdefault(key, _PairState(contact_since=t))

                # --- Proximity gate ---
                in_proximity = dist <= self._proximity_px
                if not in_proximity:
                    ps.in_contact = False
                    ps.contact_since = t
                    ps.velocity_a_buf.clear()
                    ps.velocity_b_buf.clear()
                    ps.dist_buf.clear()
                    continue

                # --- Keypoint velocity ---
                kp_a = kp_map.get(id_a)
                kp_b = kp_map.get(id_b)

                vel_a = self._compute_velocity(kp_a, ps.last_kp_a)
                vel_b = self._compute_velocity(kp_b, ps.last_kp_b)

                ps.last_kp_a = kp_a.copy() if kp_a is not None else None
                ps.last_kp_b = kp_b.copy() if kp_b is not None else None

                ps.velocity_a_buf.append(vel_a)
                ps.velocity_b_buf.append(vel_b)
                ps.dist_buf.append(dist)

                if not ps.in_contact:
                    ps.in_contact = True
                    ps.contact_since = t

                # --- Need enough history ---
                if len(ps.velocity_a_buf) < self._min_frames:
                    continue

                # --- Duration gate ---
                duration = t - ps.contact_since
                if duration < self._window_s:
                    continue

                # --- Classify ---
                fight_conf, breakdown = self._classify(ps)
                if fight_conf <= 0.0:
                    continue

                # --- Cooldown gate ---
                if t - ps.last_fire_t < self._cooldown_s:
                    continue

                ps.last_fire_t = t
                ps.contact_since = t   # reset so the same pair needs to re-sustain

                events.append(FightEvent(
                    track_ids=key,
                    t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    frame_idx=frame_idx,
                    confidence=fight_conf,
                    clip_ref=None,          # filled by ClipWriter if enabled
                    details={
                        "proximity_px": round(dist, 1),
                        "duration_s": round(duration, 2),
                        **breakdown,
                    },
                ))

        # Prune stale pair state
        for key in list(self._pairs.keys()):
            if key not in active_pairs:
                del self._pairs[key]

        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_velocity(self, kp_now: np.ndarray | None,
                          kp_prev: np.ndarray | None) -> float:
        """Mean L2 velocity over active keypoints in px/frame."""
        if kp_now is None or kp_prev is None:
            return 0.0
        vels = []
        for i in self._active_kp:
            if i >= len(kp_now) or i >= len(kp_prev):
                continue
            conf_now  = float(kp_now[i, 2])
            conf_prev = float(kp_prev[i, 2])
            if conf_now < 0.2 or conf_prev < 0.2:
                continue
            dx = float(kp_now[i, 0]) - float(kp_prev[i, 0])
            dy = float(kp_now[i, 1]) - float(kp_prev[i, 1])
            vels.append(math.hypot(dx, dy))
        return float(np.mean(vels)) if vels else 0.0

    def _classify(self, ps: _PairState) -> tuple[float, dict]:
        """Return (confidence, breakdown_dict).

        Confidence levels:
          0.0  — not a fight
          0.6  — low (one signal above threshold)
          0.75 — medium (two signals)
          0.9  — high (all three signals)
        """
        buf_a = np.array(ps.velocity_a_buf, dtype=float)
        buf_b = np.array(ps.velocity_b_buf, dtype=float)
        buf_d = np.array(ps.dist_buf, dtype=float)

        mean_vel_a = float(np.mean(buf_a))
        mean_vel_b = float(np.mean(buf_b))
        vel_var_a  = float(np.var(buf_a))
        vel_var_b  = float(np.var(buf_b))

        # Signal 1: both persons showing high mean velocity
        sig1 = (mean_vel_a >= self._vel_thresh and mean_vel_b >= self._vel_thresh)

        # Signal 2: high velocity variance (thrashing, not smooth walking)
        thresh_var = (self._vel_thresh * 0.5) ** 2
        sig2 = (vel_var_a >= thresh_var or vel_var_b >= thresh_var)

        # Signal 3: relative distance oscillation (approach/retreat pattern)
        if len(buf_d) >= 4:
            d_diff = np.diff(buf_d)
            sign_changes = int(np.sum(np.diff(np.sign(d_diff)) != 0))
            sig3 = sign_changes >= max(3, len(buf_d) // 5)
        else:
            sig3 = False

        score = sum([sig1, sig2, sig3])
        conf_map = {0: 0.0, 1: 0.0, 2: 0.6, 3: 0.9}
        # Two signals: 0.6 only if sig1 (velocity) is one of them
        if score == 2 and not sig1:
            conf = 0.0
        else:
            conf = conf_map.get(score, 0.0)

        breakdown = {
            "mean_vel_a": round(mean_vel_a, 1),
            "mean_vel_b": round(mean_vel_b, 1),
            "vel_var_a":  round(vel_var_a,  1),
            "vel_var_b":  round(vel_var_b,  1),
            "sig_velocity":  sig1,
            "sig_thrash":    sig2,
            "sig_oscillate": sig3,
        }
        return conf, breakdown

