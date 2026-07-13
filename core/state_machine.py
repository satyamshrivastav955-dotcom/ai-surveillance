"""Phase 2 fall-detection state machine (Section 5).

Rule-based and intentionally VLM-agnostic (constraint #5): emits generic
FALL events that go to the event bus later in Phase 5 whether or not the
VLM layer exists.

A fall is defined as the conjunction of three signals on a *single track*
within a short window:

  (a) bbox aspect-ratio transitions from vertical (h > w) to clearly
      horizontal (w > h * aspect_threshold) — a person lying flat, not
      merely leaning or sitting
  (b) hip / shoulder keypoints drop below a height fraction of the bbox
      (body's centre of mass is low)
  (c) the (a)+(b) transition completes within `transition_window_s`

  AND (d) the track was *stably* upright for at least `min_upright_s`
      seconds before the transition — a single upright frame during
      fidgeting does NOT re-arm the trigger (this is the key fix vs the
      original implementation, which re-fired every few seconds when a
      seated person shifted posture).

The window filters slow sitting-down: someone lowering themselves into a
chair over 3-4 seconds should NOT trip the detector, but a person's centre
of mass dropping in <1.5s should.

Per-track state is kept in a dict keyed by track_id so multiple simultaneous
falls are tracked independently. Stale tracks are GC'd when not seen for
a while.

Debug: set env FALL_DEBUG=1 to log every TRIGGER and every "would-trigger-
but-suppressed" event with the full signal breakdown (aspect, kp frac,
upright duration, cooldown status). Useful for tuning thresholds.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.pose import (
    KP_LEFT_HIP, KP_LEFT_SHOULDER, KP_RIGHT_HIP, KP_RIGHT_SHOULDER, Pose,
)

_FALL_DEBUG = os.environ.get("FALL_DEBUG") == "1"
_FALL_DEBUG_LIMIT = int(os.environ.get("FALL_DEBUG_LIMIT", "10"))


@dataclass
class FallEvent:
    track_id: int
    t_iso: str
    frame_idx: int
    aspect_now: float
    keypoints_low: bool
    kp_height_frac: float               # measured fraction at trigger time
    upright_duration_s: float           # how long the track was upright before the fall
    transition_delta_s: float           # time from last upright to fallen

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": "FALL",
            "track_id": self.track_id,
            "t_iso": self.t_iso,
            "frame_idx": self.frame_idx,
            "aspect_now": round(self.aspect_now, 3),
            "keypoints_low": bool(self.keypoints_low),
            "kp_height_frac": round(self.kp_height_frac, 3),
            "upright_duration_s": round(self.upright_duration_s, 3),
            "transition_delta_s": round(self.transition_delta_s, 3),
        }


@dataclass
class _TrackState:
    """Rolling per-track state for the fall detector."""
    last_aspect: float = 0.0           # most recent w/h
    # "Armed" semantics: the track was continuously upright for >= min_upright_s
    # at some point, and that upright period ended at `armed_at`. The track is
    # eligible to fire a fall if it sees FALLEN within (armed_at + transition_window_s).
    # Once a fall fires OR the window expires, the track disarms and must
    # re-accumulate min_upright_s of continuous uprightness to re-arm.
    armed_at: float | None = None      # time the upright period ended (start of fall window)
    upright_since: float | None = None # start of the current continuous upright run
    last_seen: float = 0.0
    in_cooldown_until: float = 0.0       # suppress re-trigger after a FALL
    # debug: count of triggers logged so we can stop spamming after N
    _debug_log_count: int = 0
    # cache of last frame's signals for "suppressed trigger" debug logging
    _last_kp_frac: float = 0.0


class FallDetector:
    """Stateful rule-based fall detector. Updates per-frame, emits FallEvent.

    Operational definition (cleaner than "flip then check"):
      UPRIGHT     = bbox vertical AND keypoints high (a standing/sitting-up person)
      FALLEN      = bbox clearly horizontal AND keypoints low
      TRIGGER     = transition UPRIGHT -> FALLEN that completes within
                    `transition_window_s`, where UPRIGHT was held continuously
                    for at least `min_upright_s` first.

    Config (`fall` section of models.yaml):
      aspect_threshold:     bbox w/h above this counts as "horizontal".
                            A genuine lying-flat fall produces ~1.5-2.0+; a
                            seated/leaning person is typically 0.8-1.2.
                            Default 1.5 (was 1.0 — too permissive, fired on
                            sitting/leaning).
      keypoint_height_frac: hip/shoulder y must be at or below this fraction of
                            the bbox's vertical extent (measured from the top)
                            to count as "low". 0.60 was too low (sitting puts
                            hips at ~0.55-0.65 of bbox height). Default 0.75.
      transition_window_s:  max elapsed time from last UPRIGHT to first FALLEN.
      cooldown_s:          suppress repeated FALL events on same track AFTER
                            a fire. Note: cooldown alone is not enough — see
                            `min_upright_s` for the re-arm requirement.
      min_upright_s:        minimum continuous upright duration before the
                            track is eligible to fire a fall. Single upright
                            frames during fidgeting no longer re-arm the
                            trigger. Default 1.0s.
      min_conf:             keypoints below this conf are ignored.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg is not None else _load_fall_cfg()
        f = self.cfg
        self.aspect_threshold = float(f.get("aspect_threshold", 1.5))
        self.kp_height_frac = float(f.get("keypoint_height_frac", 0.75))
        self.transition_window_s = float(f.get("transition_window_s", 1.5))
        self.cooldown_s = float(f.get("cooldown_s", 5.0))
        self.min_upright_s = float(f.get("min_upright_s", 1.0))
        self.min_conf = float(f.get("min_conf", 0.30))
        self._states: dict[int, _TrackState] = {}

    def update(self, poses: list[Pose], frame_idx: int, t: float | None = None) -> list[FallEvent]:
        """Process one frame's poses; return any FallEvents raised this frame.

        `t` defaults to time.perf_counter() — pass explicit t for deterministic
        testing.
        """
        if t is None:
            t = time.perf_counter()
        events: list[FallEvent] = []
        seen_ids: set[int] = set()

        for pose in poses:
            tid = pose.track_id
            if tid < 0:
                continue   # untracked detections can't hold fall state
            seen_ids.add(tid)
            st = self._states.setdefault(tid, _TrackState(last_seen=t))
            st.last_seen = t

            # --- signal (a): aspect ratio ---
            x1, y1, x2, y2 = pose.xyxy
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            aspect = w / h

            # --- signal (b): hip/shoulder low in the bbox ---
            kp_frac = self._keypoint_height_frac(pose.keypoints, y1, y2)
            kp_low = kp_frac >= self.kp_height_frac
            st._last_kp_frac = kp_frac

            # --- state classification for this frame ---
            is_vertical_now = aspect < self.aspect_threshold
            uprightness = is_vertical_now and (not kp_low)   # standing / sitting up tall
            falleness = (not is_vertical_now) and kp_low      # lying on the ground

            # --- maintain the continuous-upright run + arming ---
            if uprightness:
                if st.upright_since is None:
                    st.upright_since = t
                # check whether this run has accumulated enough to arm the track
                # (we re-check each upright frame so arming happens as soon as
                # min_upright_s is reached; the run continues to extend)
                run_duration = t - st.upright_since
                if run_duration >= self.min_upright_s:
                    # track is "armed" — the most recent upright frame refreshes
                    # the arm time so the transition window starts from the LAST
                    # upright frame, not the first
                    st.armed_at = t
            else:
                # upright run broken (track is in transition or fallen).
                # Do NOT clear armed_at here — the fall window is allowed to
                # start after the upright period ends. armed_at persists until
                # either a fall fires or the transition_window expires.
                st.upright_since = None

            # expire a stale arm if the window has passed without a fall
            if st.armed_at is not None and (t - st.armed_at) > self.transition_window_s:
                st.armed_at = None

            # --- fall trigger evaluation ---
            if falleness and (t > st.in_cooldown_until) and (st.armed_at is not None):
                delta = t - st.armed_at
                # delta is guaranteed <= transition_window_s because we just
                # expired stale arms above; check anyway for safety
                if delta <= self.transition_window_s:
                    upright_duration = (st.armed_at - (st.armed_at - self.min_upright_s))  # >= min_upright_s by construction
                    events.append(FallEvent(
                        track_id=tid,
                        t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        frame_idx=frame_idx,
                        aspect_now=aspect,
                        keypoints_low=True,
                        kp_height_frac=float(kp_frac),
                        upright_duration_s=float(upright_duration),
                        transition_delta_s=float(delta),
                    ))
                    # cooldown + disarm so the same fallen track doesn't re-fire
                    st.in_cooldown_until = t + self.cooldown_s
                    st.armed_at = None
                    st.upright_since = None
                    if _FALL_DEBUG and st._debug_log_count < _FALL_DEBUG_LIMIT:
                        st._debug_log_count += 1
                        print(f"[fall:TRIGGER] tid={tid} f={frame_idx} aspect={aspect:.2f} "
                              f"kp_frac={kp_frac:.2f} upright_dur>={self.min_upright_s:.2f}s "
                              f"delta={delta:.2f}s cooldown_until={st.in_cooldown_until:.2f}",
                              flush=True)
                elif _FALL_DEBUG and st._debug_log_count < _FALL_DEBUG_LIMIT:
                    st._debug_log_count += 1
                    print(f"[fall:STALE-ARM] tid={tid} f={frame_idx} aspect={aspect:.2f} "
                          f"kp_frac={kp_frac:.2f} delta={delta:.2f}s > window {self.transition_window_s}s",
                          flush=True)
            elif falleness and (t > st.in_cooldown_until) and (st.armed_at is None):
                # fallen but not armed — log for threshold tuning
                if _FALL_DEBUG and st._debug_log_count < _FALL_DEBUG_LIMIT:
                    st._debug_log_count += 1
                    print(f"[fall:NOT-ARMED] tid={tid} f={frame_idx} aspect={aspect:.2f} "
                          f"kp_frac={kp_frac:.2f} (track never accumulated {self.min_upright_s}s upright)",
                          flush=True)

            st.last_aspect = aspect

        # GC tracks we haven't seen for >3x the cooldown — keeps the dict small
        gc_before = t - max(15.0, 3 * self.cooldown_s)
        for tid in [k for k, v in self._states.items() if v.last_seen < gc_before]:
            del self._states[tid]

        return events

    def _keypoints_low(self, kpts: np.ndarray, y_top: int, y_bot: int) -> bool:
        """Backward-compat wrapper. See _keypoint_height_frac for the actual
        computation."""
        return self._keypoint_height_frac(kpts, y_top, y_bot) >= self.kp_height_frac

    def _keypoint_height_frac(self, kpts: np.ndarray, y_top: int, y_bot: int) -> float:
        """Return the fractional height (0..1) of the hip/shoulder keypoints
        down the bbox. 0 = at the top of the bbox, 1 = at the bottom.

        For a standing person: shoulders ~0.05, hips ~0.5.
        For a seated person (bbox shrinks vertically): hips ~0.55-0.65.
        For a person lying flat: hips+shoulders both ~0.85-0.95 of the
        (now-short, wide) bbox's vertical extent.

        Returns 0.0 if fewer than 2 of the 4 keypoints are visible above
        min_conf — i.e. we don't fire on absent evidence.
        """
        h = max(1, y_bot - y_top)
        indices = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP]
        ys = []
        for i in indices:
            x, y, c = kpts[i]
            if c >= self.min_conf:
                ys.append(float(y))
        if len(ys) < 2:
            return 0.0
        mean_y = sum(ys) / len(ys)
        return float((mean_y - y_top) / h)

    def reset(self) -> None:
        self._states.clear()


def _load_fall_cfg() -> dict[str, Any]:
    from core.config import load_models_config
    return load_models_config().get("fall", {})


# --- Phase 5 placeholder: event-bus-compatible emission ---------------------
# The FallEvent.as_dict() shape is what the Phase 5 event bus + log will
# consume. For Phase 2 we just collect + print so we don't pre-wire storage.