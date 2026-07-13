"""Fall-detection test at every-2 cadence.

The existing fall_detection_test.py feeds synthetic poses to the FallDetector
on every frame. But with pose.every=2 in pipeline.yaml, the real pipeline
only feeds poses on even frames (at 30fps that's every 33ms -> 67ms gap).
This test simulates that cadence to verify the state machine still catches
fast falls and still rejects slow sit-downs.

The critical timing math:
  - 30 fps source, pose fires every 2nd frame -> 15 pose samples/sec
  - Fast fall completes in ~0.5s -> ~7 pose samples during the fall
  - transition_window_s = 1.5s -> the state machine has a 1.5s window
  - So even at every=2, a 0.5s fall produces ample samples within the window

Run:
    python tests/fall_cadence_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pose import (
    KP_LEFT_HIP, KP_LEFT_SHOULDER, KP_RIGHT_HIP, KP_RIGHT_SHOULDER, Pose,
)
from core.state_machine import FallDetector
from pipeline.frame_router import FrameRouter
from core.config import load_pipeline_config


def _make_pose(track_id, x1, y1, x2, y2, keypoint_y_frac, keypoint_conf=0.9):
    x_center = (x1 + x2) / 2
    kp_y = y1 + (y2 - y1) * keypoint_y_frac
    kpts = np.zeros((17, 3), dtype=np.float32)
    for idx in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP):
        kpts[idx] = [x_center + (8 if idx % 2 == 0 else -8), kp_y, keypoint_conf]
    return Pose(track_id=track_id, xyxy=(x1, y1, x2, y2),
                keypoints=kpts, conf=1.0)


def _vertical_pose(tid):
    return _make_pose(tid, 300, 100, 380, 400, 0.15)


def _horizontal_pose(tid):
    return _make_pose(tid, 200, 300, 600, 380, 0.85)


def _interp(v, h, alpha):
    x1 = int(v.xyxy[0] + (h.xyxy[0] - v.xyxy[0]) * alpha)
    y1 = int(v.xyxy[1] + (h.xyxy[1] - v.xyxy[1]) * alpha)
    x2 = int(v.xyxy[2] + (h.xyxy[2] - v.xyxy[2]) * alpha)
    y2 = int(v.xyxy[3] + (h.xyxy[3] - v.xyxy[3]) * alpha)
    kpts = v.keypoints + (h.keypoints - v.keypoints) * alpha
    return Pose(track_id=v.track_id, xyxy=(x1, y1, x2, y2),
                keypoints=kpts.astype(np.float32), conf=1.0)


def _get_pose_every():
    """Read the actual pose cadence from pipeline.yaml via FrameRouter."""
    cfg = load_pipeline_config()
    router = FrameRouter(cfg.get("router", {}))
    return router.every("pose")


def test_fast_fall_at_cadence(pose_every):
    """Fast fall (0.5s transition) with pose firing every Nth frame.
    Must still trigger within the 1.5s transition window.

    Person is upright for 1.5s (clears min_upright_s=1.0 even at every=2
    cadence, which yields ~22 pose samples in 1.5s), then transitions to
    fallen over 0.5s.
    """
    det = FallDetector()
    tid = 5
    t0 = 100.0
    fps = 30.0
    v = _vertical_pose(tid)
    h = _horizontal_pose(tid)
    fall_at = None
    upright_frames = 45   # 1.5s upright — clears min_upright_s
    fall_frames = 7       # ~0.23s transition
    total_frames = upright_frames + fall_frames + 15
    for frame_idx in range(total_frames):
        if frame_idx % pose_every != 0:
            continue
        if frame_idx < upright_frames:
            pose = v
        else:
            alpha = min(1.0, (frame_idx - upright_frames) / fall_frames)
            pose = _interp(v, h, alpha)
        t = t0 + frame_idx / fps
        events = det.update([pose], frame_idx=frame_idx, t=t)
        if events:
            fall_at = frame_idx
            break
    assert fall_at is not None, (
        f"FAIL: fast fall did not trigger at every={pose_every} cadence. "
        f"Pose samples were fed at frames {[i for i in range(total_frames) if i % pose_every == 0][:10]}..."
    )
    fall_time_s = fall_at / fps
    print(f"  [ok] fast fall triggered at frame {fall_at} ({fall_time_s:.3f}s) "
          f"with every={pose_every} cadence (after {upright_frames} upright frames)")


def test_slow_sit_at_cadence(pose_every):
    """Slow sit-down (4s transition) with pose firing every Nth frame.
    Must NOT trigger (takes longer than transition_window_s=1.5s).

    Person is upright for 2s, then transitions gradually over 4s.
    """
    det = FallDetector()
    tid = 5
    t0 = 200.0
    fps = 30.0
    v = _vertical_pose(tid)
    h = _horizontal_pose(tid)
    fired = False
    upright_frames = 60    # 2s upright
    sit_frames = 120       # 4s gradual transition
    total_frames = upright_frames + sit_frames
    for frame_idx in range(total_frames):
        if frame_idx % pose_every != 0:
            continue
        if frame_idx < upright_frames:
            pose = v
        else:
            alpha = (frame_idx - upright_frames) / (sit_frames - 1)
            pose = _interp(v, h, alpha)
        t = t0 + frame_idx / fps
        events = det.update([pose], frame_idx=frame_idx, t=t)
        if events:
            fired = True
            break
    assert not fired, (
        f"FAIL: slow sit-down tripped the detector at every={pose_every} cadence"
    )
    print(f"  [ok] slow sit-down did NOT fire at every={pose_every} cadence")


def test_cooldown_at_cadence(pose_every):
    """After a fall, cooldown + min_upright_s suppress re-trigger.

    Person is upright for 1.5s (clears min_upright_s), falls fast, then stays
    fallen for 7s (longer than cooldown_s=5.0). Must NOT re-fire: even after
    cooldown expires, re-firing requires re-accumulating min_upright_s of
    continuous uprightness, which never happens.
    """
    det = FallDetector()
    det.cooldown_s = 5.0
    tid = 5
    t0 = 300.0
    fps = 30.0
    v = _vertical_pose(tid)
    h = _horizontal_pose(tid)
    first_fire = None
    second_fire = None
    upright_frames = 45
    fall_frames = 7
    post_fall_frames = int((det.cooldown_s + 2.0) * fps)   # 7s
    total_frames = upright_frames + fall_frames + post_fall_frames
    for frame_idx in range(total_frames):
        if frame_idx % pose_every != 0:
            continue
        if frame_idx < upright_frames:
            pose = v
        elif frame_idx < upright_frames + fall_frames:
            pose = _interp(v, h, (frame_idx - upright_frames) / fall_frames)
        else:
            pose = h
        t = t0 + frame_idx / fps
        events = det.update([pose], frame_idx=frame_idx, t=t)
        if events:
            if first_fire is None:
                first_fire = frame_idx
            elif second_fire is None:
                second_fire = frame_idx
                break
    assert first_fire is not None, "FAIL: first fall did not fire"
    assert second_fire is None, (
        f"FAIL: re-triggered after cooldown at frame {second_fire} "
        f"(first was at {first_fire}) — min_upright_s re-arm failed at cadence"
    )
    print(f"  [ok] cooldown + min_upright_s held at every={pose_every} cadence "
          f"(first fire at frame {first_fire}, no second within "
          f"{det.cooldown_s + 2.0:.0f}s)")


def main():
    pose_every = _get_pose_every()
    print(f"fall_cadence_test: (pose.every={pose_every} from pipeline.yaml)")
    test_fast_fall_at_cadence(pose_every)
    test_slow_sit_at_cadence(pose_every)
    test_cooldown_at_cadence(pose_every)
    print(f"fall_cadence_test: all passed at every={pose_every} cadence")


if __name__ == "__main__":
    main()