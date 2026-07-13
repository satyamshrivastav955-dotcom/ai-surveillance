"""Phase 2 fall-detection state-machine unit test.

Stages a synthetic 'fall' over a sequence of frames: a person's bbox
starts vertical (taller than wide) and gradually becomes horizontal (wider
than tall) while the hip/shoulder keypoints drop toward the bottom of the
bbox, all within the transition window. The FallDetector must emit one
FALL event. A separate 'slow sit-down' sequence (same final pose, but the
transition takes > transition_window_s) must NOT emit a fall.

Run:
    python tests/fall_detection_test.py
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


def _make_pose(track_id: int, x1: int, y1: int, x2: int, y2: int,
               keypoint_y_frac: float, keypoint_conf: float = 0.9) -> Pose:
    """Construct a synthetic Pose where hip+shoulder keypoints are placed at
    a given fractional height down the bbox. x coords are centered."""
    x_center = (x1 + x2) / 2
    kp_y = y1 + (y2 - y1) * keypoint_y_frac
    kpts = np.zeros((17, 3), dtype=np.float32)
    # set all keypoints to (x_center, kp_y, conf) — only the four we test
    # actually matter; the fall detector only looks at shoulders+hips
    for idx in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP):
        kpts[idx] = [x_center + (8 if idx % 2 == 0 else -8), kp_y, keypoint_conf]
    # keep other keypoints at conf 0 so they don't pollute any future logic
    return Pose(track_id=track_id, xyxy=(x1, y1, x2, y2),
                keypoints=kpts, conf=1.0)


def _vertical_pose(tid: int) -> Pose:
    """Tall narrow bbox (aspect=0.4), keypoints high up."""
    return _make_pose(tid, x1=300, y1=100, x2=380, y2=400, keypoint_y_frac=0.15)


def _horizontal_pose(tid: int) -> Pose:
    """Wide flat bbox (aspect=2.0), keypoints low down — a fallen person."""
    return _make_pose(tid, x1=200, y1=300, x2=600, y2=380, keypoint_y_frac=0.85)


def _interp(v_start: Pose, v_end: Pose, alpha: float) -> Pose:
    """Linearly interpolate between two poses (only bbox + kp_y used)."""
    x1 = int(v_start.xyxy[0] + (v_end.xyxy[0] - v_start.xyxy[0]) * alpha)
    y1 = int(v_start.xyxy[1] + (v_end.xyxy[1] - v_start.xyxy[1]) * alpha)
    x2 = int(v_start.xyxy[2] + (v_end.xyxy[2] - v_start.xyxy[2]) * alpha)
    y2 = int(v_start.xyxy[3] + (v_end.xyxy[3] - v_start.xyxy[3]) * alpha)
    k_start = v_start.keypoints
    k_end = v_end.keypoints
    kpts = k_start + (k_end - k_start) * alpha
    return Pose(track_id=v_start.track_id, xyxy=(x1, y1, x2, y2),
                keypoints=kpts.astype(np.float32), conf=1.0)


def test_fast_fall_triggers():
    """Vertical -> horizontal transition complete in <1.5s — must FIRE.

    The track must be upright for >= min_upright_s (default 1.0s) before the
    fall to be eligible. So we keep the person upright for 1.5s (45 frames at
    30fps), then transition to fallen over 0.5s.
    """
    det = FallDetector()
    tid = 5
    t0 = 100.0
    fps = 30.0
    v = _vertical_pose(tid)
    h = _horizontal_pose(tid)
    upright_frames = int(det.min_upright_s * fps) + 15   # 1.5s = 45 frames
    fall_frames = 7                                       # 0.23s transition
    total = upright_frames + fall_frames + 5
    fall_at = None
    for i in range(total):
        if i < upright_frames:
            pose = v
        else:
            alpha = min(1.0, (i - upright_frames) / fall_frames)
            pose = _interp(v, h, alpha)
        events = det.update([pose], frame_idx=i, t=t0 + i / fps)
        if events:
            fall_at = i
            break
    assert fall_at is not None, "FAIL: fast fall did not trigger a FALL event"
    fall_time_s = fall_at / fps
    print(f"  [ok] fast fall triggered at frame {fall_at} ({fall_time_s:.3f}s) "
          f"after {upright_frames}/{fps}s upright "
          f"(aspect={det._states[tid].last_aspect:.2f})")


def test_slow_sit_down_no_trigger():
    """Vertical -> horizontal transition over 4s — must NOT fire (sitting down).

    Person starts upright for 2s (well above min_upright_s) then transitions
    very gradually over 4s. The transition_window_s=1.5 filter rejects this.
    """
    det = FallDetector()
    tid = 5
    t0 = 200.0
    fps = 30.0
    v = _vertical_pose(tid)
    h = _horizontal_pose(tid)
    upright_frames = 60   # 2s upright
    sit_frames = 120      # 4s gradual transition
    fired = False
    for i in range(upright_frames + sit_frames):
        if i < upright_frames:
            pose = v
        else:
            alpha = (i - upright_frames) / (sit_frames - 1)
            pose = _interp(v, h, alpha)
        events = det.update([pose], frame_idx=i, t=t0 + i / fps)
        if events:
            fired = True
            print(f"  unexpected fire at frame {i} alpha={((i-upright_frames)/(sit_frames-1)):.2f}")
            break
    assert not fired, "FAIL: slow sit-down tripped the fall detector (false positive)"
    print(f"  [ok] slow sit-down did NOT fire (correctly)")

def test_no_low_keypoints_no_trigger():
    """A person who was upright for >min_upright_s, then goes horizontally wide
    but keeps keypoints high (e.g. leaning forward over a counter) must NOT
    fire. The kp_low signal is the gate — without it, "wide bbox" alone is
    just bending, not falling."""
    det = FallDetector()
    tid = 5
    t0 = 300.0
    fps = 30.0
    fired = False
    v = _vertical_pose(tid)
    upright_frames = 45   # 1.5s — clears min_upright_s
    bend_frames = 30      # 1s of bending
    for i in range(upright_frames + bend_frames):
        if i < upright_frames:
            pose = v
        else:
            j = i - upright_frames
            # bbox becomes wide (aspect up to ~1.5+) but keypoints stay at 0.15
            # (shoulders near the top of the bbox — person is bending forward,
            # not falling down)
            x1 = 200 + j * 8
            x2 = 600 - j * 8
            y1 = 200
            y2 = 340
            pose = _make_pose(tid, x1, y1, x2, y2, keypoint_y_frac=0.15)
        events = det.update([pose], frame_idx=i, t=t0 + i / fps)
        if events:
            fired = True
            break
    assert not fired, "FAIL: high keypoints + wide bbox fired (false positive)"
    print(f"  [ok] leaning-forward (wide bbox, high keypoints) did NOT fire (correctly)")


def test_cooldown_prevents_rapid_retrigger():
    """After a fall, the same track should not re-fire within cooldown_s.

    Person is upright for 1.5s (clears min_upright_s), falls fast, stays fallen
    for 6s (longer than cooldown_s=5.0). Without min_upright_s the detector
    would re-fire at the 5s mark; with it, re-firing requires re-accumulating
    1.0s of continuous upright, which never happens here.
    """
    det = FallDetector()
    det.cooldown_s = 5.0
    tid = 5
    t0 = 400.0
    fps = 30.0
    v = _vertical_pose(tid)
    h = _horizontal_pose(tid)
    upright_frames = 45   # 1.5s — clears min_upright_s=1.0
    fall_frames = 7
    post_fall_frames = int((det.cooldown_s + 2.0) * fps)   # 7s after the fall
    first_fire = None
    second_fire = None
    for i in range(upright_frames + fall_frames + post_fall_frames):
        if i < upright_frames:
            pose = v
        elif i < upright_frames + fall_frames:
            pose = _interp(v, h, (i - upright_frames) / fall_frames)
        else:
            pose = h
        events = det.update([pose], frame_idx=i, t=t0 + i / fps)
        if events:
            if first_fire is None:
                first_fire = i
            elif second_fire is None:
                second_fire = i
                break
    assert first_fire is not None, "FAIL: first fall did not fire"
    assert second_fire is None, (
        f"FAIL: re-triggered after cooldown at frame {second_fire} "
        f"(first was at {first_fire}) — min_upright_s re-arm failed"
    )
    print(f"  [ok] cooldown + min_upright_s suppressed re-trigger "
          f"(first fire at {first_fire}, no second within "
          f"{det.cooldown_s + 2.0:.0f}s)")


def test_fidget_does_not_rearm():
    """A person sitting and shifting posture (occasional single upright frames
    surrounded by slightly-horizontal frames) must NOT re-trigger a fall.

    This is the specific case the user reported: the same track fired "FALL"
    every 4-7 seconds during normal sitting/standing/moving. Root cause was
    that a single upright frame reset `upright_until`, re-arming the trigger;
    the next slightly-horizontal frame then fired again after cooldown.

    With min_upright_s=1.0, the track must be continuously upright for 1s
    before being eligible. A single upright frame in the middle of a seated
    sequence doesn't qualify.
    """
    det = FallDetector()
    det.cooldown_s = 2.0    # short cooldown to make the test fast
    tid = 5
    t0 = 500.0
    fps = 30.0
    v = _vertical_pose(tid)
    h = _horizontal_pose(tid)
    # simulate: 1.5s upright (eligible), real fall, then 10s of "fidgeting"
    # where the person is mostly horizontal (seated/leaning) but occasionally
    # has a single upright frame (posture shift). Must NOT re-fire.
    upright_frames = 45
    fall_frames = 7
    fidget_frames = 300    # 10s
    fires = []
    for i in range(upright_frames + fall_frames + fidget_frames):
        if i < upright_frames:
            pose = v
        elif i < upright_frames + fall_frames:
            pose = _interp(v, h, (i - upright_frames) / fall_frames)
        else:
            # fidgeting: mostly horizontal, but every ~60 frames (2s) insert
            # a SINGLE upright frame then back to horizontal
            j = i - (upright_frames + fall_frames)
            if j > 0 and j % 60 == 0:
                pose = v   # single upright frame
            else:
                pose = h   # horizontal
        events = det.update([pose], frame_idx=i, t=t0 + i / fps)
        if events:
            fires.append(i)
    assert len(fires) >= 1, "FAIL: initial fall did not fire"
    assert len(fires) == 1, (
        f"FAIL: fidgeting re-triggered {len(fires) - 1} extra falls "
        f"at frames {fires[1:]} — min_upright_s re-arm is broken"
    )
    print(f"  [ok] fidgeting (single upright frames) did NOT re-trigger "
          f"(only the initial fall fired at frame {fires[0]})")


def main():
    print("fall_detection_test:")
    test_fast_fall_triggers()
    test_slow_sit_down_no_trigger()
    test_no_low_keypoints_no_trigger()
    test_cooldown_prevents_rapid_retrigger()
    test_fidget_does_not_rearm()
    print("fall_detection_test: all passed")


if __name__ == "__main__":
    main()