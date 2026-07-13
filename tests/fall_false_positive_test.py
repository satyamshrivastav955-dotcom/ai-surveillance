"""Phase 2 false-positive regression tests.

Three real-world movements that should NOT trigger a fall, all of which
were reported as false triggers on the real webcam:

  - test_lean_forward_to_pickup: person bends forward (bbox goes briefly
    wide, keypoints stay high because they're still on their feet)
  - test_sit_down_normal: person sits down in a chair over ~2s (slow
    transition, exceeds transition_window_s)
  - test_rotate_in_chair: person rotates sideways (bbox aspect wobbles
    but keypoint height stays mid-range; never goes "low" enough)

All three start with the person upright for >= min_upright_s so they're
"armed" — the tests confirm the OTHER signals (kp_low, transition_window)
correctly reject them, not just the arming gate.

Run:
    python tests/fall_false_positive_test.py
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


def _make_pose(tid, x1, y1, x2, y2, kp_y_frac, kp_conf=0.9):
    x_center = (x1 + x2) / 2
    kp_y = y1 + (y2 - y1) * kp_y_frac
    kpts = np.zeros((17, 3), dtype=np.float32)
    for idx in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP):
        kpts[idx] = [x_center + (8 if idx % 2 == 0 else -8), kp_y, kp_conf]
    return Pose(track_id=tid, xyxy=(x1, y1, x2, y2),
                keypoints=kpts, conf=1.0)


def _vertical_pose(tid):
    """Standing person: 80 wide x 300 tall, aspect=0.27, kpts high (0.15)."""
    return _make_pose(tid, 300, 100, 380, 400, 0.15)


def _interp_pose(v, h, alpha):
    x1 = int(v.xyxy[0] + (h.xyxy[0] - v.xyxy[0]) * alpha)
    y1 = int(v.xyxy[1] + (h.xyxy[1] - v.xyxy[1]) * alpha)
    x2 = int(v.xyxy[2] + (h.xyxy[2] - v.xyxy[2]) * alpha)
    y2 = int(v.xyxy[3] + (h.xyxy[3] - v.xyxy[3]) * alpha)
    kpts = v.keypoints + (h.keypoints - v.keypoints) * alpha
    return Pose(track_id=v.track_id, xyxy=(x1, y1, x2, y2),
                keypoints=kpts.astype(np.float32), conf=1.0)


def _run_sequence(det, poses_with_times):
    """Feed (pose, t) tuples to the detector; return list of fire-frame indices."""
    fires = []
    for i, (pose, t) in enumerate(poses_with_times):
        events = det.update([pose], frame_idx=i, t=t)
        if events:
            fires.append(i)
    return fires


def test_lean_forward_to_pickup():
    """Person stands for 2s, then bends forward to pick something up over 0.5s,
    stays bent for 1s, then returns to upright. Should NOT fire.

    Key signal: while bent, the bbox goes wide (aspect up to ~1.4) but the
    keypoints stay HIGH in the bbox (the head/shoulders are still at the top
    of the bent-over body; hips are mid-bbox). kp_low should be False throughout.
    """
    det = FallDetector()
    tid = 5
    t0 = 100.0
    fps = 30.0
    # standing pose (vertical, kpts high)
    v = _vertical_pose(tid)
    # bent-forward pose: wider bbox (aspect ~1.4) but keypoints at 0.25
    # (shoulders still near the top of the bbox — person is bending forward,
    # not collapsing to the ground)
    bent = _make_pose(tid, 200, 200, 580, 380, 0.25)
    seq = []
    # 2s upright (clears min_upright_s)
    for i in range(int(2.0 * fps)):
        seq.append((v, t0 + i / fps))
    # 0.5s transition to bent
    bent_frames = int(0.5 * fps)
    for j in range(bent_frames):
        alpha = j / (bent_frames - 1)
        seq.append((_interp_pose(v, bent, alpha), t0 + (2.0 * fps + j) / fps))
    # 1s holding the bent pose
    for k in range(int(1.0 * fps)):
        seq.append((bent, t0 + (2.5 * fps + k) / fps))
    # 0.5s returning to upright
    for j in range(bent_frames):
        alpha = j / (bent_frames - 1)
        seq.append((_interp_pose(bent, v, alpha), t0 + (3.5 * fps + j) / fps))
    fires = _run_sequence(det, seq)
    assert not fires, (
        f"FAIL: lean-forward-to-pickup triggered {len(fires)} fall(s) at seq indices {fires}"
    )
    print(f"  [ok] lean-forward-to-pickup did NOT fire (correctly)")


def test_sit_down_normal():
    """Person stands for 2s, then sits down in a chair over ~2s, stays seated
    for 3s. Should NOT fire because the 2s transition exceeds
    transition_window_s=1.5s.

    Key signal: the transition is SLOW. Even though the final seated pose has
    a somewhat-wide bbox and somewhat-low keypoints, getting there took longer
    than the transition window, so it's a controlled sit, not a fall.
    """
    det = FallDetector()
    tid = 5
    t0 = 200.0
    fps = 30.0
    v = _vertical_pose(tid)
    # seated pose: bbox ~1.1 aspect (slightly wide), keypoints at 0.65
    # (hips low-ish in the bbox because the bbox shrunk vertically when sitting,
    # but NOT as low as a lying-flat person which would be ~0.85+)
    seated = _make_pose(tid, 250, 250, 530, 460, 0.65)
    seq = []
    # 2s upright (clears min_upright_s)
    for i in range(int(2.0 * fps)):
        seq.append((v, t0 + i / fps))
    # 2s gradual transition to seated (exceeds transition_window_s=1.5)
    sit_frames = int(2.0 * fps)
    for j in range(sit_frames):
        alpha = j / (sit_frames - 1)
        seq.append((_interp_pose(v, seated, alpha), t0 + (2.0 * fps + j) / fps))
    # 3s staying seated
    for k in range(int(3.0 * fps)):
        seq.append((seated, t0 + (4.0 * fps + k) / fps))
    fires = _run_sequence(det, seq)
    assert not fires, (
        f"FAIL: normal sit-down triggered {len(fires)} fall(s) at seq indices {fires}"
    )
    print(f"  [ok] normal sit-down did NOT fire (correctly)")


def test_rotate_in_chair():
    """Person is seated and rotates sideways (turning to face a different
    direction). The bbox aspect wobbles between ~0.9 and ~1.3 over a few
    seconds as the detector sees them from different angles, but keypoint
    height fraction stays around 0.55-0.65 (seated posture). Should NOT fire.

    Key signal: aspect sometimes crosses the old 1.0 threshold (which is why
    the old detector false-triggered), but with the new 1.5 threshold the
    aspect never gets high enough. And even if it did, kp_frac stays below
    the new 0.75 threshold.
    """
    det = FallDetector()
    tid = 5
    t0 = 300.0
    fps = 30.0
    # start upright for 1.5s to arm the detector (so we're testing the OTHER
    # signals, not just the arming gate)
    v = _vertical_pose(tid)
    seq = []
    for i in range(int(1.5 * fps)):
        seq.append((v, t0 + i / fps))
    # then sit + rotate for 5s: aspect wobbles 0.9..1.3, kp_frac wobbles 0.55..0.65
    rotate_frames = int(5.0 * fps)
    for j in range(rotate_frames):
        phase = j / fps
        # bbox aspect oscillates between 0.9 and 1.3 (period 2s)
        aspect = 1.1 + 0.2 * np.sin(2 * np.pi * phase / 2.0)
        # bbox: choose x1,x2,y1,y2 to produce that aspect with height=210
        h = 210
        w = int(aspect * h)
        cx = 400
        x1 = cx - w // 2; x2 = cx + w // 2
        y1 = 250; y2 = y1 + h
        # kp_frac oscillates 0.55..0.65 (well below the 0.75 threshold)
        kp_frac = 0.60 + 0.05 * np.sin(2 * np.pi * phase / 1.7)
        pose = _make_pose(tid, x1, y1, x2, y2, kp_frac)
        seq.append((pose, t0 + (1.5 * fps + j) / fps))
    fires = _run_sequence(det, seq)
    assert not fires, (
        f"FAIL: rotate-in-chair triggered {len(fires)} fall(s) at seq indices {fires}"
    )
    print(f"  [ok] rotate-in-chair did NOT fire (correctly)")


def test_lean_then_actual_fall_still_fires():
    """Sanity check: after a lean-forward (no fire), if the person then actually
    falls, the detector MUST still fire. Confirms the false-positive filters
    don't over-suppress and miss real falls.
    """
    det = FallDetector()
    tid = 5
    t0 = 400.0
    fps = 30.0
    v = _vertical_pose(tid)
    bent = _make_pose(tid, 200, 200, 580, 380, 0.25)
    # genuinely fallen: wide bbox, keypoints very low
    fallen = _make_pose(tid, 200, 300, 600, 380, 0.90)
    seq = []
    # 2s upright
    for i in range(int(2.0 * fps)):
        seq.append((v, t0 + i / fps))
    # 0.5s lean forward
    for j in range(int(0.5 * fps)):
        seq.append((_interp_pose(v, bent, j / (0.5 * fps - 1)),
                    t0 + (2.0 * fps + j) / fps))
    # 0.5s back upright (re-arm requires 1s continuous upright)
    for j in range(int(1.5 * fps)):
        seq.append((_interp_pose(bent, v, j / (1.5 * fps - 1)),
                    t0 + (2.5 * fps + j) / fps))
    # 1s upright (re-arms the detector)
    for i in range(int(1.0 * fps)):
        seq.append((v, t0 + (4.0 * fps + i) / fps))
    # 0.3s actual fall
    fall_frames = int(0.3 * fps)
    for j in range(fall_frames):
        seq.append((_interp_pose(v, fallen, j / (fall_frames - 1)),
                    t0 + (5.0 * fps + j) / fps))
    fires = _run_sequence(det, seq)
    assert fires, (
        "FAIL: real fall after a lean-forward did NOT fire — over-suppressed"
    )
    print(f"  [ok] real fall after lean-forward DID fire (at seq index {fires[0]}) — "
          f"false-positive filters don't over-suppress")


def main():
    print("fall_false_positive_test:")
    test_lean_forward_to_pickup()
    test_sit_down_normal()
    test_rotate_in_chair()
    test_lean_then_actual_fall_still_fires()
    print("fall_false_positive_test: all passed")


if __name__ == "__main__":
    main()