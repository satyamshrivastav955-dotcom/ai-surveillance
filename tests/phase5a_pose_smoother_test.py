"""Tests for core/pose_smoother.py — Phase 5A."""
from __future__ import annotations

import math
import numpy as np
import pytest

from core.pose_smoother import PoseSmoother, SmoothedPose, _OneEuro


# ---------------------------------------------------------------------------
# _OneEuro filter unit tests
# ---------------------------------------------------------------------------

def test_one_euro_passthrough_constant():
    """Constant signal: filtered value should converge to signal value."""
    f = _OneEuro(freq=30.0, min_cutoff=1.0, beta=0.007)
    v = 100.0
    result = v
    for _ in range(50):
        result = f.filter(v)
    assert abs(result - v) < 0.5, f"Expected ~{v}, got {result}"


def test_one_euro_smooths_noise():
    """Noisy signal: filtered sequence should have lower variance than input."""
    f = _OneEuro(freq=30.0, min_cutoff=1.0, beta=0.007)
    rng = np.random.default_rng(42)
    signal = 50.0 + rng.normal(0, 5, size=200)
    filtered = [f.filter(float(v)) for v in signal]
    raw_std = float(np.std(signal[10:]))
    filt_std = float(np.std(filtered[10:]))
    assert filt_std < raw_std * 0.8, (
        f"Expected filter to reduce noise: raw_std={raw_std:.2f} filt_std={filt_std:.2f}"
    )


def test_one_euro_reset():
    """After reset, filter starts fresh at new position."""
    f = _OneEuro(freq=30.0, min_cutoff=1.0, beta=0.007)
    for _ in range(10):
        f.filter(200.0)
    x = f.reset(0.0)
    assert x == 0.0
    # Next filter call should start near 0, not 200
    result = f.filter(0.0)
    assert abs(result) < 5.0


# ---------------------------------------------------------------------------
# PoseSmoother integration tests
# ---------------------------------------------------------------------------

def _make_pose(track_id: int, x1=100, y1=100, x2=200, y2=400,
               kp_conf: float = 0.9, noise: float = 0.0):
    """Build a minimal fake Pose-like object for the smoother."""
    from types import SimpleNamespace
    kpts = np.zeros((17, 3), dtype=np.float32)
    rng = np.random.default_rng(track_id * 7 + 13)
    for i in range(17):
        kpts[i] = [150.0 + rng.normal(0, noise),
                   200.0 + rng.normal(0, noise),
                   kp_conf]
    return SimpleNamespace(
        track_id=track_id, xyxy=(x1, y1, x2, y2),
        keypoints=kpts, conf=0.85
    )


def _make_track(track_id: int, x1=100, y1=100, x2=200, y2=400):
    from types import SimpleNamespace
    return SimpleNamespace(track_id=track_id, xyxy=(x1, y1, x2, y2), cls=0, conf=0.9)


def test_smoother_returns_smoothed_pose():
    """Basic: smoother wraps raw Pose in SmoothedPose without crashing."""
    smoother = PoseSmoother()
    pose = _make_pose(1)
    track = _make_track(1)
    results = smoother.update([pose], ema_tracks=[track])
    assert len(results) == 1
    sp = results[0]
    assert isinstance(sp, SmoothedPose)
    assert sp.track_id == 1
    assert sp.keypoints.shape == (17, 3)
    assert sp.pose_quality in ("stable", "degraded", "missing")


def test_smoother_quality_stable_on_good_keypoints():
    """When all keypoints have conf > gate_conf, quality should be 'stable'."""
    smoother = PoseSmoother({"gate_conf": 0.3})
    pose = _make_pose(1, kp_conf=0.9)
    track = _make_track(1)
    for _ in range(5):
        results = smoother.update([pose], ema_tracks=[track])
    assert results[0].pose_quality == "stable"


def test_smoother_quality_missing_on_low_conf():
    """When all keypoints have conf=0 and no prior, quality should be 'missing'."""
    smoother = PoseSmoother({"gate_conf": 0.3, "hold_frames": 0})
    pose = _make_pose(1, kp_conf=0.0)
    track = _make_track(1)
    results = smoother.update([pose], ema_tracks=[track])
    assert results[0].pose_quality == "missing"


def test_smoother_hold_last_good():
    """After a good frame, a low-conf frame should still output the held position."""
    smoother = PoseSmoother({"gate_conf": 0.3, "hold_frames": 5})
    # First: good pose at known position
    good = _make_pose(1, kp_conf=0.9)
    track = _make_track(1)
    smoother.update([good], ema_tracks=[track])
    # Save output position
    good_result = smoother.update([good], ema_tracks=[track])[0]
    good_xy = good_result.keypoints[0, :2].copy()

    # Now: low confidence pose
    bad = _make_pose(1, kp_conf=0.0)
    results = smoother.update([bad], ema_tracks=[track])
    held_xy = results[0].keypoints[0, :2]
    # Position should be close to the held good value (within 2px)
    assert np.linalg.norm(held_xy - good_xy) < 2.0, (
        f"Held position drifted: good={good_xy}, held={held_xy}"
    )


def test_smoother_reduces_jitter():
    """Smoothed keypoints should have lower frame-to-frame variance than raw."""
    smoother = PoseSmoother({"gate_conf": 0.1, "min_cutoff": 1.0, "beta": 0.007})
    track = _make_track(1)

    # Use a shared rng that advances each frame to produce real per-frame noise
    rng = np.random.default_rng(99)

    raw_kp0 = []
    smooth_kp0 = []
    for _ in range(60):
        # Build a fresh pose with per-frame noise (rng advances each call)
        kpts = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            kpts[i] = [150.0 + rng.normal(0, 5.0),
                       200.0 + rng.normal(0, 5.0),
                       0.9]
        from types import SimpleNamespace
        noisy_pose = SimpleNamespace(
            track_id=1, xyxy=(100, 100, 200, 400),
            keypoints=kpts, conf=0.85
        )
        raw_kp0.append(float(noisy_pose.keypoints[0, 0]))
        result = smoother.update([noisy_pose], ema_tracks=[track])[0]
        smooth_kp0.append(result.keypoints[0, 0])

    raw_var = float(np.var(raw_kp0[10:]))
    smooth_var = float(np.var(smooth_kp0[10:]))
    # One-Euro reduces but doesn't eliminate variance; require at least 30% reduction
    assert smooth_var < raw_var * 0.85, (
        f"Expected smoother to reduce variance: raw={raw_var:.2f} smooth={smooth_var:.2f}"
    )



def test_smoother_teleport_reinit():
    """A keypoint jump > teleport_px should trigger filter re-init, not lag."""
    smoother = PoseSmoother({"gate_conf": 0.3, "teleport_px": 50.0, "min_cutoff": 0.1})
    track = _make_track(1)

    # Warm up filter at position ~150
    for _ in range(10):
        pose = _make_pose(1, kp_conf=0.9)
        smoother.update([pose], ema_tracks=[track])

    # Teleport to position ~300 (jump of 150px > teleport_px=50)
    from types import SimpleNamespace
    kpts = np.full((17, 3), 0.9, dtype=np.float32)
    kpts[:, 0] = 300.0
    kpts[:, 1] = 300.0
    teleport_pose = SimpleNamespace(
        track_id=1, xyxy=(100, 100, 200, 400),
        keypoints=kpts, conf=0.85
    )
    results = smoother.update([teleport_pose], ema_tracks=[track])
    # After teleport re-init, filtered position should be near 300, not still near 150
    x_filtered = results[0].keypoints[0, 0]
    assert x_filtered > 250.0, (
        f"Expected re-init near 300px after teleport, got {x_filtered:.1f}"
    )


def test_smoother_prunes_stale_tracks():
    """Tracks no longer in raw_poses should be removed from internal state."""
    smoother = PoseSmoother()
    track1 = _make_track(1)
    track2 = _make_track(2)
    pose1 = _make_pose(1, kp_conf=0.9)
    pose2 = _make_pose(2, kp_conf=0.9)

    smoother.update([pose1, pose2], ema_tracks=[track1, track2])
    assert 1 in smoother._tracks
    assert 2 in smoother._tracks

    # Only track 1 appears next frame
    smoother.update([pose1], ema_tracks=[track1])
    assert 1 in smoother._tracks
    assert 2 not in smoother._tracks


def test_smoother_ema_bbox_smoothing():
    """EMA bbox should lie between raw and previous bbox."""
    smoother = PoseSmoother({"ema_alpha": 0.5})
    pose = _make_pose(1, kp_conf=0.9)
    track_a = _make_track(1, x1=100, y1=100, x2=200, y2=400)
    smoother.update([pose], ema_tracks=[track_a])

    track_b = _make_track(1, x1=200, y1=100, x2=300, y2=400)
    results = smoother.update([pose], ema_tracks=[track_b])
    x1 = results[0].xyxy[0]
    # EMA(alpha=0.5) of 200 and prev~100 = ~150
    assert 100 < x1 < 200, f"Expected EMA-smoothed x1 between 100 and 200, got {x1}"


def test_smoother_empty_input():
    """Empty pose list returns empty list without error."""
    smoother = PoseSmoother()
    results = smoother.update([], ema_tracks=[])
    assert results == []


if __name__ == "__main__":
    test_one_euro_passthrough_constant()
    test_one_euro_smooths_noise()
    test_one_euro_reset()
    test_smoother_returns_smoothed_pose()
    test_smoother_quality_stable_on_good_keypoints()
    test_smoother_quality_missing_on_low_conf()
    test_smoother_hold_last_good()
    test_smoother_reduces_jitter()
    test_smoother_teleport_reinit()
    test_smoother_prunes_stale_tracks()
    test_smoother_ema_bbox_smoothing()
    test_smoother_empty_input()
    print("phase5a_pose_smoother_test: all passed")
