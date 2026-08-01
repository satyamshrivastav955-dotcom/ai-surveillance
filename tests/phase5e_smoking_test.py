"""Phase 5E tests — SmokingDetector oscillation gate.

Tests that the SmokingDetector correctly requires rhythmic hand oscillation
(rise-lower cycles) before triggering the YOLO confirmation stage, filtering
out static hand-near-face false positives.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


@dataclass
class _MockTrack:
    track_id: int
    cls: int
    conf: float
    xyxy: tuple[int, int, int, int]


@dataclass
class _MockPose:
    track_id: int
    keypoints: np.ndarray
    conf: float = 0.9


def _make_smoking_detector(min_oscillations: int = 2,
                           oscillation_window_frames: int = 30):
    """Create a SmokingDetector with YOLO disabled (HSV fallback) for testing."""
    from core.events import SmokingDetector
    cfg = {
        "weights": None,  # force HSV fallback — no YOLO loading
        "conf": 0.25,
        "imgsz": 320,
        "min_oscillations": min_oscillations,
        "oscillation_window_frames": oscillation_window_frames,
        "min_object_area": 1,  # very low — we're testing gesture, not object detection
        "glow_hsv_low": [0, 0, 0],
        "glow_hsv_high": [180, 255, 255],  # match everything — force confirmation
    }
    det = SmokingDetector(cfg)
    return det


def _kpts_with_wrist_near_nose(y_wrist: float, person_h: float = 200.0) -> np.ndarray:
    """Create 17 COCO keypoints where right wrist is near nose at variable Y.

    Returns (17, 3) array. Nose at (100, 100), right wrist at (100+dx, y_wrist)
    where dx is small enough to pass the 0.15*H proximity check.
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    nose_x, nose_y = 100.0, 100.0
    kpts[0] = [nose_x, nose_y, 0.9]          # nose
    kpts[5] = [80, 130, 0.9]                  # left shoulder
    kpts[6] = [120, 130, 0.9]                 # right shoulder
    kpts[9] = [80, 200, 0.9]                  # left wrist (away)
    kpts[10] = [nose_x + 5, y_wrist, 0.9]    # right wrist (near nose)
    return kpts


class TestSmokingDetectorOscillation:
    """Test oscillation gate on SmokingDetector."""

    def test_static_hand_near_face_no_trigger(self):
        """Static hand near face (no oscillation) should NOT trigger smoking."""
        det = _make_smoking_detector(min_oscillations=2)
        person_h = 200.0
        track = _MockTrack(track_id=0, cls=0, conf=0.9,
                           xyxy=(50, 50, 150, 250))

        # Build a pose where wrist is near nose — static Y=105 every frame
        # (phone call, chin resting)
        static_kpts = _kpts_with_wrist_near_nose(y_wrist=105.0, person_h=person_h)
        pose = _MockPose(track_id=0, keypoints=static_kpts)

        # Simulate enough frames to pass the 2s duration check
        frame = np.zeros((300, 200, 3), dtype=np.uint8)
        all_events = []
        for i in range(40):
            t = 100.0 + i * 0.1  # 40 frames over 4 seconds
            evts = det.detect(frame, [track], [pose], frame_idx=i, t=t)
            all_events.extend(evts)

        assert len(all_events) == 0, \
            f"Static hand near face should not trigger smoking, got {len(all_events)} events"

    def test_oscillating_hand_triggers(self):
        """Rhythmic hand rise/lower pattern SHOULD trigger smoking detection."""
        det = _make_smoking_detector(min_oscillations=2, oscillation_window_frames=30)
        track = _MockTrack(track_id=1, cls=0, conf=0.9,
                           xyxy=(50, 50, 150, 250))
        frame = np.zeros((300, 200, 3), dtype=np.uint8)

        all_events = []
        for i in range(60):
            t = 100.0 + i * 0.1  # 6 seconds total
            # Oscillate Y between 95 and 115 (near nose at y=100)
            # This simulates the raise-to-mouth, lower, raise pattern
            phase = (i % 10) / 10.0  # 0 to 1 over 10 frames
            if phase < 0.5:
                y_wrist = 95.0 + phase * 40  # 95 → 115
            else:
                y_wrist = 115.0 - (phase - 0.5) * 40  # 115 → 95

            kpts = _kpts_with_wrist_near_nose(y_wrist=y_wrist)
            pose = _MockPose(track_id=1, keypoints=kpts)

            evts = det.detect(frame, [track], [pose], frame_idx=i, t=t)
            all_events.extend(evts)

        assert len(all_events) > 0, \
            "Oscillating hand-to-mouth pattern should trigger smoking"
        assert all_events[0].event_type == "SMOKING"
        assert all_events[0].details.get("oscillations", 0) >= 2

    def test_min_oscillations_configurable(self):
        """Higher min_oscillations should require more cycles before triggering."""
        det = _make_smoking_detector(min_oscillations=10, oscillation_window_frames=30)
        track = _MockTrack(track_id=2, cls=0, conf=0.9,
                           xyxy=(50, 50, 150, 250))
        frame = np.zeros((300, 200, 3), dtype=np.uint8)

        # Only 2 oscillation cycles — should NOT trigger with min_oscillations=10
        all_events = []
        for i in range(30):
            t = 100.0 + i * 0.1
            # Two gentle cycles over 30 frames
            phase = (i % 15) / 15.0
            if phase < 0.5:
                y_wrist = 95.0 + phase * 20
            else:
                y_wrist = 105.0 - (phase - 0.5) * 20

            kpts = _kpts_with_wrist_near_nose(y_wrist=y_wrist)
            pose = _MockPose(track_id=2, keypoints=kpts)
            evts = det.detect(frame, [track], [pose], frame_idx=i, t=t)
            all_events.extend(evts)

        assert len(all_events) == 0, \
            "Should not trigger with only 2 cycles when min_oscillations=10"

    def test_no_pose_no_crash(self):
        """SmokingDetector should handle tracks without pose gracefully."""
        det = _make_smoking_detector()
        track = _MockTrack(track_id=3, cls=0, conf=0.9,
                           xyxy=(50, 50, 150, 250))
        frame = np.zeros((300, 200, 3), dtype=np.uint8)

        # No poses at all
        events = det.detect(frame, [track], [], frame_idx=0, t=100.0)
        assert len(events) == 0

    def test_non_person_tracks_ignored(self):
        """Non-person tracks (cls != 0) should be ignored."""
        det = _make_smoking_detector()
        track = _MockTrack(track_id=4, cls=24, conf=0.9,  # backpack
                           xyxy=(50, 50, 150, 250))
        frame = np.zeros((300, 200, 3), dtype=np.uint8)

        events = det.detect(frame, [track], [], frame_idx=0, t=100.0)
        assert len(events) == 0
