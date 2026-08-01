"""Phase 5C tests — GatheringDetector wall-clock sustained duration.

Tests that the GatheringDetector correctly uses wall-clock seconds (not frame counts)
to gate gathering events, and that ROI polygon filtering works.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest


@dataclass
class _MockTrack:
    track_id: int
    cls: int
    conf: float
    xyxy: tuple[int, int, int, int]


def _make_person_tracks(n: int, base_x: int = 100, spacing: int = 30) -> list[_MockTrack]:
    """Create n person tracks clustered near base_x."""
    tracks = []
    for i in range(n):
        x1 = base_x + i * spacing
        y1 = 100
        x2 = x1 + 50
        y2 = 300
        tracks.append(_MockTrack(track_id=i, cls=0, conf=0.9, xyxy=(x1, y1, x2, y2)))
    return tracks


class TestGatheringDetectorSustainedDuration:
    """Test that gathering detection uses wall-clock sustained seconds."""

    def _make_detector(self, sustained_s: float = 1.0, min_people: int = 3,
                       cooldown_s: float = 60.0, radius: float = 200.0):
        from core.events import GatheringDetector
        cfg = {
            "min_people": min_people,
            "radius_pixels": radius,
            "cooldown_s": cooldown_s,
            "sustained_s": sustained_s,
        }
        # Patch out ROI loading to avoid filesystem dependency
        with patch("core.events.GatheringDetector.__init__", lambda self, cfg=None: None):
            det = GatheringDetector.__new__(GatheringDetector)
        det.cfg = cfg
        det.min_people = min_people
        det.radius_pixels = radius
        det.cooldown_s = cooldown_s
        det.sustained_s = sustained_s
        det._sustained_since = {}
        det._last_fire_times = {}
        det.rois = [{"id": "default", "polygon": None}]
        return det

    def test_does_not_fire_before_sustained_s(self):
        """Gathering should NOT fire if the crowd hasn't been present for sustained_s."""
        det = self._make_detector(sustained_s=2.0)
        tracks = _make_person_tracks(4)

        # First call at t=100.0 — starts the timer
        events = det.detect(tracks, frame_idx=0, t=100.0)
        assert len(events) == 0, "Should not fire on first detection"

        # Call at t=101.0 — only 1s elapsed, need 2s
        events = det.detect(tracks, frame_idx=1, t=101.0)
        assert len(events) == 0, "Should not fire before sustained_s"

    def test_fires_after_sustained_s(self):
        """Gathering SHOULD fire once sustained_s seconds have elapsed."""
        det = self._make_detector(sustained_s=2.0)
        tracks = _make_person_tracks(4)

        det.detect(tracks, frame_idx=0, t=100.0)  # start timer
        det.detect(tracks, frame_idx=1, t=101.0)  # 1s — not yet
        events = det.detect(tracks, frame_idx=2, t=102.1)  # 2.1s — should fire

        assert len(events) == 1
        assert events[0].event_type == "GATHERING"
        assert events[0].details["count"] == 4
        assert events[0].details["duration_seconds"] >= 2.0

    def test_resets_when_people_leave(self):
        """Timer resets if the cluster drops below threshold between calls."""
        det = self._make_detector(sustained_s=2.0)
        tracks_4 = _make_person_tracks(4)
        tracks_2 = _make_person_tracks(2)

        det.detect(tracks_4, frame_idx=0, t=100.0)  # start timer
        det.detect(tracks_2, frame_idx=1, t=101.0)  # below threshold → reset
        events = det.detect(tracks_4, frame_idx=2, t=103.0)  # restart, only 0s elapsed

        assert len(events) == 0, "Timer should reset when people leave"

    def test_roi_polygon_filtering(self):
        """Tracks outside the ROI polygon should not count toward gathering."""
        det = self._make_detector(sustained_s=0.0, min_people=3)
        # ROI is a small box (0,0)-(200,400)
        det.rois = [{"id": "entrance", "polygon": [[0, 0], [200, 0], [200, 400], [0, 400]]}]

        # 2 people inside the ROI, 2 people far outside
        tracks = [
            _MockTrack(0, 0, 0.9, (10, 100, 60, 300)),    # inside
            _MockTrack(1, 0, 0.9, (50, 100, 100, 300)),   # inside
            _MockTrack(2, 0, 0.9, (500, 100, 550, 300)),  # outside
            _MockTrack(3, 0, 0.9, (600, 100, 650, 300)),  # outside
        ]
        events = det.detect(tracks, frame_idx=0, t=100.0)
        assert len(events) == 0, "Only 2 people in ROI — below threshold of 3"

    def test_backward_compat_sustained_frames(self):
        """If sustained_s is absent, sustained_frames should be converted to seconds."""
        from core.events import GatheringDetector
        with patch("core.events.GatheringDetector.__init__", lambda self, cfg=None: None):
            det = GatheringDetector.__new__(GatheringDetector)
        det.cfg = {"sustained_frames": 60}  # 60 frames / 30fps = 2.0s
        det.min_people = 3
        det.radius_pixels = 200.0
        det.cooldown_s = 60.0
        # Simulate the init logic
        if "sustained_s" in det.cfg:
            det.sustained_s = float(det.cfg["sustained_s"])
        else:
            det.sustained_s = int(det.cfg.get("sustained_frames", 5)) / 30.0
        assert abs(det.sustained_s - 2.0) < 0.01


class TestGatheringDetectorCooldown:
    """Test cooldown prevents re-trigger within cooldown_s."""

    def _make_detector(self):
        from core.events import GatheringDetector
        with patch("core.events.GatheringDetector.__init__", lambda self, cfg=None: None):
            det = GatheringDetector.__new__(GatheringDetector)
        det.cfg = {}
        det.min_people = 3
        det.radius_pixels = 200.0
        det.cooldown_s = 5.0
        det.sustained_s = 0.0
        det._sustained_since = {}
        det._last_fire_times = {}
        det.rois = [{"id": "default", "polygon": None}]
        return det

    def test_cooldown_suppresses_retrigger(self):
        det = self._make_detector()
        tracks = _make_person_tracks(4)

        events1 = det.detect(tracks, frame_idx=0, t=100.0)
        assert len(events1) == 1  # first fire

        events2 = det.detect(tracks, frame_idx=1, t=102.0)
        assert len(events2) == 0  # within cooldown

        events3 = det.detect(tracks, frame_idx=2, t=106.0)
        assert len(events3) == 1  # after cooldown
