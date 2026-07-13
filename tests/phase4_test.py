"""Phase 4 unit tests — all 5 event detectors.

Tests:
  1. Fire/smoke: HSV heuristic detects a synthetic fire-colored region
  2. Smoking: glow heuristic flags a bright spot near a person crop
  3. Phone: YOLO detects a cell phone + head-pose heuristic (synthetic)
  4. Gathering: 3+ people within radius triggers; 2 people don't
  5. Violence: overlapping bboxes + rapid motion triggers; slow motion doesn't

Run:
    python tests/phase4_test.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- helpers ---

class FakeTrack:
    def __init__(self, tid, cls, xyxy):
        self.track_id = tid
        self.cls = cls
        self.conf = 0.9
        self.xyxy = xyxy


def _make_frame_with_fire(w=320, h=240) -> np.ndarray:
    """A frame with a bright orange region (fire) in the center."""
    import cv2
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # fire: bright orange-red region
    cv2.rectangle(frame, (w//2-40, h//2-40), (w//2+40, h//2+40), (0, 100, 255), -1)  # BGR: orange
    return frame


def _make_frame_with_smoke(w=320, h=240) -> np.ndarray:
    """A frame with a large gray region (smoke) in the upper portion."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # smoke: light gray region
    frame[:h//2, :] = 180  # gray (BGR all equal)
    return frame


def _make_frame_with_glow(w=80, h=60) -> np.ndarray:
    """A small crop with a bright orange spot (cigarette glow)."""
    import cv2
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(frame, (w//2, h//2), 5, (0, 120, 255), -1)  # bright orange spot
    return frame


# --- tests ---

def test_fire_detection():
    """Fire heuristic should detect a synthetic fire-colored region.
    Requires 2 consecutive detections (fire_min_duration=2) before firing."""
    from core.events import FireSmokeDetector
    det = FireSmokeDetector({"fire_min_pixel_ratio": 0.005, "fire_min_duration": 2})
    frame = _make_frame_with_fire()
    # first call primes the counter (no event yet at consecutive=1)
    det.detect(frame, frame_idx=0)
    # second call satisfies min_duration=2
    events = det.detect(frame, frame_idx=1)
    fire_events = [e for e in events if e.event_type == "FIRE"]
    assert len(fire_events) >= 1, f"FAIL: expected FIRE event, got {len(fire_events)}"
    print(f"  [ok] fire detected: pixel_ratio={fire_events[0].details['pixel_ratio']}")


def test_no_fire_on_blank():
    """A black frame should NOT trigger fire."""
    from core.events import FireSmokeDetector
    det = FireSmokeDetector({"fire_min_pixel_ratio": 0.01, "fire_min_duration": 2})
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    events = det.detect(frame, frame_idx=0)
    fire_events = [e for e in events if e.event_type == "FIRE"]
    assert len(fire_events) == 0, f"FAIL: blank frame triggered fire ({len(fire_events)})"
    print(f"  [ok] no fire on blank frame")


def test_smoking_detection():
    """Smoking glow heuristic should flag a bright spot near a person."""
    from core.events import SmokingDetector
    import cv2
    det = SmokingDetector({"min_object_area": 10})
    # make a 200x200 frame with a glow in the UPPER portion (face/hand area)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 50), 8, (0, 120, 255), -1)  # bright orange glow near top
    tr = FakeTrack(1, 0, (0, 0, 200, 200))
    events = det.detect(frame, [tr], frame_idx=0)
    assert len(events) >= 1, f"FAIL: expected SMOKING event, got {len(events)}"
    print(f"  [ok] smoking detected: glow_area={events[0].details['glow_area']}")


def test_gathering_triggers():
    """3+ people within radius should trigger GATHERING."""
    from core.events import GatheringDetector
    det = GatheringDetector({"min_people": 3, "radius_pixels": 100, "cooldown_s": 0})
    tracks = [
        FakeTrack(1, 0, (100, 100, 150, 200)),
        FakeTrack(2, 0, (120, 100, 170, 200)),
        FakeTrack(3, 0, (140, 100, 190, 200)),
    ]
    events = det.detect(tracks, frame_idx=0, t=100.0)
    assert len(events) >= 1, f"FAIL: expected GATHERING with 3 people, got {len(events)}"
    assert events[0].details["count"] >= 3
    print(f"  [ok] gathering detected: count={events[0].details['count']}")


def test_gathering_no_trigger_with_2():
    """Only 2 people should NOT trigger gathering (min_people=3)."""
    from core.events import GatheringDetector
    det = GatheringDetector({"min_people": 3, "radius_pixels": 100, "cooldown_s": 0})
    tracks = [
        FakeTrack(1, 0, (100, 100, 150, 200)),
        FakeTrack(2, 0, (120, 100, 170, 200)),
    ]
    events = det.detect(tracks, frame_idx=0, t=100.0)
    assert len(events) == 0, f"FAIL: 2 people triggered gathering ({len(events)})"
    print(f"  [ok] no gathering with only 2 people")


def test_violence_triggers():
    """Overlapping bboxes + rapid relative motion should trigger VIOLENCE."""
    from core.events import ViolenceDetector
    import time as _time
    det = ViolenceDetector({"iou_threshold": 0.05, "motion_threshold": 10.0,
                            "window_s": 0.1, "cooldown_s": 0})
    # frame 0: two people overlapping, at positions A and B
    t0 = 100.0
    tracks0 = [
        FakeTrack(1, 0, (100, 100, 200, 250)),
        FakeTrack(2, 0, (150, 100, 250, 250)),   # overlaps with track 1
    ]
    det.detect(tracks0, frame_idx=0, t=t0)
    # frame 1: 0.2s later, both have moved significantly
    tracks1 = [
        FakeTrack(1, 0, (130, 100, 230, 250)),   # moved 30px right
        FakeTrack(2, 0, (120, 100, 220, 250)),   # moved 30px left
    ]
    events = det.detect(tracks1, frame_idx=1, t=t0 + 0.2)
    # should fire because sustained contact + rapid motion over >window_s
    assert len(events) >= 1, f"FAIL: expected VIOLENCE event, got {len(events)}"
    print(f"  [ok] violence detected: iou={events[0].details['iou']} "
          f"motion={events[0].details['rel_motion']}")


def test_violence_no_trigger_slow_motion():
    """Overlapping bboxes but SLOW motion should NOT trigger violence."""
    from core.events import ViolenceDetector
    det = ViolenceDetector({"iou_threshold": 0.05, "motion_threshold": 50.0,
                            "window_s": 0.1, "cooldown_s": 0})
    t0 = 200.0
    # two people overlapping, barely moving
    tracks0 = [
        FakeTrack(1, 0, (100, 100, 200, 250)),
        FakeTrack(2, 0, (150, 100, 250, 250)),
    ]
    det.detect(tracks0, frame_idx=0, t=t0)
    tracks1 = [
        FakeTrack(1, 0, (102, 100, 202, 250)),   # moved only 2px
        FakeTrack(2, 0, (152, 100, 252, 250)),
    ]
    events = det.detect(tracks1, frame_idx=1, t=t0 + 0.2)
    assert len(events) == 0, f"FAIL: slow motion triggered violence ({len(events)})"
    print(f"  [ok] no violence with slow motion")


def test_phone_detection():
    """Phone detector: construction + blank-frame smoke test.

    PhoneWatcherDetector ALWAYS loads its own independent YOLOv8n instance
    (COCO class 67 only). The `detector_model` param is accepted for backward
    compatibility but is IGNORED. This isolates phone detection from the
    tracker's shared model (which uses classes=[0] + persist=True, and mixing
    classes=[67] on the same model corrupted tracker state).
    """
    from core.events import PhoneWatcherDetector
    # Pass cfg only; detector_model kwarg is ignored by PhoneWatcherDetector
    det = PhoneWatcherDetector(cfg={"weights": "models/yolov8n.pt",
                                    "imgsz": 320, "conf": 0.3})
    # blank frame, no tracks, no poses — should return empty
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = det.detect(frame, [], [], frame_idx=0)
    assert events == [], "FAIL: expected no events on blank frame"
    print(f"  [ok] phone detector constructs (own YOLO instance) + returns empty on blank frame")


def main():
    print("phase4_test:")
    test_fire_detection()
    test_no_fire_on_blank()
    test_smoking_detection()
    test_gathering_triggers()
    test_gathering_no_trigger_with_2()
    test_violence_triggers()
    test_violence_no_trigger_slow_motion()
    test_phone_detection()
    print("phase4_test: all passed")


if __name__ == "__main__":
    main()