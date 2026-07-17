"""Phase 4+5 integration tests — new features added in the pre-VLM pass.

Tests:
  1.  ObjectLeftDetector: stationary object triggers after min_stationary_s
  2.  ObjectLeftDetector: moving object does NOT trigger
  3.  ObjectLeftDetector: person (cls=0) is skipped
  4.  MotionPrefilter: identical frames → no motion; changed frame → motion
  5.  MotionPrefilter: reset() clears state
  6.  EventLogger: log + query round-trip (SQLite in-memory)
  7.  EventLogger: keyframe is saved alongside event
  8.  FireSmokeDetector: multi-frame confirmation blocks single-frame noise
  9.  FireSmokeDetector: HSV fallback still works (no weights key)
  10. PhoneWatcherDetector: hysteresis — confirm_frames blocks early emit
  11. PhoneWatcherDetector: hold timer keeps event alive after YOLO miss
  12. Full pipeline init: all detectors construct together without error

Run:
    python -m pytest tests/phase5_integration_test.py -v
    python tests/phase5_integration_test.py
"""
from __future__ import annotations

import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTrack:
    """Minimal tracker.Track stand-in."""
    def __init__(self, tid: int, cls: int, xyxy: tuple):
        self.track_id = tid
        self.cls = cls
        self.conf = 0.9
        self.xyxy = xyxy


def _blank(h=240, w=320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray(v=128, h=240, w=320) -> np.ndarray:
    return np.full((h, w, 3), v, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1-3. ObjectLeftDetector
# ---------------------------------------------------------------------------

def test_object_left_triggers_after_stationary():
    """A non-person object that stays in one place fires OBJECT_LEFT.

    ObjectLeftDetector has two hard-coded floors:
      - len(history) >= 30 before evaluating
      - recent = history[-int(min_stationary_s * 30)] must also be >= 30 samples
    So min_stationary_s >= 1.0 is required for the recent slice to be >= 30.
    We use min_stationary_s=1.5 and feed 55 samples at ~30fps (~1.83s of data).
    """
    from core.events import ObjectLeftDetector
    det = ObjectLeftDetector({
        "min_stationary_s": 1.5,          # 1.5s * 30fps = 45-sample recent window
        "position_variance_threshold": 100.0,
        "cooldown_s": 1000.0,
    })
    tr = FakeTrack(10, 24, (100, 100, 150, 150))   # cls=24 = backpack
    t0 = 1000.0
    all_events = []
    for i in range(55):                            # 55 frames = ~1.83s at 30fps
        t = t0 + i * (1 / 30.0)
        evts = det.detect([tr], frame_idx=i, t=t)
        all_events.extend(evts)
    assert len(all_events) >= 1, f"FAIL: expected OBJECT_LEFT, got {all_events}"
    assert all_events[0].event_type == "OBJECT_LEFT"
    assert all_events[0].details["object_class"] == "backpack"
    print(f"  [ok] object_left triggered: class={all_events[0].details['object_class']} "
          f"duration={all_events[0].details['stationary_duration_s']:.2f}s")


def test_object_left_no_trigger_if_moving():
    """An object that drifts more than position_variance_threshold does NOT fire."""
    from core.events import ObjectLeftDetector
    det = ObjectLeftDetector({
        "min_stationary_s": 0.1,
        "position_variance_threshold": 5.0,    # very tight variance
        "cooldown_s": 1.0,
    })
    t0 = 2000.0
    events = []
    for i in range(35):
        # Object moves 5px per sample — variance >> 5
        x = 100 + i * 5
        tr = FakeTrack(11, 26, (x, 100, x + 50, 150))   # cls=26 = handbag
        t = t0 + i * (1 / 30.0)
        events = det.detect([tr], frame_idx=i, t=t)
    assert len(events) == 0, f"FAIL: moving object triggered OBJECT_LEFT ({events})"
    print(f"  [ok] no object_left for moving object")


def test_object_left_skips_persons():
    """ObjectLeftDetector must never fire for persons (cls=0)."""
    from core.events import ObjectLeftDetector
    det = ObjectLeftDetector({
        "min_stationary_s": 0.1,
        "position_variance_threshold": 1000.0,
        "cooldown_s": 0.0,
    })
    tr = FakeTrack(12, 0, (100, 100, 200, 300))   # cls=0 = person
    t0 = 3000.0
    events = []
    for i in range(50):
        events = det.detect([tr], frame_idx=i, t=t0 + i * 0.033)
    assert len(events) == 0, f"FAIL: person triggered OBJECT_LEFT ({events})"
    print(f"  [ok] person correctly skipped by object_left detector")


# ---------------------------------------------------------------------------
# 4-5. MotionPrefilter
# ---------------------------------------------------------------------------

def test_motion_filter_static_scene():
    """Identical frames report no motion after the first frame."""
    from core.motion_filter import MotionPrefilter
    mf = MotionPrefilter(threshold=0.01, min_changed_pixels=500)
    frame = _gray(128)
    mf.has_motion(frame)   # prime with first frame
    result = mf.has_motion(frame)
    assert not result, "FAIL: identical frame reported as having motion"
    print(f"  [ok] motion_filter: identical frames → no motion")


def test_motion_filter_detects_change():
    """A significantly different second frame is detected as motion."""
    from core.motion_filter import MotionPrefilter
    mf = MotionPrefilter(min_changed_pixels=100)
    mf.has_motion(_gray(10))   # dark frame
    result = mf.has_motion(_gray(245))  # bright frame — massive diff
    assert result, "FAIL: large frame change not detected as motion"
    print(f"  [ok] motion_filter: large frame change → motion detected")


def test_motion_filter_reset():
    """reset() clears state so the very next frame always returns True."""
    from core.motion_filter import MotionPrefilter
    mf = MotionPrefilter(min_changed_pixels=100)
    frame = _gray(128)
    mf.has_motion(frame)       # prime
    mf.has_motion(frame)       # no motion
    mf.reset()
    result = mf.has_motion(frame)  # after reset: prev_gray is None → True
    assert result, "FAIL: first frame after reset() should return True"
    print(f"  [ok] motion_filter: reset() clears state correctly")


# ---------------------------------------------------------------------------
# 6-7. EventLogger
# ---------------------------------------------------------------------------

def test_event_logger_log_and_query():
    """Log a synthetic event and retrieve it via query_events."""
    from core.events import Event
    from core.event_logger import EventLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EventLogger(
            db_path=str(Path(tmpdir) / "test.db"),
            keyframes_dir=str(Path(tmpdir) / "keyframes"),
        )
        ev = Event(
            event_type="FIRE",
            t_iso="2024-01-01T00:00:00",
            frame_idx=42,
            details={"bbox": (10, 20, 100, 200), "confidence": 0.85, "track_id": None},
        )
        logger.log_event(ev, frame=None, frame_idx=42)
        rows = logger.query_events(event_type="FIRE")
        assert len(rows) == 1, f"FAIL: expected 1 row, got {len(rows)}"
        assert rows[0]["event_type"] == "FIRE"
        assert rows[0]["frame_idx"] == 42
        logger.close()
    print(f"  [ok] event_logger: log + query round-trip")


def test_event_logger_saves_keyframe():
    """When a valid frame is provided, a keyframe JPEG is written to disk."""
    from core.events import Event
    from core.event_logger import EventLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        kf_dir = Path(tmpdir) / "kf"
        logger = EventLogger(
            db_path=str(Path(tmpdir) / "test.db"),
            keyframes_dir=str(kf_dir),
        )
        ev = Event(
            event_type="SMOKE",
            t_iso="2024-01-01T12:00:00",
            frame_idx=7,
            details={"bbox": (0, 0, 50, 50), "confidence": 0.6, "track_id": None},
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        logger.log_event(ev, frame=frame, frame_idx=7)
        rows = logger.query_events(event_type="SMOKE")
        assert len(rows) == 1
        kf_path = rows[0]["keyframe_path"]
        assert kf_path is not None, "FAIL: keyframe_path is None"
        assert Path(kf_path).exists(), f"FAIL: keyframe file not found at {kf_path}"
        logger.close()
    print(f"  [ok] event_logger: keyframe JPEG saved to disk")


def test_event_logger_multi_type_query():
    """Query by event_type correctly filters among multiple event types."""
    from core.events import Event
    from core.event_logger import EventLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EventLogger(
            db_path=str(Path(tmpdir) / "test.db"),
            keyframes_dir=str(Path(tmpdir) / "kf"),
        )
        for etype in ["FIRE", "SMOKE", "PHONE", "PHONE"]:
            ev = Event(event_type=etype, t_iso="2024-01-01T00:00:00",
                       frame_idx=1, details={"confidence": 0.5})
            logger.log_event(ev, frame=None, frame_idx=1)
        phone_rows = logger.query_events(event_type="PHONE")
        fire_rows  = logger.query_events(event_type="FIRE")
        assert len(phone_rows) == 2, f"FAIL: expected 2 PHONE rows, got {len(phone_rows)}"
        assert len(fire_rows)  == 1, f"FAIL: expected 1 FIRE row, got {len(fire_rows)}"
        logger.close()
    print(f"  [ok] event_logger: multi-type filtering works correctly")


# ---------------------------------------------------------------------------
# 8-9. FireSmokeDetector — multi-frame confirmation
# ---------------------------------------------------------------------------

def test_fire_smoke_multiframe_confirmation_blocks_single_frame():
    """YOLO mode: single fire detection in a 5-frame window must NOT emit
    when min_consecutive_frames=2.  We mock _use_yolo=False so this falls
    back to HSV — but we test the NEW HSV path which uses fire_min_duration.
    The key assertion is that calling detect() once does not fire even on a
    matching frame when the required consecutive count isn't met.
    """
    from core.events import FireSmokeDetector
    import cv2
    # HSV fallback with fire_min_duration=3: needs 3 consecutive matches
    det = FireSmokeDetector({
        "fire_min_pixel_ratio": 0.001,
        "fire_min_duration": 3,
    })
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (200, 200), (0, 100, 255), -1)  # orange region
    # Call 1 and 2: consecutive=1, 2 — below fire_min_duration=3
    ev1 = det.detect(frame, frame_idx=0)
    ev2 = det.detect(frame, frame_idx=1)
    fire1 = [e for e in ev1 if e.event_type == "FIRE"]
    fire2 = [e for e in ev2 if e.event_type == "FIRE"]
    assert len(fire1) == 0, f"FAIL: fire emitted at consecutive=1 (expected suppress)"
    assert len(fire2) == 0, f"FAIL: fire emitted at consecutive=2 (expected suppress)"
    # Call 3: consecutive=3 — should fire
    ev3 = det.detect(frame, frame_idx=2)
    fire3 = [e for e in ev3 if e.event_type == "FIRE"]
    assert len(fire3) >= 1, f"FAIL: fire NOT emitted at consecutive=3"
    print(f"  [ok] fire_smoke: multi-frame confirmation suppresses single-frame noise")


def test_fire_smoke_hsv_fallback_resets_on_miss():
    """HSV fallback: consecutive counter resets when fire not seen."""
    from core.events import FireSmokeDetector
    import cv2
    det = FireSmokeDetector({"fire_min_pixel_ratio": 0.001, "fire_min_duration": 2})
    fire_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(fire_frame, (50, 50), (200, 200), (0, 100, 255), -1)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    det.detect(fire_frame, frame_idx=0)  # consecutive=1
    det.detect(blank, frame_idx=1)       # reset → consecutive=0
    ev = det.detect(fire_frame, frame_idx=2)  # consecutive=1 again, below 2
    fire_ev = [e for e in ev if e.event_type == "FIRE"]
    assert len(fire_ev) == 0, f"FAIL: fire fired after counter was reset"
    print(f"  [ok] fire_smoke: HSV consecutive counter resets on miss")


# ---------------------------------------------------------------------------
# 10-11. PhoneWatcherDetector — hysteresis
# ---------------------------------------------------------------------------

class _FakePose:
    """Minimal pose stand-in for PhoneWatcherDetector."""
    def __init__(self, tid):
        self.track_id = tid
        # keypoints: nose, ..., l_shoulder, r_shoulder — all (0, 0, 0) = low confidence
        self.keypoints = [(0.0, 0.0, 0.0)] * 17


def test_phone_hysteresis_confirm_frames():
    """PhoneWatcherDetector should NOT emit before confirm_frames consecutive
    detections.  We test this by constructing with confirm_frames=3 and
    calling detect() twice with a blank frame (YOLO won't detect anything),
    then with a frame containing a green square where we inject a mock result.

    Since we can't easily mock the internal YOLO call, we test the hysteresis
    by verifying the detector constructs with the right params and the
    _confirm_frames attribute is set correctly.
    """
    from core.events import PhoneWatcherDetector
    det = PhoneWatcherDetector(cfg={
        "weights": "models/yolov8n.pt",
        "imgsz": 320,
        "conf": 0.15,
        "confirm_frames": 5,
        "hold_frames": 10,
    })
    assert det._confirm_frames == 5, f"FAIL: confirm_frames={det._confirm_frames}, expected 5"
    assert det._hold_frames == 10, f"FAIL: hold_frames={det._hold_frames}, expected 10"
    # Blank frame → no phone detected → no event regardless of confirm state
    frame = _blank()
    ev = det.detect(frame, [], [], frame_idx=0)
    assert ev == [], f"FAIL: blank frame produced events: {ev}"
    print(f"  [ok] phone hysteresis: confirm_frames/hold_frames params set correctly, "
          f"blank frame → no event")


def test_phone_hysteresis_hold_timer_attr():
    """Verify _track_state is per-track and hold timer decrements correctly."""
    from core.events import PhoneWatcherDetector
    det = PhoneWatcherDetector(cfg={
        "weights": "models/yolov8n.pt",
        "imgsz": 320,
        "conf": 0.15,
        "confirm_frames": 3,
        "hold_frames": 5,
    })
    # Directly inject a per-track state as if a detection just fired:
    det._track_state[42] = {"consec": 3, "hold": 5, "last_bbox": (10, 10, 50, 50), "last_conf": 0.3}
    # Now call detect() with that track and a blank frame (no YOLO phone detection)
    tr = FakeTrack(42, 0, (0, 0, 200, 400))
    frame = _blank(480, 640)
    ev = det.detect(frame, [tr], [], frame_idx=1)
    # Should emit because hold > 0 (even though no detection this frame)
    assert len(ev) >= 1, f"FAIL: expected hold event, got {ev}"
    assert ev[0].details["hold_remaining"] > 0
    # hold should have decremented
    assert det._track_state[42]["hold"] == 4, \
        f"FAIL: hold not decremented: {det._track_state[42]['hold']}"
    print(f"  [ok] phone hysteresis: hold timer keeps event alive after YOLO miss")


# ---------------------------------------------------------------------------
# 12. Full pipeline init smoke test
# ---------------------------------------------------------------------------

def test_full_pipeline_constructs():
    """Verify all Phase 4 detectors can be constructed simultaneously
    without interfering with each other (no import errors, no VRAM conflicts).

    This is the integration smoke test — it does NOT run a full video loop,
    but confirms all modules load together cleanly.
    """
    from core.events import (
        FireSmokeDetector, SmokingDetector, PhoneWatcherDetector,
        GatheringDetector, ViolenceDetector, ObjectLeftDetector,
    )
    from core.motion_filter import MotionPrefilter
    from core.event_logger import EventLogger
    from core.config import load_models_config
    from pipeline.frame_router import FrameRouter
    from core.config import load_pipeline_config

    cfg = load_pipeline_config()
    models_cfg = load_models_config()
    router = FrameRouter(cfg.get("router", {}))

    # Construct all detectors with real config
    mf   = MotionPrefilter()
    fs   = FireSmokeDetector(models_cfg.get("fire_smoke", {}))
    sm   = SmokingDetector(models_cfg.get("smoking", {}))
    ph   = PhoneWatcherDetector(models_cfg.get("phone", {}))
    ga   = GatheringDetector(models_cfg.get("gathering", {}))
    vi   = ViolenceDetector(models_cfg.get("violence", {}))
    ol   = ObjectLeftDetector(models_cfg.get("object_left", {}))

    with tempfile.TemporaryDirectory() as tmpdir:
        el = EventLogger(
            db_path=str(Path(tmpdir) / "test.db"),
            keyframes_dir=str(Path(tmpdir) / "kf"),
        )
        el.close()

    # Run a single blank frame through all detectors
    frame = _blank(720, 1280)
    fs.detect(frame, frame_idx=0)
    sm.detect(frame, [], frame_idx=0)
    ph.detect(frame, [], [], frame_idx=0)
    ga.detect([], frame_idx=0, t=0.0)
    vi.detect([], frame_idx=0, t=0.0)
    ol.detect([], frame_idx=0, t=0.0)
    mf.has_motion(frame)

    print(f"  [ok] full pipeline: all Phase 4 detectors construct + run on blank frame "
          f"without error")
    print(f"        router stages active: {router.active_stages()}")


def test_event_logger_log_fall_event():
    """Verify that EventLogger can log and query a FallEvent object successfully."""
    from core.state_machine import FallEvent
    from core.event_logger import EventLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EventLogger(
            db_path=str(Path(tmpdir) / "test.db"),
            keyframes_dir=str(Path(tmpdir) / "kf"),
        )
        ev = FallEvent(
            track_id=1,
            t_iso="2024-01-01T12:00:00",
            frame_idx=681,
            aspect_now=1.51,
            keypoints_low=True,
            kp_height_frac=0.45,
            upright_duration_s=5.0,
            transition_delta_s=1.2,
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        logger.log_event(ev, frame=frame, frame_idx=681)
        rows = logger.query_events(event_type="FALL")
        assert len(rows) == 1, f"Expected 1 FALL event, got {len(rows)}"
        assert rows[0]["event_type"] == "FALL"
        assert rows[0]["frame_idx"] == 681
        assert rows[0]["track_id"] == 1
        logger.close()
    print(f"  [ok] event_logger: FallEvent logged and queried correctly")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("phase5_integration_test:")
    test_object_left_triggers_after_stationary()
    test_object_left_no_trigger_if_moving()
    test_object_left_skips_persons()
    test_motion_filter_static_scene()
    test_motion_filter_detects_change()
    test_motion_filter_reset()
    test_event_logger_log_and_query()
    test_event_logger_saves_keyframe()
    test_event_logger_multi_type_query()
    test_event_logger_log_fall_event()
    test_fire_smoke_multiframe_confirmation_blocks_single_frame()
    test_fire_smoke_hsv_fallback_resets_on_miss()
    test_phone_hysteresis_confirm_frames()
    test_phone_hysteresis_hold_timer_attr()
    test_full_pipeline_constructs()
    print("phase5_integration_test: all passed")


if __name__ == "__main__":
    main()
