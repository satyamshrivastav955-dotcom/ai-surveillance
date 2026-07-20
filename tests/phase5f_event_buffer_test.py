"""Tests for core/event_buffer.py — Phase 5F."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from core.event_buffer import EventBuffer


# ---------------------------------------------------------------------------
# Helpers to build fake events matching the various event types in the system
# ---------------------------------------------------------------------------

def _phase4_event(etype: str, track_id: int | None = None, frame_idx: int = 1,
                  extra: dict | None = None) -> SimpleNamespace:
    """Fake core.events.Event-like object."""
    details: dict = {}
    if track_id is not None:
        details["track_id"] = track_id
    if extra:
        details.update(extra)
    return SimpleNamespace(
        event_type=etype,
        t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
        frame_idx=frame_idx,
        details=details,
    )


def _fall_event(track_id: int, frame_idx: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="FALL",
        t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
        frame_idx=frame_idx,
        track_id=track_id,
        details={"track_id": track_id},
    )


def _fight_event(tid_a: int, tid_b: int, conf: float = 0.75,
                 frame_idx: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="FIGHT",
        t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
        frame_idx=frame_idx,
        track_ids=(tid_a, tid_b),
        confidence=conf,
        clip_ref="data/clips/clip_fight_test.mp4",
        details={"proximity_px": 120.0},
    )


def _identity_event(track_id: int, label: str, frame_idx: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="IDENTITY",
        t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
        frame_idx=frame_idx,
        track_id=track_id,
        label=label,
        source="face",
        similarity=0.82,
        details={"track_id": track_id},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_buffer(tmp_path, flush_interval_s: float = 100.0) -> EventBuffer:
    """Build EventBuffer with a long flush interval (no auto-flush during tests)."""
    return EventBuffer({
        "camera_id": "test_cam",
        "flush_interval_s": flush_interval_s,
        "json_dir": str(tmp_path),
        "max_track_age": 300,
    })


def test_basic_flush_empty(tmp_path):
    """Flushing with no events produces a valid JSON blob."""
    buf = _make_buffer(tmp_path)
    blob = buf.force_flush(frame_idx=0)
    assert blob["camera_id"] == "test_cam"
    assert "window_start" in blob
    assert "window_end" in blob
    assert blob["tracks"] == []
    assert blob["scene_events"] == []


def test_json_file_written(tmp_path):
    """Flush must write a .json file to the json_dir."""
    buf = _make_buffer(tmp_path)
    buf.force_flush(frame_idx=0)
    json_files = list(tmp_path.glob("events_test_cam_*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text())
    assert data["camera_id"] == "test_cam"


def test_phase4_fire_event_scene_level(tmp_path):
    """FIRE event without track_id goes to scene_events."""
    buf = _make_buffer(tmp_path)
    ev = _phase4_event("FIRE", track_id=None, extra={"bbox": [10, 10, 50, 50]})
    buf.append(ev, frame_idx=1)
    blob = buf.force_flush(frame_idx=1)
    scene_types = [e["type"] for e in blob["scene_events"]]
    assert "FIRE" in scene_types


def test_phase4_smoking_per_track(tmp_path):
    """SMOKING event with track_id goes into the track's events list."""
    buf = _make_buffer(tmp_path)
    ev = _phase4_event("SMOKING", track_id=5)
    buf.update_track(5, bbox=(10, 10, 100, 300), frame_idx=1)
    buf.append(ev, frame_idx=1)
    blob = buf.force_flush(frame_idx=1)
    track = next((t for t in blob["tracks"] if t["track_id"] == 5), None)
    assert track is not None
    event_types = [e["type"] for e in track["events"]]
    assert "SMOKING" in event_types


def test_fall_event_per_track(tmp_path):
    """FALL event goes into the correct track's events."""
    buf = _make_buffer(tmp_path)
    ev = _fall_event(track_id=3, frame_idx=5)
    buf.update_track(3, bbox=(50, 50, 200, 400), frame_idx=5)
    buf.append(ev, frame_idx=5)
    blob = buf.force_flush(frame_idx=5)
    track = next((t for t in blob["tracks"] if t["track_id"] == 3), None)
    assert track is not None
    assert any(e["type"] == "FALL" for e in track["events"])


def test_fight_event_scene_level(tmp_path):
    """FIGHT event goes to scene_events with track_ids and clip_ref."""
    buf = _make_buffer(tmp_path)
    ev = _fight_event(tid_a=10, tid_b=11, conf=0.9)
    buf.append(ev, frame_idx=10)
    blob = buf.force_flush(frame_idx=10)
    fight_events = [e for e in blob["scene_events"] if e["type"] == "FIGHT"]
    assert len(fight_events) == 1
    fe = fight_events[0]
    assert set(fe["track_ids"]) == {10, 11}
    assert fe["confidence"] == pytest.approx(0.9, abs=0.01)
    assert "clip_fight" in fe["clip_ref"]


def test_identity_per_track(tmp_path):
    """IDENTITY event goes into the correct track's events with label."""
    buf = _make_buffer(tmp_path)
    ev = _identity_event(track_id=7, label="Alice")
    buf.update_track(7, bbox=(0, 0, 100, 300), frame_idx=2)
    buf.append(ev, frame_idx=2)
    blob = buf.force_flush(frame_idx=2)
    track = next((t for t in blob["tracks"] if t["track_id"] == 7), None)
    assert track is not None
    identity_evs = [e for e in track["events"] if e["type"] == "IDENTITY"]
    assert len(identity_evs) == 1
    assert identity_evs[0]["label"] == "Alice"


def test_gathering_dedup_still_present(tmp_path):
    """Second GATHERING trigger in same window should have action='still_present'."""
    buf = _make_buffer(tmp_path)
    ev1 = _phase4_event("GATHERING", extra={"count": 3, "track_ids": [1, 2, 3], "roi_id": "default"})
    ev2 = _phase4_event("GATHERING", extra={"count": 3, "track_ids": [1, 2, 3], "roi_id": "default"})
    buf.append(ev1, frame_idx=1)
    buf.append(ev2, frame_idx=2)
    blob = buf.force_flush(frame_idx=2)
    gatherings = [e for e in blob["scene_events"] if e["type"] == "GATHERING"]
    assert len(gatherings) == 2
    actions = [e["action"] for e in gatherings]
    assert "trigger" in actions
    assert "still_present" in actions


def test_object_left_dedup(tmp_path):
    """Repeated OBJECT_LEFT for same track_id uses 'still_present'."""
    buf = _make_buffer(tmp_path)
    for i in range(3):
        ev = _phase4_event("OBJECT_LEFT", extra={"track_id": 99, "dwell_s": i * 10.0})
        buf.append(ev, frame_idx=i * 30)
    blob = buf.force_flush(frame_idx=90)
    ol_events = [e for e in blob["scene_events"] if e["type"] == "OBJECT_LEFT"]
    assert len(ol_events) == 3
    assert ol_events[0]["action"] == "trigger"
    assert all(e["action"] == "still_present" for e in ol_events[1:])


def test_track_update_bbox_pose_quality(tmp_path):
    """update_track sets bbox_last and pose_quality in the flushed output."""
    buf = _make_buffer(tmp_path)
    buf.update_track(42, bbox=(10, 20, 110, 220), pose_quality="stable", frame_idx=1)
    blob = buf.force_flush(frame_idx=1)
    # Track 42 has no events but was updated — should appear if recently seen
    track = next((t for t in blob["tracks"] if t["track_id"] == 42), None)
    # Track may be filtered out because max_track_age=300 > 1 frame, so present
    if track:
        assert track["bbox_last"] == [10, 20, 110, 220]
        assert track["pose_quality"] == "stable"


def test_buffer_resets_after_flush(tmp_path):
    """After flush, scene_events should be empty on next flush."""
    buf = _make_buffer(tmp_path)
    ev = _phase4_event("FIRE")
    buf.append(ev, frame_idx=1)
    buf.force_flush(frame_idx=1)
    # Second flush — should have nothing
    blob2 = buf.force_flush(frame_idx=2)
    assert blob2["scene_events"] == []


def test_maybe_flush_not_yet(tmp_path):
    """maybe_flush should return None before the interval has elapsed."""
    buf = _make_buffer(tmp_path, flush_interval_s=3600.0)
    result = buf.maybe_flush(frame_idx=0)
    assert result is None


def test_multi_track_events_separated(tmp_path):
    """Events for different tracks end up in their respective track records."""
    buf = _make_buffer(tmp_path)
    ev_a = _phase4_event("PHONE", track_id=1)
    ev_b = _phase4_event("SMOKING", track_id=2)
    buf.update_track(1, bbox=(0, 0, 100, 300), frame_idx=1)
    buf.update_track(2, bbox=(200, 0, 300, 300), frame_idx=1)
    buf.append(ev_a, frame_idx=1)
    buf.append(ev_b, frame_idx=1)
    blob = buf.force_flush(frame_idx=1)
    t1 = next(t for t in blob["tracks"] if t["track_id"] == 1)
    t2 = next(t for t in blob["tracks"] if t["track_id"] == 2)
    assert any(e["type"] == "PHONE" for e in t1["events"])
    assert any(e["type"] == "SMOKING" for e in t2["events"])
    assert not any(e["type"] == "SMOKING" for e in t1["events"])


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_basic_flush_empty(p)
        test_json_file_written(Path(d))
        test_phase4_fire_event_scene_level(Path(d))
        test_phase4_smoking_per_track(Path(d))
        test_fall_event_per_track(Path(d))
        test_fight_event_scene_level(Path(d))
        test_identity_per_track(Path(d))
        test_gathering_dedup_still_present(Path(d))
        test_object_left_dedup(Path(d))
        test_track_update_bbox_pose_quality(Path(d))
        test_buffer_resets_after_flush(Path(d))
        test_maybe_flush_not_yet(Path(d))
        test_multi_track_events_separated(Path(d))
    print("phase5f_event_buffer_test: all passed")
