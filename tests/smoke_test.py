"""Smoke test: VideoSource + Detector + Tracker + FrameRouter import and run.

Run with:  python -m tests.smoke_test
Or:        python tests/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# make `core` / `pipeline` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_frame_router():
    from pipeline.frame_router import FrameRouter
    r = FrameRouter({"stages": {"detect": {"enabled": True, "every": 1},
                                "track": {"enabled": True, "every": 1}}})
    assert r.should_run("detect", 5)
    assert r.should_run("track", 100)
    assert not r.should_run("pose", 0)  # not registered -> disabled
    print("  [ok] frame_router")


def test_video_source_factory():
    from core.video_source import build_source, FileSource, WebcamSource, RTSPSource
    assert isinstance(build_source({"type": "webcam"}), WebcamSource)
    assert isinstance(build_source({"type": "file", "path": "x.mp4"}), FileSource)
    assert isinstance(build_source({"type": "rtsp", "path": "rtsp://x"}), RTSPSource)
    print("  [ok] video_source factory")


def test_detector_and_tracker():
    from core.detector import Detector
    from core.tracker import Tracker
    d = Detector()
    t = Tracker(d)
    f = np.zeros((d.imgsz, d.imgsz, 3), dtype=np.uint8)
    dets = d.detect(f)
    assert isinstance(dets, list)
    tracks = t.update(f)
    assert isinstance(tracks, list)
    print(f"  [ok] detector+tracker  warmup_vram={d.vram_mb()}MB imgsz={d.imgsz}")


def main():
    print("smoke_test:")
    test_frame_router()
    test_video_source_factory()
    test_detector_and_tracker()
    print("smoke_test: all passed")


if __name__ == "__main__":
    main()