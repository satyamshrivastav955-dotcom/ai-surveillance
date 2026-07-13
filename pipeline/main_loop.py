"""Phase 1 main loop: capture -> detect -> track -> draw -> display.

No other models run here yet. The loop logs FPS and VRAM continuously and
writes a VRAM CSV sample every `perf.vram_log_every_s` seconds, so the
Phase 1 success criteria (>=20 FPS @ 640x640, <2GB VRAM) can be verified
without any extra tooling.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.config import load_pipeline_config
from core.detector import Detector
from core.tracker import Tracker
from core.video_source import VideoSource, build_source
from pipeline.frame_router import FrameRouter

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# distinct colors per track id for visual debugging
_TRACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0),
]


def _color_for(tid: int) -> tuple[int, int, int]:
    return _TRACK_COLORS[tid % len(_TRACK_COLORS)]


def _draw_tracks(frame: np.ndarray, tracks) -> None:
    for t in tracks:
        x1, y1, x2, y2 = t.xyxy
        c = _color_for(t.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f"id{t.track_id} {COCO_NAMES[t.cls] if 0 <= t.cls < len(COCO_NAMES) else t.cls} {t.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, label, (x1 + 2, max(th, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _vram_mb() -> int:
    try:
        import torch
        return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        return -1


def _vram_reserved_mb() -> int:
    try:
        import torch
        return int(torch.cuda.memory_reserved() / (1024 * 1024))
    except Exception:
        return -1


def _open_vram_log(path: str) -> tuple[Any, Any] | None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["t_iso", "frame_idx", "fps", "vram_alloc_mb", "vram_reserved_mb"])
    f.flush()
    return w, f


def run(config_path: str | None = None) -> None:
    cfg = load_pipeline_config() if config_path is None else _load_extra(config_path)
    src_cfg = cfg["source"]
    disp = cfg.get("display", {})
    perf = cfg.get("perf", {})

    router = FrameRouter(cfg.get("router", {}))

    # --- models -----------------------------------------------------------
    detector = Detector()
    tracker = Tracker(detector)

    # --- source -----------------------------------------------------------
    source: VideoSource = build_source(src_cfg)
    source.open()
    if not source.isOpened():
        raise RuntimeError("Video source failed to open")

    # --- logging ----------------------------------------------------------
    vram_writer = None
    vram_file = None
    if perf.get("vram_log_path"):
        opened = _open_vram_log(perf["vram_log_path"])
        if opened:
            vram_writer, vram_file = opened
    vram_log_every = float(perf.get("vram_log_every_s", 5))
    vram_warn_gb = float(perf.get("vram_warn_above_gb", 5))
    fps_warn = float(perf.get("fps_warn_below", 20))

    win = disp.get("window", "ai-surveillance")
    resize_w = disp.get("resize_w")
    show_fps = disp.get("show_fps", True)
    show_vram = disp.get("show_vram", True)

    frame_idx = 0
    fps = 0.0
    last_t = time.perf_counter()
    fps_window_t = last_t
    fps_window_n = 0
    last_vram_log = last_t

    print(f"[phase1] router={router}  source={src_cfg.get('type')}  "
          f"imgsz={detector.imgsz} half={detector.half} device={detector.device}")

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                # file EOF with loop=False or camera gone — bail
                print("[phase1] source ended (no frame).")
                break

            # --- pipeline stages: all gated by the FrameRouter --------------
            # Phase 1: detect + track every frame. track() internally runs the
            # shared YOLO pass; we don't call detect() separately to avoid a
            # double pass (constraint: "One shared detector").
            tracks = []
            if router.should_run("track", frame_idx):
                tracks = tracker.update(frame)
            elif router.should_run("detect", frame_idx):
                tracks = [  # reuse Track shape so draw code is uniform
                    __import__("core.tracker", fromlist=["Track"]).Track(
                        -1, d.cls, d.conf, d.xyxy)
                    for d in detector.detect(frame)
                ]

            # --- display ----------------------------------------------------
            if disp.get("enabled", True):
                vis = frame
                _draw_tracks(vis, tracks)
                if resize_w and vis.shape[1] != resize_w:
                    scale = resize_w / vis.shape[1]
                    vis = cv2.resize(vis, (resize_w, int(vis.shape[0] * scale)))
                hud = []
                if show_fps:
                    hud.append(f"FPS:{fps:5.1f}")
                if show_vram:
                    hud.append(f"VRAM:{_vram_mb()}MB/{_vram_reserved_mb()}MB")
                hud.append(f"f:{frame_idx} n:{len(tracks)}")
                cv2.putText(vis, " | ".join(hud), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    print("[phase1] quit requested.")
                    break

            # --- perf bookkeeping ------------------------------------------
            now = time.perf_counter()
            fps_window_n += 1
            if now - fps_window_t >= 0.5:
                fps = fps_window_n / (now - fps_window_t)
                fps_window_t = now
                fps_window_n = 0
                if fps < fps_warn:
                    print(f"[phase1] WARN fps={fps:.1f} below target {fps_warn}")

            if vram_writer is not None and (now - last_vram_log) >= vram_log_every:
                alloc = _vram_mb()
                vram_writer.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S"), frame_idx, f"{fps:.2f}",
                    alloc, _vram_reserved_mb(),
                ])
                # csv.writer has no flush(); flush the underlying file handle so
                # a crash / SIGINT mid-run doesn't lose buffered samples.
                vram_file.flush()
                last_vram_log = now
                if alloc / 1024.0 > vram_warn_gb:
                    print(f"[phase1] WARN VRAM {alloc}MB exceeds {vram_warn_gb}GB budget")

            frame_idx += 1
    except KeyboardInterrupt:
        print("[phase1] interrupted.")
    finally:
        source.release()
        if disp.get("enabled", True):
            cv2.destroyAllWindows()
        if vram_file is not None:
            vram_file.close()
        print(f"[phase1] done. frames={frame_idx} final_fps={fps:.1f} "
              f"vram_alloc={_vram_mb()}MB reserved={_vram_reserved_mb()}MB")


def _load_extra(path: str):
    from core.config import load_yaml
    return load_yaml(path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="path to a pipeline yaml (defaults to configs/pipeline.yaml)")
    args = p.parse_args()
    run(args.config)