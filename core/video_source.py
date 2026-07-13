"""Video input abstraction (Phase 1, Section 5).

All sources expose the same `read() -> (ret, frame)` interface so every
downstream module is source-agnostic. `RTSPSource` is a working stub — it
plugs into a real IP camera later via a config change, not a refactor.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import cv2


class VideoSource(ABC):
    """Common interface for every camera/video source."""

    def __init__(self, width: int | None = None, height: int | None = None):
        self.cap: cv2.VideoCapture | None = None
        self.width = width
        self.height = height

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> tuple[bool, Any]:
        """Return (ok, frame). `ok` is False on EOF / camera failure."""

    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _apply_resolution(self) -> None:
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    # context-manager sugar so `with WebcamSource() as src:` is valid
    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class WebcamSource(VideoSource):
    """Laptop webcam — honest live read on real-time performance."""

    def __init__(self, index: int = 0, width: int | None = 1280, height: int | None = 720):
        super().__init__(width, height)
        self.index = index

    def open(self) -> None:
        # CAP_DSHOW avoids the slow MSMF warmup on Windows
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self._apply_resolution()
        if not self.isOpened():
            raise RuntimeError(f"Could not open webcam index {self.index}")

    def read(self) -> tuple[bool, Any]:
        return self.cap.read()


class FileSource(VideoSource):
    """Recorded video — repeatable accuracy tests; loops by default."""

    def __init__(self, path: str, loop: bool = True, width: int | None = None, height: int | None = None):
        super().__init__(width, height)
        self.path = path
        self.loop = loop

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.path)
        if not self.isOpened():
            raise FileNotFoundError(f"Could not open video file: {self.path}")
        self._apply_resolution()

    def read(self) -> tuple[bool, Any]:
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame


class SyntheticSource(VideoSource):
    """In-process deterministic frame generator — no camera / lighting dependency.

    Produces a 'person-like' tall rectangle that moves horizontally across the
    frame so YOLO actually has something to detect + ByteTrack has a stable
    track ID to maintain. Respects a configurable target FPS by rate-limiting
    `read()` calls (sleeps if the consumer is faster than the target).

    Width/height default to 720p to match the webcam default; imgsz for
    inference is independent (detector downscales internally).
    """

    def __init__(self, fps: float = 30.0, width: int = 1280, height: int = 720,
                 n_persons: int = 1, seed: int = 0):
        super().__init__(width, height)
        self.target_fps = float(fps)
        self.n_persons = n_persons
        self._frame_period_s = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._frame_idx = 0
        self._next_release_at: float | None = None
        self._rng_state = seed  # deterministic motion; we don't import numpy here
        self._opened = False

    def open(self) -> None:
        # nothing to allocate; just mark ready and prime the rate-limiter clock
        self._next_release_at = time.perf_counter()
        self._opened = True

    def isOpened(self) -> bool:  # override — no cv2.VideoCapture involved
        return self._opened

    def release(self) -> None:
        self._opened = False

    def read(self) -> tuple[bool, Any]:
        import numpy as _np  # lazy import to keep core.video_source importable without numpy at module load

        if not self._opened:
            return False, None

        # rate-limit: sleep until the next frame's scheduled release time.
        # If the consumer is slower than the target FPS, we release immediately
        # (no sleep) — so throughput is min(target_fps, consumer_fps).
        now = time.perf_counter()
        if self._next_release_at is not None and now < self._next_release_at:
            time.sleep(self._next_release_at - now)
            self._next_release_at += self._frame_period_s
        else:
            # we're behind (or first frame); resync the schedule to 'now'
            self._next_release_at = now + self._frame_period_s

        w = self.width or 1280
        h = self.height or 720
        # plausible kitchen / office ambient background; static gradient
        frame = _np.zeros((h, w, 3), dtype=_np.uint8)
        # subtle horizontal gradient (BGR) so the BG isn't pure black
        for c in range(3):
            frame[:, :, c] = _np.linspace(20 + c * 5, 40 + c * 5, w, dtype=_np.uint8)[None, :]

        # draw n person-shaped tall rectangles moving across the frame
        for k in range(self.n_persons):
            # deterministic pseudo-motion: each person has its own phase + speed
            speed = 0.03 + 0.012 * k          # px/frame horizontal speed
            phase = (k * 211) % w             # offset each person
            x_center = int((phase + self._frame_idx * speed) % w)
            # person dimensions: aspect ratio ~ 1:2.4 (taller than wide)
            pw = max(40, w // 18)
            ph = max(120, h // 3)
            x1 = max(0, x_center - pw // 2)
            y1 = max(0, int(h * 0.25))
            x2 = min(w - 1, x1 + pw)
            y2 = min(h - 1, y1 + ph)
            # warm-ish color, varying slightly per person so YOLO has texture
            color = (40 + k * 11, 90 + k * 5, 180 - k * 11)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            # "head": a smaller square on top
            head_h = max(20, ph // 6)
            cv2.rectangle(frame, (x1 + 6, y1 - head_h), (x2 - 6, y1),
                          (int(color[0] * 1.2), int(color[1] * 1.1), color[2]), -1)
            # a few darker internal bands so it reads as a person to YOLO
            for band in range(3):
                by = y1 + (band + 1) * ph // 4
                cv2.line(frame, (x1 + 4, by), (x2 - 4, by),
                         (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5)), 2)

        self._frame_idx += 1
        return True, frame


class RTSPSource(VideoSource):
    """IP camera stub. Untested on real hardware in Phase 1 but ready to use.

    Uses a TCP transport + low-latency buffer to reduce frame jitter. Set
    `source.type: rtsp` and `source.path: rtsp://user:pass@ip/stream` in
    pipeline.yaml to enable.
    """

    def __init__(self, url: str, reconnect: bool = True, width: int | None = None, height: int | None = None):
        super().__init__(width, height)
        self.url = url
        self.reconnect = reconnect
        self._backoff = 0.5

    def open(self) -> None:
        self._open_internal()

    def _open_internal(self) -> None:
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.isOpened():
            raise RuntimeError(f"Could not open RTSP stream: {self.url}")

    def read(self) -> tuple[bool, Any]:
        ok, frame = self.cap.read()
        if not ok and self.reconnect:
            time.sleep(self._backoff)
            self.release()
            try:
                self._open_internal()
                ok, frame = self.cap.read()
            except RuntimeError:
                ok = False
        return ok, frame


def build_source(source_cfg: dict[str, Any]) -> VideoSource:
    """Factory: pick the source implementation from pipeline config."""
    kind = source_cfg.get("type", "webcam").lower()
    path = source_cfg.get("path")
    width = source_cfg.get("width")
    height = source_cfg.get("height")

    if kind == "webcam":
        return WebcamSource(index=int(path) if path is not None else 0, width=width, height=height)
    if kind == "file":
        if not path:
            raise ValueError("source.type=file requires source.path")
        return FileSource(path, loop=source_cfg.get("loop", True), width=width, height=height)
    if kind == "rtsp":
        if not path:
            raise ValueError("source.type=rtsp requires source.path")
        return RTSPSource(path, reconnect=source_cfg.get("reconnect", True), width=width, height=height)
    if kind == "synthetic":
        fps = float(source_cfg.get("fps", 30.0))
        return SyntheticSource(fps=fps, width=width or 1280, height=height or 720)
    raise ValueError(f"Unknown source type: {kind}")