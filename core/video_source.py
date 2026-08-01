"""Video input abstraction.

All sources expose the same `read() -> (ret, frame)` interface so every
downstream module is source-agnostic.
"""
from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2


def _get_preferred_backend() -> int:
    """Get platform-appropriate video backend."""
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    elif sys.platform == "linux":
        return cv2.CAP_V4L2
    elif sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


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

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class WebcamSource(VideoSource):
    """Laptop webcam with retry logic."""

    def __init__(self, index: int = 0, width: int | None = 1280, height: int | None = 720, max_retries: int = 3):
        super().__init__(width, height)
        self.index = index
        self.max_retries = max_retries
        self._consecutive_failures = 0

    def open(self) -> None:
        backend = _get_preferred_backend()
        self.cap = cv2.VideoCapture(self.index, backend)
        self._apply_resolution()
        if not self.isOpened():
            raise RuntimeError(f"Could not open webcam index {self.index}")

    def read(self) -> tuple[bool, Any]:
        ok, frame = self.cap.read()
        
        if not ok:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.max_retries:
                return False, None
            return False, None
        
        self._consecutive_failures = 0
        return True, frame


class FileSource(VideoSource):
    """Recorded video with loop support and corrupt frame handling."""

    def __init__(self, path: str, loop: bool = True, width: int | None = None, height: int | None = None):
        super().__init__(width, height)
        self.path = Path(path)
        self.loop = loop
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10

    def open(self) -> None:
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.isOpened():
            raise FileNotFoundError(f"Could not open video file: {self.path}")
        self._apply_resolution()

    def read(self) -> tuple[bool, Any]:
        ok, frame = self.cap.read()
        
        if not ok:
            self._consecutive_failures += 1
            
            if self.loop and self._consecutive_failures < self._max_consecutive_failures:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if ok:
                    self._consecutive_failures = 0
            
            if self._consecutive_failures >= self._max_consecutive_failures:
                return False, None
        
        if ok:
            self._consecutive_failures = 0
            
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
    """IP camera with exponential backoff reconnection."""

    def __init__(
        self,
        url: str,
        reconnect: bool = True,
        width: int | None = None,
        height: int | None = None,
        max_retries: int = 10,
        initial_backoff: float = 0.5,
        max_backoff: float = 30.0,
    ):
        super().__init__(width, height)
        self.url = url
        self.reconnect = reconnect
        self.max_retries = max_retries
        self._backoff = initial_backoff
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._retry_count = 0

    def open(self) -> None:
        self._open_internal()

    def _open_internal(self) -> None:
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.isOpened():
            raise RuntimeError(f"Could not open RTSP stream: {self.url}")
        self._retry_count = 0
        self._backoff = self._initial_backoff

    def read(self) -> tuple[bool, Any]:
        if self.cap is None:
            return False, None
            
        ok, frame = self.cap.read()
        
        if not ok:
            frame = None
            if self.reconnect:
                self._retry_count += 1
                if self._retry_count <= self.max_retries:
                    time.sleep(self._backoff)
                    self._backoff = min(self._backoff * 2, self._max_backoff)
                    self.release()
                    try:
                        self._open_internal()
                        ok, frame = self.cap.read()
                    except (RuntimeError, Exception):
                        ok = False
                        frame = None
        
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
        if "fps" not in source_cfg:
            raise KeyError(
                "[video_source] source.fps is required for a synthetic source. "
                "Refusing to default to 30 fps, which would make the synthetic "
                "stream run at a different rate than every time-based window in "
                "the pipeline assumes."
            )
        fps = float(source_cfg["fps"])
        return SyntheticSource(fps=fps, width=width or 1280, height=height or 720)
    raise ValueError(f"Unknown source type: {kind}")