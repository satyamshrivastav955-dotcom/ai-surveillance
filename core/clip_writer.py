"""ClipWriter — Phase 5D.

Maintains a rolling ring-buffer of recent video frames (``buffer_s`` seconds).
When a fight/violence event fires, call ``flush(event)`` to save the buffered
frames as an .mp4 clip to ``clips_dir``.

The saved filename is returned as a string and attached to the FightEvent's
``clip_ref`` field before it enters the event buffer.

Design constraints
------------------
- VRAM budget: ring buffer is CPU-side numpy arrays — no GPU memory used.
- Disk usage: 5s at 720p30 ~= 6 MB at OpenCV default codec.
- The buffer always runs at source fps; even when FrameRouter skips frames,
  the ClipWriter receives every raw frame (it's called in the outermost loop).

Usage::

    writer = ClipWriter(cfg)

    # inside frame loop, always:
    writer.push(frame)

    # when a FightEvent fires:
    clip_path = writer.flush(event)
    event.clip_ref = clip_path
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ClipWriter:
    """Rolling frame buffer + on-demand .mp4 flush for fight events.

    Configuration (``clip_writer`` block in pipeline.yaml):
      clips_dir   : output directory (default "data/clips")
      buffer_s    : seconds of video to keep in the ring buffer (default 5.0)
      fps         : frames-per-second of source (default 30.0)
      fourcc      : OpenCV FourCC codec string (default "mp4v")
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self._clips_dir = Path(cfg.get("clips_dir", "data/clips"))
        self._clips_dir.mkdir(parents=True, exist_ok=True)
        self._fps    = float(cfg.get("fps",       30.0))
        self._buf_s  = float(cfg.get("buffer_s",  5.0))
        self._fourcc = cfg.get("fourcc", "mp4v")
        maxlen = int(self._fps * self._buf_s)
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)
        self._frame_size: tuple[int, int] | None = None   # (width, height)

    def push(self, frame: np.ndarray) -> None:
        """Add a frame to the ring buffer. Call every frame."""
        if self._frame_size is None:
            h, w = frame.shape[:2]
            self._frame_size = (w, h)
        self._buf.append(frame.copy())

    def flush(self, label: str = "fight") -> str | None:
        """Save current ring-buffer contents as an .mp4 clip.

        Parameters
        ----------
        label : short prefix for the filename (e.g. "fight", "violence")

        Returns
        -------
        str path to the written clip, or None if the buffer is empty.
        """
        if not self._buf or self._frame_size is None:
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = self._clips_dir / f"clip_{label}_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
        writer = cv2.VideoWriter(
            str(filename), fourcc, self._fps, self._frame_size
        )
        if not writer.isOpened():
            return None

        for frm in list(self._buf):
            writer.write(frm)
        writer.release()
        return str(filename)

    def flush_for_event(self, event) -> str | None:
        """Convenience: flush and set event.clip_ref in one call.

        Returns the clip path (also stored on event).
        """
        path = self.flush(label=getattr(event, "event_type", "event").lower())
        if hasattr(event, "clip_ref"):
            event.clip_ref = path
        return path
