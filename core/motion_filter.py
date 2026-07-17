"""Motion prefilter — cheap frame differencing to skip processing on static frames.

Runs every frame (negligible cost) and gates heavier per-frame stages (pose,
reid, face) when the scene is completely static. This improves real-world FPS
without hurting accuracy since a static frame has nothing new to detect.
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionPrefilter:
    """Cheap motion detection using frame differencing."""

    def __init__(self, threshold: float = 0.01, min_changed_pixels: int = 1000):
        self.threshold = threshold
        self.min_changed_pixels = min_changed_pixels
        self._prev_gray = None

    def has_motion(self, frame: np.ndarray) -> bool:
        """Returns True if frame has significant motion compared to previous frame."""
        if frame is None or frame.size == 0:
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return True

        # Compute absolute difference
        diff = cv2.absdiff(self._prev_gray, gray)

        # Threshold the difference
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Count changed pixels
        changed_pixels = np.count_nonzero(thresh)

        # Update previous frame
        self._prev_gray = gray

        # Return True if motion detected
        return changed_pixels >= self.min_changed_pixels

    def reset(self) -> None:
        """Reset the prefilter state."""
        self._prev_gray = None
