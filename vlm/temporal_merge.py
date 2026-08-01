"""Temporal Frame Merger - Delta-based frame merging with escalation awareness.

This module implements Step 3 of the VLM spec: merging adjacent frames with minimal
visual delta into a single representation.

CRITICAL SAFETY: Frame merging is gated by escalation status, not just visual similarity.
If either frame was escalated by the detector (high priority) or forced interval,
the merge is SKIPPED. This prevents erasing motionless persons from merged frames.

Rules:
1. Compute cheap perceptual delta between consecutive frames
2. Merge only if: (a) delta below threshold AND (b) NEITHER frame was escalated
3. Escalated frames are ALWAYS processed separately
4. Forced full-fidelity interval frames bypass merging entirely

This ensures the "person might be here" constraint is honored in temporal compression.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlm.config_paths import DEFAULT_VLM_CONFIG

try:
    import cv2
    IMPORTED_CV2 = True
except ImportError:
    IMPORTED_CV2 = False
    cv2 = None


@dataclass
class TemporalConfig:
    enabled: bool = True
    
    merge_threshold: float = 0.05
    
    max_merge_span_frames: int = 5
    
    skip_escalated_frames: bool = True
    
    delta_method: str = "perceptual"
    
    hash_size: int = 8
    
    histogram_bins: int = 64
    
    @classmethod
    def from_yaml(cls, path: str = DEFAULT_VLM_CONFIG) -> "TemporalConfig":
        from vlm.config_paths import load_vlm_config

        cfg = load_vlm_config(path)
        
        temporal_cfg = cfg.get("temporal", {})
        
        return cls(
            enabled=temporal_cfg.get("enabled", True),
            merge_threshold=temporal_cfg.get("merge_threshold", 0.05),
            max_merge_span_frames=temporal_cfg.get("max_merge_span_frames", 5),
            skip_escalated_frames=temporal_cfg.get("skip_escalated_frames", True),
            delta_method=temporal_cfg.get("delta_method", "perceptual"),
        )


@dataclass
class MergedFrame:
    """Result of temporal frame merging.
    
    If frames were merged, represents the combined result.
    If merge was skipped (escalation), represents a single frame.
    """
    frame_indices: list[int]
    frames: list[np.ndarray]
    
    merged_count: int
    
    representative_frame_idx: int
    representative_frame: np.ndarray
    
    was_merged: bool
    skip_reason: str = ""
    
    timestamp: float = 0.0
    
    any_escalated: bool = False
    max_priority_score: float = 0.0


class TemporalMerger:
    """Temporal frame merger with escalation awareness.
    
    Key safety feature: NEVER merges escalated frames, ensuring that
    high-priority detections are processed at full temporal resolution.
    """
    
    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig.from_yaml()
        
        self._frame_buffer: list[tuple[np.ndarray, int, bool, float]] = []
        self._merge_count = 0
        self._skip_count = 0
        self._last_delta = 0.0
    
    def push(
        self,
        frame: np.ndarray,
        frame_idx: int,
        is_escalated: bool = False,
        priority_score: float = 0.0,
    ) -> tuple[bool, MergedFrame | None]:
        if not self.config.enabled:
            return False, None
        
        if is_escalated and self.config.skip_escalated_frames:
            result = MergedFrame(
                frame_indices=[frame_idx],
                frames=[frame.copy()],
                merged_count=1,
                representative_frame_idx=frame_idx,
                representative_frame=frame.copy(),
                was_merged=False,
                skip_reason="escalated_frame",
                timestamp=time.perf_counter(),
                any_escalated=True,
                max_priority_score=priority_score,
            )
            self._skip_count += 1
            return True, result
        
        self._frame_buffer.append((frame.copy(), frame_idx, is_escalated, priority_score))
        
        if len(self._frame_buffer) < 2:
            return False, None
        
        return self._try_merge()
    
    def _try_merge(self) -> tuple[bool, MergedFrame | None]:
        if len(self._frame_buffer) < 2:
            return False, None
        
        recent = self._frame_buffer[-2:]
        
        frame1, idx1, esc1, pri1 = recent[0]
        frame2, idx2, esc2, pri2 = recent[1]
        
        if (esc1 or esc2) and self.config.skip_escalated_frames:
            result = MergedFrame(
                frame_indices=[idx1],
                frames=[frame1.copy()],
                merged_count=1,
                representative_frame_idx=idx1,
                representative_frame=frame1.copy(),
                was_merged=False,
                skip_reason="escalation_in_window",
                any_escalated=True,
                max_priority_score=max(pri1, pri2),
            )
            self._frame_buffer.pop(0)
            return True, result
        
        delta = self._compute_delta(frame1, frame2)
        self._last_delta = delta
        
        if delta < self.config.merge_threshold:
            if len(self._frame_buffer) >= self.config.max_merge_span_frames:
                return self._finalize_merge()
            
            return False, None
        else:
            return self._finalize_merge()
    
    def _finalize_merge(self) -> tuple[bool, MergedFrame]:
        if len(self._frame_buffer) == 0:
            return False, None
        
        frames = [f for f, i, e, p in self._frame_buffer]
        indices = [i for f, i, e, p in self._frame_buffer]
        escalated = any(e for f, i, e, p in self._frame_buffer)
        max_pri = max(p for f, i, e, p in self._frame_buffer) if self._frame_buffer else 0.0
        
        result = MergedFrame(
            frame_indices=indices,
            frames=frames,
            merged_count=len(frames),
            representative_frame_idx=indices[-1],
            representative_frame=frames[-1].copy() if frames else None,
            was_merged=len(frames) > 1,
            skip_reason="",
            timestamp=time.perf_counter(),
            any_escalated=escalated,
            max_priority_score=max_pri,
        )
        
        self._frame_buffer.clear()
        self._frame_buffer.append((frames[-1].copy() if frames else None, indices[-1], escalated, max_pri))
        
        if len(frames) > 1:
            self._merge_count += 1
        
        return True, result
    
    def _compute_delta(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        if frame1 is None or frame2 is None:
            return 1.0
        
        if not IMPORTED_CV2:
            return self._compute_simple_delta(frame1, frame2)
        
        if self.config.delta_method == "perceptual":
            return self._compute_perceptual_delta(frame1, frame2)
        elif self.config.delta_method == "histogram":
            return self._compute_histogram_delta(frame1, frame2)
        elif self.config.delta_method == "hash":
            return self._compute_hash_delta(frame1, frame2)
        else:
            return self._compute_perceptual_delta(frame1, frame2)
    
    def _compute_perceptual_delta(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        if frame1 is None or frame2 is None:
            return 1.0
        
        if frame1.shape != frame2.shape:
            return 1.0
        
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
            gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
            
            diff = cv2.absdiff(gray1, gray2)
            
            mean_diff = np.mean(diff) / 255.0
            
            return float(mean_diff)
        except Exception:
            return self._compute_simple_delta(frame1, frame2)
    
    def _compute_histogram_delta(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        if not IMPORTED_CV2:
            return self._compute_simple_delta(frame1, frame2)
        
        try:
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, 
                                 [self.config.histogram_bins, self.config.histogram_bins, self.config.histogram_bins],
                                 [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None,
                                 [self.config.histogram_bins, self.config.histogram_bins, self.config.histogram_bins],
                                 [0, 256, 0, 256, 0, 256])
            
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            return 1.0 - correlation
        except Exception:
            return self._compute_simple_delta(frame1, frame2)
    
    def _compute_hash_delta(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        if not IMPORTED_CV2:
            return self._compute_simple_delta(frame1, frame2)
        
        try:
            size = self.config.hash_size
            
            resized1 = cv2.resize(frame1, (size + 1, size))
            resized2 = cv2.resize(frame2, (size + 1, size))
            
            gray1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
            
            hash1 = gray1[:, 1:] > gray1[:, :-1]
            hash2 = gray2[:, 1:] > gray2[:, :-1]
            
            hamming = np.sum(hash1 != hash2)
            max_bits = size * size
            
            return float(hamming) / max_bits
        except Exception:
            return self._compute_simple_delta(frame1, frame2)
    
    def _compute_simple_delta(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        if frame1 is None or frame2 is None:
            return 1.0
        
        if frame1.shape != frame2.shape:
            return 1.0
        
        try:
            diff = np.abs(frame1.astype(float) - frame2.astype(float))
            return float(np.mean(diff) / 255.0)
        except Exception:
            return 1.0
    
    def flush(self) -> tuple[bool, MergedFrame | None]:
        while len(self._frame_buffer) > 1:
            return self._finalize_merge()
        
        if len(self._frame_buffer) == 1:
            frame, idx, esc, pri = self._frame_buffer[0]
            result = MergedFrame(
                frame_indices=[idx],
                frames=[frame],
                merged_count=1,
                representative_frame_idx=idx,
                representative_frame=frame,
                was_merged=False,
                skip_reason="",
                any_escalated=esc,
                max_priority_score=pri,
            )
            self._frame_buffer.clear()
            return True, result
        
        return False, None
    
    def get_stats(self) -> dict:
        return {
            "merge_count": self._merge_count,
            "skip_count": self._skip_count,
            "last_delta": self._last_delta,
            "buffer_size": len(self._frame_buffer),
        }
    
    def reset(self) -> None:
        self._frame_buffer.clear()
        self._merge_count = 0
        self._skip_count = 0
        self._last_delta = 0.0


class FrameDeltaComputer:
    """Utility class for computing frame deltas with various methods."""
    
    @staticmethod
    def compute_motion_pixels(frame1: np.ndarray, frame2: np.ndarray, threshold: int = 25) -> int:
        if not IMPORTED_CV2:
            return 0
        
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            return int(cv2.countNonZero(thresh))
        except Exception:
            return 0
    
    @staticmethod
    def compute_optical_flow_magnitude(frame1: np.ndarray, frame2: np.ndarray) -> float:
        if not IMPORTED_CV2:
            return 0.0
        
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
            gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
            
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            return float(np.mean(magnitude))
        except Exception:
            return 0.0
