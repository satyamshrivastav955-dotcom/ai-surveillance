"""Escalation Handler - Manages frame escalation from detector signals.

This module implements the escalation logic for Steps 1 and 5 of the VLM spec:

1. Detector as Priority Signal (not gate):
   - High confidence → escalate to full fidelity
   - Ambiguous → escalate for multi-frame verification
   - Forced interval → escalate regardless of detector
   
2. Multi-frame reasoning for ambiguous cases:
   - Joint analysis over a window of frames
   - Resolves occlusion/dust/smoke cases

The key innovation is that the VLM's own hot-KV attention can ALSO flag escalation,
providing a second independent path beyond the detector.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlm.config_paths import DEFAULT_VLM_CONFIG

from vlm.core import VLMOutput, VLMManager


@dataclass
class EscalationConfig:
    ambiguous_threshold_low: float = 0.3
    ambiguous_threshold_high: float = 0.7
    
    multi_frame_window_s: float = 1.5
    
    forced_full_fidelity_interval: int = 100
    
    detector_escalation_weight: float = 0.8
    ambient_attention_flag_weight: float = 0.5
    
    cooldown_frames: int = 30

    # Frame rate used to size every seconds->frames window below. Sourced from
    # pipeline.yaml `source.fps`; never assumed locally.
    fps: float = 0.0

    # Seconds of context to send on a forced full-fidelity escalation.
    full_context_window_s: float = 1.5

    @classmethod
    def from_yaml(cls, path: str = DEFAULT_VLM_CONFIG) -> "EscalationConfig":
        from core.config import load_fps
        from vlm.config_paths import load_vlm_config

        cfg = load_vlm_config(path)

        esc_cfg = cfg.get("escalation", {})

        return cls(
            ambiguous_threshold_low=esc_cfg.get("ambiguous_threshold_low", 0.3),
            ambiguous_threshold_high=esc_cfg.get("ambiguous_threshold_high", 0.7),
            multi_frame_window_s=esc_cfg.get("multi_frame_window", 1.5),
            forced_full_fidelity_interval=esc_cfg.get("forced_full_fidelity_interval", 100),
            detector_escalation_weight=esc_cfg.get("detector_escalation_weight", 0.8),
            ambient_attention_flag_weight=float(esc_cfg.get("ambient_attention_flag_weight", 0.5)),
            cooldown_frames=esc_cfg.get("cooldown_frames", 30),
            fps=load_fps(),
            full_context_window_s=esc_cfg.get("full_context_window_s", 1.5),
        )


@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: str
    
    priority_score: float
    
    frame_idx: int
    timestamp: float
    
    multi_frame_window: list[int]
    
    escalate_event_dtype: str = ""


@dataclass
class EscalationResult:
    decision: EscalationDecision
    
    vlm_output: VLMOutput
    
    frames_processed: int
    latency_seconds: float


class EscalationHandler:
    """Manages the escalation logic for VLM processing.
    
    The detector is used as a PRIORITY SIGNAL, not a gate:
    - Frames with high detector confidence get escalated
    - Frames with ambiguous confidence get escalated for multi-frame analysis
    - Forced interval frames get escalated regardless of detector
    - The VLM's own attention can trigger escalation
    
    This ensures the VLM always has visibility into all frames (ambient pass)
    while focusing expensive compute where it matters most.
    """
    
    EVENT_TYPE_PRIORITIES = {
        "FALL": 1.0,
        "FIRE": 0.95,
        "SMOKE": 0.90,
        "FIGHT": 0.85,
        "VIOLENCE": 0.80,
        "SMOKING": 0.60,
        "PHONE": 0.55,
        "GATHERING": 0.50,
        "OBJECT_LEFT": 0.40,
        "IDENTITY": 0.30,
    }
    
    def __init__(self, config: EscalationConfig | None = None):
        self.config = config or EscalationConfig.from_yaml()

        if not self.config.fps:
            raise ValueError(
                "[Escalation] EscalationConfig.fps is unset. The frame buffer and "
                "context windows are sized in seconds and cannot be converted to "
                "frames without the real capture rate. Build the config via "
                "EscalationConfig.from_yaml() (which reads pipeline.yaml source.fps) "
                "or set fps explicitly. Refusing to assume 30 fps."
            )
        self.fps = float(self.config.fps)

        self._frame_buffer: deque[tuple[np.ndarray, int, float, list]] = deque(
            maxlen=int(self.config.multi_frame_window_s * self.fps) + 10
        )

        self._last_escalation_frame = -self.config.cooldown_frames - 10
        self._escalation_count = 0
        
        self._output_history: deque[VLMOutput] = deque(maxlen=100)
    
    def check_escalation(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detector_events: list,
        vlm_ambient_output: VLMOutput | None = None,
    ) -> EscalationDecision:
        timestamp = time.perf_counter()
        
        in_cooldown = frame_idx - self._last_escalation_frame < self.config.cooldown_frames
        
        detector_priority = self._compute_detector_priority(detector_events)
        
        attention_flag = 0.0
        if vlm_ambient_output is not None:
            attention_flag = vlm_ambient_output.attention_flag
        
        store_this_frame = vlm_ambient_output is not None and vlm_ambient_output.attention_flag > 0.3
        
        self._frame_buffer.append((frame.copy() if store_this_frame else None, frame_idx, timestamp, detector_events))
        
        detector_priority = self._compute_detector_priority(detector_events)
        
        if detector_priority >= self.config.ambiguous_threshold_high:
            if in_cooldown:
                return self._no_escalation(frame_idx, timestamp, "in_cooldown_high_conf")
            
            window = self._get_multi_frame_window(frame_idx)
            self._last_escalation_frame = frame_idx
            self._escalation_count += 1
            
            return EscalationDecision(
                should_escalate=True,
                reason="high_detector_confidence",
                priority_score=detector_priority,
                frame_idx=frame_idx,
                timestamp=timestamp,
                multi_frame_window=window,
            )
        
        if self.config.ambiguous_threshold_low < detector_priority < self.config.ambiguous_threshold_high:
            if in_cooldown:
                return self._no_escalation(frame_idx, timestamp, "in_cooldown_ambiguous")
            
            window = self._get_multi_frame_window(frame_idx, full_context=True)
            self._last_escalation_frame = frame_idx
            self._escalation_count += 1
            
            return EscalationDecision(
                should_escalate=True,
                reason="ambiguous_needs_verification",
                priority_score=detector_priority,
                frame_idx=frame_idx,
                timestamp=timestamp,
                multi_frame_window=window,
            )
        
        if attention_flag > 0.5:
            if in_cooldown:
                return self._no_escalation(frame_idx, timestamp, "in_cooldown_attention")
            
            window = self._get_multi_frame_window(frame_idx)
            self._last_escalation_frame = frame_idx
            self._escalation_count += 1
            
            return EscalationDecision(
                should_escalate=True,
                reason="vlm_attention_flag",
                priority_score=attention_flag,
                frame_idx=frame_idx,
                timestamp=timestamp,
                multi_frame_window=window,
            )
        
        if frame_idx % self.config.forced_full_fidelity_interval == 0 and not in_cooldown:
            window = self._get_multi_frame_window(frame_idx)
            self._last_escalation_frame = frame_idx
            self._escalation_count += 1
            
            return EscalationDecision(
                should_escalate=True,
                reason="forced_interval",
                priority_score=1.0,
                frame_idx=frame_idx,
                timestamp=timestamp,
                multi_frame_window=window,
            )
        
        return self._no_escalation(frame_idx, timestamp, "")
    
    def _no_escalation(self, frame_idx: int, timestamp: float, reason: str) -> EscalationDecision:
        return EscalationDecision(
            should_escalate=False,
            reason=reason,
            priority_score=0.0,
            frame_idx=frame_idx,
            timestamp=timestamp,
            multi_frame_window=[],
        )
    
    def _compute_detector_priority(self, detector_events: list) -> float:
        if not detector_events:
            return 0.0
        
        max_priority = 0.0
        
        for event in detector_events:
            dtype = getattr(event, 'event_type', '')
            if dtype in self.EVENT_TYPE_PRIORITIES:
                if hasattr(event, 'confidence'):
                    event_priority = event.confidence
                elif hasattr(event, 'details') and isinstance(event.details, dict):
                    event_priority = event.details.get('confidence', 0.5)
                else:
                    event_priority = 0.5
                type_weight = self.EVENT_TYPE_PRIORITIES[dtype]
                max_priority = max(max_priority, event_priority * type_weight)
        
        return max_priority
    
    def _get_multi_frame_window(self, frame_idx: int, full_context: bool = False) -> list[int]:
        # Both windows are expressed in seconds and converted with the real
        # capture rate. `-45:` and `* 15` were 1.5s and half-rate at an assumed
        # 30 fps; at any other rate they covered the wrong span of time.
        window_s = (self.config.full_context_window_s if full_context
                    else self.config.multi_frame_window_s)
        n_frames = max(1, int(round(window_s * self.fps)))
        return [fid for _, fid, _, _ in list(self._frame_buffer)[-n_frames:]]
    
    def get_escalation_context(self, window: list[int]) -> list[tuple[np.ndarray, int]]:
        context = []
        for stored_frame, fid, _, events in list(self._frame_buffer):
            if fid in window and stored_frame is not None:
                context.append((stored_frame, fid))
        return context
    
    def record_output(self, output: VLMOutput) -> None:
        self._output_history.append(output)
    
    def get_stats(self) -> dict:
        return {
            "total_escalations": self._escalation_count,
            "last_escalation_frame": self._last_escalation_frame,
            "buffer_frames": len(self._frame_buffer),
            "output_history": len(self._output_history),
        }
    
    def reset(self) -> None:
        self._frame_buffer.clear()
        self._output_history.clear()
        self._last_escalation_frame = -self.config.cooldown_frames - 10
        self._escalation_count = 0


class AlertVerifier:
    """Verifies alerts using multi-frame VLM analysis.
    
    When an event is flagged, this class uses the VLM to verify whether
    it's a true positive or false alarm.
    """
    
    VERIFICATION_PROMPT_TEMPLATE = """
Analyze this frame sequence for a potential {event_type} event.
Detector confidence: {confidence}
Frame range: {frame_range}

Key questions:
1. Is there clear visual evidence of {event_type}?
2. Are there any alternative explanations?
3. Should this alert be confirmed, rejected, or marked ambiguous?

Assessment:
"""
    
    def __init__(self, vlm_manager: VLMManager, query_engine: Any):
        self.vlm_manager = vlm_manager
        self.query_engine = query_engine
    
    def verify(
        self,
        event_type: str,
        confidence: float,
        frames: list[np.ndarray],
        frame_indices: list[int],
    ) -> tuple[str, str, float]:
        if not frames:
            return "AMBIGUOUS", "No frames available for verification", 0.5
        
        vlm_output = self.vlm_manager.process_frame(
            frames[-1],
            frame_indices[-1] if frame_indices else 0,
            detector_priority_score=confidence,
        )
        
        reasoning = f"VLM analysis of {len(frames)} frames"
        
        if vlm_output.attention_flag > 0.7:
            status = "CONFIRMED"
            verified_confidence = min(1.0, confidence + 0.1)
        elif vlm_output.attention_flag > 0.4:
            status = "AMBIGUOUS"
            verified_confidence = confidence
        else:
            status = "REJECTED"
            verified_confidence = max(0.0, confidence - 0.1)
        
        return status, reasoning, verified_confidence
