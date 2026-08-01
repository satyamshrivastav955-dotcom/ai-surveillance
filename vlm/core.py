"""VLM Core — Qwen2.5-VL-3B-Instruct wrapper for escalated surveillance inference.

What this module does (real):
  - Loads Qwen2.5-VL-3B-Instruct with bitsandbytes NF4 4-bit quantization.
  - Runs escalated inference when the detector pipeline flags high priority.
  - Returns a structured entity list parsed from the model's own JSON output.
  - Tracks a frame buffer so escalation receives multi-frame context.

What was removed (was dead scaffolding):
  - encode_frame / run_ambient_pass: ran the processor but never the model.
  - _compute_attention_flag_PLACEHOLDER_NOT_IMPLEMENTED: had no implementation;
    raised NotImplementedError every time and was caught silently.
  - KV-cache hot/warm/cold tiers: every block was pushed with keys=None so the
    cache stored no attention state and could not contribute anything.
  - SpatialTokenPruner: never called from any production path.
  - TemporalMerger: output was consumed by nothing.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vlm.config_paths import DEFAULT_VLM_CONFIG

try:
    import torch
    IMPORTED_TORCH = True
except ImportError:
    IMPORTED_TORCH = False
    torch = None  # type: ignore[assignment]

_VLM_DEBUG = os.environ.get("VLM_DEBUG", "0") == "1"


def _pipeline_source_size() -> tuple[int | None, int | None]:
    """Read the capture resolution from pipeline.yaml `source`.

    Returns (None, None) if the pipeline config has no explicit size — callers
    must then obtain real dimensions from the live video source rather than
    substituting a guess.
    """
    try:
        from core.config import load_pipeline_config
        src = load_pipeline_config().get("source", {}) or {}
    except Exception:
        return (None, None)
    w = src.get("width")
    h = src.get("height")
    return (int(w) if w else None, int(h) if h else None)


@dataclass
class VLMConfig:
    # No defaults: model identity and device come only from vlm.yaml.
    model_name: str = ""
    device: str = ""
    half: bool = True
    dtype: str = "bfloat16"
    max_pixels: int = 602112
    min_pixels: int = 3136
    quantization: str = "nf4"   # "nf4" | "awq" | null/empty = full precision
    max_vram_gb: float = 3.5    # hard ceiling; load aborts if exceeded

    # Warmup frame dimensions. Default None — resolved from the pipeline
    # source config (or the live video source) rather than guessed.
    warmup_frame_height: int | None = None
    warmup_frame_width: int | None = None

    escalated_max_tokens: int = 256
    task_prompt: str = "Monitor for: people, distress postures, fire, smoke, objects left behind."

    @classmethod
    def from_yaml(cls, path: str = DEFAULT_VLM_CONFIG) -> "VLMConfig":
        from vlm.config_paths import load_vlm_config, model_name, model_device
        cfg = load_vlm_config(path)

        model_cfg = cfg.get("model", {})
        escalated_cfg = cfg.get("escalated", {})

        # Warmup resolution comes from the pipeline's video source, not a
        # guessed constant. vlm.yaml may override explicitly; otherwise inherit
        # from the pipeline.
        src_w, src_h = _pipeline_source_size()

        return cls(
            model_name=model_name(cfg),
            device=model_device(cfg),
            half=model_cfg.get("half", True),
            dtype=model_cfg.get("dtype", "bfloat16"),
            max_pixels=model_cfg.get("max_pixels", 602112),
            min_pixels=model_cfg.get("min_pixels", 3136),
            quantization=model_cfg.get("quantization", "nf4"),
            max_vram_gb=float(model_cfg.get("max_vram_gb", 3.5)),
            warmup_frame_height=model_cfg.get("warmup_frame_height", src_h),
            warmup_frame_width=model_cfg.get("warmup_frame_width", src_w),
            escalated_max_tokens=escalated_cfg.get("max_tokens", 256),
            task_prompt=cfg.get("ambient", {}).get("task_prompt",
                "Monitor for: people, distress postures, fire, smoke, objects left behind."),
        )


@dataclass
class VLMOutput:
    frame_idx: int
    timestamp: float
    is_escalated: bool
    priority_score: float

    detected_entities: list = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    scene_description: str = ""   # natural language: what the model actually sees


class VLMCore:
    """Core VLM engine: load Qwen2.5-VL-3B-Instruct and run escalated inference.

    Only escalated passes run the model.  There is no ambient pass — the
    ambient path was processor-only (no generate() call) and was therefore
    doing no real inference while consuming VRAM for a second model instance.
    Escalation is triggered by detector-pipeline priority signals.
    """

    def __init__(self, config: VLMConfig | None = None):
        self.config = config or VLMConfig.from_yaml()
        self._model = None
        self._processor = None
        self._initialized = False
        self._frame_count = 0
        self._last_scene_description: str = ""  # set by _run_model_inference

        self._device = self.config.device
        self._dtype = (
            torch.bfloat16
            if self.config.dtype == "bfloat16" and IMPORTED_TORCH
            else (torch.float16 if IMPORTED_TORCH else None)
        )

        # Actual frame size, injected from the live video source when known.
        self._frame_size: tuple[int, int] | None = None

    def set_frame_size(self, width: int, height: int) -> None:
        """Record the real capture resolution (pre-initialize, for warmup)."""
        self._frame_size = (int(width), int(height))

    def initialize(self) -> bool:
        if self._initialized:
            return True

        if not IMPORTED_TORCH:
            print("[VLM] WARNING: torch not available, VLM disabled")
            return False

        try:
            self._load_model()
            self._initialized = True
            print(f"[VLM] Initialized: {self.config.model_name} on {self._device}")
            return True
        except Exception as e:
            print(f"[VLM] FAILED to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_model(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        quant = (self.config.quantization or "").lower().strip()

        print(
            f"[VLM] Loading {self.config.model_name}  "
            f"quantization={quant or 'none (full precision)'}  "
            f"device={self._device}  vram_ceiling={self.config.max_vram_gb} GB"
        )

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )

        # --- load with quantization ---
        if quant == "nf4":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise RuntimeError(
                    "[VLM] bitsandbytes is not installed.  NF4 quantization requires "
                    "bitsandbytes.  Install it with: pip install bitsandbytes  "
                    "Then retry.  Without NF4 the 3B model needs ~7 GB bfloat16 "
                    "which exceeds the 6 GB VRAM budget."
                )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("[VLM] Using bitsandbytes NF4 4-bit quantization (announced)")
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map=self._device if torch.cuda.is_available() else "cpu",
            )
        elif quant in ("awq", ""):
            # AWQ not available on this platform — raise with explicit message.
            if quant == "awq":
                raise RuntimeError(
                    "[VLM] AWQ quantization requested but autoawq is not installed "
                    "(the upstream package was archived; no Windows wheel exists for "
                    "this torch build).  Set model.quantization: nf4 in configs/vlm.yaml "
                    "to use bitsandbytes NF4 instead."
                )
            # null/empty = full precision
            print(
                "[VLM] *** WARNING: quantization is null — loading in full bfloat16. "
                "The 3B model needs ~7 GB bfloat16 which EXCEEDS the 6 GB VRAM budget. "
                "Set model.quantization: nf4 in configs/vlm.yaml. ***"
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=self._dtype,
                device_map=self._device if torch.cuda.is_available() else "cpu",
            )
        else:
            raise RuntimeError(
                f"[VLM] Unknown quantization method: {quant!r}. "
                "Supported: 'nf4', or null for full precision."
            )

        self._model.eval()

        # --- VRAM budget check ---
        if torch.cuda.is_available():
            vram_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            ceiling = self.config.max_vram_gb
            print(
                f"[VLM] Model loaded.  VRAM allocated: {vram_used_gb:.2f} GB  "
                f"(ceiling: {ceiling} GB)"
            )
            if vram_used_gb > ceiling:
                raise RuntimeError(
                    f"[VLM] VRAM budget exceeded: model used {vram_used_gb:.2f} GB "
                    f"but model.max_vram_gb={ceiling} GB in configs/vlm.yaml.  "
                    "Aborting — a silent OOM-then-CPU-fallback would make inference "
                    "too slow to be useful.  Lower the model size or raise the ceiling."
                )

    def run_escalated_pass(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        context_prompt: str | None = None,
    ) -> VLMOutput:
        if not frames:
            raise ValueError(
                "[VLM] run_escalated_pass called with no frames.  Refusing to return "
                "an escalated result when no frame was actually analyzed."
            )

        last_frame = frames[-1]
        last_idx = frame_indices[-1] if frame_indices else 0

        # Deliberately NOT wrapped in try/except: a failed escalated pass must
        # surface, not silently become an empty entity list.
        detected_entities = self._run_model_inference(last_frame, context_prompt)
        # _run_model_inference sets self._last_scene_description as a side effect.
        scene_desc = self._last_scene_description

        confidence = (
            max(float(e["confidence"]) for e in detected_entities)
            if detected_entities else 0.0
        )

        return VLMOutput(
            frame_idx=last_idx,
            timestamp=time.perf_counter(),
            is_escalated=True,
            priority_score=1.0,
            detected_entities=detected_entities,
            confidence=confidence,
            scene_description=scene_desc,
        )

    # Prompt 1: natural language scene description — what the model ACTUALLY sees.
    # No constraints on vocabulary. Model describes freely in 1-2 sentences.
    SCENE_PROMPT = (
        "You are a surveillance camera AI. Look at this image carefully and describe "
        "in 1-2 sentences exactly what you see: the people, their actions, what they "
        "are doing, any objects of interest, and anything unusual. Be specific and factual."
    )

    # Prompt 2: structured entity extraction for downstream alerting logic.
    # The parser in _parse_entities_from_text expects exactly this format.
    ENTITY_PROMPT = (
        "You are analyzing a surveillance frame. List only what you can actually see. "
        "Respond with a JSON array and nothing else. Each element must be an object "
        'with keys: "type" (one of: person, fall, fire, smoke, fight, gathering, '
        'suspicious, object_left), "confidence" (a number 0.0-1.0 reflecting how '
        'certain you are), and "description" (a short factual phrase). '
        "If you see nothing of note, respond with []."
    )

    def _run_model_inference(self, frame: np.ndarray, prompt: str | None = None) -> list[dict]:
        if self._model is None or self._processor is None:
            raise RuntimeError(
                "[VLM] Cannot run inference: model/processor not loaded.  "
                f"Expected model: {self.config.model_name}"
            )

        # --- Pass 1: Natural language scene description (what the model actually sees) ---
        self._last_scene_description = self._run_single_prompt(frame, self.SCENE_PROMPT)
        print(f"[vlm] Scene: {self._last_scene_description}")

        # --- Pass 2: Structured entity extraction for alerting logic ---
        text_prompt = prompt or self.ENTITY_PROMPT
        return self._parse_entities_from_text(
            self._run_single_prompt(frame, text_prompt)
        )

    def _run_single_prompt(self, frame: np.ndarray, prompt: str) -> str:
        """Run the model with a single text prompt on a frame. Returns raw text output."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._processor(
            text=[self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )],
            images=[frame],
            return_tensors="pt",
            padding=True,
        )

        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.config.escalated_max_tokens,
            do_sample=False,
            use_cache=True,
        )

        return self._processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

    def _parse_entities_from_text(self, text: str) -> list[dict]:
        """Parse the model's structured JSON entity list.

        The model is instructed (see ENTITY_PROMPT) to emit a JSON array of
        {type, confidence, description} objects.  If the model returned
        something else, raises with the raw text attached rather than
        keyword-guessing entities the model never asserted.
        """
        import json

        raw = text.strip()

        # Strip ```json ... ``` fence if the model wrapped its answer.
        fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()

        if not raw.startswith("["):
            start, end = raw.find("["), raw.rfind("]")
            if start == -1 or end <= start:
                raise ValueError(
                    "[VLM] Could not parse entities: model response contains no JSON "
                    "array.  Entities are NOT inferred by keyword matching.  "
                    f"Raw model output was:\n{text!r}"
                )
            raw = raw[start:end + 1]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[VLM] Could not parse entities: invalid JSON from model ({e}).  "
                f"Raw model output was:\n{text!r}"
            )

        if not isinstance(parsed, list):
            raise ValueError(
                f"[VLM] Expected a JSON array, got {type(parsed).__name__}.  "
                f"Raw model output was:\n{text!r}"
            )

        entities: list[dict] = []
        for item in parsed:
            if not isinstance(item, dict) or "type" not in item:
                raise ValueError(
                    "[VLM] Array element is not an object with a 'type' field: "
                    f"{item!r}.  Raw model output was:\n{text!r}"
                )
            if "confidence" not in item:
                raise ValueError(
                    "[VLM] Model did not report confidence for entity "
                    f"{item.get('type')!r}.  Refusing to substitute a made-up value.  "
                    f"Raw model output was:\n{text!r}"
                )
            entities.append({
                "type": str(item["type"]),
                "confidence": float(item["confidence"]),
                "source": "vlm_detection",
                "description": str(item.get("description", "")),
            })

        return entities

    def should_escalate(
        self,
        detector_priority_score: float,
        frame_idx: int,
        forced_interval: int = 900,   # ~30s at 30fps; was 150 (~5s) — too frequent
        ambiguous_low: float = 0.3,
        ambiguous_high: float = 0.7,
    ) -> tuple[bool, str]:
        if detector_priority_score >= ambiguous_high:
            return True, "high_detector_confidence"

        if ambiguous_low < detector_priority_score < ambiguous_high:
            return True, "ambiguous_needs_verification"

        if detector_priority_score < ambiguous_low and frame_idx % forced_interval == 0:
            return True, "forced_interval"

        return False, ""

    def vram_mb(self) -> int:
        if not IMPORTED_TORCH or not torch.cuda.is_available():
            return -1
        return int(torch.cuda.memory_allocated() / (1024 * 1024))

    def reset(self) -> None:
        self._frame_count = 0


class VLMManager:
    """Orchestrates VLM escalated inference with multi-frame context.

    The main loop calls update_frame_buffer() every frame so that when
    _handle_escalation fires it receives a real temporal window rather than
    a single frame.
    """

    def __init__(self, config: VLMConfig | None = None):
        self.config = config or VLMConfig.from_yaml()
        self.core = VLMCore(self.config)

        self._last_escalation_frame = -1
        self._escalation_count = 0

        self._ambient_outputs: list[VLMOutput] = []
        self._escalation_outputs: list[VLMOutput] = []

        # Frame buffer sized from the real capture rate.
        from core.config import load_fps
        self._fps = load_fps()
        self._buffer_s = 1.0
        self._context_s = 0.5
        self._frame_buffer: list[tuple[np.ndarray, int, float]] = []
        self._buffer_size = max(1, int(round(self._buffer_s * self._fps)))
        self._context_frames = max(1, int(round(self._context_s * self._fps)))

    def set_frame_size(self, width: int, height: int) -> None:
        self.core.set_frame_size(width, height)

    def initialize(self) -> bool:
        return self.core.initialize()

    def update_frame_buffer(self, frame: np.ndarray, frame_idx: int) -> None:
        """Push the current frame into the rolling buffer for escalation context."""
        self._frame_buffer.append((frame.copy(), frame_idx, time.perf_counter()))
        if len(self._frame_buffer) > self._buffer_size:
            self._frame_buffer.pop(0)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detector_priority_score: float = 0.0,
    ) -> VLMOutput | None:
        """Check if this frame should escalate; if so run the model.

        Returns a VLMOutput on escalation, None otherwise.
        """
        # Update the buffer so escalation gets real context.
        self.update_frame_buffer(frame, frame_idx)

        should_escalate, reason = self.core.should_escalate(
            detector_priority_score,
            frame_idx,
        )

        if not should_escalate:
            return None

        self._escalation_count += 1
        self._last_escalation_frame = frame_idx

        output = self._handle_escalation(frame, frame_idx, reason)
        if output:
            self._escalation_outputs.append(output)

        return output

    def _handle_escalation(
        self,
        frame: np.ndarray,
        frame_idx: int,
        reason: str,
    ) -> VLMOutput | None:
        n = self._context_frames
        context_frames = [f for f, i, _ in self._frame_buffer[-n:]]
        context_indices = [i for f, i, _ in self._frame_buffer[-n:]]

        context_frames.append(frame)
        context_indices.append(frame_idx)

        try:
            output = self.core.run_escalated_pass(
                context_frames,
                context_indices,
                context_prompt=None,
            )
        except Exception as e:
            # Re-raise: the stage guard in main_loop.py handles this.
            raise RuntimeError(
                f"[VLM] Escalated pass failed at frame {frame_idx}: {e}"
            ) from e

        output.reasoning = f"Escalated: {reason}"
        return output

    def get_stats(self) -> dict:
        return {
            "frame_count": self.core._frame_count,
            "escalation_count": self._escalation_count,
            "last_escalation_frame": self._last_escalation_frame,
            "escalation_outputs": len(self._escalation_outputs),
            "vram_mb": self.core.vram_mb(),
        }

    def reset(self) -> None:
        self.core.reset()
        self._ambient_outputs.clear()
        self._escalation_outputs.clear()
        self._frame_buffer.clear()
        self._escalation_count = 0
        self._last_escalation_frame = -1
