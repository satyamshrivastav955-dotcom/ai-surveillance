"""Pedestrian Attribute Recognition (Phase 5B).

Extracts attributes from person crops:
  - gender (male/female)
  - hair_length (short/long)
  - upper_color (black, white, red, blue, green, grey, yellow, orange, purple, pink)
  - lower_color (black, blue, grey, white, red, green, purple)
  - upper_type (long_sleeves/short_sleeves)
  - lower_type (pants/shorts/skirt)
  - accessories (backpack, handbag, hat, glasses)

Uses a hybrid approach:
  1. HSV-based dominant color extraction on upper/lower body segments (100% offline, highly accurate).
  2. torchvision ResNet18 deep learning feature extractor for structural attributes, falling back gracefully if offline.
"""
from __future__ import annotations

import os
import time
from typing import Any
import numpy as np

def get_color_name(h: float, s: float, v: float) -> str:
    # H: 0-180, S: 0-255, V: 0-255
    if v < 50:
        return "black"
    if s < 30 and v > 180:
        return "white"
    if s < 30:
        return "grey"
    
    if h < 10 or h > 165:
        return "red"
    elif h < 25:
        return "orange"
    elif h < 38:
        return "yellow"
    elif h < 75:
        return "green"
    elif h < 130:
        return "blue"
    elif h < 150:
        return "purple"
    else:
        return "pink"

def extract_segment_color(crop: np.ndarray, y_start_frac: float, y_end_frac: float) -> str:
    import cv2
    if crop is None or crop.size == 0:
        return "unknown"
    h, w = crop.shape[:2]
    ys = int(h * y_start_frac)
    ye = int(h * y_end_frac)
    segment = crop[ys:ye, :]
    if segment.size == 0:
        return "unknown"
    
    hsv = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    if len(pixels) > 500:
        # Sample subset for speed
        indices = np.linspace(0, len(pixels) - 1, 500, dtype=int)
        pixels = pixels[indices]
    
    counts: dict[str, int] = {}
    for p in pixels:
        cname = get_color_name(p[0], p[1], p[2])
        counts[cname] = counts.get(cname, 0) + 1
    
    if not counts:
        return "unknown"
    return max(counts, key=counts.get)


class PARDetector:
    """HSV colour-only PAR head.

    The deep-learning attributes (gender, hair_length, upper_type, lower_type,
    accessories) require a task-specific checkpoint — they are NOT provided by
    ImageNet pretrained weights.  Loading an ImageNet ResNet18 with a randomly
    initialised 8-output Linear head and running sigmoid(logits) on it produces
    random binary predictions on every call, which is fabricated output.

    This constructor raises immediately when `par.weights` is null (which it
    is in configs/models.yaml) rather than building a random head and printing
    "loaded successfully".  The colour-extraction helpers (get_color_name,
    extract_segment_color) are real and deterministic and are kept for future
    use once a trained checkpoint is available.

    To re-enable:
      1. Obtain or train a checkpoint on a PAR dataset (e.g. PA-100K, RAP).
      2. Set `par.weights: path/to/checkpoint.pt` in configs/models.yaml.
      3. Remove the RuntimeError below and load the checkpoint here.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        weights = (cfg or {}).get("weights")
        if not weights:
            raise RuntimeError(
                "[PAR] REFUSING TO LOAD: par.weights is null in configs/models.yaml. "
                "PARDetector requires a checkpoint trained on a pedestrian-attribute "
                "dataset (e.g. PA-100K, RAP).  An ImageNet ResNet18 with a randomly "
                "initialised Linear(512, 8) head produces random gender/hair/clothing "
                "predictions on every call — this is fabricated output, not a detection. "
                "Set features.par: false in configs/pipeline.yaml (already done) or "
                "supply a real checkpoint.  The HSV colour helpers remain available "
                "in core/par.py for use once a full detector is wired up."
            )
        # If a weights path is provided, add loading logic here.
        raise NotImplementedError(
            f"[PAR] weights={weights!r} supplied but checkpoint loading is not yet "
            "implemented in this refactored stub.  Add torch.load + model.load_state_dict "
            "here once the architecture is confirmed against the checkpoint."
        )

    def process_crop(self, crop: np.ndarray, track_id: int) -> dict[str, Any]:
        """Not reached — __init__ raises before this can be called."""
        raise RuntimeError("[PAR] PARDetector was not properly initialised.")
