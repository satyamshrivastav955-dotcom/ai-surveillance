"""Tests for core/par.py and core/par_aggregator.py — Phase 5B."""
from __future__ import annotations

import numpy as np
import pytest
from core.par import PARDetector, get_color_name, extract_segment_color
from core.par_aggregator import PARAggregator

def test_get_color_name():
    # Test black/white/grey
    assert get_color_name(0, 0, 10) == "black"
    assert get_color_name(0, 0, 240) == "white"
    assert get_color_name(0, 10, 120) == "grey"

    # Test hues
    assert get_color_name(5, 150, 150) == "red"
    assert get_color_name(20, 150, 150) == "orange"
    assert get_color_name(30, 150, 150) == "yellow"
    assert get_color_name(60, 150, 150) == "green"
    assert get_color_name(110, 150, 150) == "blue"
    assert get_color_name(140, 150, 150) == "purple"
    assert get_color_name(160, 150, 150) == "pink"

def test_extract_segment_color():
    # Create synthetic image with red top half and blue bottom half
    # BGR format
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :, :] = [0, 0, 255] # Red top
    img[50:, :, :] = [255, 0, 0] # Blue bottom

    # Upper segment (0.15 to 0.45) should be red
    upper = extract_segment_color(img, 0.15, 0.45)
    assert upper == "red"

    # Lower segment (0.50 to 0.85) should be blue
    lower = extract_segment_color(img, 0.50, 0.85)
    assert lower == "blue"

def test_par_detector_raises_without_weights():
    """PARDetector must raise immediately when par.weights is null.

    Previously it built an ImageNet ResNet18 + random Linear(512,8) head and
    printed 'loaded successfully', which was fabricated output.  Now it raises
    RuntimeError so the pipeline's stage() guard can catch it cleanly.
    """
    with pytest.raises(RuntimeError, match="REFUSING TO LOAD"):
        PARDetector(cfg={})  # weights=None


def test_par_detector_raises_with_empty_weights():
    """Empty string is also treated as null (no real path)."""
    with pytest.raises((RuntimeError, NotImplementedError)):
        PARDetector(cfg={"weights": ""})


def test_par_detector_raises_with_missing_weights_file():
    """A non-existent path should raise, not silently use ImageNet head."""
    with pytest.raises((RuntimeError, NotImplementedError)):
        PARDetector(cfg={"weights": "/nonexistent/checkpoint.pt"})

def test_par_aggregator_majority_vote():
    agg = PARAggregator(window_size=5)
    
    # Send 3 black and 2 white upper_colors
    m1 = {"gender": "male", "hair_length": "short", "upper_color": "black", 
          "upper_type": "short_sleeves", "lower_color": "blue", "lower_type": "pants", 
          "accessories": []}
    m2 = {"gender": "male", "hair_length": "short", "upper_color": "white", 
          "upper_type": "short_sleeves", "lower_color": "blue", "lower_type": "pants", 
          "accessories": []}
          
    agg.update(1, m1)
    agg.update(1, m1)
    agg.update(1, m2)
    agg.update(1, m1)
    res = agg.update(1, m2)
    
    assert res["upper_color"] == "black"

def test_par_aggregator_accessories():
    agg = PARAggregator(window_size=4)
    # backpack appears 3 times, handbag 1 time
    m1 = {"gender": "male", "hair_length": "short", "upper_color": "black", 
          "upper_type": "short_sleeves", "lower_color": "blue", "lower_type": "pants", 
          "accessories": ["backpack"]}
    m2 = {"gender": "male", "hair_length": "short", "upper_color": "black", 
          "upper_type": "short_sleeves", "lower_color": "blue", "lower_type": "pants", 
          "accessories": ["backpack", "handbag"]}
    m3 = {"gender": "male", "hair_length": "short", "upper_color": "black", 
          "upper_type": "short_sleeves", "lower_color": "blue", "lower_type": "pants", 
          "accessories": []}

    agg.update(1, m1)
    agg.update(1, m2)
    agg.update(1, m1)
    res = agg.update(1, m3)

    # backpack in 3/4 frames >= 50%
    # handbag in 1/4 frames < 50%
    assert "backpack" in res["accessories"]
    assert "handbag" not in res["accessories"]

def test_par_aggregator_prune():
    agg = PARAggregator()
    m = {"gender": "male", "hair_length": "short", "upper_color": "black", 
         "upper_type": "short_sleeves", "lower_color": "blue", "lower_type": "pants", 
         "accessories": []}
    agg.update(1, m)
    agg.update(2, m)
    
    assert 1 in agg._history
    assert 2 in agg._history
    
    agg.prune({1})
    assert 1 in agg._history
    assert 2 not in agg._history
