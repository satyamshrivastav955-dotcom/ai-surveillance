"""Pytest fixtures shared across test suites.

This file provides fixtures for tests that need access to configuration
values or runtime parameters from the pipeline config.
"""
from __future__ import annotations

import pytest

from core.config import load_pipeline_config
from pipeline.frame_router import FrameRouter


@pytest.fixture
def pose_every() -> int:
    """Get the actual pose cadence from pipeline.yaml via FrameRouter.
    
    Used by fall_cadence_test.py to verify fall detection works correctly
    at the configured cadence (every N frames).
    """
    cfg = load_pipeline_config()
    router = FrameRouter(cfg.get("router", {}))
    return router.every("pose")
