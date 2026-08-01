"""VLM Integration Tests — updated for Phase 4 rewrite.

Removed tests for deleted modules:
  - vlm.kv_cache (TieredKVCache, HotKVCache, ColdKVCache, KVBlock) — module removed.
  - vlm.token_pruning (SpatialTokenPruner, PruningConfig, PruningResult) — module removed.
  - vlm.temporal_merge (TemporalMerger, TemporalConfig) — module removed.
  - vlm.core.VLMCore.encode_frame / run_ambient_pass — methods removed.
  - VLMConfig.ambient_pruning_ratio / ambient_max_tokens — fields removed.

Added / retained tests:
  - VLMConfig loads correct model name (3B, not 7B).
  - VLMCore.should_escalate logic.
  - VLMCore.vram_mb when torch unavailable.
  - VLMCore._parse_entities_from_text with valid and invalid JSON.
  - VLMManager.update_frame_buffer (called every frame for multi-frame context).
  - VLMIntegration creation (no model load in test — vlm disabled).
  - VLMIntegration.process_frame returns None before initialization.
  - VLMIntegration._compute_priority_from_events.
  - EscalationConfig loads.
  - EscalationHandler.check_escalation with forced_interval.
  - QueryEngine.initialize returns bool.
  - OpenVocabWatcher add/remove.
"""
from __future__ import annotations

import time
import numpy as np
import pytest


class TestVLMConfig:
    """Tests for VLMConfig dataclass."""

    def test_vlm_config_from_yaml(self):
        from vlm.core import VLMConfig
        config = VLMConfig.from_yaml()
        # Must be 3B, not 7B — the 7B target was changed in Phase 3.1
        assert "3B" in config.model_name or "3b" in config.model_name.lower(), (
            f"Expected 3B model but got: {config.model_name!r}. "
            "Update configs/vlm.yaml model.name to a 3B variant."
        )
        assert config.device == "cuda:0"
        assert config.quantization in ("nf4", "awq", "")
        assert 0.0 < config.max_vram_gb <= 6.0

    def test_vlm_config_defaults(self):
        from vlm.core import VLMConfig
        config = VLMConfig()
        assert config.escalated_max_tokens > 0
        assert config.max_pixels > 0
        assert config.min_pixels > 0


class TestVLMCore:
    """Tests for VLMCore — model-independent paths only (no GPU required)."""

    def test_vlm_core_not_initialized_on_construction(self):
        from vlm.core import VLMCore, VLMConfig
        config = VLMConfig()
        core = VLMCore(config)
        assert core._initialized is False
        assert core._model is None

    def test_set_frame_size(self):
        from vlm.core import VLMCore, VLMConfig
        config = VLMConfig()
        core = VLMCore(config)
        core.set_frame_size(1280, 720)
        assert core._frame_size == (1280, 720)

    def test_should_escalate_high_confidence(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        should, reason = core.should_escalate(
            detector_priority_score=0.8, frame_idx=0
        )
        assert should is True
        assert reason == "high_detector_confidence"

    def test_should_escalate_ambiguous(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        should, reason = core.should_escalate(
            detector_priority_score=0.5, frame_idx=0
        )
        assert should is True
        assert "ambiguous" in reason

    def test_should_escalate_forced_interval(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        should, reason = core.should_escalate(
            detector_priority_score=0.0, frame_idx=150, forced_interval=150
        )
        assert should is True
        assert reason == "forced_interval"

    def test_should_not_escalate_low_confidence(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        should, reason = core.should_escalate(
            detector_priority_score=0.1, frame_idx=50, forced_interval=150
        )
        assert should is False
        assert reason == ""

    def test_vram_mb_without_cuda(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        result = core.vram_mb()
        # If CUDA is unavailable, returns -1; if available, returns int >= 0
        assert isinstance(result, int)

    def test_parse_entities_valid_json(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        text = '[{"type":"person","confidence":0.9,"description":"walking"}]'
        entities = core._parse_entities_from_text(text)
        assert len(entities) == 1
        assert entities[0]["type"] == "person"
        assert entities[0]["confidence"] == 0.9
        assert entities[0]["source"] == "vlm_detection"

    def test_parse_entities_empty_list(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        entities = core._parse_entities_from_text("[]")
        assert entities == []

    def test_parse_entities_json_fence(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        text = '```json\n[{"type":"fire","confidence":0.8,"description":"flames"}]\n```'
        entities = core._parse_entities_from_text(text)
        assert len(entities) == 1
        assert entities[0]["type"] == "fire"

    def test_parse_entities_no_array_raises(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        with pytest.raises(ValueError, match="no JSON array"):
            core._parse_entities_from_text("Sure, I can see a person walking.")

    def test_parse_entities_missing_confidence_raises(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        with pytest.raises(ValueError, match="confidence"):
            core._parse_entities_from_text('[{"type":"person","description":"walking"}]')

    def test_run_escalated_pass_no_frames_raises(self):
        from vlm.core import VLMCore, VLMConfig
        core = VLMCore(VLMConfig())
        with pytest.raises(ValueError, match="no frames"):
            core.run_escalated_pass([], [])


class TestVLMManager:
    """Tests for VLMManager — frame buffer and escalation logic."""

    def test_frame_buffer_grows(self):
        from vlm.core import VLMManager, VLMConfig
        manager = VLMManager(VLMConfig())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(5):
            manager.update_frame_buffer(frame, i)
        assert len(manager._frame_buffer) <= manager._buffer_size
        assert len(manager._frame_buffer) >= 1

    def test_frame_buffer_capped(self):
        from vlm.core import VLMManager, VLMConfig
        manager = VLMManager(VLMConfig())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(manager._buffer_size + 10):
            manager.update_frame_buffer(frame, i)
        assert len(manager._frame_buffer) <= manager._buffer_size

    def test_process_frame_no_model_returns_none_on_no_escalation(self):
        """With no model loaded and low priority, process_frame returns None."""
        from vlm.core import VLMManager, VLMConfig
        manager = VLMManager(VLMConfig())
        # Don't initialize (no model load)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # frame_idx that does NOT hit forced_interval (=150 by default)
        result = manager.process_frame(frame, frame_idx=1, detector_priority_score=0.0)
        assert result is None

    def test_stats(self):
        from vlm.core import VLMManager, VLMConfig
        manager = VLMManager(VLMConfig())
        stats = manager.get_stats()
        assert "escalation_count" in stats
        assert "vram_mb" in stats

    def test_reset(self):
        from vlm.core import VLMManager, VLMConfig
        manager = VLMManager(VLMConfig())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(3):
            manager.update_frame_buffer(frame, i)
        manager.reset()
        assert len(manager._frame_buffer) == 0
        assert manager._escalation_count == 0


class TestEscalation:
    """Tests for escalation handler."""

    def test_escalation_config_loads(self):
        from vlm.escalation import EscalationConfig
        config = EscalationConfig.from_yaml()
        assert 0.0 < config.ambiguous_threshold_low < 1.0
        assert 0.0 < config.ambiguous_threshold_high < 1.0

    def test_escalation_forced_interval(self):
        from vlm.escalation import EscalationHandler, EscalationConfig
        config = EscalationConfig.from_yaml()   # reads fps from pipeline.yaml
        config.forced_full_fidelity_interval = 10
        handler = EscalationHandler(config)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        decision = handler.check_escalation(frame, 10, [])
        assert decision.should_escalate is True
        assert "forced" in decision.reason.lower()

    def test_escalation_high_detector_confidence(self):
        from vlm.escalation import EscalationHandler, EscalationConfig
        from core.events import Event
        config = EscalationConfig.from_yaml()   # reads fps from pipeline.yaml
        handler = EscalationHandler(config)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        event = Event(
            event_type="FALL",
            t_iso="2026-01-01T00:00:00",
            frame_idx=0,
            details={"confidence": 0.9},
        )
        decision = handler.check_escalation(frame, 0, [event])
        assert decision.should_escalate is True


class TestQueryEngine:
    """Tests for query engine — no model needed for config tests."""

    def test_query_config_loads(self):
        from vlm.query_engine import QueryConfig
        config = QueryConfig.from_yaml()
        assert config.enabled is True
        assert config.max_context_tokens > 0

    def test_query_engine_event_type_parsing(self):
        from vlm.query_engine import QueryEngine, QueryConfig
        engine = QueryEngine(QueryConfig())
        event_types = engine._parse_event_types("Show me all fire events from today")
        assert "FIRE" in event_types

    def test_query_engine_time_range_parsing(self):
        from vlm.query_engine import QueryEngine, QueryConfig
        engine = QueryEngine(QueryConfig())
        start, end = engine._parse_time_range("Show me events from last 5 minutes")
        assert start < end
        assert end > time.time() - 400

    def test_open_vocab_watcher_add_remove(self):
        from vlm.query_engine import OpenVocabWatcher, QueryConfig
        watcher = OpenVocabWatcher()
        watcher_id = watcher.add_watcher("Notify me when someone carries a red bag")
        assert watcher_id in [w["id"] for w in watcher.list_watchers()]
        removed = watcher.remove_watcher(watcher_id)
        assert removed is True


class TestVLMIntegration:
    """Tests for complete VLM integration — no model load."""

    def test_vlm_integration_creation(self):
        """VLMIntegration creates without loading the model."""
        from vlm import VLMIntegration
        vlm = VLMIntegration()
        assert vlm.vlm_manager is not None
        assert vlm.escalation_handler is not None
        assert vlm.query_engine is not None
        assert vlm.open_vocab_watcher is not None
        # Removed: vlm.kv_cache, vlm.pruner, vlm.temporal_merger (dead scaffolding)

    def test_vlm_integration_not_initialized_by_default(self):
        from vlm import VLMIntegration
        vlm = VLMIntegration()
        assert vlm._initialized is False

    def test_process_frame_before_init_returns_none(self):
        """process_frame before initialize() returns None — no fabricated output."""
        from vlm import VLMIntegration
        vlm = VLMIntegration()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = vlm.process_frame(frame, 0, [])
        assert result is None

    def test_query_before_init_returns_none(self):
        from vlm import VLMIntegration
        vlm = VLMIntegration()
        result = vlm.query("Show me all events from today")
        assert result is None

    def test_watcher_add_remove(self):
        from vlm import VLMIntegration
        vlm = VLMIntegration()
        watcher_id = vlm.add_watcher("Test watcher")
        watchers = vlm.list_watchers()
        assert len(watchers) > 0
        removed = vlm.remove_watcher(watcher_id)
        assert removed is True

    def test_stats_structure(self):
        from vlm import VLMIntegration
        vlm = VLMIntegration()
        stats = vlm.get_stats()
        assert "initialized" in stats
        assert "frame_count" in stats
        assert "vlm" in stats
        # Removed: stats["kv_cache"] — no longer a stat
        assert "escalation" in stats
        assert "query" in stats

    def test_compute_priority_from_events_empty(self):
        from vlm import VLMIntegration
        vlm = VLMIntegration()
        assert vlm._compute_priority_from_events(None) == 0.0
        assert vlm._compute_priority_from_events([]) == 0.0

    def test_compute_priority_from_events_with_confidence(self):
        from vlm import VLMIntegration
        from core.events import Event
        vlm = VLMIntegration()

        class FakeEvent:
            confidence = 0.8

        result = vlm._compute_priority_from_events([FakeEvent(), FakeEvent()])
        assert abs(result - 0.8) < 1e-6

    def test_close_is_idempotent(self):
        from vlm import VLMIntegration
        vlm = VLMIntegration()
        vlm.close()  # no DB connection open — should not raise
        vlm.close()  # second call also safe


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
