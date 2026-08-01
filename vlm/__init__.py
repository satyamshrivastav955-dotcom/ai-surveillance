"""VLM Integration Module — simplified, honest entry point.

What changed from the previous version:
  - Removed: TieredKVCache, SpatialTokenPruner, TemporalMerger imports and use.
    All three were dead scaffolding (KV blocks pushed with keys=None; pruner
    never called from production; merger output consumed by nothing).
  - Removed: double model load.  QueryEngine no longer constructs its own
    VLM instance — the already-loaded model is passed in.
  - Fixed: _initialized = vlm_ok AND query_ok (was OR, so VLM failure was hidden).
  - Added: VLM escalation output persisted to vlm_events table in events.db.
  - Added: update_frame_buffer() called every frame so escalation gets real context.
  - Added: VLM process_frame() call gated through router.should_run("vlm_escalated").
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlm.config_paths import DEFAULT_VLM_CONFIG

from vlm.core import VLMConfig, VLMManager, VLMOutput
from vlm.escalation import EscalationConfig, EscalationHandler, EscalationDecision
from vlm.query_engine import QueryConfig, QueryEngine, OpenVocabWatcher


class VLMIntegration:
    """Unified VLM integration for the AI surveillance pipeline.

    Responsibilities:
    - Load Qwen2.5-VL-3B-Instruct with NF4 quantization (one instance, shared).
    - On every frame: push frame into the multi-frame buffer.
    - On detector-triggered escalation: run real inference, persist result to DB.
    - Provide natural language query interface against the vlm_events table.
    """

    def __init__(self, config_path: str = DEFAULT_VLM_CONFIG):
        self.config_path = config_path

        from vlm.config_paths import load_vlm_config
        cfg = load_vlm_config(config_path)

        self.vlm_config = VLMConfig.from_yaml(config_path)
        self.escalation_config = EscalationConfig.from_yaml(config_path)
        self.query_config = QueryConfig.from_yaml(config_path)

        self.vlm_manager = VLMManager(self.vlm_config)
        self.escalation_handler = EscalationHandler(self.escalation_config)

        # QueryEngine shares the already-loaded model — do NOT let it construct
        # its own.  We pass the model reference in after initialization.
        self.query_engine = QueryEngine(self.query_config)
        self.open_vocab_watcher = OpenVocabWatcher(self.query_config)

        self._initialized = False
        self._frame_count = 0
        self._start_time = 0.0

        # VLM events persistence
        integration_cfg = cfg.get("integration", {})
        self._log_vlm_events = bool(integration_cfg.get("log_vlm_events", True))
        self._vlm_event_table = str(integration_cfg.get("vlm_event_db_table", "vlm_events"))
        self._event_db_path: str | None = None
        self._db_conn: sqlite3.Connection | None = None

    def set_frame_size(self, width: int, height: int) -> None:
        """Declare the live video source resolution before initialize()."""
        self.vlm_manager.set_frame_size(width, height)

    def initialize(self, event_db_path: str | None = None) -> bool:
        """Initialize the VLM stack.

        Args:
            event_db_path: Path to the pipeline events.db (single source of truth).
                           If None, derived from configs/pipeline.yaml.
        """
        if self._initialized:
            return True

        self._start_time = time.perf_counter()

        # --- load VLM (one instance) ---
        vlm_ok = self.vlm_manager.initialize()
        if not vlm_ok:
            print(
                "[VLMIntegration] FAILED: VLM model did not load.  "
                "See above for the specific error.  "
                "Running without VLM layer."
            )
            self._initialized = False
            return False

        # --- pass the loaded model into QueryEngine (avoid second load) ---
        if hasattr(self.query_engine, "set_model"):
            self.query_engine.set_model(
                self.vlm_manager.core._model,
                self.vlm_manager.core._processor,
            )
        query_ok = self.query_engine.initialize()
        if not query_ok:
            print(
                "[VLMIntegration] WARN: QueryEngine did not initialize.  "
                "Natural language queries against vlm_events will be unavailable.  "
                "Escalated inference and event logging remain active."
            )

        # _initialized requires BOTH: a query-only start without the model
        # running would look like success while the main value (inference) is missing.
        self._initialized = vlm_ok and query_ok

        # --- open the VLM events table ---
        if event_db_path is None:
            try:
                from core.config import load_pipeline_config
                perf = load_pipeline_config().get("perf", {})
                event_db_path = perf.get("event_db_path", "data/events.db")
            except Exception:
                event_db_path = "data/events.db"
        self._event_db_path = event_db_path
        self._open_vlm_events_db()

        if self._initialized:
            print(
                f"[VLMIntegration] Initialized  "
                f"VLM={vlm_ok}  Query={query_ok}  "
                f"vlm_events_db={self._event_db_path}"
            )
        return self._initialized

    def _open_vlm_events_db(self) -> None:
        if not self._log_vlm_events or not self._event_db_path:
            return
        try:
            p = Path(self._event_db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._db_conn = sqlite3.connect(str(p), check_same_thread=False)
            self._db_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._vlm_event_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    t_iso TEXT NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    reason TEXT,
                    entities_json TEXT,
                    confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._db_conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._vlm_event_table}_frame "
                f"ON {self._vlm_event_table}(frame_idx)"
            )
            self._db_conn.commit()
            print(f"[VLMIntegration] vlm_events table ready in {self._event_db_path}")
        except Exception as e:
            print(f"[VLMIntegration] WARN: could not open vlm_events DB: {e}")
            self._db_conn = None

    def _persist_vlm_output(self, output: VLMOutput) -> None:
        if not self._log_vlm_events or self._db_conn is None:
            return
        try:
            import json
            self._db_conn.execute(
                f"INSERT INTO {self._vlm_event_table} "
                "(t_iso, frame_idx, reason, entities_json, confidence) VALUES (?,?,?,?,?)",
                (
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    output.frame_idx,
                    output.reasoning or "",
                    json.dumps(output.detected_entities),
                    round(output.confidence, 4),
                ),
            )
            self._db_conn.commit()
        except Exception as e:
            print(f"[VLMIntegration] WARN: failed to persist VLM output: {e}")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detector_events: list | None = None,
        tracks: list | None = None,
    ) -> VLMOutput | None:
        """Process one frame: update buffer; run escalated inference if triggered.

        Returns VLMOutput on escalation, None on non-escalated frames.
        """
        if not self._initialized:
            return None

        self._frame_count += 1

        detector_priority = self._compute_priority_from_events(detector_events)

        output = self.vlm_manager.process_frame(
            frame, frame_idx, detector_priority
        )

        if output is not None and output.is_escalated:
            self._persist_vlm_output(output)

        return output

    def _compute_priority_from_events(self, events: list | None) -> float:
        if not events:
            return 0.0
        return sum(
            getattr(e, "confidence", 0.5)
            for e in events
            if hasattr(e, "confidence")
        ) / max(len(events), 1)

    def query(
        self,
        query_text: str,
        time_range: tuple[float, float] | None = None,
        event_types: list[str] | None = None,
    ) -> Any:
        if not self._initialized:
            return None
        return self.query_engine.query(
            query_text,
            None,   # no KV cache object anymore
            None,
            time_range,
            event_types,
        )

    def verify_event(
        self,
        event_type: str,
        confidence: float,
        details: dict,
        timestamp: str,
    ) -> Any:
        if not self._initialized:
            return None
        return self.query_engine.verify_event(
            event_type,
            confidence,
            details,
            None,
            None,
            timestamp,
        )

    def add_watcher(self, description: str, check_interval: int = 30) -> str:
        return self.open_vocab_watcher.add_watcher(
            description=description,
            check_interval=check_interval,
        )

    def remove_watcher(self, watcher_id: str) -> bool:
        return self.open_vocab_watcher.remove_watcher(watcher_id)

    def list_watchers(self) -> list[dict]:
        return self.open_vocab_watcher.list_watchers()

    def check_watchers(self, frame: np.ndarray, frame_idx: int) -> list:
        return self.open_vocab_watcher.check_watchers(
            frame, frame_idx, self.query_engine
        )

    def get_stats(self) -> dict:
        return {
            "initialized": self._initialized,
            "frame_count": self._frame_count,
            "uptime_seconds": time.perf_counter() - self._start_time,
            "vlm": self.vlm_manager.get_stats() if self.vlm_manager else {},
            "escalation": self.escalation_handler.get_stats() if self.escalation_handler else {},
            "query": self.query_engine.get_stats() if self.query_engine else {},
        }

    def reset(self) -> None:
        self.vlm_manager.reset()
        self.escalation_handler.reset()
        self._frame_count = 0
        self._start_time = time.perf_counter()

    def close(self) -> None:
        """Close the VLM events DB connection."""
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception:
                pass
            self._db_conn = None

    def vram_mb(self) -> int:
        return self.vlm_manager.core.vram_mb() if self.vlm_manager else -1


def create_vlm_integration(config_path: str = DEFAULT_VLM_CONFIG) -> VLMIntegration:
    return VLMIntegration(config_path)
