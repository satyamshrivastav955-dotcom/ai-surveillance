"""Event logger — stores all events in SQLite for Phase 5 and VLM layer.

This module provides structured event logging that persists all event types
from the pipeline (fall, fire, smoke, phone, gathering, violence, object_left,
identity) to a SQLite database with keyframe images saved to disk.

This is required groundwork for the VLM layer's natural-language query
feature — build it now so it doesn't need retrofitting.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class EventLogger:
    """Logs all events to SQLite with optional keyframe images."""

    def __init__(self, db_path: str = "data/events.db", keyframes_dir: str = "data/keyframes"):
        self.db_path = Path(db_path)
        self.keyframes_dir = Path(keyframes_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                t_iso TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                track_id INTEGER,
                confidence REAL,
                details_json TEXT,
                keyframe_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_t_iso ON events(t_iso)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_track_id ON events(track_id)")
        self.conn.commit()

    def log_event(self, event, frame: np.ndarray | None = None, frame_idx: int = 0) -> None:
        """Log an event to the database with optional keyframe.

        Args:
            event: Event object with event_type, t_iso, frame_idx, details
            frame: Optional frame image to save as keyframe
            frame_idx: Frame index for keyframe filename
        """
        keyframe_path = None
        if frame is not None and frame.size > 0:
            keyframe_path = self._save_keyframe(event, frame, frame_idx)

        details_json = None
        if hasattr(event, 'details'):
            details_json = json.dumps(event.details)

        self.conn.execute("""
            INSERT INTO events (t_iso, frame_idx, event_type, track_id, confidence, details_json, keyframe_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event.t_iso,
            event.frame_idx,
            event.event_type,
            event.details.get("track_id") if hasattr(event, 'details') else None,
            event.details.get("confidence") if hasattr(event, 'details') else None,
            details_json,
            str(keyframe_path) if keyframe_path else None
        ))
        self.conn.commit()

    def _save_keyframe(self, event, frame: np.ndarray, frame_idx: int) -> Path | None:
        """Save a keyframe image for the event."""
        try:
            t_safe = event.t_iso.replace(":", "-").replace(" ", "_")
            filename = f"{event.event_type}_{t_safe}_f{frame_idx}_id{event.details.get('track_id', 'na')}.jpg"
            path = self.keyframes_dir / filename
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return path
        except Exception:
            return None

    def query_events(self, event_type: str | None = None,
                     track_id: int | None = None,
                     limit: int = 100) -> list[dict]:
        """Query events from the database."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if track_id is not None:
            query += " AND track_id = ?"
            params.append(track_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        cursor = self.conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        self.conn.close()
