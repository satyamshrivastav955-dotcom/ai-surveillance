"""EventBuffer — Phase 5F.

Collects all per-frame events into a rolling per-camera buffer and flushes
them every ``flush_interval_s`` seconds as a single dense JSON blob matching
the spec schema from Section 8 of the implementation spec.

Design
------
- One EventBuffer instance per camera (or per pipeline run).
- Events are appended as they fire (fall, fight, smoking, phone, fire,
  gathering, object_left, identity) via ``append()``.
- Per-track state (bbox_last, attributes, pose_quality) is updated each frame.
- ``maybe_flush()`` is called every frame and silently returns None when it's
  not yet time to flush; it returns the JSON dict (and resets) when the
  interval has elapsed.

Deduplication
-------------
Object-left and gathering events deduplicate within a window:
- On first trigger: emit with event action="trigger".
- On subsequent windows while still present: emit with action="still_present"
  and updated ``dwell_seconds``.
- Fight events always emit (with clip_ref for human review).

JSON schema (matches the spec)::

    {
      "camera_id": "cam_01",
      "window_start": "2026-07-20T10:00:00",
      "window_end": "2026-07-20T10:00:10",
      "tracks": [
        {
          "track_id": 42,
          "bbox_last": [x1, y1, x2, y2],
          "pose_quality": "stable",
          "attributes": {...},        # from PAR (Phase 5B), or {}
          "events": [
            {"type": "SMOKING", "confidence": 0.74, "t": "10:00:04"}
          ]
        }
      ],
      "scene_events": [
        {"type": "OBJECT_LEFT", "action": "trigger", ...},
        {"type": "GATHERING", ...},
        {"type": "FIGHT", "track_ids": [42, 43], "clip_ref": "...mp4", ...}
      ]
    }
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Per-track rolling state
# ---------------------------------------------------------------------------

@dataclass
class _TrackRecord:
    track_id: int
    bbox_last: tuple = (0, 0, 0, 0)
    pose_quality: str = "missing"
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    last_seen: int = 0          # frame_idx of last appearance


# ---------------------------------------------------------------------------
# EventBuffer
# ---------------------------------------------------------------------------

class EventBuffer:
    """Per-camera rolling event buffer with timed JSON flush.

    Parameters (from ``output`` block in pipeline.yaml):
      camera_id        : identifier for this camera (default "cam_01")
      flush_interval_s : seconds between JSON flushes (default 10.0)
      json_dir         : directory to write JSON files (default "data/events_json")
      max_track_age    : frames to keep a track record after last appearance
                         (default 150 = ~5 seconds at 30fps)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = cfg or {}
        self._camera_id       = str(cfg.get("camera_id",        "cam_01"))
        self._flush_interval  = float(cfg.get("flush_interval_s", 10.0))
        self._json_dir        = Path(cfg.get("json_dir",          "data/events_json"))
        self._max_track_age   = int(cfg.get("max_track_age",      150))
        self._json_dir.mkdir(parents=True, exist_ok=True)

        self._tracks: dict[int, _TrackRecord] = {}
        self._scene_events: list[dict] = []

        # Deduplication state for persistent events
        # key: event identity string -> {"first_t": str, "dwell": float}
        self._dedup: dict[str, dict] = {}

        # Window timing
        self._window_start_t: float = time.perf_counter()
        self._window_start_iso: str = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._last_flush_t: float = time.perf_counter()

    # ------------------------------------------------------------------
    # Public: per-frame update methods
    # ------------------------------------------------------------------

    def update_track(self, track_id: int, bbox: tuple,
                     pose_quality: str = "missing",
                     attributes: dict | None = None,
                     frame_idx: int = 0) -> None:
        """Update per-track display state (call every frame for visible tracks)."""
        rec = self._tracks.setdefault(track_id, _TrackRecord(track_id=track_id))
        rec.bbox_last = tuple(int(v) for v in bbox)
        rec.pose_quality = pose_quality
        rec.last_seen = frame_idx
        if attributes:
            rec.attributes = attributes

    def append(self, event, frame_idx: int = 0) -> None:
        """Append any event object to the buffer.

        Handles:
          - core.events.Event (phase4: fire, smoke, phone, gathering, object_left, violence)
          - core.state_machine.FallEvent
          - core.identity.IdentityEvent
          - core.fight_detector.FightEvent
        """
        t_str = getattr(event, "t_iso", time.strftime("%Y-%m-%dT%H:%M:%S"))
        t_clock = time.strftime("%H:%M:%S")
        etype = getattr(event, "event_type", "UNKNOWN")

        if etype in ("FIRE", "SMOKE", "SMOKING", "PHONE", "FALL"):
            # Per-track events
            details = getattr(event, "details", {})
            if isinstance(details, dict):
                tid = details.get("track_id")
            else:
                tid = getattr(event, "track_id", None)
            if tid is None:
                tid = getattr(event, "track_id", None)

            evt_dict = {
                "type": etype,
                "t": t_clock,
                "frame_idx": frame_idx,
            }
            # add confidence if available
            conf = (details.get("confidence") or details.get("phone_conf")
                    if isinstance(details, dict) else None)
            if conf is not None:
                evt_dict["confidence"] = round(float(conf), 3)

            if tid is not None:
                rec = self._tracks.setdefault(int(tid), _TrackRecord(track_id=int(tid)))
                rec.events.append(evt_dict)
                rec.last_seen = frame_idx
            else:
                # scene-level (fire/smoke without track)
                self._scene_events.append({**evt_dict, "type": etype})

        elif etype == "GATHERING":
            details = getattr(event, "details", {})
            dedup_key = f"GATHERING_{details.get('roi_id', 'default')}"
            self._emit_scene_dedup(dedup_key, {
                "type": "GATHERING",
                "roi_id": details.get("roi_id", "default"),
                "count": details.get("count", 0),
                "track_ids": details.get("track_ids", []),
                "t": t_clock,
                "frame_idx": frame_idx,
            }, dwell_key="duration_seconds")

        elif etype == "OBJECT_LEFT":
            details = getattr(event, "details", {})
            bbox = details.get("bbox") or details.get("last_bbox")
            dedup_key = f"OBJECT_LEFT_{details.get('track_id', 'x')}"
            self._emit_scene_dedup(dedup_key, {
                "type": "OBJECT_LEFT",
                "object_class": details.get("class_name", "object"),
                "track_id": details.get("track_id"),
                "location_bbox": list(bbox) if bbox else None,
                "dwell_seconds": details.get("dwell_s", 0.0),
                "t": t_clock,
                "frame_idx": frame_idx,
            }, dwell_key="dwell_seconds")

        elif etype == "VIOLENCE":
            # Old heuristic violence — treated as low-confidence fight scene event
            details = getattr(event, "details", {})
            self._scene_events.append({
                "type": "VIOLENCE",
                "track_ids": details.get("pair", []),
                "confidence": 0.4,   # heuristic — low confidence
                "t": t_clock,
                "frame_idx": frame_idx,
                "details": {k: v for k, v in details.items()
                             if k not in ("pair",)},
            })

        elif etype == "FIGHT":
            # New skeleton-based FightEvent
            self._scene_events.append({
                "type": "FIGHT",
                "track_ids": list(getattr(event, "track_ids", [])),
                "confidence": round(float(getattr(event, "confidence", 0.0)), 3),
                "clip_ref": getattr(event, "clip_ref", None),
                "t": t_clock,
                "frame_idx": frame_idx,
                "details": getattr(event, "details", {}),
            })

        elif etype == "IDENTITY":
            details = getattr(event, "details", {})
            tid = getattr(event, "track_id", details.get("track_id"))
            if tid is not None:
                rec = self._tracks.setdefault(int(tid), _TrackRecord(track_id=int(tid)))
                rec.events.append({
                    "type": "IDENTITY",
                    "label": getattr(event, "label", None),
                    "source": getattr(event, "source", ""),
                    "similarity": round(float(getattr(event, "similarity", 0)), 3),
                    "t": t_clock,
                })
                rec.last_seen = frame_idx

    def maybe_flush(self, frame_idx: int = 0) -> dict | None:
        """Check if it's time to flush; if so, build + save JSON and return it.

        Returns the flushed dict (or None if not yet time).
        """
        t = time.perf_counter()
        if t - self._last_flush_t < self._flush_interval:
            return None
        return self._do_flush(frame_idx)

    def force_flush(self, frame_idx: int = 0) -> dict:
        """Immediately flush regardless of timer (for shutdown)."""
        return self._do_flush(frame_idx)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit_scene_dedup(self, key: str, evt: dict, dwell_key: str) -> None:
        """Emit a scene event; use 'still_present' action for repeated triggers."""
        if key in self._dedup:
            evt["action"] = "still_present"
        else:
            evt["action"] = "trigger"
            self._dedup[key] = {"first_t": evt.get("t", "")}
        self._scene_events.append(evt)

    def _do_flush(self, frame_idx: int) -> dict:
        """Build the JSON blob, write to disk, reset buffer."""
        now = time.perf_counter()
        window_end_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Build tracks list (only those with events OR recently seen)
        tracks_out = []
        stale_ids = []
        for tid, rec in self._tracks.items():
            age = frame_idx - rec.last_seen
            if age > self._max_track_age and not rec.events:
                stale_ids.append(tid)
                continue
            t_entry: dict[str, Any] = {
                "track_id": rec.track_id,
                "bbox_last": list(rec.bbox_last),
                "pose_quality": rec.pose_quality,
                "events": rec.events.copy(),   # always include, even if empty
            }
            if rec.attributes:
                t_entry["attributes"] = rec.attributes
            tracks_out.append(t_entry)

        for tid in stale_ids:
            del self._tracks[tid]

        blob: dict[str, Any] = {
            "camera_id": self._camera_id,
            "window_start": self._window_start_iso,
            "window_end": window_end_iso,
            "tracks": tracks_out,
            "scene_events": self._scene_events.copy(),
        }

        # Write JSON to disk
        ts_safe = window_end_iso.replace(":", "-").replace("T", "_")
        fname = self._json_dir / f"events_{self._camera_id}_{ts_safe}.json"
        try:
            fname.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[event_buffer] WARN: failed to write {fname}: {e}")

        # Reset for next window
        self._scene_events.clear()
        self._dedup.clear()
        for rec in self._tracks.values():
            rec.events.clear()
        self._window_start_iso = window_end_iso
        self._window_start_t = now
        self._last_flush_t = now

        return blob
