"""Pedestrian Attribute Recognition Aggregator (Phase 5B).

Maintains a per-track rolling window of pedestrian attributes and aggregates
them using majority voting to eliminate frame-to-frame flickering.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any

class PARAggregator:
    """Aggregates frame-level pedestrian attributes over a rolling window."""

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        # track_id -> deque of attribute dicts
        self._history: dict[int, deque[dict[str, Any]]] = {}

    def update(self, track_id: int, attributes: dict[str, Any]) -> dict[str, Any]:
        """Insert a new attribute measurement and return the aggregated record."""
        if not attributes:
            return {}

        history = self._history.setdefault(track_id, deque(maxlen=self.window_size))
        history.append(attributes)

        # Aggregate categorical attributes via majority vote (mode)
        agg: dict[str, Any] = {}
        keys = ["gender", "hair_length", "upper_color", "upper_type", "lower_color", "lower_type"]
        for key in keys:
            vals = [h[key] for h in history if key in h]
            if vals:
                agg[key] = Counter(vals).most_common(1)[0][0]
            else:
                agg[key] = "unknown"

        # Aggregate list attributes (accessories)
        # An accessory is confirmed if it appears in >= 50% of the history frames
        accessory_counts: Counter = Counter()
        valid_frames = 0
        for h in history:
            acc_list = h.get("accessories", [])
            if isinstance(acc_list, list):
                valid_frames += 1
                for acc in acc_list:
                    accessory_counts[acc] += 1

        agg_accessories = []
        if valid_frames > 0:
            threshold = valid_frames / 2.0
            for acc, count in accessory_counts.items():
                if count >= threshold:
                    agg_accessories.append(acc)
        agg["accessories"] = agg_accessories

        return agg

    def get_attributes(self, track_id: int) -> dict[str, Any]:
        """Return the current aggregated attributes for a track, or empty dict."""
        history = self._history.get(track_id)
        if not history:
            return {}
        agg: dict[str, Any] = {}
        keys = ["gender", "hair_length", "upper_color", "upper_type", "lower_color", "lower_type"]
        for key in keys:
            vals = [h[key] for h in history if key in h]
            if vals:
                agg[key] = Counter(vals).most_common(1)[0][0]
            else:
                agg[key] = "unknown"

        accessory_counts = Counter()
        valid_frames = 0
        for h in history:
            acc_list = h.get("accessories", [])
            if isinstance(acc_list, list):
                valid_frames += 1
                for acc in acc_list:
                    accessory_counts[acc] += 1

        agg_accessories = []
        if valid_frames > 0:
            threshold = valid_frames / 2.0
            for acc, count in accessory_counts.items():
                if count >= threshold:
                    agg_accessories.append(acc)
        agg["accessories"] = agg_accessories
        return agg

    def prune(self, active_track_ids: set[int]) -> None:
        """Remove history for tracks that are no longer active."""
        for tid in list(self._history.keys()):
            if tid not in active_track_ids:
                self._history.pop(tid, None)

