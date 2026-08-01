"""Identity fusion.

When a face match succeeds, propagate that identity label onto the current
track ID so the track keeps its identity even when the face isn't visible
in later frames. When ReID re-links a new track to a lost track, propagate
the old track's identity to the new one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.events import BaseEvent
from core.face import FaceMatch
from core.reid import ReIDMatch


@dataclass
class IdentityEvent(BaseEvent):
    """Emitted when a track's identity changes (face match, re-link, or loss)."""
    track_id: int = 0
    label: str | None = None
    source: str = ""
    similarity: float = 0.0
    matched_track_id: int | None = None

    def __init__(
        self,
        track_id: int = 0,
        label: str | None = None,
        source: str = "",
        similarity: float = 0.0,
        matched_track_id: int | None = None,
        t_iso: str = "",
        frame_idx: int = 0,
    ):
        super().__init__(
            event_type="IDENTITY",
            t_iso=t_iso,
            frame_idx=frame_idx,
            details={
                "track_id": track_id,
                "label": label,
                "source": source,
                "similarity": round(similarity, 3),
                "matched_track_id": matched_track_id,
            },
        )
        self.track_id = track_id
        self.label = label
        self.source = source
        self.similarity = similarity
        self.matched_track_id = matched_track_id


class IdentityManager:
    """Maps track_id → identity label, with fusion from face + ReID signals.

    Priority: face match > ReID re-link. Once a track has a face-confirmed
    identity, ReID can't override it (face is the stronger signal). ReID
    can *propagate* a face-confirmed identity to a re-linked track, but can't
    assign a new label on its own.
    """

    def __init__(self):
        # track_id -> {label, source, confirmed_at}
        self._identities: dict[int, dict] = {}

    def on_face_match(self, match: FaceMatch, t: float | None = None,
                      frame_idx: int = 0) -> IdentityEvent | None:
        """Called when a face recognition match is found for a track.

        Returns an IdentityEvent if the track's identity changed, else None.
        """
        if t is None:
            t = time.perf_counter()
        if match.name is None:
            return None
        tid = match.track_id
        existing = self._identities.get(tid)
        if existing and existing["source"] == "face" and existing["label"] == match.name:
            existing["confirmed_at"] = t
            return None
        self._identities[tid] = {
            "label": match.name,
            "source": "face",
            "confirmed_at": t,
            "similarity": match.similarity,
        }
        return IdentityEvent(
            track_id=tid,
            label=match.name,
            source="face",
            similarity=match.similarity,
            t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
            frame_idx=frame_idx,
        )

    def on_reid_relink(self, match: ReIDMatch, t: float | None = None,
                       frame_idx: int = 0) -> IdentityEvent | None:
        """Called when ReID re-links a new track to a lost track.

        Propagates the lost track's identity to the new track, but only if
        the new track doesn't already have a face-confirmed identity (face
        wins over ReID).
        """
        if t is None:
            t = time.perf_counter()
        if match.matched_track_id is None:
            return None
        old_id = match.matched_track_id
        new_id = match.new_track_id
        old = self._identities.get(old_id)
        if old is None or old["label"] is None:
            return None
        existing = self._identities.get(new_id)
        if existing and existing["source"] == "face":
            return None
        self._identities[new_id] = {
            "label": old["label"],
            "source": "reid",
            "confirmed_at": t,
            "similarity": match.similarity,
        }
        return IdentityEvent(
            track_id=new_id,
            label=old["label"],
            source="reid",
            similarity=match.similarity,
            matched_track_id=old_id,
            t_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
            frame_idx=frame_idx,
        )

    def on_track_lost(self, track_id: int) -> IdentityEvent | None:
        """Called when a track disappears. Does NOT clear the identity — it
        may be needed for ReID re-linking later. The identity is GC'd when
        the ReID index prunes the lost entry."""
        # just note it; we keep the identity for potential re-link propagation
        return None

    def get_label(self, track_id: int) -> str | None:
        entry = self._identities.get(track_id)
        return entry["label"] if entry else None

    def get_source(self, track_id: int) -> str | None:
        entry = self._identities.get(track_id)
        return entry["source"] if entry else None

    def reset(self) -> None:
        self._identities.clear()