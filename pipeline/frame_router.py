"""FrameRouter — the single scheduler every stage queries (constraint #4).

Phase 1 only activates `detect` and `track` (both every frame). But the
scheduler is already built generically: adding Phase 2+ stages is a config
edit in pipeline.yaml (`router.stages.<name>`) plus a stage handler in
main_loop — no scattered `if frame_count % N == 0` anywhere.
"""
from __future__ import annotations

from typing import Any


class FrameRouter:
    """Config-driven scheduler. `should_run(stage, frame_idx)` is the only API
    the hot loop calls."""

    def __init__(self, router_cfg: dict[str, Any]):
        stages = router_cfg.get("stages", {})
        # normalize: both shorthand `every: N` and full `{enabled, every}` dicts
        self._table: dict[str, tuple[bool, int]] = {}
        for name, spec in stages.items():
            if isinstance(spec, dict):
                enabled = bool(spec.get("enabled", True))
                every = int(spec.get("every", 1))
            else:
                enabled = True
                every = int(spec)
            self._table[name] = (enabled, max(1, every))

    def is_enabled(self, stage: str) -> bool:
        return self._table.get(stage, (False, 1))[0]

    def every(self, stage: str) -> int:
        return self._table.get(stage, (False, 1))[1]

    def should_run(self, stage: str, frame_idx: int) -> bool:
        enabled, every = self._table.get(stage, (False, 1))
        if not enabled:
            return False
        if every == 1:
            return True
        return (frame_idx % every) == 0

    def active_stages(self) -> list[str]:
        return [s for s, (e, _) in self._table.items() if e]

    def __repr__(self) -> str:
        return f"FrameRouter(active={self.active_stages()})"