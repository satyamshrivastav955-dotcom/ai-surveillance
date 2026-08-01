"""Spatial Token Pruning - ZSPAPrune-style prompt-aware token selection.

This module implements prompt-aware token pruning that:
1. Balances task-relevance (what matters for detection) with diversity (cover the frame)
2. Preserves person-related tokens with high priority
3. Uses hierarchical selection for efficiency
4. Maintains attention sinks for streaming stability

Key insight: Generic/background-only pruning doesn't know what matters. A motionless
person in a corner can get pruned. Prompt-conditioning aligns pruning to the detection goal.

Implementation based on ZSPAPrune (Zero-Shot Prompt-Aware Pruning) principles:
- Hierarchical token selection (coarse → fine)
- Task prompt guides importance scoring
- Diversity preservation prevents attention collapse
- Person-aware overrides protect detection targets
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlm.config_paths import DEFAULT_VLM_CONFIG

try:
    import torch
    import torch.nn.functional as F
    IMPORTED_TORCH = True
except ImportError:
    IMPORTED_TORCH = False
    torch = None
    F = None


@dataclass
class PruningConfig:
    method: str = "prompt_aware"
    
    diversity_weight: float = 0.3
    relevance_weight: float = 0.7
    
    hierarchical: bool = True
    preserve_attention_sinks: int = 4
    
    person_aware: bool = True
    person_token_boost: float = 2.0
    
    min_tokens: int = 32
    max_tokens: int = 512
    
    spatial_grid_size: int = 8

    # Relevance/diversity scoring is NOT implemented against real model signals
    # (attention maps / token embeddings). The placeholder scorers below are
    # positional heuristics and produce meaningless importance scores. They are
    # disabled by default and raise; set this true ONLY to knowingly run fake
    # scoring for plumbing tests.
    allow_placeholder_scoring: bool = False

    @classmethod
    def from_yaml(cls, path: str = DEFAULT_VLM_CONFIG) -> "PruningConfig":
        from vlm.config_paths import load_vlm_config
        cfg = load_vlm_config(path)
        
        pruning_cfg = cfg.get("pruning", {})
        
        return cls(
            method=pruning_cfg.get("method", "prompt_aware"),
            diversity_weight=pruning_cfg.get("diversity_weight", 0.3),
            relevance_weight=pruning_cfg.get("relevance_weight", 0.7),
            hierarchical=pruning_cfg.get("hierarchical", True),
            preserve_attention_sinks=pruning_cfg.get("preserve_attention_sinks", 4),
            person_aware=pruning_cfg.get("person_aware", True),
            person_token_boost=pruning_cfg.get("person_token_boost", 2.0),
            spatial_grid_size=pruning_cfg.get("spatial_grid_size", 8),
            allow_placeholder_scoring=pruning_cfg.get("allow_placeholder_scoring", False),
        )


@dataclass
class TokenMetadata:
    token_idx: int
    importance_score: float = 0.0
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    
    spatial_position: tuple[int, int] = (0, 0)
    is_person_related: bool = False
    is_attention_sink: bool = False
    
    visual_features: Any = None


@dataclass
class PruningResult:
    kept_indices: list[int]
    removed_indices: list[int]
    kept_tokens: Any
    kept_attention_mask: Any
    
    original_token_count: int = 0
    kept_token_count: int = 0
    pruning_ratio: float = 0.0
    
    metadata: list[TokenMetadata] = field(default_factory=list)
    person_tokens_preserved: int = 0


class SpatialTokenPruner:
    """Prompt-aware spatial token pruning for VLM vision encoders.
    
    This class implements the token pruning strategy for Step 2 of the VLM spec:
    - Ambient pass: aggressive pruning (~75% reduction)
    - Escalated pass: conservative pruning (~40-60% reduction)
    
    The key innovation is using the task prompt to guide importance scoring,
    ensuring the pruning preserves tokens that matter for the detection task.
    """
    
    PERSON_KEYWORDS = [
        "person", "people", "human", "man", "woman", "child", "body",
        "face", "hand", "standing", "sitting", "lying", "falling",
        "walking", "running", "fight", "distress", "injured",
    ]
    
    PERSON_VISUAL_TOKENS = set(range(0, 256))
    
    def __init__(self, config: PruningConfig | None = None):
        self.config = config or PruningConfig.from_yaml()
        
        self._attention_sink_positions: set[int] = set()
        self._person_token_positions: set[int] = set()
        self._placeholder_warned = False

    def prune(
        self,
        tokens: Any,
        attention_mask: Any,
        target_token_count: int,
        task_prompt: str = "",
        person_bboxes: list[tuple] | None = None,
        frame_shape: tuple[int, int, int] = (720, 1280, 3),
    ) -> PruningResult:
        if not IMPORTED_TORCH:
            return self._dummy_prune(tokens, attention_mask, target_token_count)
        
        if tokens is None:
            return PruningResult(
                kept_indices=[],
                removed_indices=[],
                kept_tokens=None,
                kept_attention_mask=None,
                original_token_count=target_token_count,
                kept_token_count=target_token_count,
            )
        
        total_tokens = tokens.shape[1] if hasattr(tokens, 'shape') else len(tokens)
        
        if total_tokens <= target_token_count:
            return PruningResult(
                kept_indices=list(range(total_tokens)),
                removed_indices=[],
                kept_tokens=tokens,
                kept_attention_mask=attention_mask,
                original_token_count=total_tokens,
                kept_token_count=total_tokens,
                pruning_ratio=0.0,
            )
        
        token_metadata = self._compute_token_importance(
            tokens, task_prompt, person_bboxes, frame_shape
        )
        
        self._mark_attention_sinks(token_metadata)
        
        if person_bboxes and self.config.person_aware:
            self._mark_person_tokens(token_metadata, person_bboxes, frame_shape, total_tokens)
        
        kept_indices, removed_indices = self._select_tokens(
            token_metadata, target_token_count, total_tokens
        )
        
        kept_tokens = self._gather_tokens(tokens, kept_indices)
        kept_attention_mask = self._gather_attention(attention_mask, kept_indices)
        
        person_preserved = sum(1 for idx in kept_indices if token_metadata[idx].is_person_related)
        
        return PruningResult(
            kept_indices=kept_indices,
            removed_indices=removed_indices,
            kept_tokens=kept_tokens,
            kept_attention_mask=kept_attention_mask,
            original_token_count=total_tokens,
            kept_token_count=len(kept_indices),
            pruning_ratio=1.0 - len(kept_indices) / total_tokens,
            metadata=token_metadata,
            person_tokens_preserved=person_preserved,
        )
    
    def _compute_token_importance(
        self,
        tokens: Any,
        task_prompt: str,
        person_bboxes: list[tuple] | None,
        frame_shape: tuple[int, int, int],
    ) -> list[TokenMetadata]:
        total_tokens = tokens.shape[1] if hasattr(tokens, 'shape') else 1
        
        metadata = []
        h, w = frame_shape[:2]
        grid_h, grid_w = self.config.spatial_grid_size, self.config.spatial_grid_size
        
        for i in range(total_tokens):
            relevance = self._compute_relevance_PLACEHOLDER_NOT_IMPLEMENTED(
                tokens, i, task_prompt)
            diversity = self._compute_diversity_PLACEHOLDER_NOT_IMPLEMENTED(
                tokens, i, total_tokens)
            
            grid_y = (i * grid_h) // max(1, total_tokens)
            grid_x = i % grid_w
            spatial_pos = (grid_x, grid_y)
            
            importance = (
                self.config.relevance_weight * relevance +
                self.config.diversity_weight * diversity
            )
            
            metadata.append(TokenMetadata(
                token_idx=i,
                importance_score=importance,
                relevance_score=relevance,
                diversity_score=diversity,
                spatial_position=spatial_pos,
            ))
        
        return metadata
    
    _PLACEHOLDER_MSG = (
        "[SpatialTokenPruner] {fn} is a PLACEHOLDER, not a real implementation. "
        "It scores tokens from their positional index and prompt keywords, NOT from "
        "model attention weights or token embeddings, so every importance score it "
        "produces is fabricated and the resulting pruning is arbitrary. Real scoring "
        "requires attention maps / hidden states from the VLM forward pass, which are "
        "not wired up. Refusing to emit fake scores. To knowingly run fake scoring for "
        "plumbing tests only, set pruning.allow_placeholder_scoring: true in vlm.yaml."
    )

    def _placeholder_guard(self, fn: str) -> None:
        """Raise unless the operator explicitly opted into fake scoring."""
        if not self.config.allow_placeholder_scoring:
            raise NotImplementedError(self._PLACEHOLDER_MSG.format(fn=fn))
        if not self._placeholder_warned:
            print(
                f"[SpatialTokenPruner] *** WARNING: PLACEHOLDER SCORING ACTIVE *** "
                f"allow_placeholder_scoring=true, so token importance scores are "
                f"FABRICATED positional heuristics, not model-derived. Pruning "
                f"decisions from this run are NOT real results."
            )
            self._placeholder_warned = True

    def _compute_relevance_PLACEHOLDER_NOT_IMPLEMENTED(
        self, tokens: Any, idx: int, task_prompt: str
    ) -> float:
        self._placeholder_guard("_compute_relevance")
        relevance = 0.5

        prompt_lower = task_prompt.lower()
        
        person_keyword_present = any(kw in prompt_lower for kw in self.PERSON_KEYWORDS)
        if person_keyword_present:
            relevance += 0.2
        
        if idx < 32:
            relevance += 0.15
        
        return min(1.0, relevance)
    
    def _compute_diversity_PLACEHOLDER_NOT_IMPLEMENTED(
        self, tokens: Any, idx: int, total_tokens: int
    ) -> float:
        self._placeholder_guard("_compute_diversity")
        if total_tokens <= 1:
            return 1.0
            
        grid_pos = (idx % self.config.spatial_grid_size)
        spatial_diversity = 1.0 - abs(grid_pos - self.config.spatial_grid_size // 2) / self.config.spatial_grid_size
        
        token_diversity = 1.0 - abs(idx - total_tokens // 2) / max(1, total_tokens)
        
        return 0.5 * spatial_diversity + 0.5 * token_diversity
    
    def _mark_attention_sinks(self, metadata: list[TokenMetadata]) -> None:
        n_sinks = self.config.preserve_attention_sinks
        total = len(metadata)
        
        for i in range(min(n_sinks, total)):
            metadata[i].is_attention_sink = True
            metadata[i].importance_score = 1.0
            self._attention_sink_positions.add(i)
    
    def _mark_person_tokens(
        self,
        metadata: list[TokenMetadata],
        person_bboxes: list[tuple],
        frame_shape: tuple[int, int, int],
        total_tokens: int,
    ) -> None:
        if not person_bboxes:
            return
            
        h, w = frame_shape[:2]
        tokens_per_row = int(math.sqrt(total_tokens))
        
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            
            token_x1 = int(x1 / w * tokens_per_row)
            token_x2 = int(x2 / w * tokens_per_row)
            token_y1 = int(y1 / h * tokens_per_row)
            token_y2 = int(y2 / h * tokens_per_row)
            
            for ty in range(token_y1, min(token_y2 + 1, tokens_per_row)):
                for tx in range(token_x1, min(token_x2 + 1, tokens_per_row)):
                    token_idx = ty * tokens_per_row + tx
                    if 0 <= token_idx < len(metadata):
                        metadata[token_idx].is_person_related = True
                        metadata[token_idx].importance_score *= self.config.person_token_boost
                        self._person_token_positions.add(token_idx)
    
    def _select_tokens(
        self,
        metadata: list[TokenMetadata],
        target_count: int,
        total_tokens: int,
    ) -> tuple[list[int], list[int]]:
        sink_indices = [m.token_idx for m in metadata if m.is_attention_sink]
        person_indices = [m.token_idx for m in metadata if m.is_person_related]
        
        forced_keep = set(sink_indices) | set(person_indices)
        
        remaining_tokens = target_count - len(forced_keep)
        remaining_tokens = max(self.config.min_tokens - len(forced_keep), remaining_tokens)
        
        other_metadata = [m for m in metadata if m.token_idx not in forced_keep]
        other_metadata.sort(key=lambda m: -m.importance_score)
        
        selected_others = [m.token_idx for m in other_metadata[:remaining_tokens]]
        
        kept = list(forced_keep) + selected_others
        removed = [i for i in range(total_tokens) if i not in kept]
        
        return kept, removed
    
    def _gather_tokens(self, tokens: Any, indices: list[int]) -> Any:
        if not IMPORTED_TORCH or tokens is None:
            return tokens
            
        try:
            if hasattr(tokens, 'index_select'):
                idx_tensor = torch.tensor(indices, device=tokens.device)
                return tokens.index_select(1, idx_tensor)
            else:
                return tokens
        except Exception:
            return tokens
    
    def _gather_attention(self, attention_mask: Any, indices: list[int]) -> Any:
        if attention_mask is None:
            return None
            
        if not IMPORTED_TORCH:
            return attention_mask
            
        try:
            if hasattr(attention_mask, 'index_select'):
                idx_tensor = torch.tensor(indices, device=attention_mask.device)
                return attention_mask.index_select(1, idx_tensor)
            else:
                return attention_mask
        except Exception:
            return attention_mask
    
    def _dummy_prune(self, tokens: Any, attention_mask: Any, target_count: int) -> PruningResult:
        total = target_count
        return PruningResult(
            kept_indices=list(range(target_count)),
            removed_indices=[],
            kept_tokens=tokens,
            kept_attention_mask=attention_mask,
            original_token_count=total,
            kept_token_count=target_count,
            pruning_ratio=0.0,
        )
    
    def get_ambient_pruning_ratio(self) -> float:
        return 0.75
    
    def get_escalated_pruning_ratio(self) -> float:
        return 0.40


class HierarchicalPruner(SpatialTokenPruner):
    """Two-stage hierarchical pruning for better efficiency.
    
    Stage 1: Coarse grid-level selection (fast, low precision)
    Stage 2: Fine token-level selection within kept regions
    
    This significantly reduces compute for large token counts.
    """
    
    def __init__(self, config: PruningConfig | None = None):
        super().__init__(config)
        self.coarse_ratio = 0.5
    
    def prune_hierarchical(
        self,
        tokens: Any,
        attention_mask: Any,
        target_count: int,
        task_prompt: str = "",
        person_bboxes: list[tuple] | None = None,
        frame_shape: tuple[int, int, int] = (720, 1280, 3),
    ) -> PruningResult:
        if not self.config.hierarchical:
            return self.prune(tokens, attention_mask, target_count, task_prompt, person_bboxes, frame_shape)
        
        coarse_result = self._coarse_stage(tokens, attention_mask, target_count, person_bboxes, frame_shape)
        
        if coarse_result.kept_token_count <= target_count:
            return coarse_result
        
        fine_result = self._fine_stage(
            coarse_result.kept_tokens,
            coarse_result.kept_attention_mask,
            target_count,
            task_prompt,
            coarse_result.metadata,
            person_bboxes,
        )
        
        final_indices = [coarse_result.kept_indices[i] for i in fine_result.kept_indices]
        
        return PruningResult(
            kept_indices=final_indices,
            removed_indices=[i for i in range(tokens.shape[1] if hasattr(tokens, 'shape') else 1) if i not in final_indices],
            kept_tokens=fine_result.kept_tokens,
            kept_attention_mask=fine_result.kept_attention_mask,
            original_token_count=coarse_result.original_token_count,
            kept_token_count=len(final_indices),
            pruning_ratio=1.0 - len(final_indices) / coarse_result.original_token_count,
            metadata=coarse_result.metadata,
        )
    
    def _coarse_stage(
        self,
        tokens: Any,
        attention_mask: Any,
        target_count: int,
        person_bboxes: list[tuple] | None,
        frame_shape: tuple[int, int, int],
    ) -> PruningResult:
        coarse_target = int(target_count / self.coarse_ratio)
        return self.prune(tokens, attention_mask, coarse_target, "", person_bboxes, frame_shape)
    
    def _fine_stage(
        self,
        tokens: Any,
        attention_mask: Any,
        target_count: int,
        task_prompt: str,
        parent_metadata: list[TokenMetadata],
        person_bboxes: list[tuple] | None,
    ) -> PruningResult:
        return self.prune(tokens, attention_mask, target_count, task_prompt, person_bboxes)
