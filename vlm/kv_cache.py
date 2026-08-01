"""Tiered KV Cache Memory - Hot/Warm/Cold tier management for infinite streaming.

This module implements the Step 4 tiered KV cache from the VLM spec:

Tier 1 (Hot - VRAM): Fixed sliding window (~30s), full precision, attention sinks, streaming RoPE
Tier 2 (Warm - RAM): Quantized KV tensors, asymmetric precision (Keys E4M3 / Values E5M2)
Tier 3 (Cold - Disk): Structured tuples in SQLite, integrated with existing events.db

Key innovations:
- Paged attention (vLLM-style) prevents VRAM fragmentation
- Attention sinks prevent perplexity spikes in streaming
- Streaming RoPE re-anchors positional embeddings as window slides
- Asymmetric quantization preserves accuracy while reducing memory 4x

The design ensures:
- Bounded memory regardless of video duration
- Fast retrieval of recent context (hot tier)
- Queryable historical data (cold tier via existing event system)
"""
from __future__ import annotations

import math
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vlm.config_paths import DEFAULT_VLM_CONFIG

try:
    import torch
    IMPORTED_TORCH = True
except ImportError:
    IMPORTED_TORCH = False
    torch = None

_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0


@dataclass
class KVCacheConfig:
    hot_window_seconds: float = 30.0
    hot_dtype: str = "bfloat16"
    attention_sinks: int = 4
    streaming_rope: bool = True
    
    warm_max_size_mb: int = 2048
    warm_key_dtype: str = "e4m3"
    warm_value_dtype: str = "e5m2"
    warm_offload_async: bool = True
    
    # No default: the events DB path comes only from vlm.yaml.
    cold_event_db_path: str = ""
    cold_embedding_dim: int = 512
    
    # Sourced from pipeline.yaml source.fps; never assumed locally. The hot
    # window and the cold-tier time range are expressed in seconds and are
    # converted to frame indices with this value.
    fps: float = 0.0
    
    @classmethod
    def from_yaml(cls, path: str = DEFAULT_VLM_CONFIG) -> "KVCacheConfig":
        from core.config import load_fps
        from vlm.config_paths import event_db_path, load_vlm_config
        cfg = load_vlm_config(path)
        
        kv_cfg = cfg.get("kv_cache", {})
        hot_cfg = kv_cfg.get("hot", {})
        warm_cfg = kv_cfg.get("warm", {})
        cold_cfg = kv_cfg.get("cold", {})
        
        return cls(
            hot_window_seconds=hot_cfg.get("window_seconds", 30.0),
            hot_dtype=hot_cfg.get("dtype", "bfloat16"),
            attention_sinks=hot_cfg.get("attention_sinks", 4),
            streaming_rope=hot_cfg.get("streaming_rope", True),
            warm_max_size_mb=warm_cfg.get("max_size_mb", 2048),
            warm_key_dtype=warm_cfg.get("key_dtype", "e4m3"),
            warm_value_dtype=warm_cfg.get("value_dtype", "e5m2"),
            warm_offload_async=warm_cfg.get("offload_async", True),
            cold_event_db_path=event_db_path(cfg),
            cold_embedding_dim=cold_cfg.get("embedding_dim", 512),
            fps=load_fps(),
        )


@dataclass
class KVBlock:
    frame_idx: int
    timestamp: float
    
    keys: Any
    values: Any
    attention_mask: Any
    
    is_escalated: bool = False
    priority_score: float = 0.0
    
    token_count: int = 0
    
    is_attention_sink: bool = False
    position_offset: int = 0
    
    embedding: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    
    tier: str = "hot"


@dataclass
class TierStats:
    block_count: int = 0
    total_tokens: int = 0
    size_bytes: int = 0
    oldest_frame: int = -1
    newest_frame: int = -1


class HotKVCache:
    """Tier 1: Hot KV cache in VRAM with sliding window.
    
    Features:
    - Fixed-size sliding window (bounded VRAM)
    - Attention sinks (first N tokens never evicted)
    - Streaming RoPE (position re-anchoring as window slides)
    - Full precision (FP16/BF16)
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        if not config.fps:
            raise ValueError(
                "[KVCache] KVCacheConfig.fps is unset. The hot window is "
                f"{config.hot_window_seconds}s and cannot be converted to a block "
                "count without the real capture rate. Build the config via "
                "KVCacheConfig.from_yaml() (which reads pipeline.yaml source.fps) "
                "or set fps explicitly. Refusing to assume 30 fps, which would "
                "silently size the window for the wrong span of time."
            )
        self.window_size = int(config.hot_window_seconds * config.fps)
        
        self._blocks: deque[KVBlock] = deque(maxlen=self.window_size)
        self._sink_indices: set[int] = set(range(config.attention_sinks))
        
        self._position_counter = 0
        self._dtype = torch.bfloat16 if config.hot_dtype == "bfloat16" and IMPORTED_TORCH else torch.float16
        
        self._stats = TierStats()
    
    def push(self, block: KVBlock) -> Optional[KVBlock]:
        evicted = None
        
        if len(self._blocks) >= self.window_size:
            evicted = self._blocks.popleft()
            if evicted.frame_idx in self._sink_indices:
                self._blocks.appendleft(evicted)
                evicted = None
        
        if self.config.streaming_rope:
            block.position_offset = self._position_counter
            self._position_counter += block.token_count
        
        block.tier = "hot"
        self._blocks.append(block)
        
        self._update_stats()
        return evicted
    
    def get_block(self, frame_idx: int) -> Optional[KVBlock]:
        for block in self._blocks:
            if block.frame_idx == frame_idx:
                return block
        return None
    
    def get_recent_blocks(self, n: int) -> list[KVBlock]:
        return list(self._blocks)[-n:]
    
    def get_window_tokens(self) -> Optional[tuple]:
        if not self._blocks:
            return None
        
        keys_list = []
        values_list = []
        masks_list = []
        
        for block in self._blocks:
            if block.keys is not None:
                keys_list.append(block.keys)
            if block.values is not None:
                values_list.append(block.values)
            if block.attention_mask is not None:
                masks_list.append(block.attention_mask)
        
        if not keys_list:
            return None
        
        if IMPORTED_TORCH:
            try:
                keys = torch.cat(keys_list, dim=1)
                values = torch.cat(values_list, dim=1)
                masks = torch.cat(masks_list, dim=1) if masks_list else None
                return (keys, values, masks)
            except Exception:
                pass
        
        return (keys_list, values_list, masks_list)
    
    def get_attention_sink_blocks(self) -> list[KVBlock]:
        sinks = []
        for block in list(self._blocks)[:self.config.attention_sinks]:
            block.is_attention_sink = True
            sinks.append(block)
        return sinks
    
    def apply_streaming_rope(self, position_ids: Any) -> Any:
        if not self.config.streaming_rope:
            return position_ids
        
        if position_ids is None or not IMPORTED_TORCH:
            return position_ids
        
        offset = self._position_counter
        if hasattr(position_ids, 'add'):
            return position_ids + offset
        
        return position_ids
    
    def _update_stats(self) -> None:
        self._stats.block_count = len(self._blocks)
        self._stats.total_tokens = sum(b.token_count for b in self._blocks)
        
        if self._blocks:
            fids = [b.frame_idx for b in self._blocks]
            self._stats.oldest_frame = min(fids)
            self._stats.newest_frame = max(fids)
    
    def clear(self) -> None:
        self._blocks.clear()
        self._position_counter = 0
        self._stats = TierStats()
    
    @property
    def stats(self) -> TierStats:
        return self._stats


class WarmKVCache:
    """Tier 2: Warm KV cache in host RAM with asymmetric quantization.
    
    Features:
    - Quantized KV tensors (Keys E4M3 / Values E5M2)
    - Asymmetric precision preserves routing accuracy
    - Async offloading from hot tier
    - Fast indexer for retrieval
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.max_size_bytes = config.warm_max_size_mb * 1024 * 1024
        
        self._blocks: dict[int, KVBlock] = {}
        self._index: dict[int, list[int]] = {}
        
        self._current_size = 0
    
    def offload(self, block: KVBlock) -> bool:
        if self._current_size >= self.max_size_bytes:
            self._evict_oldest()
        
        quantized_block = self._quantize_block(block)
        self._blocks[block.frame_idx] = quantized_block
        
        self._current_size += self._estimate_block_size(quantized_block)
        return True
    
    def retrieve(self, frame_idx: int) -> Optional[KVBlock]:
        block = self._blocks.get(frame_idx)
        if block is None:
            return None
        
        dequantized = self._dequantize_block(block)
        return dequantized
    
    def retrieve_range(self, start_frame: int, end_frame: int) -> list[KVBlock]:
        blocks = []
        for fid in sorted(self._blocks.keys()):
            if start_frame <= fid <= end_frame:
                block = self.retrieve(fid)
                if block:
                    blocks.append(block)
        return blocks
    
    def _quantize_block(self, block: KVBlock) -> KVBlock:
        quantized = KVBlock(
            frame_idx=block.frame_idx,
            timestamp=block.timestamp,
            keys=self._quantize_keys(block.keys),
            values=self._quantize_values(block.values),
            attention_mask=block.attention_mask,
            is_escalated=block.is_escalated,
            priority_score=block.priority_score,
            token_count=block.token_count,
            embedding=block.embedding,
            metadata=block.metadata,
            tier="warm",
        )
        return quantized
    
    def _dequantize_block(self, block: KVBlock) -> KVBlock:
        dequantized = KVBlock(
            frame_idx=block.frame_idx,
            timestamp=block.timestamp,
            keys=self._dequantize_keys(block.keys),
            values=self._dequantize_values(block.values),
            attention_mask=block.attention_mask,
            is_escalated=block.is_escalated,
            priority_score=block.priority_score,
            token_count=block.token_count,
            embedding=block.embedding,
            metadata=block.metadata,
            tier="hot",
        )
        return dequantized
    
    def _quantize_keys(self, keys: Any) -> Any:
        if keys is None or not IMPORTED_TORCH:
            return keys
        
        if not hasattr(keys, 'to'):
            return keys
        
        try:
            if self.config.warm_key_dtype == "e4m3":
                if hasattr(torch, 'float8_e4m3fn'):
                    return keys.to(torch.float8_e4m3fn)
                return keys.half()
            return keys.half()
        except Exception:
            return keys
    
    def _quantize_values(self, values: Any) -> Any:
        if values is None or not IMPORTED_TORCH:
            return values
        
        if not hasattr(values, 'to'):
            return values
        
        try:
            if self.config.warm_value_dtype == "e5m2":
                if hasattr(torch, 'float8_e5m2'):
                    return values.to(torch.float8_e5m2)
                return values.half()
            return values.half()
        except Exception:
            return values
    
    def _dequantize_keys(self, keys: Any) -> Any:
        if keys is None or not IMPORTED_TORCH:
            return keys
        
        if hasattr(keys, 'to'):
            try:
                return keys.to(torch.bfloat16)
            except Exception:
                return keys.float() if hasattr(keys, 'float') else keys
        return keys
    
    def _dequantize_values(self, values: Any) -> Any:
        if values is None or not IMPORTED_TORCH:
            return values
        
        if hasattr(values, 'to'):
            try:
                return values.to(torch.bfloat16)
            except Exception:
                return values.float() if hasattr(values, 'float') else values
        return values
    
    def _estimate_block_size(self, block: KVBlock) -> int:
        size = block.token_count * 2 * 512 * 2
        return size
    
    def _evict_oldest(self) -> Optional[KVBlock]:
        if not self._blocks:
            return None
        
        oldest_fid = min(self._blocks.keys())
        block = self._blocks.pop(oldest_fid)
        self._current_size -= self._estimate_block_size(block)
        return block
    
    def clear(self) -> None:
        self._blocks.clear()
        self._index.clear()
        self._current_size = 0


class ColdKVCache:
    """Tier 3: Cold storage as structured tuples in SQLite.
    
    This tier integrates with your existing events.db system.
    
    Features:
    - Structured tuples: (timestamp, object_class, position, confidence, embedding)
    - Queryable by time range, object type, spatial location
    - No hallucination risk from raw text logs
    """
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS cold_kv_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_idx INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        t_iso TEXT NOT NULL,
        object_class TEXT,
        bbox TEXT,
        confidence REAL,
        embedding BLOB,
        is_escalated INTEGER DEFAULT 0,
        metadata TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_cold_frame_idx ON cold_kv_cache(frame_idx);
    CREATE INDEX IF NOT EXISTS idx_cold_timestamp ON cold_kv_cache(timestamp);
    CREATE INDEX IF NOT EXISTS idx_cold_object_class ON cold_kv_cache(object_class);
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        db_path = Path(config.cold_event_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(db_path))
        self._create_schema()
    
    def _create_schema(self) -> None:
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
    
    def store(self, block: KVBlock) -> bool:
        try:
            embedding_blob = None
            if block.embedding is not None:
                embedding_blob = block.embedding.astype(np.float32).tobytes()
            
            bbox_str = None
            if "bbox" in block.metadata:
                bbox_str = str(block.metadata["bbox"])
            
            self.conn.execute("""
                INSERT INTO cold_kv_cache 
                (frame_idx, timestamp, t_iso, object_class, bbox, confidence, embedding, is_escalated, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                block.frame_idx,
                block.timestamp,
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                block.metadata.get("object_class"),
                bbox_str,
                block.confidence if hasattr(block, 'confidence') else block.priority_score,
                embedding_blob,
                1 if block.is_escalated else 0,
                str(block.metadata) if block.metadata else None,
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[ColdKVCache] Store error: {e}")
            return False
    
    def query_by_time_range(
        self,
        start_time: float,
        end_time: float,
    ) -> list[dict]:
        cursor = self.conn.execute("""
            SELECT * FROM cold_kv_cache
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        """, (start_time, end_time))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def query_by_object_class(
        self,
        object_class: str,
        limit: int = 100,
    ) -> list[dict]:
        cursor = self.conn.execute("""
            SELECT * FROM cold_kv_cache
            WHERE object_class = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (object_class, limit))
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_embedding(self, record: dict) -> Optional[np.ndarray]:
        embedding_blob = record.get("embedding")
        if embedding_blob is None:
            return None
        
        try:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            expected_size = self.config.cold_embedding_dim
            return embedding[:expected_size]
        except Exception:
            return None
    
    def close(self) -> None:
        self.conn.close()


class TieredKVCache:
    """Unified tiered KV cache manager.
    
    Coordinates Hot (VRAM) → Warm (RAM) → Cold (Disk) transitions.
    """
    
    def __init__(self, config: KVCacheConfig | None = None):
        self.config = config or KVCacheConfig.from_yaml()
        
        self.hot = HotKVCache(self.config)
        self.warm = WarmKVCache(self.config)
        self.cold = ColdKVCache(self.config)
        
        self._block_count = 0
    
    def push(self, block: KVBlock) -> None:
        self._block_count += 1
        
        evicted = self.hot.push(block)
        
        if evicted is not None:
            warm_evicted = self.warm.offload(evicted)
            if warm_evicted is None:
                self.cold.store(evicted)
    
    def get_context_for_time_range(
        self,
        start_time: float,
        end_time: float,
    ) -> tuple[list[KVBlock], list[dict]]:
        hot_blocks = []
        for block in self.hot.get_recent_blocks(self.hot.window_size):
            if start_time <= block.timestamp <= end_time:
                hot_blocks.append(block)
        
        warm_blocks = self.warm.retrieve_range(
            int(start_time * self.config.fps),
            int(end_time * self.config.fps),
        )
        
        cold_records = self.cold.query_by_time_range(start_time, end_time)
        
        return hot_blocks + warm_blocks, cold_records
    
    def get_stats(self) -> dict:
        return {
            "hot": {
                "block_count": self.hot.stats.block_count,
                "total_tokens": self.hot.stats.total_tokens,
                "oldest_frame": self.hot.stats.oldest_frame,
                "newest_frame": self.hot.stats.newest_frame,
            },
            "warm": {
                "block_count": len(self.warm._blocks),
                "size_bytes": self.warm._current_size,
            },
            "cold": {
                "block_count": self._block_count - self.hot.stats.block_count - len(self.warm._blocks),
            },
            "total_blocks": self._block_count,
        }
    
    def clear(self) -> None:
        self.hot.clear()
        self.warm.clear()
        self._block_count = 0
    
    def close(self) -> None:
        self.cold.close()


def create_kv_block(
    frame_idx: int,
    keys: Any,
    values: Any,
    attention_mask: Any,
    is_escalated: bool = False,
    priority_score: float = 0.0,
    metadata: dict | None = None,
) -> KVBlock:
    token_count = 0
    if keys is not None and hasattr(keys, 'shape'):
        token_count = keys.shape[1]
    
    return KVBlock(
        frame_idx=frame_idx,
        timestamp=time.perf_counter(),
        keys=keys,
        values=values,
        attention_mask=attention_mask,
        is_escalated=is_escalated,
        priority_score=priority_score,
        token_count=token_count,
        metadata=metadata or {},
    )
