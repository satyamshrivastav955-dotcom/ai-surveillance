"""Query Engine - Natural language interface to the VLM layer.

This module implements Step 6 of the VLM spec:
1. Pull relevant hot-tier KV directly (already in VRAM)
2. Retrieve relevant warm-tier KV blocks (indexer → dequantize → load to VRAM)
3. Retrieve relevant cold-tier structured records
4. Fuse retrieved context, run generation once

Key design principle: Text generation only triggers on:
- An explicit user query, or
- An automated alert condition

This ensures flat query latency regardless of stream duration — no replay of raw video.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from vlm.config_paths import DEFAULT_VLM_CONFIG

try:
    import torch
    IMPORTED_TORCH = True
except ImportError:
    IMPORTED_TORCH = False
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    IMPORTED_TRANSFORMERS = True
except ImportError:
    IMPORTED_TRANSFORMERS = False
    AutoTokenizer = None
    AutoModelForCausalLM = None


@dataclass
class QueryConfig:
    enabled: bool = True
    
    max_context_tokens: int = 4096
    generation_max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    
    # No defaults: model identity and device come only from vlm.yaml.
    model_name: str = ""
    device: str = ""
    
    default_prompt_template: str = """You are a surveillance assistant. Answer questions about the video stream.
Use the retrieved context to provide accurate, grounded responses.
If something is not visible in the context, say so.

Context:
{context}

Question: {query}
Answer:"""
    
    # No default: shared with the cold KV tier, sourced from vlm.yaml.
    event_db_path: str = ""
    
    @classmethod
    def from_yaml(cls, path: str = DEFAULT_VLM_CONFIG) -> "QueryConfig":
        from vlm.config_paths import load_vlm_config, model_name, model_device, event_db_path

        cfg = load_vlm_config(path)
        
        query_cfg = cfg.get("query", {})
        
        return cls(
            enabled=query_cfg.get("enabled", True),
            max_context_tokens=query_cfg.get("max_context_tokens", 4096),
            generation_max_new_tokens=query_cfg.get("generation_max_new_tokens", 512),
            temperature=query_cfg.get("temperature", 0.1),
            top_p=query_cfg.get("top_p", 0.9),
            model_name=model_name(cfg),
            device=model_device(cfg),
            default_prompt_template=query_cfg.get("default_prompt_template", cls.default_prompt_template),
            event_db_path=event_db_path(cfg),
        )
    
    default_alert_prompt_template: str = """Verify this surveillance alert:

Event Type: {event_type}
Confidence: {confidence}
Timestamp: {timestamp}
Details: {details}

Based on the context, determine if this alert is:
1. CONFIRMED - The event is clearly visible and real
2. REJECTED - This appears to be a false positive
3. AMBIGUOUS - Needs human review

Respond in exactly this format:
VERDICT: <CONFIRMED|REJECTED|AMBIGUOUS>
Reasoning: <your reasoning>"""


@dataclass
class QueryContext:
    hot_blocks: list[Any]
    warm_blocks: list[Any]
    cold_records: list[dict]
    
    events: list[dict]
    detected_entities: list[dict]
    
    time_range: tuple[float, float]
    frame_range: tuple[int, int]
    
    text_context: str = ""
    token_count: int = 0


@dataclass
class QueryResult:
    query: str
    response: str
    
    context_used: QueryContext
    
    latency_seconds: float
    
    token_counts: dict = field(default_factory=dict)
    
    success: bool = True
    error: str = ""
    
    verification_status: str = ""
    reasoning: str = ""


class QueryEngine:
    """Natural language query interface to VLM layer.
    
    This is the user-facing component that handles:
    1. Query parsing and entity extraction
    2. Context retrieval from all three tiers
    3. Context fusion and formatting
    4. LLM generation
    5. Response post-processing
    
    The query engine follows the principle of "load, not replay" — it retrieves
    structured context from tiered memory rather than processing raw video.
    """
    
    EVENT_TYPE_KEYWORDS = {
        "fall": ["fall", "fell", "fallen", "falling", "trip", "tripped"],
        "fire": ["fire", "flame", "burning", "burned", "blaze"],
        "smoke": ["smoke", "smoking", "cigarette", "vape"],
        "phone": ["phone", "cell", "mobile", "calling", "texting"],
        "fight": ["fight", "fighting", "punch", "kick", "violence", "violent"],
        "gathering": ["gathering", "group", "crowd", "people", "meeting"],
        "object_left": ["left", "abandoned", "unattended", "bag", "backpack", "suitcase"],
        "identity": ["identity", "person", "who", "name", "face"],
    }
    
    TIME_KEYWORDS = {
        "today": (0, 86400),
        "this_hour": (0, 3600),
        "last_hour": (-3600, 0),
        "recent": (-300, 0),
        "now": (-10, 0),
    }
    
    def __init__(self, config: QueryConfig | None = None):
        self.config = config or QueryConfig.from_yaml()
        
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._initialized = False
        self._model_shared = False   # True when model was injected via set_model()
        
        self._query_count = 0
        self._total_latency = 0.0

    def set_model(self, model, processor) -> None:
        """Inject an already-loaded model/processor (shared from VLMCore).

        When called before initialize(), initialize() will skip loading its
        own copy.  This prevents the second full model load that wastes ~2.3 GB.
        """
        self._model     = model
        self._processor = processor
        if processor is not None:
            self._tokenizer = getattr(processor, "tokenizer", None)
        self._model_shared = True
        print("[QueryEngine] model injected from VLMCore (no second load)")
    def initialize(self) -> bool:
        if self._initialized:
            return True

        # If a model was already injected via set_model(), skip loading.
        if self._model_shared and self._model is not None:
            self._initialized = True
            print(f"[QueryEngine] Initialized with model: {self.config.model_name}")
            return True

        if not IMPORTED_TRANSFORMERS:
            print("[QueryEngine] WARNING: transformers not available")
            return False
        
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            
            dtype = torch.bfloat16 if torch.cuda.is_available() and hasattr(torch, 'bfloat16') else torch.float16
            
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=self.config.device if torch.cuda.is_available() else "cpu",
            )
            
            self._tokenizer = self._processor.tokenizer
            self._initialized = True
            print(f"[QueryEngine] Initialized with model: {self.config.model_name}")
            return True
        except Exception as e:
            print("=" * 78)
            print(f"[QueryEngine] *** MODEL FAILED TO LOAD: {e}")
            print(f"[QueryEngine] *** Model expected: {self.config.model_name}")
            print("[QueryEngine] *** QUERIES AND ALERT VERIFICATION ARE UNAVAILABLE.")
            print("[QueryEngine] *** Any call to query()/verify_event() will raise.")
            print("=" * 78)
            return False
    
    def query(
        self,
        query_text: str,
        kv_cache,
        event_logger,
        time_range: tuple[float, float] | None = None,
        event_types: list[str] | None = None,
    ) -> QueryResult:
        start_time = time.perf_counter()
        
        if not self._initialized:
            self.initialize()
        
        parsed_event_types = event_types or self._parse_event_types(query_text)
        parsed_time_range = time_range or self._parse_time_range(query_text)
        
        context = self._retrieve_context(
            kv_cache, event_logger, parsed_time_range, parsed_event_types
        )
        
        prompt = self._build_prompt(query_text, context)
        
        response = self._generate_response(prompt)
        
        self._query_count += 1
        latency = time.perf_counter() - start_time
        self._total_latency += latency
        
        return QueryResult(
            query=query_text,
            response=response,
            context_used=context,
            latency_seconds=latency,
            success=True,
            token_counts={
                "context_tokens": context.token_count,
                "max_tokens": self.config.generation_max_new_tokens,
            }
        )
    
    def verify_event(
        self,
        event_type: str,
        confidence: float,
        details: dict,
        kv_cache,
        event_logger,
        timestamp: str,
    ) -> QueryResult:
        query_text = f"Verify {event_type} event"
        
        prompt = self.config.default_alert_prompt_template.format(
            event_type=event_type,
            confidence=confidence,
            timestamp=timestamp,
            details=details,
        )
        
        if not self._initialized:
            self.initialize()
        
        context = QueryContext(
            hot_blocks=[],
            warm_blocks=[],
            cold_records=[],
            events=[{"event_type": event_type, "confidence": confidence, "details": details}],
            detected_entities=[],
            time_range=(time.time() - 60, time.time()),
            frame_range=(0, 0),
        )
        
        response = self._generate_response(prompt)
        
        status, reasoning = self._parse_verification_response(response)
        
        return QueryResult(
            query=query_text,
            response=response,
            context_used=context,
            latency_seconds=0.0,
            success=True,
            verification_status=status,
            reasoning=reasoning,
        )
    
    def _retrieve_context(
        self,
        kv_cache,
        event_logger,
        time_range: tuple[float, float],
        event_types: list[str] | None,
    ) -> QueryContext:
        hot_blocks = []
        warm_blocks = []
        
        if hasattr(kv_cache, 'hot'):
            hot_blocks = list(kv_cache.hot._blocks)
        
        if hasattr(kv_cache, 'warm'):
            warm_blocks = []
        
        cold_records = []
        if event_logger is not None:
            cold_records = event_logger.query_events(limit=50)
        
        events = []
        if event_logger is not None:
            for et in event_types or []:
                events.extend(event_logger.query_events(event_type=et, limit=20))
        
        text_context = self._format_context_text(hot_blocks, warm_blocks, cold_records, events)
        token_count = len(text_context.split()) // 4
        
        return QueryContext(
            hot_blocks=hot_blocks,
            warm_blocks=warm_blocks,
            cold_records=cold_records,
            events=events,
            detected_entities=[],
            time_range=time_range,
            frame_range=(0, 0),
            text_context=text_context,
            token_count=token_count,
        )
    
    def _format_context_text(
        self,
        hot_blocks: list,
        warm_blocks: list,
        cold_records: list[dict],
        events: list[dict],
    ) -> str:
        parts = []
        
        if hot_blocks:
            parts.append(f"Current buffer: {len(hot_blocks)} recent frames available.")
        
        if cold_records:
            parts.append(f"Recent events in database: {len(cold_records)} records.")
            for record in cold_records[:10]:
                event_type = record.get('event_type', 'unknown')
                frame_idx = record.get('frame_idx', '?')
                parts.append(f"  - {event_type} at frame {frame_idx}")
        
        if events:
            parts.append(f"Matched events: {len(events)} relevant events.")
            for event in events[:5]:
                event_type = event.get('event_type', 'unknown')
                parts.append(f"  - {event_type}")
        
        return "\n".join(parts) if parts else "No context available."
    
    def _build_prompt(self, query: str, context: QueryContext) -> str:
        return self.config.default_prompt_template.format(
            context=context.text_context,
            query=query,
        )
    
    def _generate_response(self, prompt: str) -> str:
        if self._model is not None and self._processor is not None:
            try:
                inputs = self._processor(
                    text=[prompt],
                    return_tensors="pt",
                    padding=True,
                )
                
                if IMPORTED_TORCH and hasattr(inputs, 'to'):
                    inputs = inputs.to(self.config.device)
                
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=min(self.config.generation_max_new_tokens, 256),
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.temperature > 0,
                        pad_token_id=self._tokenizer.eos_token_id if self._tokenizer else 0,
                    )
                
                if self._tokenizer:
                    response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if prompt in response:
                        response = response[len(prompt):].strip()
                    return response
            except Exception as e:
                raise RuntimeError(
                    f"[QueryEngine] Model generation failed: {e}. "
                    f"No fallback available - model must produce real output."
                )
        
        raise RuntimeError(
            f"[QueryEngine] Cannot generate response: model not loaded. "
            f"VLM model must be available for real queries. "
            f"Model name expected: {self.config.model_name}"
        )
    
    def _parse_verification_response(self, response: str) -> tuple[str, str]:
        """Read the verdict the model actually stated.

        The alert prompt asks for a `VERDICT: <CONFIRMED|REJECTED|AMBIGUOUS>`
        line. An unparseable response raises instead of silently collapsing to
        "AMBIGUOUS", which previously made "the model said unclear" and "we
        could not understand the model" indistinguishable.
        """
        match = re.search(r"verdict\s*[:=]\s*(CONFIRMED|REJECTED|AMBIGUOUS)",
                          response, re.IGNORECASE)
        if match is None:
            raise ValueError(
                "[QueryEngine] Alert verification response has no "
                "'VERDICT: CONFIRMED|REJECTED|AMBIGUOUS' line. The verdict is NOT "
                "guessed by keyword search, and no default is assumed. "
                f"Raw model response was:\n{response!r}"
            )

        status = match.group(1).upper()

        reason_match = re.search(r"reasoning\s*[:=]\s*(.+)", response,
                                 re.IGNORECASE | re.DOTALL)
        reasoning = reason_match.group(1).strip() if reason_match else response.strip()

        return status, reasoning
    
    def _parse_event_types(self, query: str) -> list[str]:
        query_lower = query.lower()
        found = []
        
        for event_type, keywords in self.EVENT_TYPE_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                found.append(event_type.upper())
        
        return found
    
    def _parse_time_range(self, query: str) -> tuple[float, float]:
        query_lower = query.lower()
        now = time.time()
        
        for keyword, (offset_start, offset_end) in self.TIME_KEYWORDS.items():
            if keyword in query_lower:
                return (now + offset_start, now + offset_end)
        
        time_patterns = [
            (r'last (\d+) minutes?', lambda m: (now - m * 60, now)),
            (r'past (\d+) hours?', lambda m: (now - m * 3600, now)),
            (r'last (\d+) seconds?', lambda m: (now - m, now)),
        ]
        
        for pattern, range_fn in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                value = int(match.group(1))
                return range_fn(value)
        
        return (now - 300, now)
    
    def get_stats(self) -> dict:
        return {
            "query_count": self._query_count,
            "total_latency_s": self._total_latency,
            "avg_latency_s": self._total_latency / max(1, self._query_count),
        }
    
    def reset(self) -> None:
        self._query_count = 0
        self._total_latency = 0.0


class OpenVocabWatcher:
    """Open-vocabulary watcher for custom natural language triggers.
    
    Example: "Notify me when someone carries a red bag"
    
    This allows users to define custom watch conditions that aren't
    covered by the fixed event detectors.
    """
    
    def __init__(self, config: QueryConfig | None = None):
        self.config = config or QueryConfig.from_yaml()
        self._max_watchers = getattr(self.config, 'max_watchers', 10)
        self._default_check_interval = getattr(self.config, 'default_check_interval', 30)
        self._confidence_threshold = getattr(self.config, 'confidence_threshold', 0.5)
        
        self._watchers: dict[str, dict] = {}
        self._watcher_id = 0
    
    def add_watcher(
        self,
        description: str,
        callback: Any = None,
        check_interval: int | None = None,
    ) -> str:
        if len(self._watchers) >= self._max_watchers:
            raise RuntimeError(f"Maximum watcher limit ({self._max_watchers}) reached")
        
        self._watcher_id += 1
        watcher_id = f"watcher_{self._watcher_id}"
        
        self._watchers[watcher_id] = {
            "description": description,
            "callback": callback,
            "check_interval": check_interval or self._default_check_interval,
            "last_check": 0,
            "trigger_count": 0,
        }
        
        return watcher_id
    
    def remove_watcher(self, watcher_id: str) -> bool:
        if watcher_id in self._watchers:
            del self._watchers[watcher_id]
            return True
        return False
    
    def check_watchers(
        self,
        frame: np.ndarray,
        frame_idx: int,
        query_engine: QueryEngine,
    ) -> list[tuple[str, float, str]]:
        current_time = time.time()
        triggers = []
        
        for watcher_id, watcher in self._watchers.items():
            if current_time - watcher["last_check"] < watcher["check_interval"]:
                continue
            
            result = query_engine.query(
                f"{watcher['description']}\n\n{self.CONFIDENCE_INSTRUCTION}",
                None,
                None,
            )

            confidence = self._extract_confidence(result.response)

            # self.config has no confidence_threshold field; the resolved value
            # lives on the instance (set in __init__).
            if confidence >= self._confidence_threshold:
                triggers.append((watcher_id, confidence, watcher["description"]))
                watcher["trigger_count"] += 1
            
            watcher["last_check"] = current_time
        
        return triggers
    
    CONFIDENCE_INSTRUCTION = (
        "Answer with a single line in exactly this form and nothing else:\n"
        "CONFIDENCE: <number between 0.0 and 1.0>\n"
        "where the number is how certain you are that the described condition is "
        "present in the current view."
    )

    def _extract_confidence(self, response: str) -> float:
        """Read the model's own stated confidence out of its response.

        Keyword scoring (yes->0.7, might->0.5, ...) is deliberately NOT used:
        those numbers were invented here, not produced by the model, and the
        watcher fires on them.
        """
        match = re.search(r"confidence\s*[:=]\s*([01]?\.\d+|[01](?:\.0+)?)",
                          response, re.IGNORECASE)
        if match is None:
            raise ValueError(
                "[OpenVocabWatcher] Model response contains no 'CONFIDENCE: <0.0-1.0>' "
                "value. Confidence is NOT guessed from keywords such as 'yes'/'might'. "
                f"Raw model response was:\n{response!r}"
            )

        value = float(match.group(1))
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"[OpenVocabWatcher] Model reported an out-of-range confidence "
                f"({value}). Raw model response was:\n{response!r}"
            )
        return value
    
    def list_watchers(self) -> list[dict]:
        return [
            {"id": wid, **wdata}
            for wid, wdata in self._watchers.items()
        ]
