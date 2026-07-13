"""VRAM profiler — constraint #2 in Section 6.

Run after every model addition. Loads each model in isolation, reports
peak allocated/reserved VRAM, and flags if the *combined* estimate exceeds
the 5GB headroom budget. Phase 1 only profiles the detector+tracker
pair (which share one YOLO instance).
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _gpu_stats() -> dict[str, int]:
    import torch
    return {
        "alloc_mb": int(torch.cuda.memory_allocated() / (1024 * 1024)),
        "reserved_mb": int(torch.cuda.memory_reserved() / (1024 * 1024)),
    }


def _peak(fn, iters: int = 8) -> dict[str, int]:
    import torch
    torch.cuda.reset_peak_memory_stats()
    stats = fn()
    for i in range(iters):
        stats = fn()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return {
        "steady_alloc_mb": stats["alloc_mb"],
        "steady_reserved_mb": stats["reserved_mb"],
        "peak_alloc_mb": int(peak_alloc),
        "peak_reserved_mb": int(peak_reserved),
    }


def profile_detector() -> dict:
    from core.detector import Detector
    d = Detector()
    frame = np.zeros((d.imgsz, d.imgsz, 3), dtype=np.uint8)

    def fn():
        d.detect(frame)
        return _gpu_stats()

    res = _peak(fn)
    res["model"] = "yolov8n (shared detector + tracker, FP16)"
    res["imgsz"] = d.imgsz
    res["half"] = d.half
    del d
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return res


def profile_all() -> list[dict]:
    # each phase adds an entry here. Phase 1: just the detector.
    return [profile_detector()]


def main() -> None:
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available — profiler is GPU-only.")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    rows = profile_all()
    total_peak = 0
    for r in rows:
        print("-" * 60)
        for k, v in r.items():
            print(f"  {k:24} {v}")
        total_peak += r["peak_alloc_mb"]
    print("-" * 60)
    print(f"  combined peak alloc (sum)   {total_peak} MB")
    if total_peak > 5 * 1024:
        print(f"  !!! EXCEEDS 5GB BUDGET ({total_peak / 1024:.2f}GB) — revisit model choices")
    else:
        print(f"  within 5GB budget ({total_peak / 1024:.2f}GB) — OK")


if __name__ == "__main__":
    main()