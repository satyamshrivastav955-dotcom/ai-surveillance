"""Quick GPU diagnostic — prints CUDA availability, where the detector's
parameters live, and current allocated/reserved VRAM.

Run idle (no warm model in the process) and immediately after fps_bench.py
to see how much VRAM the model + CUDA context actually hold.

    python benchmarks/gpu_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_models_config


def _print_header(label: str) -> None:
    print(f"=== {label} ===")


def _print_raw() -> None:
    print(f"torch.cuda.is_available()        : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    print(f"torch.cuda.memory_allocated()     : {torch.cuda.memory_allocated() / (1024*1024):.1f} MB")
    print(f"torch.cuda.memory_reserved()      : {torch.cuda.memory_reserved() / (1024*1024):.1f} MB")


def _print_with_detector() -> None:
    from core.detector import Detector
    cfg = load_models_config()
    d = Detector()
    param = next(d.model.model.parameters())
    print(f"next(model.model.parameters()).device : {param.device}")
    print(f"param.dtype                            : {param.dtype}")
    print(f"torch.cuda.memory_allocated()          : {torch.cuda.memory_allocated() / (1024*1024):.1f} MB")
    print(f"torch.cuda.memory_reserved()           : {torch.cuda.memory_reserved() / (1024*1024):.1f} MB")
    print(f"torch.cuda.max_memory_allocated()      : {torch.cuda.max_memory_allocated() / (1024*1024):.1f} MB")
    print(f"(detector.imgsz, half)                 : {d.imgsz}, {d.half}")


def main() -> None:
    _print_header("idle (no model loaded yet)")
    _print_raw()
    print()
    _print_header("after constructing Detector")
    _print_with_detector()


if __name__ == "__main__":
    main()