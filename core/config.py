"""Console / entrypoint config helpers shared across the pipeline."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import yaml

# Ultralytics 8.4 emits a per-call logging.WARNING about the `half` kwarg being
# deprecated in favor of `quantize`. FP16 still works through `half`; the
# alternative kwarg only exists in newer builds. We suppress the noise so
# logs/CSV stay readable. Re-enable by removing this filter rule.
logging.getLogger("ultralytics").addFilter(
    lambda r: "'half' is deprecated" not in r.getMessage()
)
# pynvml (used by phase2_static_bench for VRAM in onnx_direct mode) emits a
# FutureWarning on import. Silence it for the same reason.
warnings.filterwarnings("ignore", message=".*The pynvml package is deprecated.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = CONFIG_DIR / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models_config() -> dict[str, Any]:
    return load_yaml("models.yaml")


def load_pipeline_config() -> dict[str, Any]:
    return load_yaml("pipeline.yaml")