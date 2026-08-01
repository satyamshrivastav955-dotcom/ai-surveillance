"""Single source of truth for VLM configuration loading.

Every VLM submodule imports from here instead of repeating a default path, a
model name, a device string or a database path. Three problems this fixes:

1. Three different default config paths existed ("vlm.yaml", "configs/vlm.yaml"),
   so which file loaded depended on which entry point you came through.
2. Each `from_yaml` guarded with `Path(path).exists()`, which resolves against
   the *current working directory*. From the repo root `Path("vlm.yaml")` does
   not exist, so the guard silently produced `cfg = {}` and every setting fell
   back to a hardcoded default. configs/vlm.yaml was never read at all.
3. Model name / device / event-db path were duplicated as literals across
   files, so "fixing" one left the others pointing somewhere else.

`load_vlm_config()` resolves through core.config.load_yaml (which anchors
relative paths at configs/) and raises if the file is missing, rather than
degrading to an empty dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import CONFIG_DIR, load_yaml

# The one default path. Relative names resolve against configs/.
DEFAULT_VLM_CONFIG = "vlm.yaml"

_cache: dict[str, dict[str, Any]] = {}


def load_vlm_config(path: str | Path = DEFAULT_VLM_CONFIG) -> dict[str, Any]:
    """Load and cache the VLM config, raising if it is not present.

    Never returns {} for a missing file: an empty config means every VLM
    setting silently becomes a hardcoded default, which is exactly the failure
    this module exists to prevent.
    """
    key = str(path)
    if key in _cache:
        return _cache[key]

    p = Path(path)
    resolved = p if p.is_absolute() else CONFIG_DIR / p
    if not resolved.exists():
        raise FileNotFoundError(
            f"[vlm] VLM config not found at {resolved}. Every VLM setting "
            f"(model name, device, prompts, thresholds, cache paths) comes from "
            f"this file. Refusing to continue with hardcoded defaults, which "
            f"would look like a working VLM while ignoring your configuration."
        )

    cfg = load_yaml(path) or {}
    _cache[key] = cfg
    return cfg


def require(cfg: dict[str, Any], dotted_key: str, what: str) -> Any:
    """Fetch a required config value or raise naming exactly what is missing."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"[vlm] Required config key '{dotted_key}' is missing from "
                f"{CONFIG_DIR / DEFAULT_VLM_CONFIG}. It supplies {what}. "
                f"Refusing to substitute a guessed value."
            )
        node = node[part]
    return node


def model_name(cfg: dict[str, Any]) -> str:
    """The VLM model id. Config-only — no hardcoded fallback."""
    return str(require(cfg, "model.name", "the VLM model to load"))


def model_device(cfg: dict[str, Any]) -> str:
    """The device the VLM runs on. Config-only — no hardcoded fallback."""
    return str(require(cfg, "model.device", "the device the VLM runs on"))


def event_db_path(cfg: dict[str, Any] | None = None) -> str:
    """Path to the events database — single source of truth is pipeline.yaml.

    Previously this read kv_cache.cold.event_db_path from vlm.yaml, which
    could silently diverge from configs/pipeline.yaml perf.event_db_path.
    Both files are now consistent: vlm.yaml defers to pipeline.yaml.

    The ``cfg`` argument is accepted for backward compatibility but is not used
    for the path — the path always comes from pipeline.yaml.
    """
    try:
        from core.config import load_pipeline_config
        return str(
            load_pipeline_config().get("perf", {}).get("event_db_path", "data/events.db")
        )
    except Exception:
        return "data/events.db"

