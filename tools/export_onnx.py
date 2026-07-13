"""Export YOLOv8n + YOLOv8n-pose to ONNX FP16 for ONNX Runtime CUDA EP.

Reproducible export. Produces:
  models/yolov8n.onnx         (detector, 640x640, FP16)
  models/yolov8n-pose.onnx    (pose, 160x160, FP16)

Run:
    python tools/export_onnx.py
    python tools/export_onnx.py --imgsz 640 640    # custom detector size
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_models_config


def export_one(weights: str, imgsz: int, half: bool, opset: int = 12) -> str:
    from ultralytics import YOLO
    print(f"exporting {weights}  imgsz={imgsz}  half={half}  opset={opset}")
    model = YOLO(weights)
    # ultralytics' export writes <stem>.onnx next to the .pt
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        opset=opset,
        dynamic=False,        # fixed shape for max TRT/ORT throughput
        simplify=True,        # onnx-simplifier pass — smaller graph, fewer nodes
    )
    print(f"  -> {out}")
    return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, nargs=2, default=None,
                   help="override detector imgsz as W H (e.g. --imgsz 640 640)")
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()

    cfg = load_models_config()
    d = cfg["detector"]
    p_cfg = cfg["pose"]

    # export detector
    det_imgsz = args.imgsz[0] if args.imgsz else d["imgsz"]
    export_one(d["weights"], det_imgsz, d["half"], opset=args.opset)

    # export pose
    export_one(p_cfg["weights"], p_cfg["imgsz"], p_cfg["half"], opset=args.opset)

    print("\nexports complete. Set `runtime: onnx` in configs/models.yaml "
          "for both detector and pose to switch inference to ONNX Runtime.")


if __name__ == "__main__":
    main()