# ai-surveillance

Real-time AI security camera system — **pure-DL pipeline** as the real-time
core, with a **VLM layer** added later (Phase 6) for alert verification,
open-vocabulary descriptions, and natural-language querying.

Target hardware: **single RTX 4050 laptop GPU, 6GB VRAM.** Every model choice
is constrained by this — see the build spec in the repo's master prompt.

## Status: Phase 1 (core detection loop)

| Phase | What | Status |
|---|---|---|
| 1 | shared YOLO detector + ByteTrack + live display | **active** |
| 2 | pose / fall detection | pending |
| 3 | ReID + face recognition | pending |
| 4 | fire/smoke, smoking, phone, gathering, violence | pending |
| 5 | event bus + structured logging | pending |
| 6 | VLM layer (verifier, open-vocab watcher, NL query) | pending |

## Quick start (Phase 1)

```bash
cd ai-surveillance
python pipeline/main_loop.py                 # webcam, default 640x640 FP16
python pipeline/main_loop.py --config configs/pipeline.yaml
```

Switch source in `configs/pipeline.yaml`:
```yaml
source:
  type: file        # file | webcam | rtsp
  path: tests/clip.mp4
```

Press **q** or **ESC** to quit.

## Layout

```
configs/        models.yaml, pipeline.yaml — single source of truth
core/           detector, tracker, video_source, config helpers
pipeline/       frame_router (scheduler), main_loop
benchmarks/     vram_profile.py — run after every model addition
tests/          smoke_test.py
models/         weights (gitignored; auto-downloaded)
vlm/            Phase 6 stub
```

## Hardware-budget rules (enforced in code)

- One shared YOLO instance for *all* detection (Phase 1 reuses it for tracking).
- FP16 on CUDA (`configs/models.yaml: detector.half: true`).
- Every stage is gated by the `FrameRouter`, never scattered modulo checks.
- Every feature is toggleable in `configs/pipeline.yaml` — no code changes.
- VRAM logged every 5s to `benchmarks/vram_log.csv`; flagged above 5GB.

## Phase 1 success criteria

- ≥20 FPS sustained at 640×640 input on the RTX 4050.
- Detector + tracker VRAM < ~2GB.
- No double YOLO pass: tracker rides on the shared detector.

## Verification

```bash
python tests/smoke_test.py            # imports + one-frame inference
python benchmarks/vram_profile.py     # isolated VRAM report
python pipeline/main_loop.py          # live FPS/VRAM read on webcam
```