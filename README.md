# ai-surveillance

Real-time AI security camera system — **pure deep-learning pipeline** as the real-time
core, with a **VLM layer** planned for Phase 6 (alert verification, open-vocabulary
descriptions, natural-language querying).

**Target hardware:** RTX 4050 Laptop GPU, 6 GB VRAM. Every model choice and config
value is constrained by this budget — ~70–100 MB VRAM used across all active models (<2% of budget).

---

## Status: Phase 4 Complete — Pre-VLM Baseline

| Phase | What | Status |
|---|---|---|
| 1 | Shared YOLO detector + ByteTrack + live display | ✅ complete |
| 2 | YOLOv8n-pose on crops + rule-based fall detector | ✅ complete |
| ONNX pass | onnx_direct for pose — 46% overhead reduction | ✅ complete |
| 3 | ReID (ResNet18) + face recognition (SCRFD/MobileFaceNet) + identity fusion | ✅ complete |
| 4 | Fire/smoke · smoking · phone · gathering · violence · object-left detection | ✅ complete |
| — | Motion prefilter (frame differencing gates heavy stages) | ✅ complete |
| — | Structured event logging (SQLite + keyframes) | ✅ complete |
| 5 | VLM layer — alert verifier, open-vocab watcher, NL query engine | pending |

---

## Quick start

```bash
# run the full live pipeline (all features enabled)
python -m pipeline.main_loop

# run all tests (42 tests across 7 suites)
python -m pytest tests/ -v
```

Press **q** or **ESC** to quit.

### Switch video source (`configs/pipeline.yaml`)

```yaml
source:
  type: webcam       # webcam | file | rtsp
  # type: file
  # path: tests/clip.mp4
```

### Enable / disable features

Every feature is a one-line toggle in `configs/pipeline.yaml` — no code changes needed:

```yaml
features:
  pose: true
  fall_detection: true
  reid: true
  face: true
  fire_smoke: true
  smoking: true
  phone: true
  gathering: true
  violence: true
  object_left: true
```

---

## What each detector does

| Detector | Signal | Status |
|---|---|---|
| **Fall** | YOLOv8n-pose keypoints + bbox aspect ratio, rule-based state machine | ✅ tuned, live-tested |
| **Fire/Smoke** | YOLOv8n fine-tuned on D-Fire (conf=0.45, multi-frame confirmation ≥2/5 frames) | ✅ real model, HSV fallback |
| **Smoking** | YOLOv8n fine-tuned for cigarette/vape detection near tracked person | ✅ real model, HSV fallback |
| **Phone** | Dedicated YOLOv8n (COCO cls 67), imgsz=480, confirm+hold hysteresis (3/15) | ✅ stable with hysteresis |
| **Gathering** | Fixed-radius centroid clustering, fires on 3+ people within 150 px | ✅ works reliably |
| **Violence** | Bbox overlap (IoU ≥ 0.3) + rapid motion (≥ 40 px/frame) sustained 1.5 s | ⚠️ heuristic placeholder — VLM will disambiguate |
| **Object-Left** | Tracks stationary non-person objects (bags, suitcases) for >30s | ✅ complete |

### Models downloaded (free, open-source)

| Model | Source | License |
|---|---|---|
| `fire_smoke_yolov8n.pt` | [rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) (HuggingFace) | AGPL-3.0 |
| `smoking_yolov8n.pt` | [cadilak/smoking-detection-yolov8](https://huggingface.co/cadilak/smoking-detection-yolov8) (HuggingFace) | AGPL-3.0 |

### HUD visualization

Events are drawn on the live video window (not just logged to terminal):

- 🟠 **FIRE** — orange/red bounding box + label
- ⬜ **SMOKE** — light gray bounding box + label
- 🟣 **PHONE** — magenta bounding box + `PHONE id{N}` label
- 📦 **OBJECT_LEFT** — cyan bounding box + label + duration
- **FALL / GATHERING / VIOLENCE / SMOKING** — appear in the HUD text bar

---

## Motion prefilter

Cheap frame-differencing runs every frame and skips heavy stages (pose, reid, face,
smoking, phone, gathering, violence, object_left) when the scene is completely static.
Fire/smoke detection is NOT gated — fire can happen in a static monitoring scene.

---

## Event logging (Phase 5 foundation)

All events are logged to SQLite at `data/events.db` with keyframe images saved to
`data/keyframes/`. Schema: `id, t_iso, frame_idx, event_type, track_id, confidence,
details_json, keyframe_path, created_at`. Indexed on `event_type`, `t_iso`, `track_id`.

Query API: `event_logger.query_events(event_type=None, track_id=None, limit=100)`

This is the foundation for the VLM layer's natural-language query engine.

---

## Performance (measured, RTX 4050 Laptop)

| Metric | Value |
|---|---|
| Webcam FPS (all features active) | ~15–20 FPS (camera-bound) |
| Peak FPS (synthetic source) | ~100+ FPS |
| Total VRAM | ~70–100 MB alloc / ~94 MB reserved |
| Budget used | <2% of 6 GB |

> FPS drops to ~15 when face recognition (buffalo_s, 5 ONNX models) runs; held to ≥20
> at other frames. Cadence tuning (face/8, reid/15, pose/3) recovers throughput.

---

## Test suites

```bash
python -m pytest tests/ -v                    # run all 43 tests
python tests/smoke_test.py                    # Phase 1: imports + one-frame inference
python tests/fall_detection_test.py           # 5 fall detector tests
python tests/fall_cadence_test.py             # 3 fall tests at configured cadence
python tests/fall_false_positive_test.py      # 4 false-positive guard tests
python tests/phase3_test.py                   # 5 ReID + face recognition tests
python tests/phase4_test.py                   # 8 Phase 4 event detector tests
python tests/phase5_integration_test.py       # 15 integration tests (new features)
```

**43/43 tests pass across 7 suites.**

Tests cover: ObjectLeftDetector (3), MotionPrefilter (3), EventLogger (4),
fire/smoke multi-frame confirmation (2), phone hysteresis (2), full pipeline
construction (1), fall detection + FP suppression (12), face/ReID (5),
all Phase 4 detectors (8), core infrastructure (3).

---

## Debug flags

| Env var | Effect |
|---|---|
| `PHONE_DEBUG=1` | Print raw YOLO confidences/classes for every phone-detector call (even below threshold) |
| `FALL_DEBUG=1` | Log next 10 fall trigger candidates with full signal breakdown |
| `DEBUG_DEVICE=1` | Print model device before each inference call |

Example:
```bash
PHONE_DEBUG=1 python -m pipeline.main_loop
```

---

## Project layout

```
configs/
  models.yaml        model registry: weights, thresholds, hysteresis params
  pipeline.yaml      source, display, FrameRouter stages, feature toggles
core/
  detector.py        YOLOv8n wrapper (pytorch/onnx/onnx_direct)
  tracker.py         ByteTrack wrapper
  pose.py            YOLOv8n-pose wrapper (onnx_direct)
  state_machine.py   Rule-based fall detector
  reid.py            ResNet18 + FAISS re-identification
  face.py            SCRFD + MobileFaceNet + FAISS face index
  identity.py        Identity fusion (face > ReID priority)
  events.py          Phase 4: fire/smoke, smoking, phone, gathering, violence, object-left
  motion_filter.py   Frame differencing prefilter
  event_logger.py    SQLite event logger + keyframe storage
  video_source.py    Webcam / file / RTSP / synthetic source abstraction
  config.py          YAML loaders + warning filters
pipeline/
  frame_router.py    Config-driven stage scheduler (no scattered modulo checks)
  main_loop.py       Full pipeline: capture→detect→track→pose→reid→face→events→log→display
tools/
  export_onnx.py     Reproducible FP16 ONNX export for detector + pose
  enroll_face.py     Webcam face enrollment script
benchmarks/
  vram_profile.py    Isolated VRAM profiler for all models
  fps_bench.py       Headless throughput benchmark
tests/               Unit + integration test suites (42 tests, 7 suites)
models/              Weights directory (gitignored — see models/README.md for download links)
data/                Event log database + keyframe images (runtime-generated)
vlm/                 Phase 6 stub
```

---

## Hardware-budget rules (enforced in code)

- **One shared YOLO instance** for all person detection + tracking. Exception: `PhoneWatcherDetector` loads its own independent YOLOv8n (COCO class 67 only) — sharing caused tracker state corruption.
- **FP16 on CUDA** everywhere (`half: true` in `models.yaml`).
- **FrameRouter** gates every stage — no scattered `frame_count % N` checks. Cadence changes are config edits.
- **Motion prefilter** skips heavy stages on static frames (pose, reid, face, phone, smoking, gathering, violence, object_left).
- **Every feature is toggleable** in `configs/pipeline.yaml` — no code changes to disable a phase.
- **VRAM logged** every 5 s to `benchmarks/vram_log.csv`; warning printed above 5 GB.

---

## License

- Code: MIT
- YOLO models: AGPL-3.0 (Ultralytics)
- D-Fire model: AGPL-3.0 (rabahdev/fire-smoke-yolov8n)
- Smoking model: AGPL-3.0 (cadilak/smoking-detection-yolov8)
- InsightFace: MIT