# ai-surveillance

Real-time AI security camera system — **pure deep-learning pipeline** as the real-time
core, with a **VLM layer** planned for Phase 6 (alert verification, open-vocabulary
descriptions, natural-language querying).

**Target hardware:** RTX 4050 Laptop GPU, 6 GB VRAM. Every model choice and config
value is constrained by this budget — 43 MB VRAM used across all active models (0.86% of budget).

---

## Status: Phase 4 complete

| Phase | What | Status |
|---|---|---|
| 1 | Shared YOLO detector + ByteTrack + live display | ✅ complete |
| 2 | YOLOv8n-pose on crops + rule-based fall detector | ✅ complete |
| ONNX pass | onnx_direct for pose — 46% overhead reduction | ✅ complete |
| 3 | ReID (ResNet18) + face recognition (SCRFD/MobileFaceNet) + identity fusion | ✅ complete |
| 4 | Fire/smoke · smoking · phone-watching · gathering · violence detection | ✅ finalized |
| 5 | Event bus + structured logging (Redis Streams + SQLite) | pending |
| 6 | VLM layer — alert verifier, open-vocab watcher, NL query engine | pending |

---

## Quick start

```bash
# run the full live pipeline (all Phase 1-4 features enabled)
python pipeline/main_loop.py

# with explicit config
python pipeline/main_loop.py --config configs/pipeline.yaml
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
  phone: true
  gathering: true
  violence: true
```

---

## What each detector does

| Detector | Signal | Status |
|---|---|---|
| **Fall** | YOLOv8n-pose keypoints + bbox aspect ratio, rule-based state machine | ✅ tuned, live-tested |
| **Violence** | Bbox overlap (IoU ≥ 0.3) + rapid relative motion (≥ 40 px/frame) sustained 1.5 s | ✅ heuristic, false-positives tightened |
| **Gathering** | Fixed-radius centroid clustering, fires on 3+ people within 150 px | ✅ works reliably |
| **Smoking** | Cigarette-glow HSV heuristic near face/hand crop | ⚠️ rough placeholder — bright reflections can trigger |
| **Phone** | Separate YOLOv8n instance, COCO class 67, proximity to tracked person + head-tilt pose heuristic | ✅ fixed (3 bugs patched in Phase 4 finalization) |
| **Fire/Smoke** | HSV color thresholding | ❌ confirmed unreliable — FP and FN both observed in live testing. Needs D-Fire fine-tuned model for production |

### Phase 4 HUD visualization

Events are drawn on the live video window (not just logged to terminal):

- 🟠 **FIRE** — orange/red bounding box + label
- ⬜ **SMOKE** — light gray bounding box + label
- 🟣 **PHONE** — magenta bounding box + `PHONE id{N}` label
- **FALL / GATHERING / VIOLENCE / SMOKING** — appear in the HUD text bar

---

## Performance (measured, RTX 4050 Laptop)

| Metric | Value |
|---|---|
| Webcam FPS (all features active) | ~17–30 FPS (camera-bound, not GPU-bound) |
| Synthetic source ceiling | 109.6 FPS |
| Combined VRAM (all models) | 43 MB alloc / ~51 MB peak |
| Budget used | 0.86% of 5 GB |

> FPS varies with lighting — webcam auto-exposure sets the effective frame rate.
> The pipeline is GPU-bound only above ~110 FPS (synthetic source).

---

## Test suites

```bash
python tests/smoke_test.py              # Phase 1: imports + one-frame inference
python tests/phase2_smoke.py            # Phase 2: pose + fall detector construction
python tests/phase3_smoke.py            # Phase 3: ReID + face + identity (isolated index)
python tests/fall_detection_test.py     # 5 fall tests
python tests/fall_cadence_test.py       # 3 fall tests at every=2 cadence
python tests/fall_false_positive_test.py # 4 false-positive guard tests
python tests/phase3_test.py             # 5 ReID + face recognition tests
python tests/phase4_test.py             # 8 Phase 4 event detector tests
```

**All 32 tests pass across 8 suites.**

---

## Debug flags

| Env var | Effect |
|---|---|
| `PHONE_DEBUG=1` | Print raw YOLO confidences/classes for every phone-detector call (even below threshold) — fastest way to diagnose missed detections |
| `FALL_DEBUG=1` | Log next 10 fall trigger candidates with full signal breakdown (aspect, kp_frac, upright duration) |

Example:
```bash
PHONE_DEBUG=1 python pipeline/main_loop.py
```

---

## Project layout

```
configs/
  models.yaml        model registry: weights, thresholds, HSV params
  pipeline.yaml      source, display, FrameRouter stages, feature toggles
core/
  detector.py        YOLOv8n wrapper (pytorch/onnx/onnx_direct)
  tracker.py         ByteTrack wrapper
  pose.py            YOLOv8n-pose wrapper (onnx_direct)
  state_machine.py   Rule-based fall detector
  reid.py            ResNet18 + FAISS re-identification
  face.py            SCRFD + MobileFaceNet + FAISS face index
  identity.py        Identity fusion (face > ReID priority)
  events.py          Phase 4: fire/smoke, smoking, phone, gathering, violence
  video_source.py    Webcam / file / RTSP / synthetic source abstraction
  config.py          YAML loaders + warning filters
pipeline/
  frame_router.py    Config-driven stage scheduler (no scattered modulo checks)
  main_loop.py       Full pipeline: capture→detect→track→pose→reid→face→events→display
tools/
  export_onnx.py     Reproducible FP16 ONNX export for detector + pose
  enroll_face.py     Webcam face enrollment script
benchmarks/
  vram_profile.py    Isolated VRAM profiler for all models
  fps_bench.py       Headless throughput benchmark
tests/               Unit + integration test suites (see above)
models/              Weights directory (gitignored — see models/README.md for download links)
vlm/                 Phase 6 stub
storage/             Phase 5 stub
```

---

## Hardware-budget rules (enforced in code)

- **One shared YOLO instance** for all person detection + tracking. Exception: `PhoneWatcherDetector` loads its own independent YOLOv8n (COCO class 67 only) — sharing caused tracker state corruption.
- **FP16 on CUDA** everywhere (`half: true` in `models.yaml`).
- **FrameRouter** gates every stage — no scattered `frame_count % N` checks. Cadence changes are config edits.
- **Every feature is toggleable** in `configs/pipeline.yaml` — no code changes to disable a phase.
- **VRAM logged** every 5 s to `benchmarks/vram_log.csv`; warning printed above 5 GB.