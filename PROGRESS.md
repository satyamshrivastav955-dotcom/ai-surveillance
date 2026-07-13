# AI Surveillance System — Consolidated Technical Summary (Phases 1-3)

## Project Overview

A real-time AI security camera system built on a pure deep-learning pipeline (Phases 1-3) with a VLM layer planned for Phase 6. Target hardware: **RTX 4050 Laptop GPU, 6GB VRAM**. Every model choice, config value, and architectural decision is constrained by this VRAM budget.

**Final measured performance (Phase 3, all features active):**
- Webcam: **29.9 FPS** (camera-bound, not GPU-bound)
- Synthetic ceiling: **109.6 FPS**
- Combined VRAM: **43 MB** (0.86% of the 5GB budget — 116x headroom)
- All 17 unit tests pass across 5 test suites

---

## Repository Structure (46 files, ~3500 lines of Python)

```
ai-surveillance/
├── configs/
│   ├── models.yaml            (92 lines)  — model registry: weights, runtime, thresholds
│   └── pipeline.yaml          (60 lines)  — source, display, FrameRouter stages, features
├── core/
│   ├── config.py              (29 lines)  — yaml loaders + warning filters
│   ├── detector.py            (252 lines) — YOLOv8n wrapper (pytorch/onnx/onnx_direct)
│   ├── tracker.py             (139 lines) — ByteTrack wrapper (ultralytics + direct path)
│   ├── pose.py                (205 lines) — YOLOv8n-pose wrapper (pytorch/onnx/onnx_direct)
│   ├── state_machine.py       (244 lines) — rule-based fall detector
│   ├── reid.py                (280 lines) — ReID: ResNet18 + FAISS + re-linking
│   ├── face.py                (167 lines) — SCRFD + MobileFaceNet + FAISS face index
│   ├── identity.py            (120 lines) — identity fusion (face > ReID priority)
│   └── video_source.py        (192 lines) — Webcam/File/RTSP/Synthetic source abstraction
├── pipeline/
│   ├── frame_router.py        (38 lines)  — config-driven stage scheduler
│   └── main_loop.py           (348 lines) — capture -> detect -> track -> pose -> reid -> face -> display
├── tools/
│   ├── export_onnx.py         (47 lines)  — reproducible ONNX FP16 export for detector + pose
│   └── enroll_face.py         (101 lines) — webcam face enrollment script
├── benchmarks/
│   ├── vram_profile.py        (160 lines) — isolated VRAM profiler for all 4 models
│   ├── fps_bench.py           (183 lines) — headless throughput bench (3 source types)
│   ├── phase2_static_bench.py (151 lines) — GPU-only A/B bench with CUDA events
│   ├── gpu_check.py           (39 lines)  — quick CUDA device/VRAM diagnostic
│   ├── webcam_diag.py         (102 lines) — webcam capture bottleneck diagnostic
│   ├── onnx_ab.py             (22 lines)  — ultralytics-ONNX vs PyTorch quick compare
│   └── vram_log.csv                       — runtime VRAM samples (appended every 5s)
├── tests/
│   ├── smoke_test.py          (43 lines)  — Phase 1 imports + one-frame inference
│   ├── phase2_smoke.py        (13 lines)  — Phase 2 pose + fall detector construction
│   ├── phase3_smoke.py        (59 lines)  — Phase 3 ReID + face + identity construction
│   ├── fall_detection_test.py (239 lines) — 5 fall tests (fast fall, slow sit, lean, cooldown, fidget)
│   ├── fall_cadence_test.py   (173 lines) — 3 fall tests at every=2 cadence
│   ├── fall_false_positive_test.py (212 lines) — 4 false-positive tests (lean, sit, rotate, sanity)
│   ├── phase3_test.py         (172 lines) — 5 ReID + face tests (isolated temp index)
│   ├── capture_person_frame.py (45 lines) — captures a real-person webcam frame for benching
│   ├── graceful_exit.py       (21 lines)  — SIGINT test runner for clean shutdown verification
│   └── real_person_frame.npz  (~1MB)      — captured 1280x720 frame with 1 person
├── models/
│   ├── README.md              — weight download links
│   ├── .gitignore             — gitignores .pt/.onnx/.engine
│   ├── face_index.json        — ["Satyam", "Chirandilal"] (2 enrolled faces)
│   ├── face_index.faiss       — FAISS IndexFlatIP with 2 face embeddings
│   ├── yolov8n.pt             (~6.2MB)    — detector weights
│   ├── yolov8n.onnx           (~6.2MB)    — detector FP16 ONNX export
│   ├── yolov8n-pose.pt        (~6.5MB)    — pose weights
│   └── yolov8n-pose.onnx      (~6.4MB)    — pose FP16 ONNX export
├── vlm/                       — Phase 6 stub (empty __init__.py)
├── storage/                   — Phase 5 stub (empty __init__.py)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Models and Config Values

### Model Registry (`configs/models.yaml`)

| Model | Weights | Runtime | imgsz | half | conf | Other |
|---|---|---|---|---|---|---|
| **Detector** (YOLOv8n) | `models/yolov8n.pt` / `.onnx` | `pytorch` | 640 | true | 0.35 | iou=0.5, classes=[0] (persons-only) |
| **Pose** (YOLOv8n-pose) | `models/yolov8n-pose.pt` / `.onnx` | `onnx_direct` | 160 | true | 0.25 | — |
| **ReID** (ResNet18) | torchvision pretrained | `pytorch` (FP16) | 128 | true | — | dim=512, match_threshold=0.95, lost_ttl=30s |
| **Face** (buffalo_s pack) | insightface auto-download | `onnx` (insightface) | det_size=320 | — | — | dim=512, match_threshold=0.4, index_path=models/face_index |

**Model sizes:** YOLOv8n 3.15M params / 8.7 GFLOPs | YOLOv8n-pose 3.29M params / 9.2 GFLOPs | ResNet18 ~11M params | buffalo_s ~5M params total (SCRFD-500M + MobileFaceNet).

### Fall Detector Thresholds (`configs/models.yaml: fall:`)

| Parameter | Value | Rationale |
|---|---|---|
| `aspect_threshold` | 1.5 | was 1.0 (too permissive — sitting=0.8-1.2 fired); genuine lying-flat=1.5-2.0+ |
| `keypoint_height_frac` | 0.75 | was 0.60 (sitting hips at 0.55-0.65 fired); lying-flat=0.85-0.95 |
| `transition_window_s` | 1.5 | UPRIGHT->FALLEN must complete within 1.5s (filters slow sit-down) |
| `cooldown_s` | 5.0 | suppress re-trigger after a FALL |
| `min_upright_s` | 1.0 | **new field** — track must be continuously upright >=1s before eligible to fire |
| `min_conf` | 0.30 | keypoints below this confidence are ignored |

### Pipeline Config (`configs/pipeline.yaml`)

| Stage | enabled | every | Notes |
|---|---|---|---|
| detect | true | 1 | every frame — shared YOLO pass |
| track | true | 1 | ByteTrack, CPU-cheap |
| pose | true | 2 | every other frame — halves pose overhead |
| reid | true | 10 | only on new/re-appearing tracks |
| face | true | 4 | every 3rd-5th frame on person crops |
| fire_smoke | false | 15 | Phase 4 stub |
| crowd | false | 30 | Phase 4 stub |
| violence | false | 30 | Phase 4 stub |
| smoking | false | 10 | Phase 4 stub |
| phone | false | 10 | Phase 4 stub |

**Features:** detection, tracking, pose, fall_detection, reid, face, identity_fusion — all `true`.

**Source defaults:** webcam, 1280x720, CAP_DSHOW. **Display:** 960px resize, FPS+VRAM HUD. **Perf:** fps_warn<20, vram_warn>5GB, vram_log every 5s to `benchmarks/vram_log.csv`.

---

## Benchmark Results

### Throughput (wall-clock, all features active)

| Source | Phase 1 (detect+track) | Phase 2 (+pose every=2) | Phase 3 (+reid+face) | VRAM (alloc/peak) |
|---|---|---|---|---|
| Synthetic @ 30 fps | 30.0 FPS | 30.1 FPS | **30.0 FPS** | 28 / 35 MB |
| Synthetic @ 200 fps (ceiling) | 105.8 FPS | 103.6 FPS | **109.6 FPS** | 28 / 35 MB |
| Webcam (live, person in frame) | 20-30 FPS* | 29.8 FPS | **29.9 FPS** | 28 / 35 MB |

*Webcam FPS is lighting-dependent (auto-exposure): 7.4 FPS in dim light, 30 FPS in good light.

### GPU-Only A/B (CUDA events, static real-person frame, isolated from host contention)

| Path | Phase 2 (all PyTorch) | Phase 2+ (mixed ONNX) | Phase 3 (+reid+face) |
|---|---|---|---|
| Detector+tracker only | 9.45 ms (105.8 FPS) | 9.46 ms (105.7 FPS) | — |
| + pose (every=1) | 18.55 ms (53.9 FPS) | 14.41 ms (69.4 FPS) | — |
| + pose (every=2, avg) | — | ~11.5 ms (~87 FPS) | — |
| Pose overhead per frame | 9.10 ms | 4.95 ms (-46%) | 4.95 ms |

### VRAM Profile (all models in isolation)

| Model | Peak VRAM (torch) | Peak VRAM (NVML) | Runtime |
|---|---|---|---|
| YOLOv8n detector | 16 MB | — | pytorch FP16 |
| YOLOv8n-pose | 0 MB (ORT) | ~8 MB | onnx_direct |
| ResNet18 ReID | 27 MB | — | pytorch FP16 |
| buffalo_s (SCRFD+MobileFaceNet) | 0 MB (ORT) | ~16 MB | onnx via insightface |
| **Combined** | **43 MB** | ~51 MB | — |
| **vs 5GB budget** | **0.86%** | ~1.0% | **116x headroom** |

### Webcam Diagnostic (capture-only, no inference)

| Backend | Driver FPS | Raw read FPS | Read latency (mean) |
|---|---|---|---|
| DEFAULT (->MSMF) | 30.0 | 29.94 | 33.33 ms |
| CAP_DSHOW | -1.0 (undisclosed) | 24.29 | 41.16 ms |
| CAP_MSMF | 30.0 | 29.94 | 33.38 ms |

---

## Bugs Found and Fixed

### Bug 1: `csv.writer.flush()` AttributeError (Phase 1)
- **File:** `pipeline/main_loop.py:189`
- **Symptom:** `AttributeError: '_csv.writer' object has no attribute 'flush'`
- **Root cause:** `csv.writer` has no `.flush()` — only the underlying file handle does
- **Fix:** Changed `vram_writer.flush()` -> `vram_file.flush()`; also confirmed `vram_file.close()` in the `finally` block

### Bug 2: Inference running on CPU despite model weights on cuda:0 (Phase 1)
- **File:** `core/detector.py`, `core/tracker.py`
- **Symptom:** FPS dropped to 7.4, nvidia-smi showed 0% GPU-Util
- **Root cause:** `device="cuda:0"` (string) was passed but ultralytics sometimes silently falls back to CPU; `DEBUG_DEVICE=1` confirmed `self.model.device=cuda:0` but the GPU was still idle (webcam auto-exposure was the actual cause — not a code bug, but the investigation revealed the device kwarg needed to be `device=0` int for robustness)
- **Fix:** Changed all inference calls to pass `device=0` (int) explicitly on every `predict()`/`track()` call, not just at construction

### Bug 3: Letterbox remap math wrong — boxes appeared at wrong y-coords (ONNX pass)
- **File:** `core/detector.py:_OnnxDirectDetector`
- **Symptom:** ONNX direct detection box `(534, 0, 764, 154)` vs PyTorch `(533, 410, 765, 716)` — x matched but y was wrong
- **Root cause:** Pad is at bottom/right of the letterboxed canvas, so original image lives at top-left. The remap was subtracting `pad_h` from y, but should have just divided by ratio (no subtraction needed since pad is below, not above)
- **Fix:** `bx1 = boxes_xyxy[i, 0] / ratio` (removed `- pad_w` and `/ ratio` combination)

### Bug 4: ONNX Runtime CUDA EP silently falling back to CPU (ONNX pass)
- **File:** `core/detector.py:_OnnxDirectDetector.__init__`, `core/pose.py:_OnnxDirectPose.__init__`
- **Symptom:** ORT printed `Error loading cudnn64_9.dll which is missing` and fell back to CPUExecutionProvider
- **Root cause:** `onnxruntime-gpu` needs `cudnn64_9.dll` which PyTorch ships in `torch/lib/` but that dir wasn't on PATH
- **Fix:** Added code to find `torch/lib/` and prepend it to `PATH` before creating ORT sessions; also added a warning print if CUDA EP isn't in the active providers list

### Bug 5: BYTETracker.update() output row had 8 values, not 7 (ONNX pass)
- **File:** `core/tracker.py:_update_direct`
- **Symptom:** `ValueError: too many values to unpack (expected 7)`
- **Root cause:** BYTETracker's `STrack.result` property returns `[x1, y1, x2, y2, id, score, cls, idx]` (8 values) — I assumed 7
- **Fix:** Changed `x1, y1, x2, y2, tid, conf, cls = row` -> `x1, y1, x2, y2, tid, conf, cls = row[:7]` (drop the idx column)

### Bug 6: Fall detector false-triggering every 4-7 seconds during normal movement (Phase 2)
- **File:** `core/state_machine.py`
- **Symptom:** Same track ID triggered "FALL" repeatedly during sitting/standing/moving; aspect ratios at trigger were 1.0-1.2 (near-square), not the expected 1.5-2.0+ of a genuine fall
- **Root causes (3):**
  1. `aspect_threshold=1.0` too permissive — sitting/leaning produces 0.8-1.2 w/h
  2. `kp_height_frac=0.60` too permissive — sitting puts hips at 0.55-0.65 of bbox height
  3. **Re-arm bug:** any single upright frame reset `upright_until = t`, instantly re-arming the trigger. After cooldown expired, the next slightly-horizontal frame fired again — exactly the "every 4-7 seconds" pattern
- **Fix (3 changes):**
  1. Raised `aspect_threshold` from 1.0 -> 1.5
  2. Raised `kp_height_frac` from 0.60 -> 0.75
  3. Added `min_upright_s=1.0` config field + rewrote state machine with explicit "armed" semantics: track must be **continuously** upright for >=1s before being armed. Single upright frames during fidgeting no longer re-arm. After a fall fires, the track disarms and must re-accumulate 1s of continuous uprightness
- **Tests added:** `test_fidget_does_not_rearm` (reproduces the exact bug), 4 false-positive tests (lean-forward, sit-down, rotate-in-chair, real-fall-after-lean sanity check)

### Bug 7: Fall state machine didn't fire on fast fall after min_upright_s was added (Phase 2)
- **File:** `core/state_machine.py`
- **Symptom:** `test_fast_fall_triggers` failed — fall didn't fire
- **Root cause:** Initial implementation reset `upright_since = float("inf")` on ANY non-upright frame, including the intermediate transition frames. By the time the first "fallen" frame appeared, `upright_since` was inf and `upright_duration` was 0
- **Fix:** Rewrote with explicit "armed" state — `armed_at` persists through the transition (only cleared when a fall fires or the window expires). `upright_since` is cleared on non-upright frames but `armed_at` is not

### Bug 8: phase3_test.py polluting production face index (Phase 3)
- **File:** `tests/phase3_test.py`
- **Symptom:** `models/face_index.json` contained `["Satyam", "Satyam", "Chirandilal"]` — a duplicate Satyam and the test's "TestPerson" had been enrolled into the production index
- **Root cause:** `FaceRecognizer()` constructor creates a fresh index, but the test called `rec.reset()` which clears whatever was loaded. However, an earlier version of the test may have called `save_index()` or the `enroll_face.py` script was run twice for Satyam. The test also didn't use an isolated config
- **Fix:**
  1. Added `_make_isolated_face_cfg()` helper that overrides `index_path` to a `tempfile.mkdtemp()` directory
  2. Both face test functions now use this isolated config
  3. Cleaned the production index: reconstructed vectors from FAISS, deduplicated by first occurrence per name, rewrote `face_index.faiss` + `face_index.json` -> `["Satyam", "Chirandilal"]` (2 vectors)
  4. Same fix applied to `phase3_smoke.py`

---

## Architectural Decisions (with rationale)

### 1. Single shared YOLO detector (Phase 1, constraint #1)
All object/person detection goes through one YOLOv8n instance. The tracker rides on the same model via `model.track()`. No second YOLO is ever spawned — fire/smoke and other Phase 4 detections will reuse the same pass or use the detector's crops.

### 2. FrameRouter scheduler (Phase 1, constraint #4)
One `FrameRouter` class gates every pipeline stage. No scattered `if frame_count % N == 0` checks. Adding a phase = adding a config entry + a stage handler. Cadence changes (e.g. `pose.every: 1->2`) are config edits, not code changes.

### 3. ONNX direct path for pose only (ONNX pass)
A/B benchmarking showed ultralytics' Python wrapper dominates per-call overhead. Direct ORT inference cut pose overhead by 46% (9.10ms->4.95ms). But for the detector, ultralytics' optimized NMS beats my hand-written NMS — so the detector stays on `runtime: pytorch` while pose uses `runtime: onnx_direct`. This mixed config is the default in `models.yaml`.

### 4. ReID: ResNet18 (not OSNet) with threshold 0.95 (Phase 3)
`torchreid` had Python 3.13 compatibility issues. torchvision ResNet18 (FC stripped, 512-dim avgpool) works out of the box and is the same scale as YOLOv8n (~11M params). ImageNet-pretrained features aren't person-discriminative (same person ~0.99, different ~0.94), so the threshold is 0.95 (conservative). OSNet would allow 0.6-0.7; the config is structured for a drop-in swap.

### 5. Face: insightface buffalo_s (Phase 3)
SCRFD-500M (the "small variant" per spec) + MobileFaceNet (w600k ArcFace-MobileNet) in one pack. CUDA EP confirmed active. Separate FAISS index from ReID. Identity fusion priority: face > ReID (face is the stronger signal; ReID can propagate a face-confirmed identity to a re-linked track but can't assign new labels).

### 6. FAISS (not Milvus) for vector indices (Phase 3)
`IndexFlatIP` with L2-normalized embeddings = cosine similarity. Separate indices for body (ReID) and face embeddings. Milvus is the documented production swap — the index interface is isolated in `ReIDIndex` and `FaceRecognizer` classes so the migration touches only those two files.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | 3.13.13 | — |
| torch | 2.12.1+cu126 | CUDA 12.6, cuDNN 9.x |
| torchvision | 0.27.1+cu126 | ResNet18 for ReID |
| ultralytics | 8.4.84 | YOLOv8n + YOLOv8n-pose |
| opencv-python | 5.0.0.93 | Video capture, display, image ops |
| onnxruntime-gpu | 1.23.2 | CUDA EP for ONNX inference |
| faiss-cpu | 1.14.3 | Vector indices for ReID + face |
| insightface | 1.0.1 | SCRFD + MobileFaceNet (buffalo_s pack) |
| numpy | 2.5.0 | — |
| pyyaml | 6.0.3 | Config loading |
| pynvml | 13.0.1 | NVML VRAM measurement for ONNX direct mode |
| onnx | 1.22.0 | ONNX model format (auto-installed by export) |
| onnxslim | 0.1.94 | ONNX graph simplification (auto-installed by export) |

**System:** Windows 11, NVIDIA driver 581.86, RTX 4050 Laptop GPU (6141 MiB).

---

## Test Suite Summary (24 tests across 7 suites)

| Suite | Tests | What it covers | Result |
|---|---|---|---|
| `smoke_test.py` | 3 | Phase 1 imports + FrameRouter + VideoSource factory + one-frame inference | pass |
| `phase2_smoke.py` | 1 | Phase 2 pose + fall detector construction | pass |
| `phase3_smoke.py` | 3 | Phase 3 ReID + face + identity construction (isolated index) | pass |
| `fall_detection_test.py` | 5 | Fast fall, slow sit, lean-forward, cooldown+min_upright, fidget re-arm | pass |
| `fall_cadence_test.py` | 3 | Fast fall, slow sit, cooldown at every=2 cadence | pass |
| `fall_false_positive_test.py` | 4 | Lean-forward, sit-down, rotate-in-chair, real-fall-after-lean sanity | pass |
| `phase3_test.py` | 5 | ReID self-match, different-persons reject, re-link, face enroll+recognize, face rejects unknown | pass |
| **Total** | **24** | | **24/24 pass** |

---

## What's Built vs What's Left

| Phase | Status | Key deliverable |
|---|---|---|
| 1 — Core Detection Loop | complete | Shared YOLO + ByteTrack + live display |
| 2 — Fall Detection | complete | YOLOv8n-pose on crops + rule-based state machine (tuned) |
| ONNX Export Pass | complete | Pose on onnx_direct (46% overhead reduction) |
| 3 — ReID + Face | complete | ResNet18 + SCRFD/MobileFaceNet + identity fusion |
| 4 — Remaining Events | pending | Fire/smoke, smoking, phone, gathering, violence |
| 5 — Event Bus + Logging | pending | Redis Streams + SQLite event log |
| 6 — VLM Layer | pending | Alert verifier, open-vocab watcher, NL query engine |
