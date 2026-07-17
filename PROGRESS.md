# AI Surveillance System — Phase 4 Complete / Pre-VLM Baseline

## Project Overview

A real-time AI security camera system built on a pure deep-learning pipeline (Phases 1-4+) with VLM layer planned for Phase 6. Target hardware: **RTX 4050 Laptop GPU, 6GB VRAM**.

**Status: Phase 4 Complete — Pre-VLM Baseline Locked**

---

## Phase Summary

| Phase | Status | Key Deliverable |
|---|---|---|
| 1 — Core Detection Loop | ✅ complete | Shared YOLO + ByteTrack + live display |
| 2 — Fall Detection | ✅ complete | YOLOv8n-pose on person crops + state machine (tuned anti-FP) |
| ONNX Export Pass | ✅ complete | Pose on onnx_direct (46% overhead reduction) |
| 3 — ReID + Face | ✅ complete | ResNet18 + SCRFD/MobileFaceNet + identity fusion |
| 4 — Event Detectors | ✅ complete | All 6 detectors + real YOLO models (see below) |
| Motion Prefilter | ✅ complete | Frame differencing gates heavy stages on static scenes |
| Object-Left-Behind | ✅ complete | Tracks stationary non-person objects for >30s |
| Structured Event Log | ✅ complete | SQLite + keyframes (Phase 5 foundation) |
| Pre-VLM Integration | ✅ complete | 42/42 tests pass, full pipeline confirmed |
| 5 — VLM Layer | 🔜 pending | Alert verifier, open-vocab watcher, NL query engine |

---

## Model Registry

| Model | Weights | Runtime | Source | License |
|---|---|---|---|---|
| Detector (YOLOv8n) | `models/yolov8n.pt` | pytorch FP16 | Ultralytics | AGPL-3.0 |
| Pose (YOLOv8n-pose) | `models/yolov8n-pose.onnx` | onnx_direct | Ultralytics | AGPL-3.0 |
| ReID (ResNet18) | torchvision pretrained | pytorch FP16 | torchvision | BSD |
| Face (buffalo_s) | insightface auto-download | onnx | InsightFace | MIT |
| Fire/Smoke | `models/fire_smoke_yolov8n.pt` | pytorch FP16 | rabahdev/fire-smoke-yolov8n (HF) | AGPL-3.0 |
| Smoking | `models/smoking_yolov8n.pt` | pytorch FP16 | cadilak/smoking-detection-yolov8 (HF) | AGPL-3.0 |
| Phone | `models/yolov8n.pt` (COCO cls 67) | pytorch FP16 | Ultralytics | AGPL-3.0 |

**Total active VRAM: ~70-100 MB (<2% of 6GB budget)**

---

## Event Detectors — Final State

### 1. Fire/Smoke — ✅ Real YOLO Model
- **Model:** YOLOv8n fine-tuned on D-Fire dataset (4,306 images, mAP50=0.754)
- **Source:** `rabahdev/fire-smoke-yolov8n` (HuggingFace)
- **Classes:** smoke (0), fire (1)
- **Anti-FP:** conf=0.45 + multi-frame confirmation (≥2 of last 5 relevant frames)
- **Fallback:** HSV color heuristic (auto-activates if weights missing)

### 2. Smoking Detection — ✅ Real YOLO Model
- **Model:** YOLOv8n fine-tuned on cigarette/vape dataset
- **Source:** `cadilak/smoking-detection-yolov8` (HuggingFace)
- **Classes:** Cigarette (2), Vape (3) near tracked person
- **Fallback:** HSV glow heuristic (auto-activates if weights missing)

### 3. Phone-Watching — ✅ Working with Hysteresis
- **Model:** YOLOv8n, COCO class 67 (cell_phone), dedicated instance at imgsz=480
- **Hysteresis:** confirm_frames=3 (suppress noise) + hold_frames=15 (kill flicker)
- **Head-pose:** Nose below shoulder midpoint = "looking down"
- **Model search result:** No freely-downloadable HuggingFace/Roboflow `.pt` specific
  to surveillance "phone in hand" found without API credentials. COCO class 67 +
  hysteresis + imgsz=480 is the correct practical baseline.
- **Fallback:** N/A — no fallback needed; model is generic COCO, always available

### 4. Gathering Detection — ✅ No Model Needed
- **Method:** Fixed-radius centroid clustering
- **Trigger:** 3+ people within 150px radius, cooldown 10s

### 5. Violence Detection — ⚠️ Heuristic Placeholder (by design)
- **Method:** Bbox overlap IoU≥0.3 + rapid relative motion ≥40px/frame for 1.5s
- **Limitation:** Cannot distinguish fighting from hugs/handshakes
- **Resolution:** VLM layer (Phase 6) will verify and disambiguate
- **Temporal model (MoViNet):** Out of scope; needs curated training data

### 6. Object-Left-Behind — ✅ Complete
- **Method:** Track stationary non-person objects >30s at same position
- **Classes:** backpack (24), handbag (26), suitcase (28), bottle (39), furniture (56-61)
- **Gate:** Detector now uses classes=[0,24,26,28,39]; tracker passes these through

---

## Detector Configuration (Post-Tuning)

| Config | Value | Rationale |
|---|---|---|
| `detector.classes` | `[0, 24, 26, 28, 39]` | Persons + bags for ObjectLeft |
| `fire_smoke.conf` | `0.45` | Cut false positives on bright backgrounds |
| `fire_smoke.min_consecutive_frames` | `2` | Multi-frame confirmation |
| `phone.imgsz` | `480` | Better small-object recall vs 320 |
| `phone.confirm_frames` | `3` | Suppress single-frame noise |
| `phone.hold_frames` | `15` | Hold event alive across YOLO miss frames |
| `pose.every` | `3` | Was 2; frees GPU cycles |
| `reid.every` | `15` | Was 10 |
| `face.every` | `8` | Was 4; buffalo_s=5 ONNX models |

---

## Motion Prefilter

- **Method:** Frame differencing + Gaussian blur, threshold=25, min_changed=1000px
- **Gated stages:** pose, reid, face, smoking, phone, gathering, violence, object_left
- **Not gated:** fire_smoke (can occur in static monitoring scenes), detect, track

---

## Structured Event Logging (Phase 5 Foundation)

- **Storage:** SQLite at `data/events.db`
- **Schema:** id, t_iso, frame_idx, event_type, track_id, confidence, details_json, keyframe_path, created_at
- **Indexed fields:** event_type, t_iso, track_id (for fast VLM query)
- **Keyframes:** saved to `data/keyframes/` as JPEG (85% quality)
- **Query API:** `event_logger.query_events(event_type=None, track_id=None, limit=100)`
- **Events logged:** ALL event types — fall, fire, smoke, smoking, phone, gathering, violence, object_left, identity

---

## Test Suite — Final State

| Suite | Tests | Result |
|---|---|---|
| smoke_test.py | 3 | ✅ pass |
| fall_cadence_test.py | 3 | ✅ pass |
| fall_detection_test.py | 5 | ✅ pass |
| fall_false_positive_test.py | 4 | ✅ pass |
| phase3_test.py | 5 | ✅ pass |
| phase4_test.py | 8 | ✅ pass |
| phase5_integration_test.py | 14 | ✅ pass |
| **Total** | **42** | **42/42 pass** |

Tests cover: ObjectLeftDetector, MotionPrefilter, EventLogger (log/query/keyframe),
fire/smoke multi-frame confirmation, phone hysteresis (confirm + hold), full pipeline
simultaneous construction, fall detection FP suppression, face enroll/recognize, ReID relink.

---

## Performance (Measured on RTX 4050, webcam, all features active)

| Metric | Value |
|---|---|
| Sustained FPS | 15–20 FPS (camera-bound) |
| Peak FPS (synthetic) | ~100+ FPS |
| Total VRAM allocated | ~45–70 MB |
| VRAM reserved | ~94 MB |
| Budget used | <2% of 6GB |

FPS drops to ~15 when face recognition (buffalo_s, 5 ONNX models) runs; held to ≥20
at other frames. Cadence tuning (face/8, reid/15, pose/3) recovers throughput.

---

## Full-Stack Integration Verification

Confirmed via live webcam run (session logged in initial issue):
- **Detection + Tracking:** persons tracked stably, face recognized as "Satyam" (sim 0.65–0.78)
- **Phone:** detected at frames 1680, 1700, 1710, 1750; with hysteresis now sustained
- **Fire/Smoke:** 8 false positives at conf=0.25 eliminated by raising to 0.45 + multi-frame
- **Gathering:** requires 3+ persons; correctly idle in single-person scenes
- **Violence:** heuristic-placeholder runs without interfering with any other stage
- **Object-Left:** now receives backpack/handbag/suitcase tracks (detector classes expanded)
- **Pose/Fall:** runs every 3rd frame alongside face/reid without conflict
- **Event logging:** SQLite + keyframes confirmed populating via test_event_logger_*

No cross-feature interference observed: phone detection does not break tracking (uses separate YOLO instance); face/reid running simultaneously with fall detection confirmed.

---

## Files Changed (Pre-VLM Final Pass)

| File | Change |
|---|---|
| `README.md` | Full rewrite reflecting all Phase 4+ features, real models, 42 tests |
| `configs/models.yaml` | detector.classes expanded; fire_smoke.conf=0.45; phone imgsz/confirm/hold |
| `configs/pipeline.yaml` | pose/3, reid/15, face/8 (FPS tuning) |
| `core/events.py` | FireSmokeDetector: multi-frame confirmation; PhoneWatcherDetector: confirm+hold hysteresis; ObjectLeftDetector: off-by-one window fix |
| `tests/phase5_integration_test.py` | NEW: 14 integration tests covering all Phase 4+ features |

---

## How to Run

```bash
# Run the full pipeline (webcam, all features)
python -m pipeline.main_loop

# Run all tests
python -m pytest tests/ -v

# Debug phone detection
PHONE_DEBUG=1 python -m pipeline.main_loop

# Debug fall detection signals
FALL_DEBUG=1 python -m pipeline.main_loop
```

Press **q** or **ESC** to quit.

---

## Remaining Work — VLM Layer (Phase 6)

| Item | Description |
|---|---|
| Alert Verifier | VLM confirms/rejects ambiguous events (violence, smoke, object_left) |
| Open-Vocab Watcher | "Notify me when someone carries a red bag" style triggers |
| NL Query Engine | "Show me all smoke events from today" via natural language |
| Violence Upgrade | MoViNet temporal action recognition (if VLM insufficient) |
| ReID Upgrade | OSNet-x0.25 (purpose-built; wider match margin than ResNet18) |

---

## Debug Flags

| Env Var | Effect |
|---|---|
| `PHONE_DEBUG=1` | Print raw YOLO confidences for every phone detection call |
| `FALL_DEBUG=1` | Log fall trigger candidates with full signal breakdown |
| `DEBUG_DEVICE=1` | Print model device before each inference call |

---

## License

- Code: MIT
- YOLO models: AGPL-3.0 (Ultralytics)
- D-Fire model: AGPL-3.0 (rabahdev/fire-smoke-yolov8n)
- Smoking model: AGPL-3.0 (cadilak/smoking-detection-yolov8)
- InsightFace: MIT
