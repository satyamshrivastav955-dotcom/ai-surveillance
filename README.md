# AI Surveillance

Real-time AI security camera system — end-to-end deep learning pipeline with a fully integrated **Vision Language Model (VLM) layer** for natural-language scene understanding, **Bitchat mesh network alerting** to push alerts directly to your Android phone, **Phase 5D Fight & Pose Analysis**, and automated **WiFi & Network Diagnostic Utilities**.

**Target hardware:** RTX 4050 Laptop GPU, 6 GB VRAM.

---

## Status: Phase 6 Complete — VLM + Mesh Alerts Live

| Phase | What | Status |
|---|---|---|
| 1 | Shared YOLO detector + ByteTrack + live display | ✅ complete |
| 2 | YOLOv8n-pose on crops + rule-based fall detector | ✅ complete |
| ONNX | onnx_direct for pose — 46% overhead reduction | ✅ complete |
| 3 | ReID (OSNet-x0.25) + face recognition (SCRFD/MobileFaceNet) + identity fusion | ✅ complete |
| 4 | Fire/smoke · smoking · phone · gathering · violence · object-left detection | ✅ complete |
| — | Motion prefilter (frame differencing gates heavy stages) | ✅ complete |
| — | Structured event logging (SQLite + keyframes) | ✅ complete |
| 5 | PAR (Pedestrian Attribute Recognition), event buffer, ROI zones | ✅ complete |
| 5D | Skeleton fight detector + OneEuro pose smoother + event clip writer | ✅ complete |
| **6** | **VLM layer — Qwen2.5-VL-3B, scene description, entity detection, NL query** | ✅ **live** |
| **6+** | **Bitchat mesh alerting & network diagnostics — push alerts + images to Android** | ✅ **live** |

---

## Quick Start

```bash
# Full pipeline (all detectors + VLM + Bitchat alerts)
python -m pipeline.main_loop

# VLM only — just open the camera and describe what it sees
python run_vlm_only.py

# Standalone VLM to Bitchat relay
python run_vlm_to_bitchat.py

# Diagnose Android Bitchat WiFi & API connection
python diagnose_connection.py --ip 172.20.73.217

# Run all unit and integration tests
python -m pytest tests/ -v
```

Press **Q** or **ESC** to quit the video window.

---

## VLM Layer (Phase 6)

### Model
- **Qwen/Qwen2.5-VL-3B-Instruct** — quantized NF4 via `bitsandbytes`
- **VRAM:** ~2.3 GB (NF4 quantized, fits comfortably in RTX 4050 6 GB)
- **Inference:** runs in a **background thread** — webcam stays at full FPS (20–25) during the 10–15s inference window

### How it works
Every ~30 seconds (or immediately on a FIRE/FALL/FIGHT event) the VLM runs two passes on the current camera frame:

1. **SCENE_PROMPT** — *"Describe in 1-2 sentences exactly what you see"*  
   → Real natural language: *"A man is sitting at a desk in front of a monitor. He appears to be looking down at his phone in his right hand."*

2. **ENTITY_PROMPT** — structured JSON entity detection  
   → `[{"type": "person", "confidence": 0.92, "description": "male sitting at desk"}, ...]`

### What you see

**Terminal:**
```
[vlm] 📷 SCENE: A man is sitting at a desk in front of a monitor. He is looking down at his phone.
[vlm] ESCALATED frame=900  reason='Escalated: forced_interval'  entities=2  conf=0.92
  → [PERSON] conf=92%  male sitting at desk
  → [PHONE] conf=85%  holding smartphone in right hand
```

**Video window** — scene description overlaid at the bottom in green text, updates every ~30s.

### VLM-only mode

Run just the VLM with no other detectors (minimal VRAM, clean output):

```bash
python run_vlm_only.py
```

Controls: `SPACE` = force inference now, `Q`/`ESC` = quit.

---

## Bitchat Mesh Alerting & WiFi Diagnostics

Every surveillance alert and VLM scene description is pushed to your **Android phone via Bitchat's mesh network** as a text message + camera keyframe image (using multipart caption encoding).

### Quick WiFi Setup & Connection Guide

See `QUICKSTART_WIFI.txt` for detailed step-by-step connection setup.

1. Install [Bitchat](https://github.com/permissionlesstech/bitchat-android) on your Android phone.
2. Open Bitchat → **Settings → VLM API Settings → Enable**.
3. Ensure phone and PC are connected to the same WiFi network.
4. Find your device IP automatically or run diagnostic tools:
   ```bash
   # Network diagnostic & port scanner
   python diagnose_connection.py --ip 172.20.73.217
   ```
5. Edit `configs/pipeline.yaml`:
   ```yaml
   bitchat:
     enabled: true
     ip: "172.20.73.217"   # Your Android WiFi IP
     port: 8765
     channel: "surveillance"
   ```
6. Run the main pipeline — alerts and keyframes flow automatically.

### What arrives on your phone

| Event | Bitchat message |
|---|---|
| Fire/Smoke | 🔥 `[FIRE] 15:42:10 — Fire detected in camera view` + 📷 image |
| Person falls | 🆘 `[FALL] 15:43:22 — Person id:1 has fallen` + 📷 image |
| Fight | ⚠️ `[FIGHT] 15:44:01 — Fight between persons detected` + 📷 image |
| Phone use | 📱 `[PHONE] 15:45:00 — Person id:1 using phone` + 📷 image |
| Crowd | 👥 `[GATHERING] 15:46:10 — 4 people gathered` + 📷 image |
| VLM scene | 📷 `[CAM] A man is sitting at a desk...` + 📷 image |

Priority alerts (FIRE, FALL, FIGHT, VIOLENCE) bypass the rate limiter and send immediately.

---

## What Each Detector Does

| Detector | Signal | Status |
|---|---|---|
| **Fall** | YOLOv8n-pose keypoints + bbox aspect ratio, rule-based state machine | ✅ live-tested |
| **Fight (5D)** | Pose-based kinematic interaction analysis (proximity + rapid arm/body keypoint velocity) | ✅ live-tested |
| **Fire/Smoke** | YOLOv8n fine-tuned on D-Fire (conf=0.45, multi-frame ≥2/5) | ✅ real model |
| **Smoking** | YOLOv8n fine-tuned for cigarette/vape near tracked person | ✅ real model |
| **Phone** | YOLOv8n (COCO cls 67), imgsz=480, confirm+hold hysteresis (3/15) | ✅ stable |
| **Gathering** | Fixed-radius centroid clustering, fires on 3+ people within 150 px | ✅ works |
| **Violence** | Bbox overlap (IoU ≥ 0.3) + rapid motion ≥ 40 px/frame for 1.5s | ✅ heuristic + VLM verify |
| **Object-Left** | Stationary non-person objects (bags, suitcases) tracked for >30s | ✅ complete |
| **PAR** | Pedestrian Attribute Recognition + temporal track aggregation | ✅ complete |
| **ReID** | OSNet-x0.25 (512-dim, half), match_threshold=0.65 | ✅ complete |
| **Face** | SCRFD detection + MobileFaceNet embedding + FAISS index | ✅ complete |
| **VLM** | Qwen2.5-VL-3B-Instruct NF4, scene description + entity JSON | ✅ live |
| **Clip Writer** | Pre/post buffer ring for MP4 clip exports on high-priority alerts | ✅ live |

---

## Performance (RTX 4050 Laptop)

| Metric | Value |
|---|---|
| Webcam FPS — detectors only (no VLM) | ~18–25 FPS |
| Webcam FPS — VLM running in background | ~18–25 FPS (not blocked) |
| VLM inference time | ~10–15s per pass (background thread) |
| Total VRAM — all detectors | ~70–100 MB |
| Total VRAM — detectors + VLM NF4 | ~2.3–2.5 GB |
| Budget used | ~40% of 6 GB (detectors + VLM) |

---

## Project Layout

```
configs/
  models.yaml          model registry: weights, thresholds, hysteresis
  pipeline.yaml        source, feature toggles, Bitchat config, FrameRouter
  vlm.yaml             VLM model config (model name, device, quantization)
core/
  detector.py          YOLOv8n wrapper (pytorch/onnx/onnx_direct)
  tracker.py           ByteTrack wrapper
  pose.py              YOLOv8n-pose wrapper (onnx_direct)
  pose_smoother.py     Keypoint smoothing (OneEuro / EMA filter)
  fight_detector.py    Phase 5D skeleton fight detector
  clip_writer.py       Automatic MP4 video clip recorder
  state_machine.py     Rule-based fall detector
  reid.py              OSNet-x0.25 + FAISS re-identification
  face.py              SCRFD + MobileFaceNet + FAISS face index
  identity.py          Identity fusion (face > ReID priority)
  events.py            Phase 4: fire/smoke, smoking, phone, gathering, violence, object-left
  par.py               Pedestrian Attribute Recognition
  par_aggregator.py    Temporal PAR aggregation per track
  motion_filter.py     Frame differencing prefilter
  event_logger.py      SQLite event logger + keyframe storage (thread-safe)
  event_buffer.py      Windowed JSON event flush
  bitchat.py           Bitchat mesh alert client (HTTP REST, background queue)
  network_scanner.py   Subnet IP & port scanner for Bitchat API discovery
  video_source.py      Webcam / file / RTSP / synthetic source abstraction
  config.py            YAML loaders + warning filters
vlm/
  core.py              VLMCore: Qwen2.5-VL-3B NF4, dual-pass inference, SCENE+ENTITY prompts
  __init__.py          VLMIntegration: lifecycle, escalation router, SQLite persistence
  query_engine.py      Natural language query interface (shared model, no second load)
  escalation.py        Escalation scoring and trigger logic
  kv_cache.py          Hot/warm/cold KV cache tiers
  temporal_merge.py    Multi-frame entity merging
  token_pruning.py     Attention-based token pruning for VRAM efficiency
  config_paths.py      Config path resolution
pipeline/
  frame_router.py      Config-driven stage scheduler
  main_loop.py         Full pipeline: capture→detect→track→pose→reid→face→events→VLM→Bitchat→display
run_vlm_only.py        Standalone VLM-only webcam demo (no other models loaded)
run_vlm_to_bitchat.py  Standalone VLM scene describer with live Bitchat pushing
diagnose_connection.py Network connection & Bitchat API diagnostic tool
enable_bitchat_api.py  Bitchat API helper configuration script
QUICKSTART_WIFI.txt    Step-by-step Android WiFi setup guide
tools/
  export_onnx.py       FP16 ONNX export for detector + pose
  enroll_face.py       Webcam face enrollment script
tests/                 Unit + integration test suites
models/                Weights directory (gitignored — see models/README.md)
data/                  SQLite event DB + keyframe images (runtime-generated)
```

---

## Configuration

### Enable / Disable Features (`configs/pipeline.yaml`)

```yaml
features:
  pose: true
  fall_detection: true
  fight: true        # Phase 5D skeleton fight detector
  reid: true
  face: true
  fire_smoke: true
  smoking: true
  phone: true
  gathering: true
  violence: false    # ResNet50 violence (requires custom checkpoint)
  object_left: true
  par: true
  vlm: true          # loads Qwen2.5-VL-3B NF4 (~2.3 GB VRAM)
```

### Bitchat Alerts (`configs/pipeline.yaml`)

```yaml
bitchat:
  enabled: true
  ip: "172.20.73.217"    # Android device WiFi IP
  port: 8765
  channel: "surveillance"
  rate_limit_s: 5.5
  send_keyframes: true   # attach camera image with each alert
  send_scene: true       # send ambient VLM scene descriptions
```

---

## License

- Code: MIT
- YOLO models: AGPL-3.0 (Ultralytics)
- D-Fire model: AGPL-3.0 (rabahdev)
- Smoking model: AGPL-3.0 (cadilak)
- InsightFace: MIT
- Qwen2.5-VL: Apache-2.0