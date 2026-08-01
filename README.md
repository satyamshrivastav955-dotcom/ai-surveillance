# AI Surveillance

Real-time AI security camera system — end-to-end deep learning pipeline with a fully integrated **Vision Language Model (VLM) layer** for natural-language scene understanding and **Bitchat mesh network alerting** to push alerts directly to your Android phone.

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
| **6** | **VLM layer — Qwen2.5-VL-3B, scene description, entity detection, NL query** | ✅ **live** |
| **6+** | **Bitchat mesh alerting — push every alert + image to Android phone** | ✅ **live** |

---

## Quick Start

```bash
# Full pipeline (all detectors + VLM + Bitchat alerts)
python -m pipeline.main_loop

# VLM only — just open the camera and describe what it sees
python run_vlm_only.py

# Run all tests
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

## Bitchat Mesh Alerting

Every surveillance alert and VLM scene description is pushed to your **Android phone via Bitchat's mesh network** as a message + camera keyframe image.

### Setup

1. Install [Bitchat](https://github.com/permissionlesstech/bitchat-android) on your Android phone
2. Open Bitchat → **Settings → VLM API Settings → Enable**
3. Make sure your phone and PC are on the **same WiFi**
4. Edit `configs/pipeline.yaml`:
   ```yaml
   bitchat:
     enabled: true
     ip: "10.99.134.211"   # your Android WiFi IP (auto-scan: python core/bitchat.py)
     channel: "surveillance"
   ```
5. Run the pipeline — alerts flow automatically

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

### Find your Android IP automatically

```bash
python -c "
import socket, concurrent.futures, requests
local = socket.gethostbyname(socket.gethostname())
subnet = '.'.join(local.split('.')[:3])
def check(ip):
    try:
        r = requests.get(f'http://{ip}:8765/status', timeout=1)
        return ip, r.json()
    except: return None
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
    for r in concurrent.futures.as_completed([ex.submit(check, f'{subnet}.{i}') for i in range(1,255)]):
        if r.result(): print('Found:', r.result())
"
```

---

## What Each Detector Does

| Detector | Signal | Status |
|---|---|---|
| **Fall** | YOLOv8n-pose keypoints + bbox aspect ratio, rule-based state machine | ✅ live-tested |
| **Fire/Smoke** | YOLOv8n fine-tuned on D-Fire (conf=0.45, multi-frame ≥2/5) | ✅ real model |
| **Smoking** | YOLOv8n fine-tuned for cigarette/vape near tracked person | ✅ real model |
| **Phone** | YOLOv8n (COCO cls 67), imgsz=480, confirm+hold hysteresis (3/15) | ✅ stable |
| **Gathering** | Fixed-radius centroid clustering, fires on 3+ people within 150 px | ✅ works |
| **Violence** | Bbox overlap (IoU ≥ 0.3) + rapid motion ≥ 40 px/frame for 1.5s | ✅ heuristic + VLM verify |
| **Object-Left** | Stationary non-person objects (bags, suitcases) tracked for >30s | ✅ complete |
| **PAR** | Pedestrian Attribute Recognition on tracked crops | ✅ complete |
| **ReID** | OSNet-x0.25 (512-dim, half), match_threshold=0.65 | ✅ complete |
| **Face** | SCRFD detection + MobileFaceNet embedding + FAISS index | ✅ complete |
| **VLM** | Qwen2.5-VL-3B-Instruct NF4, scene description + entity JSON | ✅ live |

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

> The VLM runs in a `threading.Thread` daemon — the main capture loop never waits for it. SPACE key in `run_vlm_only.py` forces inference immediately.

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
  reid: true
  face: true
  fire_smoke: true
  smoking: true
  phone: true
  gathering: true
  violence: true
  object_left: true
  par: true
  vlm: true          # loads Qwen2.5-VL-3B NF4 (~2.3 GB VRAM)
```

### Bitchat Alerts (`configs/pipeline.yaml`)

```yaml
bitchat:
  enabled: true
  ip: "10.99.134.211"    # Android device WiFi IP
  port: 8765
  channel: "surveillance"
  rate_limit_s: 5.5
  send_keyframes: true   # attach camera image with each alert
  send_scene: true       # send ambient VLM scene descriptions
```

### VLM Config (`configs/vlm.yaml`)

```yaml
model:
  name: Qwen/Qwen2.5-VL-3B-Instruct
  device: cuda:0
  dtype: float16
  quantization: nf4      # NF4 4-bit via bitsandbytes (~2.3 GB VRAM)
  forced_interval: 900   # frames between ambient VLM passes (~30s at 30fps)
```

---

## Debug Flags

| Env var | Effect |
|---|---|
| `PHONE_DEBUG=1` | Print raw YOLO confidences for every phone-detector call |
| `FALL_DEBUG=1` | Log next 10 fall trigger candidates with full signal breakdown |
| `DEBUG_DEVICE=1` | Print model device before each inference call |

```bash
PHONE_DEBUG=1 python -m pipeline.main_loop
```

---

## Models

| Model | Source | License |
|---|---|---|
| `fire_smoke_yolov8n.pt` | [rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) | AGPL-3.0 |
| `smoking_yolov8n.pt` | [cadilak/smoking-detection-yolov8](https://huggingface.co/cadilak/smoking-detection-yolov8) | AGPL-3.0 |
| `osnet_x0_25_market_duke.pth` | [torchreid pretrained](https://kaiyangzhou.github.io/deep-person-reid/) | MIT |
| `Qwen2.5-VL-3B-Instruct` | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Apache-2.0 |
| `buffalo_s` (InsightFace) | [InsightFace model zoo](https://github.com/deepinsight/insightface) | MIT |

Weights go in `models/`. See `models/README.md` for download links.

---

## License

- Code: MIT
- YOLO models: AGPL-3.0 (Ultralytics)
- D-Fire model: AGPL-3.0 (rabahdev)
- Smoking model: AGPL-3.0 (cadilak)
- InsightFace: MIT
- Qwen2.5-VL: Apache-2.0