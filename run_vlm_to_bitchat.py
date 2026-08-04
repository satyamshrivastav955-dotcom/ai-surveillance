"""VLM + Bitchat integration: Send VLM scene descriptions and images to Bitchat.

This script runs the VLM on your webcam and sends:
  - Scene descriptions as text messages
  - Keyframe images with analysis captions

Setup:
  1. Update BITCHAT_IP in configs/pipeline.yaml OR set env var BITCHAT_IP
  2. Ensure Bitchat VLM API is enabled on your Android device
  3. Run: python run_vlm_to_bitchat.py

Controls:
  Q / ESC  — quit
  SPACE    — force immediate VLM inference and send to Bitchat
"""
from __future__ import annotations

import os
import sys
import time
import textwrap
import threading

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.bitchat import BitchatAlertClient, build_client_from_config
from core.config import load_pipeline_config


CAMERA_ID       = 0
CAPTURE_W       = 1280
CAPTURE_H       = 720
DISPLAY_W       = 960
VLM_INTERVAL_S  = 20.0
DEVICE          = "cuda:0"
MAX_NEW_TOKENS  = 256

SCENE_PROMPT = (
    "You are a surveillance camera AI. Look at this image carefully and describe "
    "in 1-2 sentences exactly what you see: the people, their actions, what they "
    "are doing, any objects of interest, and anything unusual. Be specific and factual."
)

ENTITY_PROMPT = (
    "You are analyzing a surveillance frame. List only what you can actually see. "
    "Respond with a JSON array and nothing else. Each element must be an object "
    'with keys: "type" (one of: person, fall, fire, smoke, fight, gathering, '
    'suspicious, object_left), "confidence" (a number 0.0-1.0), '
    'and "description" (a short factual phrase). '
    "If you see nothing notable, respond with []."
)


print("[vlm] Loading Qwen2.5-VL-3B-Instruct with NF4 quantization...")
print("[vlm] This takes ~30-45s on first run...")

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    quantization_config=bnb_cfg,
    device_map=DEVICE,
    torch_dtype=torch.float16,
)
model.eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"[vlm] Model loaded. VRAM: {vram_gb:.2f} GB")


cfg = load_pipeline_config()
bitchat_client = build_client_from_config(cfg)

if bitchat_client is None:
    print("[bitchat] ERROR: Bitchat client not configured. Check pipeline.yaml")
    sys.exit(1)


def run_prompt(frame_bgr: np.ndarray, prompt: str) -> str:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame_rgb},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[frame_rgb],
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    result = processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    return result.strip()


def vlm_inference(frame_bgr: np.ndarray) -> tuple[str, str]:
    t0 = time.perf_counter()

    scene = run_prompt(frame_bgr, SCENE_PROMPT)
    print(f"[vlm] SCENE: {scene}")

    entities_raw = run_prompt(frame_bgr, ENTITY_PROMPT)
    print(f"[vlm] ENTITIES: {entities_raw}")

    elapsed = time.perf_counter() - t0
    print(f"[vlm] inference done in {elapsed:.1f}s")
    return scene, entities_raw


_lock             = threading.Lock()
_vlm_scene: list  = ["Waiting for first VLM inference..."]
_vlm_entities: list = [""]
_vlm_running: list  = [False]
_vlm_last_t: list   = [0.0]
_force_vlm: list    = [False]


def _vlm_thread_fn(frame_snap: np.ndarray) -> None:
    with _lock:
        _vlm_running[0] = True
    try:
        scene, entities = vlm_inference(frame_snap)
        with _lock:
            _vlm_scene[0]    = scene
            _vlm_entities[0] = entities
            _vlm_last_t[0]   = time.perf_counter()
        
        print(f"[bitchat] Sending scene: {scene[:80]}...")
        bitchat_client.send_scene(scene, frame_snap)
        
    except Exception as e:
        print(f"[vlm] ERROR: {e}")
        import traceback
        traceback.print_exc()
        with _lock:
            _vlm_scene[0] = f"VLM error: {e}"
    finally:
        with _lock:
            _vlm_running[0] = False


def maybe_launch_vlm(frame: np.ndarray, forced: bool = False) -> None:
    with _lock:
        running = _vlm_running[0]
        last_t  = _vlm_last_t[0]

    if running:
        return

    elapsed_since_last = time.perf_counter() - last_t
    if not forced and elapsed_since_last < VLM_INTERVAL_S:
        return

    t = threading.Thread(target=_vlm_thread_fn, args=(frame.copy(),), daemon=True)
    t.start()


def draw_overlay(vis: np.ndarray, scene: str, entities: str, running: bool) -> None:
    h, w = vis.shape[:2]

    status = "🤔 VLM thinking..." if running else f"✅ VLM→Bitchat | SPACE=force | Q=quit"
    bar_color = (60, 30, 0) if running else (0, 50, 0)
    cv2.rectangle(vis, (0, 0), (w, 28), bar_color, -1)
    cv2.putText(vis, status, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    if scene:
        chars_per_line = max(40, w // 9)
        lines = []
        for para in scene.split("\n"):
            lines.extend(textwrap.wrap(para, chars_per_line) or [""])
        lines = lines[:4]

        box_h = len(lines) * 20 + 12
        cv2.rectangle(vis, (0, h - box_h - 4), (w, h), (0, 0, 0), -1)

        for i, line in enumerate(lines):
            y = h - box_h + i * 20 + 16
            cv2.putText(vis, line, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 160), 1)

    if entities and entities.strip() not in ("[]", ""):
        import json, re
        try:
            raw = re.sub(r"```(?:json)?|```", "", entities).strip()
            ents = json.loads(raw)
            x_start = w - 300
            cv2.rectangle(vis, (x_start - 6, 32), (w, 32 + len(ents) * 22 + 10),
                          (0, 0, 60), -1)
            for i, e in enumerate(ents):
                label = (f"[{e.get('type','?').upper()}] "
                         f"{int(e.get('confidence', 0) * 100)}%  "
                         f"{e.get('description', '')[:30]}")
                cv2.putText(vis, label, (x_start, 50 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 100), 1)
        except Exception:
            pass


print(f"\n[cam] Opening camera {CAMERA_ID} at {CAPTURE_W}x{CAPTURE_H} ...")
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

print("[cam] Camera open. Starting live feed.")
print(f"[vlm] First inference in ~{VLM_INTERVAL_S:.0f}s or press SPACE to force now.\n")

WIN = "VLM → Bitchat"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, DISPLAY_W, int(DISPLAY_W * CAPTURE_H / CAPTURE_W))

frame_idx = 0
fps_t0    = time.perf_counter()
fps_count = 0
fps       = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("[cam] Frame read failed — retrying...")
        time.sleep(0.05)
        continue

    frame_idx += 1
    fps_count  += 1
    now = time.perf_counter()
    if now - fps_t0 >= 0.5:
        fps    = fps_count / (now - fps_t0)
        fps_t0    = now
        fps_count = 0

    with _lock:
        forced = _force_vlm[0]
        if forced:
            _force_vlm[0] = False

    maybe_launch_vlm(frame, forced=forced)

    scale = DISPLAY_W / frame.shape[1]
    vis   = cv2.resize(frame, (DISPLAY_W, int(frame.shape[0] * scale)))

    with _lock:
        scene    = _vlm_scene[0]
        entities = _vlm_entities[0]
        running  = _vlm_running[0]

    draw_overlay(vis, scene, entities, running)

    fps_str = f"FPS:{fps:4.1f}"
    cv2.putText(vis, fps_str, (vis.shape[1] - 90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WIN, vis)
    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), 27):
        print("[cam] Quit.")
        break
    elif key == ord(" "):
        with _lock:
            _force_vlm[0] = True
        print("[vlm] Force inference requested by user.")

cap.release()
cv2.destroyAllWindows()
print("Done.")
