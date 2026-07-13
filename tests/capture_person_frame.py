"""Capture N webcam frames into a .npz so we can re-run benchmarks against a
real 'person in frame' still deterministically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from core.detector import Detector
from core.tracker import Tracker

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("webcam failed")

print("warming up camera + detector...")
det = Detector(); tr = Tracker(det)
# warmup
for _ in range(15): cap.read()

print("looking for a person in frame (try to be in view)...")
best_frame = None
best_n_persons = 0
for i in range(60):
    ok, f = cap.read()
    if not ok or f is None: continue
    trks = tr.update(f)
    n_persons = sum(1 for t in trks if t.cls == 0)
    if n_persons > best_n_persons:
        best_n_persons = n_persons
        best_frame = f.copy()
        print(f"  frame {i}: {n_persons} persons (best yet)")
    if i % 10 == 0:
        print(f"  frame {i}: {n_persons} persons")

cap.release()

if best_frame is None or best_n_persons == 0:
    print("NO PERSON detected in 60 frames — saving last frame anyway for posterity")
    if best_frame is None:
        # take whatever we got
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        for _ in range(15): cap.read()
        ok, best_frame = cap.read()
        cap.release()
out = Path("tests/real_person_frame.npz")
np.savez_compressed(str(out), frame=best_frame)
print(f"saved -> {out}  (best had {best_n_persons} persons)")