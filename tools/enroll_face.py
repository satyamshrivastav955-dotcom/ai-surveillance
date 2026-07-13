"""Face enrollment script — capture a person's face and register it under a name.

Captures N frames from the webcam, detects the largest face in each frame,
extracts a MobileFaceNet embedding, averages them for robustness, and saves
the averaged embedding to the persistent face FAISS index.

Usage:
    python tools/enroll_face.py --name "Alice"
    python tools/enroll_face.py --name "Bob" --frames 30 --camera 0

The enrolled face is saved to models/face_index.faiss + models/face_index.json
(the index_path from configs/models.yaml). The main loop loads this index on
startup so enrolled identities are recognized automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_models_config
from core.face import FaceRecognizer


def main():
    p = argparse.ArgumentParser(description="Enroll a face into the recognition index")
    p.add_argument("--name", required=True, help="Name to register the face under")
    p.add_argument("--frames", type=int, default=20, help="Number of frames to capture")
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--delay", type=float, default=0.2, help="Seconds between captures")
    args = p.parse_args()

    cfg = load_models_config()
    face_cfg = cfg.get("face", {})
    index_path = face_cfg.get("index_path", "models/face_index")

    print(f"Enrolling '{args.name}' — capturing {args.frames} frames...")
    print("Look at the camera. Press ESC to abort.")

    # load the face recognizer (downloads buffalo_s on first run)
    recognizer = FaceRecognizer(cfg)
    # load existing index if present (so we add to it, not replace it)
    recognizer.load_index(index_path)
    print(f"  existing enrollments: {len(recognizer._labels)} "
          f"({recognizer._labels})")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {args.camera}")

    # warmup camera + detector
    for _ in range(10):
        cap.read()

    embeddings = []
    captured = 0
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # detect faces in the full frame (not just a crop — enrollment should
        # see the face in context)
        detections = recognizer.detect_and_embed(frame)
        if not detections:
            print(f"  frame {i+1}/{args.frames}: no face detected")
            # still show the frame so the user can position themselves
            cv2.imshow("enroll", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("aborted")
                break
            time.sleep(args.delay)
            continue
        # take the highest-confidence face
        best = max(detections, key=lambda d: d.conf)
        if best.embedding is not None:
            embeddings.append(best.embedding)
            captured += 1
            x1, y1, x2, y2 = best.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"captured {captured}/{args.frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("enroll", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("aborted")
            break
        time.sleep(args.delay)

    cap.release()
    cv2.destroyAllWindows()

    if captured < 5:
        print(f"ERROR: only captured {captured} frames with faces (need >= 5). "
              f"Try again with better lighting / face the camera directly.")
        return

    # average the embeddings for robustness, then re-normalize
    avg_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm < 1e-6:
        print("ERROR: averaged embedding has zero norm — something went wrong.")
        return
    avg_emb = (avg_emb / norm).astype(np.float32)

    # add to the index and save
    recognizer.enroll(args.name, avg_emb)
    recognizer.save_index(index_path)
    print(f"\nEnrolled '{args.name}' from {captured} face captures.")
    print(f"  saved to {index_path}.faiss + {index_path}.json")
    print(f"  total enrollments: {len(recognizer._labels)}")
    print(f"\nThe main loop will now recognize this face automatically.")


if __name__ == "__main__":
    main()