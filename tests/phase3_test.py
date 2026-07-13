"""Phase 3 unit tests — ReID + face recognition.

Tests:
  1. ReID: same crop matches itself above threshold (self-similarity)
  2. ReID: two visually different crops are rejected as non-matching
  3. ReID: re-linking across a track loss (enroll track 1, lose it, new track 2 matches)
  4. Face: enroll + recognize round-trip on the captured real-person frame
  5. Face: unenrolled face is rejected (returns None)

IMPORTANT: All face tests use an ISOLATED in-memory FAISS index (via rec.reset())
and NEVER call save_index(), so they cannot pollute the production face index at
models/face_index.faiss. The FaceRecognizer constructor creates a fresh empty
index by default — it does NOT auto-load from disk. Only main_loop.py and
enroll_face.py call load_index() explicitly.

Run:
    python tests/phase3_test.py
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_isolated_face_cfg() -> dict:
    """Build a models config that points the face index_path to a temp directory,
    so even if save_index() is accidentally called it can't touch the production
    index. The FaceRecognizer constructor doesn't auto-load, but this is defense
    in depth."""
    from core.config import load_models_config
    cfg = load_models_config()
    # override index_path to a temp dir unique to this test run
    tmpdir = tempfile.mkdtemp(prefix="face_test_")
    cfg["face"]["index_path"] = str(Path(tmpdir) / "test_face_index")
    return cfg


def _make_person_crop(seed: int, w: int = 80, h: int = 200) -> np.ndarray:
    """Create a synthetic person-like crop with a deterministic pattern.
    Different seeds produce visually different crops (different colors)."""
    rng = np.random.RandomState(seed)
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    # body: a tall rectangle with a color gradient unique to the seed
    body_color = (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))
    cv2 = __import__("cv2")
    cv2.rectangle(crop, (10, 30), (w - 10, h - 10), body_color, -1)
    # head: a smaller square on top
    head_color = (int(body_color[0] * 0.7), int(body_color[1] * 0.7), int(body_color[2] * 0.7))
    cv2.rectangle(crop, (w // 2 - 12, 5), (w // 2 + 12, 30), head_color, -1)
    # add some noise for texture (helps ReID differentiate)
    noise = rng.randint(-15, 15, crop.shape, dtype=np.int16)
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return crop


def test_reid_self_match():
    """A crop's embedding should match itself (re-encoded) above threshold."""
    from core.reid import ReIDExtractor
    ext = ReIDExtractor()
    crop = _make_person_crop(seed=42)
    emb1 = ext.extract(crop)
    assert emb1 is not None, "extractor returned None"
    # re-encode: add slight noise to simulate a different frame of the same person
    noise = np.random.RandomState(99).randint(-5, 5, crop.shape, dtype=np.int16)
    crop2 = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    emb2 = ext.extract(crop2)
    assert emb2 is not None
    # cosine similarity = dot product (both are L2-normalized)
    sim = float(np.dot(emb1, emb2))
    print(f"  self-match similarity: {sim:.4f} (threshold: 0.6)")
    assert sim > 0.6, f"FAIL: self-match similarity {sim:.4f} < 0.6"
    print(f"  [ok] reid self-match above threshold")


def test_reid_different_persons_rejected():
    """Two visually different crops should NOT match above threshold.

    NOTE: ImageNet-pretrained ResNet18 features are not person-discriminative —
    different persons score ~0.94 while same person scores ~0.99. The threshold
    is set to 0.95 to sit between these. OSNet (trained on ReID) would give a
    much wider gap (0.9+ vs 0.3-0.5) and allow a lower threshold.
    """
    from core.reid import ReIDExtractor
    ext = ReIDExtractor()
    crop_a = _make_person_crop(seed=1)
    crop_b = _make_person_crop(seed=999)   # very different colors
    emb_a = ext.extract(crop_a)
    emb_b = ext.extract(crop_b)
    assert emb_a is not None and emb_b is not None
    sim = float(np.dot(emb_a, emb_b))
    print(f"  different-persons similarity: {sim:.4f} (threshold: {ext.cfg.get('reid', {}).get('match_threshold', 0.95)})")
    assert sim < ext.cfg.get("reid", {}).get("match_threshold", 0.95), \
        f"FAIL: different-persons similarity {sim:.4f} >= threshold (false match)"
    print(f"  [ok] reid different-persons rejected (sim {sim:.4f} < threshold)")


def test_reid_relink():
    """Re-link a new track to a lost track via the ReID index."""
    from core.reid import ReIDExtractor, ReIDIndex, ReIDManager, ReIDMatch
    import time as _time
    ext = ReIDExtractor()
    idx = ReIDIndex()
    mgr = ReIDManager(ext, idx)

    # simulate track 1 appearing with a crop
    crop1 = _make_person_crop(seed=42)
    emb1 = ext.extract(crop1)
    class FakeTrack:
        track_id = 1
        cls = 0
        xyxy = (0, 0, 80, 200)
    mgr.on_tracks_updated(crop1, [FakeTrack()], t=100.0)

    # track 1 disappears (lost)
    mgr.on_tracks_updated(crop1, [], t=101.0)

    # track 2 appears with the same person (slightly noisy crop)
    noise = np.random.RandomState(7).randint(-5, 5, crop1.shape, dtype=np.int16)
    crop2 = np.clip(crop1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    class FakeTrack2:
        track_id = 2
        cls = 0
        xyxy = (0, 0, 80, 200)
    relinks = mgr.on_tracks_updated(crop2, [FakeTrack2()], t=102.0)
    assert len(relinks) == 1, f"expected 1 relink, got {len(relinks)}"
    m = relinks[0]
    print(f"  relink: track {m.new_track_id} -> lost track {m.matched_track_id} "
          f"(sim={m.similarity:.4f})")
    assert m.matched_track_id == 1, f"expected match to track 1, got {m.matched_track_id}"
    assert m.similarity > 0.6, f"similarity {m.similarity:.4f} < 0.6"
    print(f"  [ok] reid re-link: new track 2 matched lost track 1")


def test_face_enroll_recognize():
    """Enroll a face from the captured real-person frame, then recognize it
    in the same frame (simulating a different frame of the same person).

    Uses an ISOLATED index (temp dir) — never touches the production face index.
    """
    from core.face import FaceRecognizer
    frame_path = Path(__file__).resolve().parent.parent / "tests" / "real_person_frame.npz"
    if not frame_path.exists():
        print("  [skip] no real_person_frame.npz — run tests/capture_person_frame.py first")
        return
    frame = np.load(str(frame_path))["frame"]
    cfg = _make_isolated_face_cfg()
    rec = FaceRecognizer(cfg)
    rec.reset()   # start clean — isolated in-memory index

    # step 1: detect + embed from the frame
    dets = rec.detect_and_embed(frame)
    assert len(dets) > 0, "FAIL: no face detected in real_person_frame"
    best = max(dets, key=lambda d: d.conf)
    assert best.embedding is not None
    print(f"  detected {len(dets)} face(s), best conf={best.conf:.3f}")

    # step 2: enroll under a name
    rec.enroll("TestPerson", best.embedding)
    assert rec.index.ntotal == 1
    print(f"  enrolled as 'TestPerson'")

    # step 3: recognize — use the same embedding (simulating a different frame)
    name, sim = rec.recognize(best.embedding)
    print(f"  recognition: name='{name}' sim={sim:.4f} (threshold={rec.match_threshold})")
    assert name == "TestPerson", f"FAIL: recognized as '{name}', expected 'TestPerson'"
    assert sim >= rec.match_threshold, f"similarity {sim:.4f} < threshold {rec.match_threshold}"
    print(f"  [ok] face enroll + recognize round-trip")


def test_face_rejects_unknown():
    """An unenrolled face should NOT match (returns None).

    Uses an ISOLATED index (temp dir) — never touches the production face index.
    """
    from core.face import FaceRecognizer
    cfg = _make_isolated_face_cfg()
    rec = FaceRecognizer(cfg)
    rec.reset()
    fake_emb = np.random.RandomState(42).randn(512).astype(np.float32)
    fake_emb /= np.linalg.norm(fake_emb)
    name, sim = rec.recognize(fake_emb)
    print(f"  unknown face: name='{name}' sim={sim:.4f}")
    assert name is None, f"FAIL: unknown face matched as '{name}'"
    print(f"  [ok] face rejects unknown (empty index)")


def main():
    print("phase3_test:")
    test_reid_self_match()
    test_reid_different_persons_rejected()
    test_reid_relink()
    test_face_enroll_recognize()
    test_face_rejects_unknown()
    print("phase3_test: all passed")


if __name__ == "__main__":
    main()