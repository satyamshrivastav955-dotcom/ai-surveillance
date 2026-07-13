"""Phase 3 smoke test — verify ReID + face + identity modules import and construct.

Uses isolated config for face tests (temp dir index_path) so the production
face index at models/face_index.faiss is never touched.
"""
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_reid():
    from core.reid import ReIDExtractor, ReIDIndex, ReIDManager
    import numpy as np
    ext = ReIDExtractor()
    print(f"  [ok] reid extractor  dim={ext.input_size} half={ext.half}")
    idx = ReIDIndex()
    print(f"  [ok] reid index  dim={idx.dim} threshold={idx.match_threshold}")
    # quick extraction test on a synthetic crop
    crop = np.zeros((128, 64, 3), dtype=np.uint8)
    emb = ext.extract(crop)
    assert emb is not None and emb.shape == (512,), f"bad embedding shape: {None if emb is None else emb.shape}"
    print(f"  [ok] reid extract  shape={emb.shape} norm={np.linalg.norm(emb):.4f}")


def test_face():
    from core.face import FaceRecognizer
    from core.config import load_models_config
    import numpy as np
    cfg = load_models_config()
    tmpdir = tempfile.mkdtemp(prefix="face_smoke_")
    cfg["face"]["index_path"] = str(Path(tmpdir) / "smoke_index")
    rec = FaceRecognizer(cfg)
    rec.reset()   # isolated in-memory index
    print(f"  [ok] face recognizer  det_size={rec.det_size} dim={rec.dim}")
    # quick detect test on a blank frame (should find 0 faces)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = rec.detect_and_embed(frame)
    assert isinstance(dets, list)
    print(f"  [ok] face detect  blank-frame dets={len(dets)} (expected 0)")


def test_identity():
    from core.identity import IdentityManager
    from core.face import FaceMatch
    mgr = IdentityManager()
    # simulate a face match
    m = FaceMatch(track_id=1, name="Alice", similarity=0.85, face_bbox=(10, 10, 50, 50))
    ev = mgr.on_face_match(m, t=100.0)
    assert ev is not None and ev.label == "Alice"
    assert mgr.get_label(1) == "Alice"
    print(f"  [ok] identity fusion  face match -> label='Alice'")
    # simulate a second face match for same track (no event — already has identity)
    ev2 = mgr.on_face_match(m, t=101.0)
    assert ev2 is None
    print(f"  [ok] identity fusion  re-confirm same label -> no event")


def main():
    print("phase3_smoke:")
    test_reid()
    test_face()
    test_identity()
    print("phase3_smoke: all passed")


if __name__ == "__main__":
    main()