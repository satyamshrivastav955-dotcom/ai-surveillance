"""Phase 2 import + construction smoke test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print('importing...')
from core.pose import PoseEstimator, KP_LEFT_HIP
from core.state_machine import FallDetector, FallEvent
print('constructing pose (downloads ~7MB on first run)...')
p = PoseEstimator()
print(f'  pose ok  imgsz={p.imgsz} half={p.half}')
f = FallDetector()
print(f'  fall ok  aspect_thresh={f.aspect_threshold} window={f.transition_window_s}s')
print('SMOKE OK')