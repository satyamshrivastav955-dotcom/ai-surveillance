"""Quick A/B: ultralytics-ONNX vs ultralytics-PyTorch on the same frame.
Helps decide whether to use YOLO.onnx (safe, same outputs) or write direct
ORT (faster but needs hand-written pre/post)."""
import sys, statistics, time
sys.path.insert(0, '.')
import numpy as np, torch
from ultralytics import YOLO

frame = np.load('tests/real_person_frame.npz')['frame']

def bench(model, label, n=200):
    for _ in range(10): model.predict(frame, imgsz=640, conf=0.35, iou=0.5, classes=[0], verbose=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); model.predict(frame, imgsz=640, conf=0.35, iou=0.5, classes=[0], verbose=False); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    print(f'{label:20} median={statistics.median(times):.2f}ms mean={statistics.mean(times):.2f}ms max={max(times):.2f}ms')

pt = YOLO('models/yolov8n.pt')
ox = YOLO('models/yolov8n.onnx', task='detect')
bench(pt, 'PyTorch (FP16)')
bench(ox, 'ONNX (FP16, CUDA EP)')
print(f'vram after both: {torch.cuda.memory_allocated()//(1024*1024)}/{torch.cuda.max_memory_allocated()//(1024*1024)} MB')