"""Honest Phase 2 throughput bench: runs the FULL detector+tracker+pose+fall
path against a captured real-person frame so all four stages actually fire
on every iteration. This isolates the GPU cost of pose from the synthetic
source's 0-tracks problem (rectangles don't classify as `person`).

    python benchmarks/phase2_static_bench.py
    python benchmarks/phase2_static_bench.py --no-pose          # A/B vs Phase 1
    python benchmarks/phase2_static_bench.py --runtime pytorch  # rollback compare
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _vram_via_nvml() -> tuple[int, int]:
    """For ONNX direct mode ORT allocates outside the PyTorch cache, so
    torch.cuda.memory_allocated reads 0. Use NVML (pynvml) to read the
    process's actual GPU memory instead."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(h) if hasattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2") else pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        import os
        my_pid = os.getpid()
        for pc in procs:
            if pc.pid == my_pid:
                return int(pc.usedGpuMemory / (1024 * 1024)), 0
        return -1, -1
    except Exception:
        # fallback to torch stats (will be 0 in onnx_direct mode)
        try:
            import torch
            return (int(torch.cuda.memory_allocated() / (1024 * 1024)),
                    int(torch.cuda.max_memory_allocated() / (1024 * 1024)))
        except Exception:
            return -1, -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--no-pose", action="store_true",
                   help="skip the pose model (A/B vs full Phase 2 path)")
    p.add_argument("--runtime", choices=["pytorch", "onnx", "onnx_direct"],
                   default=None, help="override detector+pose runtime")
    p.add_argument("--cuda-event", action="store_true",
                   help="use CUDA events for GPU-only timing (excludes host contention)")
    args = p.parse_args()

    from core.config import load_models_config, load_pipeline_config
    from core.detector import Detector
    from core.tracker import Tracker
    from pipeline.frame_router import FrameRouter

    cfg_m = load_models_config()
    cfg_p = load_pipeline_config()
    if args.runtime:
        cfg_m["detector"]["runtime"] = args.runtime
        cfg_m["pose"]["runtime"] = args.runtime
        print(f"runtime override: {args.runtime}")
    det = Detector(cfg_m); tr = Tracker(det)
    router = FrameRouter(cfg_p.get("router", {}))
    pose_every = router.every("pose")
    pose_est = None
    fall_det = None
    if not args.no_pose and router.is_enabled("pose"):
        from core.pose import PoseEstimator
        from core.state_machine import FallDetector
        pose_est = PoseEstimator(cfg_m)
        fall_det = FallDetector(cfg_m.get("fall", {}))
        print(f"pose cadence: every={pose_every} (via FrameRouter)")

    import torch
    torch.cuda.reset_peak_memory_stats()

    # load the captured real-person frame
    frame_path = Path(__file__).resolve().parent.parent / "tests" / "real_person_frame.npz"
    if not frame_path.exists():
        print(f"missing {frame_path}; run tests/capture_person_frame.py first")
        return
    frame = np.load(str(frame_path))["frame"]
    print(f"loaded frame: shape={frame.shape} dtype={frame.dtype}")

    # warmup
    trks = tr.update(frame)
    n_persons = sum(1 for t in trks if t.cls == 0)
    print(f"  warmup tracks: {len(trks)} total, {n_persons} persons")
    if pose_est is not None and n_persons > 0:
        persons = [t for t in trks if t.cls == 0]
        poses = pose_est.estimate_crops(frame, persons)
        print(f"  warmup poses: {len(poses)}")
        if poses:
            print(f"  kpts shape: {poses[0].keypoints.shape}  conf mean: {poses[0].keypoints[:, 2].mean():.3f}")
    torch.cuda.synchronize()

    # timed loop
    n = 0
    n_poses = 0
    n_falls = 0
    if args.cuda_event:
        times = []
        for i in range(args.iters):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            trks = tr.update(frame)
            if pose_est is not None and router.should_run("pose", i):
                persons = [t for t in trks if t.cls == 0]
                if persons:
                    poses = pose_est.estimate_crops(frame, persons)
                    n_poses += len(poses)
                    if fall_det is not None:
                        events = fall_det.update(poses, frame_idx=i, t=time.perf_counter())
                        n_falls += len(events)
            e.record(); e.synchronize()
            times.append(s.elapsed_time(e))
            n += 1
        import statistics
        el = sum(times) / 1000.0
        fps_median = 1000.0 / statistics.median(times)
        fps_mean = 1000.0 / statistics.mean(times)
        print(f"  GPU-only: median={statistics.median(times):.2f}ms mean={statistics.mean(times):.2f}ms")
        print(f"  implied FPS: median={fps_median:.1f}  mean={fps_mean:.1f}")
    else:
        t0 = time.perf_counter()
        for i in range(args.iters):
            trks = tr.update(frame)
            if pose_est is not None and router.should_run("pose", i):
                persons = [t for t in trks if t.cls == 0]
                if persons:
                    poses = pose_est.estimate_crops(frame, persons)
                    n_poses += len(poses)
                    if fall_det is not None:
                        events = fall_det.update(poses, frame_idx=i, t=time.perf_counter())
                        n_falls += len(events)
            n += 1
        torch.cuda.synchronize()
        el = time.perf_counter() - t0

    # VRAM: use NVML when on onnx_direct (PyTorch cache is empty); else torch stats
    if cfg_m["detector"].get("runtime") == "onnx_direct":
        alloc, peak = _vram_via_nvml()
        vram_label = "nvml"
    else:
        alloc = int(torch.cuda.memory_allocated() / (1024 * 1024))
        peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
        vram_label = "torch"

    print("-" * 50)
    print(f"iters               {n}")
    print(f"elapsed             {el:.2f}s")
    if not args.cuda_event:
        print(f"throughput          {n / el:.1f} FPS")
    print(f"vram {vram_label:5}        {alloc} / {peak} MB")
    if pose_est is not None:
        print(f"poses (running tot) {n_poses}  falls: {n_falls}")
    print(f"runtime             det={cfg_m['detector'].get('runtime')} pose={cfg_m['pose'].get('runtime')}")
    print(f"config              pose={'on (every='+str(pose_every)+')' if pose_est else 'off'}")


if __name__ == "__main__":
    main()