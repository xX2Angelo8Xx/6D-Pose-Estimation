#!/usr/bin/env python3
"""
Comprehensive YOLO Optimization Test for Jetson Orin Nano.
Tests different configurations to reach <200ms target.
"""

import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

print("=" * 70)
print("YOLO Optimization Test - Target: <200ms for Real-Time")
print("=" * 70)

# Setup
yolo_model_path = Path("/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt")
test_image_path = Path("/media/angelo/DRONE_DATA1/SVO2_Frame_Export/1.1/frame_000350.jpg")

print(f"\nDevice: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Model: {yolo_model_path.name}")

# Load image
img = cv2.imread(str(test_image_path))
print(f"Image: {img.shape}")

# Load model
print("\nLoading model...")
model = YOLO(str(yolo_model_path))

# Test configurations
configs = [
    # (device, imgsz, half, description)
    ("cpu", 640, False, "Baseline CPU (current)"),
    ("cuda:0", 640, False, "GPU FP32"),
    ("cuda:0", 640, True, "GPU FP16 (Half Precision)"),
    ("cuda:0", 512, True, "GPU FP16 + Smaller Input (512)"),
    ("cuda:0", 416, True, "GPU FP16 + Smaller Input (416)"),
    ("cuda:0", 320, True, "GPU FP16 + Smallest Input (320)"),
]

print("\n" + "=" * 70)
print("Testing Configurations")
print("=" * 70)

results_data = []

for device, imgsz, half, desc in configs:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"\n⊘ {desc}: Skipped (CUDA not available)")
        continue
    
    print(f"\n📊 {desc}")
    print(f"   Settings: device={device}, imgsz={imgsz}, half={half}")
    
    # Warmup
    if device.startswith("cuda"):
        print("   Warming up GPU...")
        for _ in range(3):
            _ = model(img, verbose=False, device=device, imgsz=imgsz, half=half)
    
    # Benchmark
    times = []
    print("   Running benchmark (10 iterations)...")
    for i in range(10):
        start = time.perf_counter()
        result = model(img, verbose=False, device=device, imgsz=imgsz, half=half)
        end = time.perf_counter()
        t = (end - start) * 1000.0
        times.append(t)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Check if target met
    target_met = "✓ TARGET MET!" if avg_time < 200 else "✗ Too slow"
    fps = 1000.0 / avg_time
    
    print(f"   Results:")
    print(f"     Average: {avg_time:.2f} ms ({fps:.1f} FPS)")
    print(f"     Min: {min_time:.2f} ms, Max: {max_time:.2f} ms")
    print(f"     Status: {target_met}")
    
    # Store results
    results_data.append({
        'desc': desc,
        'device': device,
        'imgsz': imgsz,
        'half': half,
        'avg_ms': avg_time,
        'fps': fps,
        'target_met': avg_time < 200
    })

# Summary
print("\n" + "=" * 70)
print("SUMMARY - Ranked by Speed")
print("=" * 70)

results_data.sort(key=lambda x: x['avg_ms'])

print(f"\n{'Rank':<6} {'Configuration':<40} {'Time (ms)':<12} {'FPS':<8} {'Target'}")
print("-" * 70)

for i, r in enumerate(results_data, 1):
    target_str = "✓ YES" if r['target_met'] else "✗ NO"
    print(f"{i:<6} {r['desc']:<40} {r['avg_ms']:>8.2f} ms  {r['fps']:>6.1f}  {target_str}")

# Recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

fastest = results_data[0]
print(f"\n🏆 Fastest Configuration:")
print(f"   {fastest['desc']}")
print(f"   Time: {fastest['avg_ms']:.2f} ms ({fastest['fps']:.1f} FPS)")

if fastest['target_met']:
    print(f"\n✓ Target <200ms achieved!")
    print(f"   Use: device='{fastest['device']}', imgsz={fastest['imgsz']}, half={fastest['half']}")
else:
    print(f"\n⚠️  Target not met with PyTorch inference.")
    print(f"   Current best: {fastest['avg_ms']:.2f} ms (need {fastest['avg_ms'] - 200:.2f} ms improvement)")
    print(f"\n💡 TensorRT Solution:")
    print(f"   TensorRT can provide 10-20x speedup on Jetson devices.")
    print(f"   Estimated TensorRT time: {fastest['avg_ms'] / 15:.2f} ms (15x speedup)")
    print(f"\n   To export TensorRT engine:")
    print(f"   >>> model.export(format='engine', half=True, imgsz={fastest['imgsz']})")
    print(f"   >>> # Takes ~5 minutes, but only needs to be done once")

# TensorRT check
print("\n" + "=" * 70)
print("TENSORRT STATUS")
print("=" * 70)

engine_path = yolo_model_path.parent / f"{yolo_model_path.stem}_fp16.engine"
if engine_path.exists():
    print(f"\n✓ TensorRT engine found: {engine_path}")
    print(f"  Use this for maximum speed!")
else:
    print(f"\n✗ No TensorRT engine found")
    print(f"  Expected location: {engine_path}")
    print(f"\n  Create with: python -c \"from ultralytics import YOLO; YOLO('{yolo_model_path}').export(format='engine', half=True, imgsz=416)\"")

print("\n" + "=" * 70)
