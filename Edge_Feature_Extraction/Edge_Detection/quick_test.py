#!/usr/bin/env python3
"""Quick test of optimized YOLO settings."""

from pathlib import Path
import cv2
from ultralytics import YOLO
import time
import torch

print("Quick Optimization Test")
print("=" * 50)

# Check CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Load
model_path = "/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt"
img_path = "/media/angelo/DRONE_DATA1/SVO2_Frame_Export/1.1/frame_000350.jpg"

print(f"\nLoading model: {Path(model_path).name}")
model = YOLO(model_path)

print(f"Loading image: {Path(img_path).name}")
img = cv2.imread(img_path)
print(f"Image shape: {img.shape}")

# Test configurations
configs = [
    ("cuda:0", 320, True, "Optimized (GPU FP16 320)"),
    ("cuda:0", 416, True, "GPU FP16 416"),
    ("cpu", 320, False, "CPU 320"),
]

print("\n" + "=" * 50)
print("Testing...")
print("=" * 50)

for device, imgsz, half, desc in configs:
    print(f"\n{desc}")
    print(f"  Settings: device={device}, imgsz={imgsz}, half={half}")
    
    try:
        # Warmup
        print("  Warming up...")
        for _ in range(2):
            _ = model(img, verbose=False, device=device, imgsz=imgsz, half=half and device.startswith("cuda"))
        
        # Test
        print("  Running 5 iterations...")
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = model(img, verbose=False, device=device, imgsz=imgsz, half=half and device.startswith("cuda"))
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg = sum(times) / len(times)
        fps = 1000 / avg
        target_met = "✓" if avg < 200 else "✗"
        
        print(f"  Result: {avg:.2f} ms ({fps:.1f} FPS) {target_met}")
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 50)
print("Test Complete")
print("=" * 50)
