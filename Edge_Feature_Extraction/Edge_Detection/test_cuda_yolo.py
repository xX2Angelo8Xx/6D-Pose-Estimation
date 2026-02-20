#!/usr/bin/env python3
"""Test CUDA detection and YOLO performance."""

import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

print("=" * 60)
print("CUDA & YOLO Performance Test")
print("=" * 60)

# Check CUDA
print(f"\n1. PyTorch CUDA Status:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("   WARNING: CUDA not available!")

# Load YOLO model
yolo_model_path = Path("/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt")
print(f"\n2. Loading YOLO model:")
print(f"   Path: {yolo_model_path}")

model = YOLO(str(yolo_model_path))
print(f"   ✓ Model loaded")

# Test image
test_image_path = Path("/media/angelo/DRONE_DATA1/SVO2_Frame_Export/1.1/frame_000350.jpg")
print(f"\n3. Loading test image:")
print(f"   Path: {test_image_path}")
img = cv2.imread(str(test_image_path))
print(f"   Shape: {img.shape}")

# Test CPU
print(f"\n4. YOLO Inference Test (CPU):")
start = time.perf_counter()
results_cpu = model(img, verbose=False, device='cpu')
end = time.perf_counter()
cpu_time = (end - start) * 1000.0
print(f"   Time: {cpu_time:.2f} ms")
if len(results_cpu[0].boxes) > 0:
    print(f"   Detections: {len(results_cpu[0].boxes)}")
    box = results_cpu[0].boxes[0]
    conf = box.conf[0].item()
    print(f"   Confidence: {conf:.3f}")

# Test CUDA
if torch.cuda.is_available():
    print(f"\n5. YOLO Inference Test (CUDA):")
    
    # Warmup
    print(f"   Warming up GPU...")
    for _ in range(3):
        _ = model(img, verbose=False, device='cuda:0')
    
    # Actual test
    print(f"   Running inference...")
    times = []
    for i in range(5):
        start = time.perf_counter()
        results_cuda = model(img, verbose=False, device='cuda:0')
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    
    avg_time = sum(times) / len(times)
    print(f"   Average Time: {avg_time:.2f} ms (over 5 runs)")
    print(f"   Min: {min(times):.2f} ms, Max: {max(times):.2f} ms")
    if len(results_cuda[0].boxes) > 0:
        print(f"   Detections: {len(results_cuda[0].boxes)}")
    
    # Speedup
    speedup = cpu_time / avg_time
    print(f"\n6. Performance Comparison:")
    print(f"   CPU: {cpu_time:.2f} ms")
    print(f"   GPU: {avg_time:.2f} ms")
    print(f"   Speedup: {speedup:.2f}x faster on GPU")
else:
    print(f"\n5. CUDA test skipped (not available)")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
