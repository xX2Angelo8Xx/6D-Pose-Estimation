#!/usr/bin/env python3
"""
Minimaler GPU Test - Findet die cuDNN Engine das Problem?
"""

import torch
import torch.nn as nn
import time

print("=" * 70)
print("GPU Diagnose - Minimaler Test")
print("=" * 70)
print()

# System Info
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
print(f"GPU available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

# Test 1: Einfache Matrix Operation
print("Test 1: Einfache GPU Operation...")
try:
    x = torch.randn(100, 100).cuda()
    y = x @ x.T
    print("✅ Einfache Matrix-Multiplikation funktioniert")
except Exception as e:
    print(f"❌ Fehler: {e}")
    exit(1)

# Test 2: Convolution (Das ist wo es scheitert)
print()
print("Test 2: Convolution (das Problem)...")

# Disable benchmarking (kann helfen)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    # Kleine Convolution
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()
    x = torch.randn(1, 3, 32, 32).cuda()
    
    print("   Versuche Convolution mit allow_tf32=True...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    y = conv(x)
    print("   ✅ Funktioniert mit allow_tf32=True")
    
except RuntimeError as e:
    if "GET was unable" in str(e):
        print(f"   ❌ cuDNN Engine Error (wie erwartet)")
        print()
        print("   Versuche Workaround...")
        
        # Workaround 1: Disable cuDNN
        try:
            torch.backends.cudnn.enabled = False
            conv = nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()
            x = torch.randn(1, 3, 32, 32).cuda()
            y = conv(x)
            print("   ✅ Funktioniert OHNE cuDNN (langsamer aber funktioniert)")
            torch.backends.cudnn.enabled = True  # Re-enable für weitere Tests
        except Exception as e2:
            print(f"   ❌ Auch ohne cuDNN Fehler: {e2}")
            
    else:
        print(f"   ❌ Anderer Fehler: {e}")

# Test 3: YOLO auf GPU (mit Workaround)
print()
print("Test 3: YOLO auf GPU mit Workaround...")
print("=" * 70)

# Workaround Settings
torch.backends.cudnn.enabled = False  # Disable cuDNN, use native CUDA
print("⚙️  cuDNN disabled, using native CUDA kernels")

from ultralytics import YOLO
import cv2

MODEL_PATH = "/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt"
TEST_IMG = "./data/debug_images/originals/frame_000544-1.5-SE_Top-depth-4.94m-std-0.38m.jpg"

img = cv2.imread(TEST_IMG)
print(f"📁 Bild: {img.shape[1]}x{img.shape[0]}")

print("🔄 Lade YOLO...")
model = YOLO(MODEL_PATH)

# Warmup
print("🔥 Warmup...")
try:
    _ = model(img, verbose=False, device='cuda:0', imgsz=320, half=False)
    print("✅ Warmup erfolgreich!")
    
    # Benchmark
    print()
    print("📊 Benchmark...")
    times = []
    for i in range(5):
        t0 = time.time()
        results = model(img, verbose=False, device='cuda:0', imgsz=320, half=False)
        t1 = time.time()
        times.append((t1 - t0) * 1000)
        print(f"   Run {i+1}: {times[-1]:.1f} ms")
    
    avg_time = sum(times) / len(times)
    print()
    print(f"⏱️  Durchschnitt: {avg_time:.1f} ms")
    
    if avg_time < 200:
        print(f"✅ Unter 200ms! GPU funktioniert (ohne cuDNN)")
    else:
        print(f"⚠️  Über 200ms (aber GPU wird genutzt)")
        
except Exception as e:
    print(f"❌ Fehler: {e}")

print()
print("=" * 70)
print("💡 FAZIT:")
print("=" * 70)
print()
print("Wenn 'ohne cuDNN' funktioniert:")
print("  → GPU wird genutzt, nur nicht cuDNN optimiert")
print("  → Immer noch schneller als CPU")
print("  → Für Produktion: torch.backends.cudnn.enabled = False setzen")
print()
print("Alternative:")
print("  → Jetson-optimiertes PyTorch installieren")
print("  → TensorRT export (kompliziert aber 10-20x schneller)")
print()
