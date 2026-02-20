#!/usr/bin/env python3
"""
Export YOLO model to TensorRT Engine for maximum performance on Jetson.
This bypasses PyTorch/cuDNN issues and provides 10-20x speedup.
"""

from pathlib import Path
from ultralytics import YOLO
import torch

print("=" * 70)
print("TensorRT Export for Jetson Orin Nano")
print("=" * 70)

# Check CUDA
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. TensorRT requires CUDA.")
    exit(1)

print(f"Device: {torch.cuda.get_device_name(0)}")

# Model path
model_path = Path("/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt")
print(f"\nModel: {model_path}")

if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    exit(1)

# Load model
print("\nLoading YOLO model...")
model = YOLO(str(model_path))
print("✓ Model loaded")

# Export configurations
configs = [
    (320, "Ultra-Fast (320x320, best for real-time)"),
    (416, "Balanced (416x416, good speed/accuracy)"),
    (640, "High-Accuracy (640x640, slower)"),
]

print("\n" + "=" * 70)
print("Available Export Options")
print("=" * 70)

for i, (imgsz, desc) in enumerate(configs, 1):
    print(f"\n{i}. {desc}")
    print(f"   Input size: {imgsz}x{imgsz}")
    print(f"   Estimated time: ~{imgsz/100:.0f}-{imgsz/50:.0f} minutes")
    engine_path = model_path.parent / f"best_{imgsz}_fp16.engine"
    exists_str = "✓ EXISTS" if engine_path.exists() else "✗ Not created"
    print(f"   Output: {engine_path.name} {exists_str}")

# Recommended configuration
print("\n" + "=" * 70)
print("RECOMMENDED: Option 1 (320x320)")
print("=" * 70)
print("\nThis configuration:")
print("  • Meets <200ms target (estimated 10-20ms with TensorRT)")
print("  • Best for real-time inference")
print("  • Fastest export time (~3-6 minutes)")

choice = input("\nExport Option 1 (320x320)? [y/N]: ").strip().lower()

if choice == 'y':
    imgsz = 320
    output_name = f"best_{imgsz}_fp16.engine"
    
    print("\n" + "=" * 70)
    print(f"Exporting TensorRT Engine ({imgsz}x{imgsz})")
    print("=" * 70)
    print("\nSettings:")
    print(f"  • Input size: {imgsz}x{imgsz}")
    print(f"  • FP16: Yes (half precision)")
    print(f"  • Device: cuda:0")
    print("\nThis will take approximately 3-6 minutes...")
    print("Please wait...\n")
    
    try:
        model.export(
            format='engine',
            half=True,
            imgsz=imgsz,
            device=0,
            workspace=4,  # GB
            verbose=True
        )
        
        print("\n" + "=" * 70)
        print("✓ EXPORT SUCCESSFUL!")
        print("=" * 70)
        
        engine_path = model_path.parent / output_name
        print(f"\nTensorRT engine created:")
        print(f"  Location: {engine_path}")
        
        if engine_path.exists():
            size_mb = engine_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")
        
        print("\nTo use the optimized engine:")
        print("  1. Restart the GUI")
        print("  2. YOLO will automatically use the .engine file")
        print("  3. Expect 10-20x speedup (10-20ms per inference)")
        
        print("\nNOTE: The .engine file is specific to:")
        print(f"  • This GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • Input size: {imgsz}x{imgsz}")
        print("  • FP16 precision")
        print("  If you change these, you need to re-export.")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ EXPORT FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  • Ensure TensorRT is installed")
        print("  • Check CUDA/cuDNN versions")
        print("  • Try smaller workspace: workspace=2")

else:
    print("\nExport cancelled.")
    print("\nTo export later, run:")
    print(f"  python {Path(__file__).name}")

print("\n" + "=" * 70)
