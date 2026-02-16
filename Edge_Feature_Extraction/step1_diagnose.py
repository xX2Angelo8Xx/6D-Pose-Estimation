#!/usr/bin/env python3
"""
Schritt 1: System-Diagnose
Sammelt alle relevanten Informationen über Ihr System.
"""

import sys
print("=" * 70)
print("SCHRITT 1: System-Diagnose")
print("=" * 70)

# Python Version
print(f"\n1️⃣ Python:")
print(f"   Version: {sys.version}")

# PyTorch
print(f"\n2️⃣ PyTorch:")
try:
    import torch
    print(f"   ✓ Installiert")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print(f"   ⚠️  CUDA nicht verfügbar in PyTorch!")
except ImportError:
    print(f"   ✗ PyTorch nicht installiert")

# cuDNN
print(f"\n3️⃣ cuDNN:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   Version: {torch.backends.cudnn.version()}")
        print(f"   Enabled: {torch.backends.cudnn.enabled}")
    else:
        print(f"   N/A (CUDA nicht verfügbar)")
except:
    print(f"   Keine Info verfügbar")

# Ultralytics YOLO
print(f"\n4️⃣ Ultralytics YOLO:")
try:
    import ultralytics
    print(f"   ✓ Installiert")
    print(f"   Version: {ultralytics.__version__}")
except ImportError:
    print(f"   ✗ Nicht installiert")

# TensorRT
print(f"\n5️⃣ TensorRT:")
try:
    import tensorrt
    print(f"   ✓ Installiert")
    print(f"   Version: {tensorrt.__version__}")
except ImportError:
    print(f"   ✗ Nicht installiert")
    print(f"   Hinweis: TensorRT ist optional, aber empfohlen für Jetson")

# System CUDA
print(f"\n6️⃣ System CUDA:")
import subprocess
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"   {line.strip()}")
    else:
        print(f"   nvcc nicht gefunden")
except FileNotFoundError:
    print(f"   nvcc nicht im PATH")

# JetPack (nur auf Jetson)
print(f"\n7️⃣ JetPack Info:")
try:
    with open('/etc/nv_tegra_release', 'r') as f:
        print(f"   {f.read().strip()}")
except FileNotFoundError:
    print(f"   Kein Jetson Device oder /etc/nv_tegra_release nicht gefunden")

print("\n" + "=" * 70)
print("Diagnose abgeschlossen!")
print("=" * 70)
print("\n💡 Was bedeutet das?")
print("   • CUDA Available: True = GPU kann genutzt werden ✓")
print("   • TensorRT installiert = Optimierung möglich ✓")
print("   • cuDNN Version = Wichtig für Kompatibilität")
print("\nWeiter mit Schritt 2...")
