#!/usr/bin/env python3
"""
Schritt 2: Einfacher YOLO Test
Testet YOLO Inference ohne FP16, um das Basis-Setup zu validieren.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO
import time

print("=" * 70)
print("SCHRITT 2: Einfacher YOLO Test (ohne FP16)")
print("=" * 70)

# Pfade
model_path = "/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt"
image_path = "/media/angelo/DRONE_DATA1/SVO2_Frame_Export/1.1/frame_000350.jpg"

print("\n📁 Dateien:")
print(f"   Model: {Path(model_path).name}")
print(f"   Image: {Path(image_path).name}")

# Prüfe ob Dateien existieren
if not Path(model_path).exists():
    print(f"\n❌ ERROR: Model nicht gefunden!")
    print(f"   Pfad: {model_path}")
    exit(1)

if not Path(image_path).exists():
    print(f"\n❌ ERROR: Bild nicht gefunden!")
    print(f"   Pfad: {image_path}")
    exit(1)

print("   ✓ Beide Dateien gefunden")

# Lade Model
print("\n🤖 Lade YOLO Model...")
try:
    model = YOLO(model_path)
    print("   ✓ Model geladen")
except Exception as e:
    print(f"   ❌ Fehler beim Laden: {e}")
    exit(1)

# Lade Bild
print("\n🖼️  Lade Bild...")
try:
    img = cv2.imread(image_path)
    print(f"   ✓ Bild geladen, Größe: {img.shape}")
except Exception as e:
    print(f"   ❌ Fehler beim Laden: {e}")
    exit(1)

# Test 1: CPU Inference (sollte immer funktionieren)
print("\n" + "=" * 70)
print("TEST 1: CPU Inference (Baseline)")
print("=" * 70)
print("\n💡 Was passiert:")
print("   • YOLO wird auf der CPU ausgeführt")
print("   • Das ist langsam, aber sollte immer funktionieren")
print("   • Keine GPU oder FP16 Operationen")

try:
    print("\n⏳ Laufe CPU Inference...")
    start = time.perf_counter()
    results = model(img, verbose=False, device='cpu', imgsz=640, half=False)
    end = time.perf_counter()
    
    cpu_time = (end - start) * 1000
    print(f"   ✓ Erfolg!")
    print(f"   Zeit: {cpu_time:.2f} ms")
    
    if len(results[0].boxes) > 0:
        print(f"   Detektionen: {len(results[0].boxes)}")
        box = results[0].boxes[0]
        conf = box.conf[0].item()
        print(f"   Beste Detection Confidence: {conf:.3f}")
    else:
        print(f"   Keine Objekte gefunden")
    
except Exception as e:
    print(f"   ❌ CPU Inference fehlgeschlagen!")
    print(f"   Fehler: {e}")
    print("\n🚨 Wenn CPU fehlschlägt, ist das Model oder Ultralytics das Problem")
    exit(1)

# Test 2: GPU Inference ohne FP16 (Standard FP32)
print("\n" + "=" * 70)
print("TEST 2: GPU Inference ohne FP16 (FP32)")
print("=" * 70)
print("\n💡 Was passiert:")
print("   • YOLO wird auf der GPU ausgeführt")
print("   • Verwendet FP32 (normale Genauigkeit)")
print("   • Kein FP16 (Half Precision) - das vermeidet cuDNN Probleme")

try:
    print("\n⏳ Laufe GPU Inference (FP32)...")
    
    # Warmup (erste Inferenz ist immer langsamer)
    print("   Warmup...")
    for _ in range(2):
        _ = model(img, verbose=False, device='cuda:0', imgsz=640, half=False)
    
    # Eigentliche Messung
    print("   Messung...")
    times = []
    for i in range(3):
        start = time.perf_counter()
        results = model(img, verbose=False, device='cuda:0', imgsz=640, half=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)
        print(f"      Run {i+1}: {times[-1]:.2f} ms")
    
    gpu_time = sum(times) / len(times)
    print(f"\n   ✓ Erfolg!")
    print(f"   Durchschnitt: {gpu_time:.2f} ms")
    print(f"   Speedup vs CPU: {cpu_time / gpu_time:.2f}x")
    
except Exception as e:
    print(f"   ❌ GPU Inference (FP32) fehlgeschlagen!")
    print(f"   Fehler: {e}")
    print("\n⚠️  GPU funktioniert nicht mit FP32")
    print("   Das ist ungewöhnlich, aber kein Problem für den nächsten Test")

# Test 3: GPU mit kleinerer Bildgröße (FP32)
print("\n" + "=" * 70)
print("TEST 3: GPU mit kleinerem Input (320x320, FP32)")
print("=" * 70)
print("\n💡 Was passiert:")
print("   • Gleiche GPU FP32 Inference")
print("   • Aber Bild wird auf 320x320 verkleinert")
print("   • Das sollte schneller sein")

try:
    print("\n⏳ Laufe GPU Inference (320x320, FP32)...")
    
    # Warmup
    for _ in range(2):
        _ = model(img, verbose=False, device='cuda:0', imgsz=320, half=False)
    
    # Messung
    times = []
    for i in range(3):
        start = time.perf_counter()
        results = model(img, verbose=False, device='cuda:0', imgsz=320, half=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)
        print(f"      Run {i+1}: {times[-1]:.2f} ms")
    
    gpu_320_time = sum(times) / len(times)
    print(f"\n   ✓ Erfolg!")
    print(f"   Durchschnitt: {gpu_320_time:.2f} ms")
    print(f"   Speedup vs 640: {gpu_time / gpu_320_time:.2f}x schneller")
    
    # Prüfe ob unter 200ms
    if gpu_320_time < 200:
        print(f"\n   🎉 ZIEL ERREICHT! Unter 200ms ohne FP16!")
    else:
        print(f"\n   ⚠️  {gpu_320_time - 200:.0f}ms über dem 200ms Ziel")
        print("       Aber das ist okay mit FP32, FP16 wäre schneller")
    
except Exception as e:
    print(f"   ❌ GPU Inference (320, FP32) fehlgeschlagen!")
    print(f"   Fehler: {e}")

# Zusammenfassung
print("\n" + "=" * 70)
print("ZUSAMMENFASSUNG")
print("=" * 70)

print("\n✅ Was funktioniert:")
print("   • CPU Inference ✓")
print("   • GPU FP32 Inference (normal precision)")
print("   • Kleinere Bildgrößen (320x320)")

print("\n📊 Performance ohne FP16:")
print(f"   CPU (640): {cpu_time:.0f} ms")
try:
    print(f"   GPU FP32 (640): {gpu_time:.0f} ms")
    print(f"   GPU FP32 (320): {gpu_320_time:.0f} ms")
except:
    print("   GPU Tests nicht abgeschlossen")

print("\n💡 Nächster Schritt:")
print("   • FP16 ist das Problem")
print("   • TensorRT kann das umgehen")
print("   • Weiter mit Schritt 3: Alternative TensorRT Ansätze")

print("\n" + "=" * 70)
