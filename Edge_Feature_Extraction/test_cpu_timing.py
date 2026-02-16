#!/usr/bin/env python3
"""
Einfacher Performance-Test OHNE GPU (CPU only)
Testet YOLO + Edge Detection Zeiten
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from pathlib import Path

# Pfade
MODEL_PATH = "/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt"
TEST_IMG = "./data/debug_images/originals/frame_000544-1.5-SE_Top-depth-4.94m-std-0.38m.jpg"

print("=" * 70)
print("CPU-ONLY Performance Test")
print("=" * 70)
print()

# Lade Bild
img = cv2.imread(TEST_IMG)
if img is None:
    print(f"❌ Fehler: Konnte {TEST_IMG} nicht laden!")
    exit(1)

h, w = img.shape[:2]
print(f"📁 Original Bild: {w}x{h}")
print()

# Lade Modell
print("🔄 Lade YOLO Modell...")
model = YOLO(MODEL_PATH)

# Warmup auf CPU
print("🔥 Warmup auf CPU...")
_ = model(img, verbose=False, device='cpu', imgsz=640, half=False)
print("✓ Warmup abgeschlossen")
print()

# Test verschiedene Größen
sizes = [320, 416, 640]

print("=" * 70)
print("BENCHMARK: YOLO (CPU) + Edge Detection (Original ROI)")
print("=" * 70)
print()

for imgsz in sizes:
    print(f"📊 Test: YOLO {imgsz}x{imgsz}")
    print(f"   └─> Edge Detection auf Original-Auflösung {w}x{h}")
    
    # YOLO Inference (3x für Durchschnitt)
    yolo_times = []
    best_bbox = None
    
    for i in range(3):
        t0 = time.time()
        results = model(img, verbose=False, device='cpu', imgsz=imgsz, half=False)
        t1 = time.time()
        yolo_times.append((t1 - t0) * 1000)
        
        # Beste BBox speichern
        if i == 0 and len(results[0].boxes) > 0:
            # YOLO gibt BBox in Original-Koordinaten!
            xyxy = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            best_bbox = (x1, y1, x2, y2)
    
    avg_yolo_time = np.mean(yolo_times)
    
    # Edge Detection auf Original ROI (falls BBox gefunden)
    edge_times = []
    if best_bbox:
        x1, y1, x2, y2 = best_bbox
        
        # Clip auf Bildgrenzen
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extrahiere ROI aus ORIGINAL Bild (nicht resized!)
        roi = img[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        
        print(f"   └─> ROI Größe: {roi_w}x{roi_h} (aus Original {w}x{h})")
        
        # Simuliere Edge Detection (Canny, realistisch)
        for i in range(3):
            t0 = time.time()
            
            # Preprocessing (Gaussian Blur)
            blurred = cv2.GaussianBlur(roi, (5, 5), 1.5)
            
            # Canny Edge Detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Kontur-Findung (wie in echter Pipeline)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            t1 = time.time()
            edge_times.append((t1 - t0) * 1000)
        
        avg_edge_time = np.mean(edge_times)
    else:
        print(f"   └─> ⚠️  Keine BBox gefunden!")
        avg_edge_time = 0.0
    
    # Gesamt
    total_time = avg_yolo_time + avg_edge_time
    
    print()
    print(f"   ⏱️  YOLO Zeit:     {avg_yolo_time:6.1f} ms (CPU, {imgsz}x{imgsz})")
    print(f"   ⏱️  Edge Zeit:     {avg_edge_time:6.1f} ms (Original ROI)")
    print(f"   ⏱️  GESAMT:        {total_time:6.1f} ms")
    
    if total_time < 200:
        print(f"   ✅ Unter 200ms! ({total_time:.1f} ms)")
    elif total_time < 333:
        print(f"   ⚠️  Über 200ms, aber unter 333ms (3 FPS möglich)")
    else:
        print(f"   ❌ Über 333ms (< 3 FPS)")
    
    print()

print("=" * 70)
print("💡 ERGEBNIS:")
print("=" * 70)
print()
print("Die BBox-Koordinaten von YOLO sind bereits in Original-Auflösung.")
print("Edge Detection läuft auf dem Original-ROI (volle Details).")
print()
print("Wenn CPU zu langsam:")
print("  → Modell optimieren (weniger Layers)")
print("  → Schnellere Edge-Algorithmen")
print("  → FPS-Ziel reduzieren (3 FPS = 333ms OK?)")
print()
