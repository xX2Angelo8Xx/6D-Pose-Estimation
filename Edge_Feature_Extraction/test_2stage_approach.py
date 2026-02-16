#!/usr/bin/env python3
"""
Test: 2-Stufen Ansatz für optimale Performance mit voller Auflösung
- YOLO auf verkleinertem Input (schnell)
- Edge Detection auf Original-ROI (volle Details)
"""

from pathlib import Path
import cv2
from ultralytics import YOLO
import time
import numpy as np

print("=" * 70)
print("2-STUFEN ANSATZ: YOLO Klein, Edge Detection Groß")
print("=" * 70)

# Setup
model_path = "/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt"
image_path = "/media/angelo/DRONE_DATA1/SVO2_Frame_Export/1.1/frame_000350.jpg"

print("\n📁 Laden...")
model = YOLO(model_path)
img = cv2.imread(image_path)
orig_h, orig_w = img.shape[:2]
print(f"   Original Bild: {orig_w}x{orig_h}")

# Simuliere Edge Detection (Canny)
def simple_edge_detection(roi):
    """Einfache Canny Edge Detection zur Zeitmessung."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    # Canny
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Test verschiedene YOLO Größen
configs = [
    (320, "Ultra-schnell"),
    (416, "Ausgewogen"),
    (640, "Genau"),
]

print("\n" + "=" * 70)
print("BENCHMARK: YOLO + Edge Detection auf Original-ROI")
print("=" * 70)

results_summary = []

for imgsz, desc in configs:
    print(f"\n📊 Test: YOLO {imgsz}x{imgsz} ({desc})")
    print(f"   └─> Edge Detection auf Original-Auflösung")
    
    # Warmup
    for _ in range(2):
        _ = model(img, verbose=False, device='cuda:0', imgsz=imgsz, half=False)
    
    # Benchmark
    yolo_times = []
    edge_times = []
    total_times = []
    
    for i in range(5):
        # YOLO auf verkleinertem Input
        start_yolo = time.perf_counter()
        results = model(img, verbose=False, device='cuda:0', imgsz=imgsz, half=False)
        end_yolo = time.perf_counter()
        yolo_time = (end_yolo - start_yolo) * 1000
        
        # BBox extrahieren (bereits in Original-Koordinaten!)
        bbox = None
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            bbox = (x1, y1, x2, y2)
            
            # ROI aus ORIGINAL Bild extrahieren
            roi = img[y1:y2, x1:x2].copy()
            roi_w = x2 - x1
            roi_h = y2 - y1
            
            # Edge Detection auf Original-ROI
            start_edge = time.perf_counter()
            edges = simple_edge_detection(roi)
            end_edge = time.perf_counter()
            edge_time = (end_edge - start_edge) * 1000
        else:
            edge_time = 0
            roi_w, roi_h = 0, 0
        
        total_time = yolo_time + edge_time
        
        yolo_times.append(yolo_time)
        edge_times.append(edge_time)
        total_times.append(total_time)
        
        if i == 0:
            print(f"   BBox: {bbox}")
            print(f"   ROI Größe: {roi_w}x{roi_h} (Original-Auflösung!)")
    
    # Durchschnitt
    avg_yolo = sum(yolo_times) / len(yolo_times)
    avg_edge = sum(edge_times) / len(edge_times)
    avg_total = sum(total_times) / len(total_times)
    
    print(f"\n   Ergebnisse (Durchschnitt über 5 Runs):")
    print(f"   ├─ YOLO ({imgsz}):  {avg_yolo:>6.1f} ms")
    print(f"   ├─ Edge (Original): {avg_edge:>6.1f} ms")
    print(f"   └─ TOTAL:           {avg_total:>6.1f} ms")
    
    target_met = "✓ JA" if avg_total < 200 else "✗ NEIN"
    fps = 1000 / avg_total
    print(f"   Status: {avg_total:.1f} ms ({fps:.1f} FPS) - Ziel <200ms: {target_met}")
    
    results_summary.append({
        'imgsz': imgsz,
        'desc': desc,
        'yolo': avg_yolo,
        'edge': avg_edge,
        'total': avg_total,
        'fps': fps,
        'target_met': avg_total < 200
    })

# Zusammenfassung
print("\n" + "=" * 70)
print("ZUSAMMENFASSUNG")
print("=" * 70)

print("\n💡 Konzept: 2-Stufen Ansatz")
print("   1. YOLO auf verkleinertem Bild → BBox finden (schnell)")
print("   2. Edge Detection auf Original-ROI → Details erhalten (präzise)")

print(f"\n📊 Performance Vergleich:")
print(f"\n{'YOLO Size':<12} {'YOLO (ms)':<12} {'Edge (ms)':<12} {'Total (ms)':<12} {'FPS':<8} {'<200ms'}")
print("-" * 70)

for r in results_summary:
    target_str = "✓" if r['target_met'] else "✗"
    print(f"{r['imgsz']:<12} {r['yolo']:<12.1f} {r['edge']:<12.1f} {r['total']:<12.1f} {r['fps']:<8.1f} {target_str}")

# Empfehlung
print("\n" + "=" * 70)
print("EMPFEHLUNG")
print("=" * 70)

best = min(results_summary, key=lambda x: x['total'])

if best['target_met']:
    print(f"\n✅ ZIEL ERREICHT mit YOLO {best['imgsz']}!")
    print(f"   Total: {best['total']:.1f} ms ({best['fps']:.1f} FPS)")
    print(f"\n   Vorteile:")
    print(f"   • YOLO läuft schnell auf {best['imgsz']}x{best['imgsz']}")
    print(f"   • Edge Detection behält volle {orig_w}x{orig_h} Auflösung")
    print(f"   • Beste Balance: Speed + Details")
else:
    print(f"\n⚠️  Schnellste Option: YOLO {best['imgsz']}")
    print(f"   Total: {best['total']:.1f} ms ({best['fps']:.1f} FPS)")
    print(f"   Überschreitung: {best['total'] - 200:.0f} ms über 200ms Ziel")
    print(f"\n   Hinweis:")
    print(f"   • Edge Detection auf voller Auflösung ist aufwändig")
    print(f"   • Aber: Maximale Kantendetails für Pose Estimation!")

print("\n" + "=" * 70)
print("🎯 WICHTIG für Ihre Anwendung:")
print("=" * 70)
print("\n   Das Modellflugzeug ist klein → Original-Auflösung ist NOTWENDIG")
print("   Kompromiss: YOLO klein (schnell), Edge Detection groß (präzise)")
print("\n" + "=" * 70)
