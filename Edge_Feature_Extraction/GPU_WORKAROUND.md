# GPU Workaround für Jetson Orin Nano

## 🎯 Problem

**Fehlermeldung:**
```
RuntimeError: GET was unable to find an engine to execute this computation
```

**Ursache:**
- PyTorch 2.3.0 wurde für CUDA 11.x kompiliert
- Jetson Orin Nano hat CUDA 12.2
- cuDNN v8 findet keine kompatible "Engine" für Convolution-Operationen

## ✅ Aktuelle Lösung (Workaround)

**Code:**
```python
import torch
torch.backends.cudnn.enabled = False  # Disable cuDNN
```

**Effekt:**
- GPU wird trotzdem genutzt (native CUDA kernels statt cuDNN)
- ✅ 2-3x schneller als CPU
- ⏱️ YOLO 320x320: ~50ms (vs. 130ms auf CPU)
- Kein cuDNN Error mehr

**In der GUI implementiert:**
- Automatisch aktiviert wenn CUDA verfügbar
- Standard: 320x320 Eingabegröße
- Edge Detection auf Original 1280x720 ROI

## 📊 Performance

| Konfiguration | Zeit | FPS | Status |
|---------------|------|-----|--------|
| **GPU 320x320 + Edge** | **~52ms** | **~19 FPS** | ✅ Produktionsbereit |
| CPU 320x320 + Edge | ~132ms | ~7.5 FPS | ✅ Funktioniert |
| GPU 416x416 + Edge | ~70ms | ~14 FPS | ✅ Gut |
| GPU 640x640 + Edge | ~130ms | ~7.7 FPS | ⚠️ Akzeptabel |

**Ziel erreicht:** <200ms ✅ (Ziel war <200ms)

## 🔮 Zukünftige Optimierungen

### Option 1: Jetson-optimiertes PyTorch (EMPFOHLEN)

**Problem:** Aktuelles PyTorch von PyPI ist nicht für Jetson optimiert

**Lösung:** NVIDIA bietet spezielle PyTorch Wheels für Jetson

**Installation:**
```bash
# 1. Deinstalliere aktuelles PyTorch
pip uninstall torch torchvision

# 2. Installiere Jetson PyTorch (von NVIDIA)
# Siehe: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

# Für JetPack 5.x / 6.x (Orin):
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl
pip install torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl

# Oder via NVIDIA Container:
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
```

**Vorteile:**
- ✅ cuDNN funktioniert out-of-the-box
- ✅ Optimiert für Jetson Hardware
- ✅ 10-30% schneller als PyPI Version
- ✅ Bessere Memory Management

**Aufwand:** ~30 Minuten (Download + Installation)

### Option 2: TensorRT Export

**Was ist TensorRT?**
- NVIDIA's High-Performance Inference Library
- Optimiert speziell für NVIDIA GPUs
- 10-20x Speedup möglich

**Vorgehen:**
```python
from ultralytics import YOLO

model = YOLO("best.pt")

# Export zu TensorRT (dauert 5-10 Minuten)
model.export(
    format='engine',      # TensorRT format
    imgsz=320,           # Muss fixiert sein
    half=True,           # FP16 für Jetson
    device=0,            # CUDA device
    workspace=4,         # GB workspace für Optimierungen
)

# Lädt automatisch .engine wenn vorhanden
results = model("image.jpg")  # Nutzt TensorRT automatisch
```

**Performance Erwartung:**
- YOLO 320x320: 50ms → **5-10ms** 🚀
- YOLO 640x640: 130ms → **15-25ms**

**Nachteile:**
- ⏰ Export dauert 5-10 Minuten
- 📦 Große .engine Datei (~100-200 MB)
- 🔒 Gebunden an spezifische Eingabegröße
- ❓ Kann fehlschlagen (wie bereits getestet)

**Mit Jetson-PyTorch:** Höhere Erfolgsrate

### Option 3: ONNX Runtime

**Alternative zu TensorRT:**
```bash
pip install onnxruntime-gpu

# Export
model.export(format='onnx', simplify=True)

# Inference mit ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("best.onnx", providers=['CUDAExecutionProvider'])
```

**Vorteile:**
- ✅ Stabiler als TensorRT
- ✅ 3-5x Speedup (weniger als TensorRT aber zuverlässiger)
- ✅ Flexible Eingabegrößen

**Aufwand:** ~1 Stunde (Export + Integration)

## 📋 Empfehlung (Priorität)

### 1. **Jetson-PyTorch installieren** (JETZT)
   - ⏰ Aufwand: 30 Minuten
   - ✅ Nutzen: cuDNN funktioniert, 10-30% schneller
   - 🎯 Risiko: Niedrig

### 2. **TensorRT Export testen** (NACH Jetson-PyTorch)
   - ⏰ Aufwand: 1-2 Stunden
   - ✅ Nutzen: 10-20x Speedup möglich
   - ⚠️ Risiko: Kann fehlschlagen

### 3. **ONNX Runtime** (Fallback für TensorRT)
   - ⏰ Aufwand: 1 Stunde
   - ✅ Nutzen: 3-5x Speedup
   - 🎯 Risiko: Niedrig

## 🚀 Aktueller Status

**Workaround implementiert in:**
- `src/gui/svo_edge_detection_gui.py`
- Zeile 131-145: cuDNN disable
- Standard: GPU + 320x320 + FP32

**Performance:**
- ✅ ~52ms total (YOLO + Edge)
- ✅ ~19 FPS
- ✅ 4x schneller als 200ms Ziel

**Produktionsbereit:** JA ✅

## 📚 Ressourcen

- [PyTorch für Jetson (NVIDIA Forum)](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [Jetson Zoo (Pre-compiled Wheels)](https://elinux.org/Jetson_Zoo)
- [NVIDIA L4T PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [Ultralytics Export Docs](https://docs.ultralytics.com/modes/export/)

## ⚠️ Wichtig

**Nicht mischen:**
- Jetson-PyTorch + PyPI-PyTorch = Konflikt
- Immer nur EINE PyTorch Installation

**Backup vor Änderungen:**
```bash
# Aktuelle Umgebung sichern
pip freeze > requirements_backup.txt

# Environment clonen (falls conda/venv)
```

---

**Stand:** 2026-02-13  
**Status:** ✅ Workaround funktioniert, Performance exzellent  
**Nächster Schritt:** Jetson-optimiertes PyTorch installieren
