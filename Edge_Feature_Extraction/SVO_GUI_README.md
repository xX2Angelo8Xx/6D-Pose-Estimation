# SVO Edge Detection GUI

Standalone GUI für Edge Detection mit YOLO Object Detection auf SVO2 Frame Export Daten.

## 🚀 Start

```bash
./start_svo_mode.sh
```

Oder direkt:
```bash
python src/gui/svo_edge_detection_gui.py
```

## ✨ Features

### 📂 Folder Selection
- Dropdown mit allen verfügbaren SVO Ordnern (1.1 - 1.7)
- Automatisches Laden aller .jpg Dateien
- Info-Anzeige: Anzahl der Bilder pro Ordner

### 🤖 YOLO Object Detection
- **Automatisch** bei jedem Bildladen
- Model: `/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt`
- Device: CPU (sicher, keine CUDA Probleme)
- Performance: ~1.2-1.7 Sekunden pro Bild (CPU)
- Grüne Bounding Box mit Confidence

### 🧮 Edge Detection Algorithmen
1. **Canny (Auto)** - Automatische Schwellwerte (Median-basiert)
2. **CLAHE + Canny** - Contrast Limited Adaptive Histogram Equalization
3. **Canny + Contours** - Kantenerkennung + Konturensuche

Alle Algorithmen arbeiten **nur innerhalb der YOLO BBox** (ROI).

### 🌊 Depth Data Integration
- Automatisches Laden von `.npy` Dateien (wenn vorhanden)
- **Depth Filter** (Checkbox aktivieren):
  - Filtert Konturen basierend auf Tiefeninformation
  - **Radius Slider** (1-50 Pixel): Suchradius für valide Tiefenwerte
  - Nur Konturen mit mindestens einem Tiefenwert > 0 im Radius werden behalten

### ⏱️ Performance Messung
Detaillierte Zeitmessung in 3 Kategorien:
- **YOLO Zeit**: Objekterkennung (ms)
- **Edge Zeit**: Edge Detection Algorithmus (ms)
- **Total**: Gesamtzeit (YOLO + Edge)

### 🎚️ Parameter Controls
- **Gaussian σ** (0.1 - 5.0): Gaußscher Weichzeichner vor Edge Detection
- **CLAHE Clip** (1.0 - 5.0): Clip-Limit für Kontrastverbesserung
- **Depth Filter** (Checkbox): Tiefenfilterung aktivieren
- **Radius** (1 - 50): Suchradius für Tiefenwerte

### 🖼️ Image Viewer
- **Zoom**: Mausrad zum Zoomen
- **Pan**: Drag & Drop zum Verschieben (bei Zoom)
- **Fit to View**: Automatisch beim Bildwechsel

### ⌨️ Navigation
- **◀ Previous**: Vorheriges Bild
- **Next ▶**: Nächstes Bild
- **Image Info**: Aktueller Index + Dateiname

## 📊 Workflow

1. **Folder auswählen** → Dropdown (z.B. "1.1")
2. **Bilder laden** → Automatisch (1136-1429 Bilder je nach Ordner)
3. **YOLO Detection** → Automatisch bei jedem Bild
4. **Algorithmus wählen** → Dropdown
5. **Parameter einstellen** → Slider
6. **Apply Algorithm** → Button drücken
7. **Ergebnis sehen** → Edges/Konturen in der BBox
8. **Navigieren** → Next/Previous Buttons

## 🔒 Read-Only Garantie

**Alle Operationen sind Read-Only:**
- `cv2.imread()` für Bilder
- `np.load()` für Depth-Daten
- **Keine Schreiboperationen** auf Original-Daten
- Alle Modifikationen nur im RAM

## 📁 Daten-Struktur

```
/media/angelo/DRONE_DATA1/SVO2_Frame_Export/
├── 1.1/
│   ├── frame_000350.jpg
│   ├── frame_000350.npy  (Depth)
│   ├── frame_000351.jpg
│   ├── frame_000351.npy
│   └── ...
├── 1.2/
├── 1.3/
└── ...
```

## ⚙️ Technische Details

### Abhängigkeiten
- `PySide6` - GUI Framework
- `opencv-python` - Image Processing
- `numpy` - Array Operations
- `ultralytics` - YOLO Model
- `scikit-image` - (falls Active Contours verwendet wird)

### YOLO Performance
- **CPU Mode**: ~1.2-1.7s pro Bild (ARM Prozessor)
- **GPU Mode**: Deutlich schneller (falls CUDA verfügbar)
- Um GPU zu nutzen: `device='cuda:0'` in `load_current_image()`

### Depth Filter Algorithmus
```python
def has_valid_depth(contour, bbox_offset):
    # Sample 20 Punkte entlang Kontur
    # Für jeden Punkt: Suche im Radius nach Tiefenwert > 0
    # Wenn mind. 1 valider Wert gefunden → True
    # Sonst → False (Kontur wird verworfen)
```

## 🐛 Bekannte Probleme

### CUDA Warnungen
Wenn Sie CUDA-Warnungen sehen:
```
CuDNNError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
```
→ **Normal!** Die GUI nutzt automatisch CPU als Fallback.

Um GPU zu erzwingen (falls funktioniert):
```python
# Zeile 328 in svo_edge_detection_gui.py ändern:
results = self.yolo_model(self.current_image, verbose=False, device='cuda:0')
```

### Langsame YOLO Inference
CPU Mode ist langsamer (~1.5s pro Bild). Alternativen:
1. GPU nutzen (wenn CUDA funktioniert)
2. Kleineres YOLO Model (z.B. YOLOv8n statt YOLOv8m)
3. Geringere Auflösung (Bilder vor Inference downscalen)

## 📈 Performance Expectations

**Typische Zeiten (CPU):**
- YOLO: 1200-1700 ms
- Canny: 10-50 ms
- CLAHE + Canny: 20-80 ms
- Canny + Contours: 30-150 ms (je nach Anzahl Konturen)
- Depth Filter: +10-50 ms (wenn aktiv)

**Total pro Bild: ~1.3 - 1.8 Sekunden**

## 🆚 Vergleich: Neue GUI vs. Integration

### ✅ Vorteile der separaten GUI
- **Einfacher Code**: 530 Zeilen vs. 1300+ Zeilen
- **Fokussiert**: Nur SVO-spezifische Features
- **Wartbar**: Klare Zuständigkeiten
- **Stabil**: Keine Mode-Switching Komplexität
- **Schneller zu debuggen**: Weniger Interaktionen

### 📋 Features
- ✅ SVO Folder Loading
- ✅ YOLO Inference
- ✅ Depth Data
- ✅ Depth Filter
- ✅ 3 Edge Algorithmen
- ✅ Performance Messung
- ✅ Parameter Controls
- ✅ Navigation
- ✅ Zoom/Pan

## 🔧 Anpassungen

### YOLO Model Pfad ändern
```python
# Zeile 94 in svo_edge_detection_gui.py
self.yolo_model_path = Path("/ihr/pfad/zu/best.pt")
```

### SVO Base Directory ändern
```python
# Zeile 93
self.svo_base_dir = Path("/ihr/pfad/zu/SVO2_Frame_Export")
```

### Mehr Algorithmen hinzufügen
```python
# In init_ui() die ComboBox erweitern:
self.combo_algorithm.addItems([
    "Canny (Auto)",
    "CLAHE + Canny", 
    "Canny + Contours",
    "Ihr Neuer Algorithmus"  # ← Hier
])

# Dann in apply_algorithm() einen neuen elif-Block:
elif algo_name == "Ihr Neuer Algorithmus":
    result = self.apply_ihr_algorithmus()
```

## 📝 Nächste Schritte

Mögliche Erweiterungen:
1. **GPU Support togglen** (Radio Button CPU/GPU)
2. **Batch Processing** (Alle Bilder automatisch verarbeiten)
3. **Export Funktionen** (Edges als .png speichern)
4. **Statistiken** (Histogramme, Edge-Dichte)
5. **Vergleichs-View** (Original + Result nebeneinander)
6. **Mehr Algorithmen** (Sobel, LoG, Active Contours)

## 🎓 Code-Struktur

```python
class SVOEdgeDetectionGUI(QMainWindow):
    __init__()              # Setup paths, state, load YOLO
    init_ui()               # Create UI elements
    load_folder_list()      # Scan SVO directory
    on_folder_changed()     # Load image list for folder
    load_current_image()    # Load image + depth + YOLO
    display_image()         # Show in viewer with bbox
    prev_image() / next_image()  # Navigation
    apply_algorithm()       # Run selected edge detection
    apply_canny_auto()      # Canny with auto threshold
    apply_clahe_canny()     # CLAHE + Canny
    apply_canny_contours()  # Canny + Contours + Depth Filter
    has_valid_depth()       # Check contour depth validity
```

## 🎯 Zusammenfassung

**Die neue SVO GUI ist:**
- ✅ **Einfacher** als Integration in bestehende GUI
- ✅ **Funktional** mit allen geforderten Features
- ✅ **Stabil** ohne Mode-Switching Probleme
- ✅ **Wartbar** mit klarer Struktur
- ✅ **Erweiterbar** für zukünftige Features

**Empfehlung:** Diese GUI für produktive SVO-Arbeit nutzen, alte GUI für Debug Mode behalten.
