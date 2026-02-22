# System Overview

## Ziel des Unterprojekts
`Edge_Detection` extrahiert Aircraft-Silhouettenkonturen aus Kamerabildern (Debug-Modus mit YOLO-Labels oder SVO-Modus mit Live-YOLO-Inferenz + ZED-Tiefendaten) und exportiert die Ergebnisse als NPZ-Dateien für den Pose Optimizer.

## Modulstruktur

```
Edge_Detection/
├── src/
│   ├── gui/
│   │   ├── edge_detection_gui.py          # Haupt-GUI (PySide6, 1670 Zeilen)
│   │   ├── svo_edge_detection_gui.py      # SVO-spezialisierte GUI-Variante
│   │   └── svo_edge_detection_gui_enhanced.py
│   └── algorithms/
│       ├── preproc.py                     # CLAHE, Gaussian, Bilateral Filter
│       ├── canny_pipeline.py              # auto_canny + canny_edge_detection
│       ├── contour_silhouette.py          # Contour → gefüllte Silhouette
│       ├── hough_ransac.py                # Probabilistischer Hough-Transform
│       └── adaptive_pipeline.py          # AdaptiveEdgePipeline (temporal)
├── data/
│   ├── debug_images/originals/            # Quellbilder (aus YoloTrainingImagesV1)
│   ├── debug_images/labels/               # YOLO-Label-Dateien (.txt)
│   ├── algorithm_outputs/                 # Session-Ergebnisse je Algorithmus
│   └── exported_silhouettes/              # NPZ-Exporte für Pose Optimizer
└── run_gui.py                             # Startskript
```

## Zwei Betriebsmodi

### Debug Mode
- Quellbilder: `data/debug_images/originals/` (via ExtractionWorker aus YoloTrainingImagesV1)
- BBox-Quelle: YOLO-Label-Dateien `.txt` (kein Inferenz-Overhead)
- Geeignet für: Algorithmen-Parametrierung, reproduzierbare Batch-Tests

### SVO Mode
- Quellbilder: `/media/angelo/DRONE_DATA1/SVO2_Frame_Export/{folder}/*.jpg`
- BBox-Quelle: Live-YOLO-Inferenz (`best.pt` auf GPU)
- Tiefendaten: zugehörige `.npy`-Dateien (optionaler Depth-Filter)
- Geeignet für: Validierung auf echten Kamera-Frames

## Algorithmen (10)

| Name                      | Kurzbeschreibung                             | Typische Farbe |
|---------------------------|----------------------------------------------|----------------|
| Canny (Auto)              | Median-basierte Schwellenwert-Auswahl        | Grün           |
| CLAHE + Canny             | Kontrastverbesserung → Canny                 | Blau           |
| Canny (Manual)            | Feste Schwellenwerte per Slider              | Grün           |
| Sobel Magnitude           | Gradientenbetrag → Schwellenwert             | Magenta        |
| Laplacian of Gaussian     | LoG → Absolutwert → Schwellenwert            | Gelb           |
| Multi-scale Canny         | σ ∈ {0.5, 1.0, 2.0} → logisches ODER        | Orange         |
| Structured Edges          | Multi-Scale-Sobel + adaptiver Schwellenwert  | Lila           |
| Active Contours (Snake)   | skimage active_contour, rechteckige Init     | Orange         |
| Canny + Contour Silhouette| Canny → Konturen → gefüllte Maske           | Cyan           |
| Canny + Hough Lines       | Canny → probabilistischer Hough             | Rot            |

Alle Algorithmen operieren auf einem um `bbox_padding` (Standard 5 px) erweiterten ROI aus `_clean_image` (unveränderliche Kopie des Originals).

## Silhouette-Export → Pose Optimizer

`📐 Export Silhouette` speichert:
- `silhouette_{timestamp}_{img_stem}_{algo}.npz` — `edge_pixels (N,2)`, `image_size`, `bbox`
- `silhouette_{timestamp}_{img_stem}_{algo}.json` — Metadaten (Algorithmus, Laufzeit, Quelle…)
- `silhouette_{timestamp}_{img_stem}_{algo}_preview.png` — Vorschau

Der NPZ wird direkt von `Pose_Optimizer/src/run_on_real_silhouette.py` konsumiert.

## Kamera-Intrinsics (ZED 2i @ HD720)

SDK-kalibriert, S/N 34754237:
- `fx = fy = 951.1 px`, `cx = 638.9 px`, `cy = 348.0 px`
- HFOV = 67.8°, VFOV = 40.2°
- Gespeichert in `tools/zed2i_intrinsics_hd720.json`

## Wichtige Hinweise

1. **`_clean_image` niemals modifizieren.** Alle Algorithmen arbeiten auf `.copy()`.
2. **Bbox-Padding** (5 px) verhindert, dass der harte Schnittrand als Kante erkannt wird.
3. **Adaptive Pipeline** (`adaptive_pipeline.py`) ist noch nicht in die Haupt-GUI eingebunden — separate Klasse mit temporaler Filterung.
4. **YOLO-Modell-Pfad** ist hardcoded auf `/media/angelo/DRONE_DATA1/...` — bei Gerätewechsel anpassen.
