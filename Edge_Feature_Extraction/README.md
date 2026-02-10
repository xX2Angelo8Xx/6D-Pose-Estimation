# Edge Feature Extraction (Subproject)

Dieses Subproject bereitet Kantenfeatures aus YOLO-annotierten Trainingsbildern vor und bietet eine interaktive GUI zum Testen verschiedener Edge-Detection-Algorithmen.

## Struktur

- `src/extract_debug_images.py` — Hilfsskript zum selektiven Kopieren von Bildern (depth ≤ 15m)
- `src/algorithms/` — Edge-Detection-Algorithmen (Canny, Contour-Silhouette, Hough, etc.)
- `src/gui/` — Interaktive GUI zum Testen und Vergleichen von Algorithmen
- `data/` — Zielordner für kopierte Originalbilder, Labels und Debug-Bilder
- `data/algorithm_outputs/` — Session-weise Outputs der Algorithmen
- `requirements.txt` — minimale Abhängigkeiten
- `run_gui.py` — Launcher für die GUI

## Installation

```bash
# Aktiviere virtual environment (optional, empfohlen)
python3 -m venv .venv
source .venv/bin/activate

# Installiere Dependencies
pip install -r requirements.txt
```

## 1) Debug-Bilder extrahieren

Kopiere Bilder mit depth ≤ 15 m aus dem YOLO-Trainingsordner:

```bash
python3 src/extract_debug_images.py \
  --source "/media/angelo/DRONE_DATA1/YoloTrainingImagesV1/1_S/Horizon/mid" \
  --dest data/debug_images \
  --max-depth 15.0
```

Das Skript kopiert nur passende Bilder (depth ≤ 15 m) und manipuliert niemals die Originale — es arbeitet ausschließlich mit Kopien.

## 2) Interaktive GUI starten

Die GUI ermöglicht visuelles Testen und Vergleichen verschiedener Edge-Detection-Algorithmen:

```bash
python3 run_gui.py
```

### GUI-Features:
- **Bildnavigation**: Durchsuche Debug-Bilder mit Previous/Next
- **Algorithmus-Auswahl**: Wähle aus verschiedenen Methoden (Canny, Contour, Hough, etc.)
- **Echtzeit-Anwendung**: Klicke "Apply Algorithm" → sehe sofort Ergebnis + Runtime/FPS
- **Zoom**: Mausrad zum Zoomen (zoomt an Mauszeiger-Position)
- **Session-basiertes Speichern**: Klicke "Save Result" → Output landet in `data/algorithm_outputs/<algorithmus>/session_XXX/`
- Jede neue GUI-Session erstellt automatisch einen neuen Unterordner pro Algorithmus

### Verfügbare Algorithmen (Stand jetzt):
1. **Canny (Auto)** — Auto-Threshold Canny-Edge-Detection ⭐ **Robust**
2. **CLAHE + Canny** — Kontrast-Enhancement + Canny ⭐ **Robust**
3. **Canny (Manual)** — Manuelle Threshold-Kontrolle via Slider
4. **Sobel Magnitude** — Sobel-Gradienten-basierte Edge-Detection
5. **Laplacian of Gaussian** — LoG für feine Kanten
6. **Multi-scale Canny** — Canny auf mehreren Skalen kombiniert
7. **Canny + Contour Silhouette** — Silhouetten-Extraktion via Contour
8. **Canny + Hough Lines** — Linien-Detektion für Flügel/Kanten

### Parameter-Anpassung:
Die GUI bietet Live-Slider für:
- **Canny Low/High Thresholds** (10-300)
- **Gaussian σ** (0.1-5.0) — Blur vor Edge-Detection
- **CLAHE Clip Limit** (1.0-10.0) — Kontrast-Enhancement
- **BBox Padding** (0-20 px) — Reduziert BBox-Kanten-Artefakte

### GUI-Features:
- ✅ **Zoom-Erhaltung**: Zoom bleibt beim Algorithmus-Wechsel erhalten
- ✅ **BBox-Padding**: Vermeidet falsche Edge-Detection an BBox-Kanten
- ✅ **Echtzeit-Parameter**: Slider für sofortige Anpassungen
- ✅ **Session-Management**: Automatische Versionierung der Outputs

## Output-Struktur

Nach dem Speichern von Ergebnissen:

```
data/
├── debug_images/          # Von extract_debug_images.py erstellt
│   ├── originals/
│   ├── labels/
│   ├── debug/
│   └── metadata.csv
└── algorithm_outputs/     # Von GUI erstellt
    ├── canny_auto/
    │   ├── session_001/
    │   └── session_002/
    ├── canny_contour_silhouette/
    │   └── session_001/
    └── ...
```

Jede Session enthält:
- Ergebnis-Bilder (Overlay mit detektierten Kanten)
- Metadaten-JSONs (Runtime, FPS, Algorithmus, Timestamp, BBox-Koordinaten)

## Nächste Schritte

- Weitere Algorithmen hinzufügen (GrabCut, Active Contours, leichte DL-Modelle)
- Quantitative Evaluation mit Ground-Truth-Masken
- Performance-Optimierung für Jetson Orin Nano
