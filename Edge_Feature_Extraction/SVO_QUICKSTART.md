# SVO Mode - Quick Start Guide

## 🚀 Schnellstart (5 Sekunden)

```bash
cd /home/angelo/Projects/6D_Pose_Estimation/Edge_Feature_Extraction
./start_svo.sh
```

## 📋 Was passiert beim Start?

1. **YOLO Model Loading** (~3 Sekunden)
   ```
   Loading YOLO model from .../best.pt...
   YOLO model loaded successfully!
   ```

2. **Folder Dropdown gefüllt** mit 1.1, 1.2, 1.3, ...

3. **Automatisch Folder 1.1 geladen**
   ```
   Loading folder: .../1.1
   Found 1136 images in 1.1
   ```

4. **Erstes Bild geladen** (frame_000350.jpg)
   ```
   Loading image: frame_000350.jpg
   Loaded depth data: (720, 1280)
   Running YOLO inference...
   YOLO inference took 1438.49 ms
   Detected bbox: (627, 234, 641, 242), confidence: 0.766
   ```

5. **GUI ist bereit!** 🎉

## 🎮 Bedienung

### Bilder navigieren
- **Next ▶** Button → Nächstes Bild (frame_000351, 352, ...)
- **◀ Previous** Button → Vorheriges Bild

### Algorithmus anwenden
1. **Algorithmus wählen** im Dropdown (z.B. "Canny + Contours")
2. **Parameter einstellen** mit Slidern
3. **Apply Algorithm** Button drücken
4. **Ergebnis sehen** in der Image View

### Depth Filter nutzen
1. **✓ Depth Filter** Checkbox aktivieren
2. **Radius Slider** einstellen (z.B. 5-10 Pixel)
3. **Apply Algorithm** erneut drücken
4. Konturen ohne valide Tiefe werden gefiltert

### Zoom & Pan
- **Mausrad** → Rein/raus zoomen
- **Maus ziehen** → Bild verschieben (bei Zoom)

## ⏱️ Typische Performance

**Pro Bild:**
- YOLO Inference: ~1.3-1.7 Sekunden (CPU)
- Edge Detection: 10-150ms (je nach Algorithmus)
- **Total: ~1.3-1.8 Sekunden**

**Navigation ist schnell:**
- Next/Previous ohne Apply: ~1.5s (nur YOLO)
- Mit Apply: +0.01-0.15s für Edge Detection

## 🧪 Test-Workflow

### Test 1: Basis-Funktionalität
1. GUI starten → ✅ Sollte ohne Fehler laden
2. Folder 1.1 ausgewählt → ✅ 1136 images gefunden
3. Erstes Bild geladen → ✅ Grüne BBox sichtbar
4. Next Button → ✅ Nächstes Bild lädt

### Test 2: Edge Detection
1. "Canny (Auto)" wählen
2. Apply Button drücken
3. Ergebnis: ✅ Grüne Kanten in BBox

### Test 3: CLAHE
1. "CLAHE + Canny" wählen
2. CLAHE Clip Slider ändern (z.B. 3.0)
3. Apply Button
4. Ergebnis: ✅ Rote Kanten (durch CLAHE verstärkt)

### Test 4: Contours + Depth Filter
1. "Canny + Contours" wählen
2. ✓ Depth Filter aktivieren
3. Radius auf 10 setzen
4. Apply Button
5. Ergebnis: ✅ Gelbe Konturen (nur mit valider Tiefe)

## 📊 Beispiel-Output (Console)

```
Loading YOLO model from /media/angelo/DRONE_DATA1/.../best.pt...
YOLO model loaded successfully!
Found 7 SVO folders
Loading folder: /media/angelo/DRONE_DATA1/SVO2_Frame_Export/1.1
Found 1136 images in 1.1
Loading image: frame_000350.jpg
Loaded depth data: (720, 1280)
Running YOLO inference...
YOLO inference took 1438.49 ms
Detected bbox: (627, 234, 641, 242), confidence: 0.766
```

## ❓ Häufige Fragen

### Q: YOLO ist langsam (1.5s pro Bild)?
**A:** Normal auf CPU. Alternativen:
- GPU nutzen (wenn CUDA funktioniert): Zeile 328 ändern zu `device='cuda:0'`
- Kleineres Model verwenden (YOLOv8n statt YOLOv8m)

### Q: Keine Depth Data?
**A:** Prüfen ob `.npy` Files vorhanden:
```bash
ls /media/angelo/DRONE_DATA1/SVO2_Frame_Export/1.1/*.npy | head
```
Falls nicht vorhanden: Depth Filter funktioniert nicht, aber Rest läuft.

### Q: No Detection in einigen Bildern?
**A:** Normal - YOLO findet nicht in jedem Frame ein Objekt.
→ Weiter navigieren zu Frames mit Detection.

### Q: CUDA Warnungen?
**A:** Ignorieren - GUI nutzt CPU Fallback automatisch.

## 🔧 Anpassungen

### YOLO auf GPU umschalten
```python
# Datei: src/gui/svo_edge_detection_gui.py, Zeile 328
# Ändern von:
results = self.yolo_model(self.current_image, verbose=False, device='cpu')
# Zu:
results = self.yolo_model(self.current_image, verbose=False, device='cuda:0')
```

### Anderen SVO Folder als Default
```python
# Datei: src/gui/svo_edge_detection_gui.py, Zeile 93
self.svo_base_dir = Path("/ihr/pfad/zu/SVO2_Frame_Export")
```

### Anderes YOLO Model
```python
# Zeile 94
self.yolo_model_path = Path("/pfad/zu/ihrem/best.pt")
```

## 📈 Performance Optimierung

### Schnelleres YOLO (Optionen):
1. **GPU nutzen** (wenn verfügbar)
2. **Batch Processing** (mehrere Bilder auf einmal)
3. **Lower Resolution** (Bilder vor YOLO downsamplen)
4. **Leichteres Model** (YOLOv8n)

### Schnellere Edge Detection:
- Gaussian Sigma reduzieren (z.B. 0.5 statt 1.0)
- ROI verkleinern (falls BBox zu groß)

## 🎯 Typische Arbeitsschritte

### Scenario 1: Edge Parameter finden
1. Ein gutes Bild mit klarer Detection finden (Next/Previous)
2. Algorithmus wählen (z.B. Canny + Contours)
3. Parameter variieren (Sigma, CLAHE Clip)
4. Apply mehrfach drücken mit verschiedenen Werten
5. Beste Parameter merken

### Scenario 2: Depth Filter testen
1. Bild mit Depth Data laden
2. "Canny + Contours" wählen
3. Ohne Filter: Apply → Viele Konturen
4. Mit Filter: ✓ aktivieren, Radius 5-10 → Apply
5. Vergleich: Weniger aber bessere Konturen

### Scenario 3: Verschiedene Folder vergleichen
1. Folder 1.1 laden → Parameter finden
2. Folder 1.2 wählen → Gleiche Parameter testen
3. Unterschiede in Performance/Qualität notieren

## 📝 Nächste Schritte

Nach dem Quick Start können Sie:
- **Alle Algorithmen testen** (3 verfügbar)
- **Optimale Parameter finden** für Ihre Use Cases
- **Depth Filter evaluieren** (wie viel bringt es?)
- **Performance messen** (ist YOLO der Bottleneck?)

Für Details siehe: `SVO_GUI_README.md`

---

**Support:** Bei Problemen Terminal Output prüfen (siehe Console)  
**Dokumentation:** Siehe `SVO_GUI_README.md`, `DECISION_SEPARATE_GUI.md`  
**Code:** `src/gui/svo_edge_detection_gui.py` (530 Zeilen, gut kommentiert)
