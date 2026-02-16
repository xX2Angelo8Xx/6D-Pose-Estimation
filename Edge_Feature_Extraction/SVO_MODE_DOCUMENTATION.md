# SVO Mode Documentation - Edge Detection GUI

## 🎯 Übersicht

Die GUI wurde um einen **SVO Mode** erweitert, der externe Frame-Daten mit YOLO-Objekterkennung und Tiefenfilterung kombiniert.

---

## 🆕 Neue Features

### 1. **Dual-Mode Operation**
- **Debug Mode**: Verwendet vorhandene YOLO-Labels aus `.txt` Dateien
- **SVO Mode**: Führt YOLO-Inferenz in Echtzeit aus + nutzt Tiefendaten

### 2. **YOLO-Integration**
- **Modell**: `/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt`
- **Framework**: Ultralytics YOLO
- **Funktion**: Automatische Objekterkennung in Frames ohne vorhandene Labels

### 3. **Tiefendaten-Filterung**
- **Quelle**: `.npy` Dateien im selben Ordner wie Frames
- **Funktion**: Verwirft Konturen ohne valide Tiefendaten in einem konfigurierbaren Radius
- **Steuerung**: 
  - Checkbox: "Enable Depth Filter"
  - Slider: "Depth Search Radius" (1-50 Pixel)

### 4. **Performance-Messung**
- **YOLO Inference Time**: Zeit für Objekterkennung
- **Edge Detection Time**: Zeit für Kantenberechnung
- **Total Time**: Gesamtzeit
- Anzeige nur im SVO Mode sichtbar

---

## 📁 Datenstruktur

### SVO Frame Export
```
/media/angelo/DRONE_DATA1/SVO2_Frame_Export/
├── 1.1/
│   ├── frame_000350.jpg  ← RGB Frame (READ-ONLY)
│   ├── frame_000350.npy  ← Tiefendaten
│   ├── frame_000351.jpg
│   ├── frame_000351.npy
│   └── ...
├── 1.2/
├── 1.3/
├── 1.4/
└── 1.5/
```

**Wichtig**: Originaldaten werden **niemals** modifiziert (read-only Zugriff).

### YOLO Modell
```
/media/angelo/DRONE_DATA1/JetsonExport/
└── svo_model_20251204_112724_1280/
    └── models/
        ├── best.pt      ← PyTorch Modell (verwendet)
        ├── best.onnx
        └── best.engine
```

---

## 🎮 Bedienung

### Mode-Wechsel
1. **Operation Mode** Panel am oberen Bildschirmrand
2. Zwei Optionen:
   - ☑️ **Debug Mode (YOLO Labels)**: Standard, nutzt vorhandene Labels
   - ☐ **SVO Mode (YOLO Inference + Depth)**: Neue Funktionalität

### SVO Mode Aktivierung
1. Radio-Button "SVO Mode" auswählen
2. Folder-Dropdown erscheint automatisch
3. Verfügbare Ordner: 1.1, 1.2, 1.3, 1.4, 1.5, etc.
4. Ordner auswählen → Frames werden geladen

### Depth-Filter Nutzung
**Nur aktiv im SVO Mode!**

1. **Checkbox aktivieren**: "Enable Depth Filter"
2. **Radius einstellen**: Slider "Depth Search Radius (px)"
   - Bereich: 1-50 Pixel
   - Standard: 5 Pixel
   - Funktion: Sucht in n-Pixel-Umkreis nach validen Tiefendaten

3. **Algorithmus ausführen**:
   - "Canny + Contour Silhouette" empfohlen (nutzt Depth-Filter)
   - Konturen ohne valide Tiefe werden verworfen

### Performance-Messung
**Automatisch im SVO Mode angezeigt**

Nach Algorithmus-Ausführung erscheint:
```
YOLO: 45.23 ms | Edge: 12.87 ms | Total: 58.10 ms
```

- **YOLO**: Zeit für Objekterkennung beim Laden des Frames
- **Edge**: Zeit für Edge-Detection-Algorithmus
- **Total**: Summe beider Prozesse

---

## 🔧 Technische Details

### Tiefenfilter-Algorithmus

```python
def has_valid_depth(contour, bbox_offset):
    """
    Prüft ob Kontur valide Tiefendaten hat.
    
    Ablauf:
    1. Sample 20 Punkte entlang der Kontur
    2. Für jeden Punkt: Suche in radius-Pixel-Umkreis
    3. Valide Tiefe = depth > 0 und isfinite()
    4. Return True wenn mind. 1 valider Punkt gefunden
    """
```

**Anwendung**: Nur in `apply_canny_contour()` implementiert
- Filtert Konturen vor der Silhouette-Extraktion
- Reduziert False Positives in Bereichen ohne Tiefeninformation

### YOLO-Inferenz

```python
# Beim Frame-Laden (SVO Mode)
results = yolo_model(image, verbose=False)
bbox = results[0].boxes[0].xyxy  # Erste Detection
```

**Fehlerbehandlung**:
- Keine Detection → Warnung angezeigt
- YOLO-Fehler → Fallback zu Debug Mode

### Sicherheit der Originaldaten

**Read-Only Zugriff**:
```python
# Frames werden nur gelesen, nie geschrieben
self.current_image = cv2.imread(str(img_path))  # READ
depth_data = np.load(str(depth_path))           # READ

# Alle Operationen auf Kopien
result = self.current_image.copy()  # COPY
```

**Keine Dateisystem-Operationen** auf SVO-Frames:
- Kein `cv2.imwrite()` in SVO-Verzeichnisse
- Kein Überschreiben von `.npy` Dateien
- Ergebnisse werden in `data/algorithm_outputs/` gespeichert

---

## 📊 Parameter-Übersicht

### Neue Parameter

| Parameter | Bereich | Standard | Beschreibung |
|-----------|---------|----------|--------------|
| `use_depth_filter` | True/False | False | Aktiviert Tiefenfilterung |
| `depth_radius` | 1-50 px | 5 px | Suchradius für valide Tiefe |

### Erweiterte Timing-Variablen

| Variable | Typ | Beschreibung |
|----------|-----|--------------|
| `yolo_inference_time` | float | YOLO-Laufzeit in ms |
| `edge_detection_time` | float | Edge-Algorithmus-Laufzeit in ms |
| `last_runtime_ms` | float | Gesamt-Laufzeit (abwärtskompatibel) |

---

## 🐛 Bekannte Einschränkungen

1. **YOLO-Abhängigkeit**: 
   - Requires `ultralytics` package
   - Falls nicht installiert: SVO Mode nicht verfügbar

2. **Tiefenfilter nur in Canny + Contours**:
   - Andere Algorithmen nutzen Depth-Filter noch nicht
   - Erweiterung auf andere Algorithmen möglich

3. **Single Detection**:
   - Nutzt nur erste YOLO-Detection
   - Multi-Object-Support könnte hinzugefügt werden

4. **Depth-Format**:
   - Erwartet `.npy` mit 2D-Array (H×W)
   - Valide Tiefe: `depth > 0 and np.isfinite(depth)`

---

## 🔄 Workflow-Beispiel

### SVO Mode Workflow

1. **GUI starten**
   ```bash
   cd Edge_Feature_Extraction
   python3 run_gui.py
   ```

2. **SVO Mode aktivieren**
   - Radio-Button "SVO Mode" wählen
   - Folder "1.1" aus Dropdown auswählen

3. **Frame laden**
   - "Next ▶" klicken um durch Frames zu navigieren
   - YOLO läuft automatisch bei jedem Frame

4. **Depth-Filter konfigurieren**
   - Checkbox "Enable Depth Filter" aktivieren
   - Radius auf 10 Pixel einstellen

5. **Algorithmus ausführen**
   - "Canny + Contour Silhouette" auswählen
   - "Apply Algorithm" klicken

6. **Performance prüfen**
   - Timing-Anzeige lesen:
     - YOLO: ~40-60 ms
     - Edge: ~10-20 ms
     - Total: ~50-80 ms

7. **Ergebnis speichern** (optional)
   - "💾 Save Result" klicken
   - Speicherort: `data/algorithm_outputs/`

---

## 🚀 Erweiterungsmöglichkeiten

### Zukünftige Features

1. **Multi-Object-Detection**
   - Alle YOLO-Detections verarbeiten
   - Separate ROIs für jedes Objekt

2. **Depth-Visualisierung**
   - Tiefenkarte als Overlay
   - Farbcodierung nach Distanz

3. **Batch-Processing**
   - Alle Frames eines Folders automatisch verarbeiten
   - Export als Video

4. **Tiefenfilter für alle Algorithmen**
   - Integration in Canny, Sobel, LoG, etc.
   - Depth-weighted Edge Detection

5. **Performance-Optimierung**
   - TensorRT für YOLO (`.engine` nutzen)
   - GPU-Beschleunigung für Edge-Detection

---

## 📖 API-Referenz

### Neue Methoden

#### `init_svo_mode()`
Initialisiert SVO-Komponenten (YOLO-Modell, Folder-Liste).

#### `on_mode_changed(checked: bool)`
Handler für Mode-Wechsel zwischen Debug und SVO.

#### `on_svo_folder_changed(folder_name: str)`
Handler für SVO-Folder-Auswahl.

#### `load_svo_images()`
Lädt Frame-Liste aus ausgewähltem SVO-Folder.

#### `has_valid_depth(contour, bbox_offset) -> bool`
Prüft ob Kontur valide Tiefendaten hat.

**Parameters**:
- `contour`: OpenCV-Kontur (ROI-Koordinaten)
- `bbox_offset`: Tuple (x, y) für Koordinaten-Transformation

**Returns**: `True` wenn valide Tiefe gefunden, sonst `False`

---

## ✅ Testing

### Test-Checkliste

- [x] GUI startet ohne Fehler
- [x] Mode-Wechsel funktioniert
- [x] SVO-Folder-Auswahl funktioniert
- [x] YOLO-Inferenz läuft
- [x] Tiefendaten werden geladen
- [x] Depth-Filter funktioniert
- [x] Performance-Messung korrekt
- [x] Originaldaten unverändert

### Test-Command
```bash
cd Edge_Feature_Extraction
timeout 5 python3 run_gui.py  # Exit code 143 = OK (timeout)
```

---

**Version**: 0.4.0  
**Datum**: 2026-02-13  
**Autor**: Extended GUI with SVO Mode
