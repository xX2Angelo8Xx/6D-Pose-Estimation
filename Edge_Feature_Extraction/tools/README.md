# tools/

Hilfsskripte für Datenanalyse, Kalibrierung und LUT-Qualitätsbewertung.

---

## `get_zed_intrinsics.py`
Liest kalibrierte ZED-2i-Kameradaten direkt aus dem SDK aus.

```bash
# ZED 2i muss per USB angeschlossen sein
python3 tools/get_zed_intrinsics.py
```

Ausgabe: `tools/zed2i_intrinsics_hd720.json`

**Gemessene Werte (S/N 34754237 @ HD720):**
- fx = fy = 951.1 px, cx = 638.9, cy = 348.0
- HFOV = 67.8°, VFOV = 40.2°

---

## `check_lut_quality.py`
Systematische Qualitätsprüfung der gesamten LUT-Datenbank in drei Phasen.

```bash
# Phase 1 nur (JSON-Statistiken, schnell):
python3 tools/check_lut_quality.py --no-render --out-dir tools/lut_quality_output

# Vollständig (Top-200 gerendert):
python3 tools/check_lut_quality.py --top 200 --out-dir tools/lut_quality_output
```

**Ausgaben:**
| Datei | Inhalt |
|---|---|
| `lut_quality_heatmap.png` | 22×8"-Heatmap: az × el, rote ✗ auf verdächtigen Zellen |
| `lut_quality_bad_renders.png` | 10×10-Kontaktbogen der 100 schlechtesten Posen |
| `lut_quality_report.csv` | Alle 8296 Posen, sortiert nach n_points (asc) |

**Qualitätsmetriken pro Pose:**
- `fill_ratio` = morphologisch-geschlossene Pixelfläche / konvexe Hüllenfläche
- `gap_fraction` = größter angularer Abstand im Konturlinienzug / 2π
- `quality_score = fill_ratio × (1 − gap_fraction)`

**Befunde (Stand Feb 2026):**
- 830/8296 roll=0-Posen als verdächtig markiert
- Beide Fehlercluster haben gap_fraction ≈ 0.28 (vs. 0.013 bei Referenz-Posen)
- Betroffen: az ≈ 85–120° und 240–275° bei el ≤ 0° — dünne Seitenansichten und hintere Diagonalen
- Ursache: LUT-Punkte in der Optimizer-Kostenfunktion waren eingefroren → behoben in `fix: LUT-refresh in optimizer`

---

## `lut_quality_deep_inspect.py`
Rendert gezielte Azimut-Sweeps über den Elevationsbereich für detaillierte Diagnose.

```bash
python3 tools/lut_quality_deep_inspect.py
```

Ausgabe: `tools/lut_quality_output/lut_quality_sweep_detail.png`
- Zeigt Cluster A (az 86–94°, 266–274°), Cluster B (az 108–116°, 242–250°) und Referenzposen nebeneinander.

---

## `lut_quality_gallery.py`
Erzeugt eine interaktive HTML-Galerie der schlechtesten N Posen in voller Auflösung (1280×720).

```bash
python3 tools/lut_quality_gallery.py --top 200 --out-dir tools/lut_quality_output
```

Ausgabe: `tools/lut_quality_output/lut_quality_gallery.html` (~9 MB)

**Features:**
- Klick auf Bild → Lightbox (vollauflösend, beliebig zoombar im Browser)
- ← / → Pfeiltasten zum Durchblättern
- Filter nach Fehlertyp: GAP / SPARSE / alle fehlerhaften / OK
- Filter nach Azimuth, Sortierung nach Quality/Gap/n_points
- Farb-codierte Rahmen: Rot = Gap-Fehler (>0.15), Orange = Sparse (<2200 Punkte), Grün = OK

---

## Ausgabe-Verzeichnis `lut_quality_output/`
Erzeugte Dateien sind in `.gitignore` ausgenommen (regenerierbar).
