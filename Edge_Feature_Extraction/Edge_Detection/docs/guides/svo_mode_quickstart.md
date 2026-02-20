# SVO Mode Quickstart (Konsolidiert)

Diese Anleitung fasst `SVO_QUICKSTART`, `SVO_GUI_README` und `SVO_MODE_DOCUMENTATION` zusammen.

## Start
```bash
cd /home/angelo/Projects/6D_Pose_Estimation/Edge_Feature_Extraction
./Edge_Detection/start_svo_mode.sh
```

## Standard-Workflow
1. SVO-Folder waehlen (z. B. `1.1`).
2. Bild laden und YOLO-BBox pruefen.
3. Algorithmus waehlen (z. B. Canny/Adaptive).
4. Optional Depth Filter aktivieren und Radius setzen.
5. Ergebnis anwenden, mit Next/Previous navigieren.

## Depth Filter (wichtig)
- Konturen ohne valide Tiefe werden verworfen.
- Radius steuert, wie weit um Konturpunkte nach Tiefenwerten gesucht wird.

## Read-only Sicherheit
- Frames (`.jpg`) und Depth (`.npy`) werden nur gelesen.
- Keine Modifikation an SVO-Quelldaten.

## Typische Ursachen bei Problemen
- Keine YOLO-Detection in einem Frame: naechstes Bild pruefen.
- Keine Depth-Werte: `.npy`-Verfuegbarkeit/Gueltigkeit pruefen.
- Langsame Inferenz: Inputgroesse und Device in Performance-Doku abstimmen.
