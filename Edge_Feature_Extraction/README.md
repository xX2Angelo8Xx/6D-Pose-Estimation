# Edge Feature Extraction (Subproject)

Dieses Subproject bereitet Kantenfeatures aus YOLO-annotierten Trainingsbildern vor.

Wichtigste Artefakte:

- `src/extract_debug_images.py` — Hilfsskript zum selektiven Kopieren von Bildern (depth ≤ 15m), Erzeugen von Debugbildern mit Bounding-Boxes und Sammeln von Metadaten.
- `data/` — Zielordner für kopierte Originalbilder, Labels und Debug-Bilder (wird beim Ausführen des Skripts erstellt).
- `requirements.txt` — minimale Abhängigkeiten.

Benutzung (Beispiel):

```bash
python3 src/extract_debug_images.py \
  --source "/media/angelo/DRONE_DATA1/YoloTrainingImagesV1/1_S/Horizon/mid" \
  --dest data/debug_images \
  --max-depth 15.0
```

Das Skript kopiert nur passende Bilder (depth ≤ 15 m) und manipuliert niemals die Originale — es arbeitet ausschließlich mit Kopien.
