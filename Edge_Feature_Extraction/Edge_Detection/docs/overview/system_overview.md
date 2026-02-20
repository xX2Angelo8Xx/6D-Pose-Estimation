# System Overview

## Ziel des Unterprojekts
`Edge_Detection` dient zur visuellen Analyse von Aircraft-Konturen in YOLO-basierten ROIs, inklusive SVO-Frames und optionaler Tiefendaten.

## Wichtige Bausteine
- GUI fuer interaktive Edge-Analyse (`src/gui/`)
- Edge-Algorithmen (`src/algorithms/`), inkl. adaptiver Pipeline
- SVO-Betrieb mit YOLO-Inferenz und Depth-Filter
- Test-/Benchmark-Skripte fuer YOLO und Pipeline-Performance

## Konsolidierte Kernthemen aus den alten Docs
1. Adaptive Pipeline ist der robusteste Modus fuer schwierige Szenen und Sequenzen.
2. SVO-Workflow ist stabil, inkl. Folder-Auswahl, YOLO-BBox, Depth-Filter.
3. YOLO-Latenz wurde mehrfach optimiert und ist unter 200 ms erreichbar.
4. Interaktiver Debug-Modus macht die Pipeline-Schritte nachvollziehbar.

## Aktueller Betriebsfokus
- Fuer schnelle Einzelbildanalyse: einfache Pipeline/klassische Canny-Varianten.
- Fuer robuste Sequenzanalyse: adaptive Pipeline mit temporaler Stabilisierung.
- Fuer Jetson-Betrieb: GPU + kleine YOLO-Inputgroesse (typisch 320).
