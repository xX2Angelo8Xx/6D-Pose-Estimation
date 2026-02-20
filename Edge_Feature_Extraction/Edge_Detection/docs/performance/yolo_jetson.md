# YOLO Performance auf Jetson - Konsolidierung

Diese Zusammenfassung vereint `YOLO_OPTIMIZATION`, `YOLO_SOLUTION_EXPLAINED`, `GPU_WORKAROUND`, `QUICKSTART_OPTIMIZED`.

## Ziel
- Inferenzzeit < 200 ms fuer praxistaugliche Interaktivitaet

## Beobachtete, dokumentierte Betriebsmodi
1. GPU + FP32 + Input 320: stabiler Betrieb, mehrfach < 200 ms dokumentiert (z. B. ~114 ms).
2. GPU mit cuDNN-Workaround (cuDNN aus): in Notizen teils nochmals schneller berichtet.
3. TensorRT: als optionaler naechster Schritt mit deutlich kleinerer Latenz (projektionsbasiert).

## Pragmatische Empfehlung
1. Erst stabile Basis fahren: GPU + Input 320, ohne riskante Spezialpfade.
2. Dann optional Workaround/TensorRT nur, wenn wirklich mehr FPS noetig sind.
3. Immer mit den lokalen Testskripten gegenmessen (`test_yolo_optimization.py`, `test_cuda_yolo.py`, etc.).

## Warum diese Konsolidierung wichtig ist
Die alten Notizen enthalten mehrere Zwischenstaende mit teils unterschiedlichen Zahlen. Entscheidend ist:
- Das <200 ms Ziel wurde erreicht.
- Der stabile Standardpfad sollte Prioritaet vor maximaler, aber komplexerer Optimierung haben.

## Verwandte Referenz
- `Edge_Detection/docs/performance/cpu_gpu_yolo_test_scripts.md` fuer die Einordnung der Benchmark- und Diagnose-Skripte.
