# CPU/GPU/YOLO Test Scripts - Einordnung

Diese Datei fasst die verbleibenden Performance- und Diagnose-Skripte im Root von `Edge_Detection` zusammen.

## Zweck
- Historische "YOLO/GPU/CPU Docs" liegen hier als ausfuehrbare Testskripte vor.
- Statt weiterer verstreuter Markdown-Dateien ist die Referenz jetzt zentral in `docs/performance/`.

## Skripte und Rolle
1. `Edge_Detection/step1_diagnose.py`
- Systemdiagnose: Python, PyTorch, CUDA, cuDNN, TensorRT, JetPack.

2. `Edge_Detection/step2_simple_test.py`
- Baseline-Validierung: CPU Inference, GPU FP32, kleineres `imgsz=320`.

3. `Edge_Detection/test_cuda_yolo.py`
- Kurzer Vergleich CPU vs GPU (inkl. Warmup) mit einfacher Speedup-Ausgabe.

4. `Edge_Detection/test_gpu_simple.py`
- Fokus auf cuDNN-Problemfaelle; testet Workaround ohne cuDNN.

5. `Edge_Detection/test_cpu_timing.py`
- CPU-only Benchmark fuer YOLO + Edge Detection (mehrere Inputgroessen).

6. `Edge_Detection/test_2stage_approach.py`
- Zwei-Stufen-Ansatz: YOLO auf kleiner Eingabe, Edge Detection auf Original-ROI.

7. `Edge_Detection/test_yolo_optimization.py`
- Matrix-Benchmark mehrerer Konfigurationen (device/imgsz/half) inkl. Ranking.

8. `Edge_Detection/quick_test.py`
- Schneller Smoke-Test fuer optimierte YOLO-Settings.

## Empfohlene Reihenfolge
1. `step1_diagnose.py`
2. `step2_simple_test.py`
3. `test_cuda_yolo.py`
4. `test_yolo_optimization.py`
5. `test_2stage_approach.py` und `test_gpu_simple.py` bei Vertiefung
6. `quick_test.py` fuer schnelle Re-Checks

## Operative Empfehlung
- Stabiler Standardpfad: GPU + FP32 + `imgsz=320` pruefen.
- Erweiterungen (cuDNN-Workaround, TensorRT) nur bei echtem FPS-Bedarf und nach Messung.
