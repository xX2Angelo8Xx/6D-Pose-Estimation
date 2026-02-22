#!/usr/bin/env python3
"""
YOLO imgsz Switching Benchmark
================================
Misst für jede imgsz-Stufe:
  • N-Frame Steady-State Timing (Mittelwert, Std, Min, Max, P95, FPS)
  • Detektions-Check: erkennt das Modell das Objekt noch bei dieser Auflösung?
  • Wechsel-Kosten: wie lange dauert Frame 1 nach einem imgsz-Wechsel?

Verwendet ein echtes Trainingsbild (~9.69m Zielentfernung).
NMS-Overhead ist damit realistisch inkludiert.

Ausführung:
    python3 Edge_Detection/test_imgsz_switching.py

Konfiguration direkt im Skript anpassbar (MODEL_PATH, IMAGE_PATH, N_FRAMES, ...)
"""

import time
import statistics
import warnings
from pathlib import Path

import numpy as np
import torch

# ── Orin-spezifisch: cuDNN liefert keine Execution-Plans für diese PyTorch-
# ── Installation. Ohne cuDNN laufen Pure-CUDA-Kernel, die korrekt funktionieren.
# ── (Entspricht dem tatsächlichen Laufzeitverhalten der App: 13 FPS @ 1280)
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path("/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt")
IMAGE_PATH  = Path("/home/angelo/Projects/6D_Pose_Estimation/Edge_Feature_Extraction/Edge_Detection/data/debug_images/originals/frame_000428-1.3-SE_Horizon-depth-9.69m-std-1.40m.jpg")
N_FRAMES    = 200     # Frames pro Stufe für Steady-State Messung
N_WARMUP    = 10      # Warmup-Frames (verworfen)
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF        = True    # FP16 auf CUDA, automatisch False auf CPU
CONF_THRESH = 0.25    # Mindest-Konfidenz für Detektion

# Alle zu testenden Auflösungen: (Label, imgsz)
# Alle müssen Vielfache von 32 sein
STAGES = [
    ("256  (sehr klein)",  256),
    ("320",                320),
    ("384",                384),
    ("416",                416),
    ("448",                448),
    ("480",                480),
    ("512",                512),
    ("640  (Standard)",    640),
    ("960",                960),
    ("1280 (Training)",   1280),
]

# Wechsel-Sequenz für Switch-Cost-Test
SWITCH_SEQUENCE = [1280, 960, 640, 512, 384, 512, 640, 960, 1280]

# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Lädt Bild von Pfad. Fallback auf schwarzes 1280×720 wenn nicht gefunden."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        print(f"  WARNUNG: Bild nicht gefunden: {path}")
        print("  → Verwende schwarzes Fallback-Bild (720×1280)")
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    return img


def infer_with_detections(model: YOLO, img: np.ndarray, imgsz: int,
                          half: bool, device: str, conf: float
                          ) -> tuple[float, list[dict]]:
    """Einzelne Inferenz. Gibt (Zeit in ms, Liste der Detektionen) zurück."""
    use_half = half and device.startswith("cuda")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = model(img, verbose=False, device=device, imgsz=imgsz,
                    half=use_half, conf=conf)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detections.append({
                "cls":  int(boxes.cls[i].item()),
                "conf": float(boxes.conf[i].item()),
                "w_px": x2 - x1,
                "h_px": y2 - y1,
            })
    return elapsed_ms, detections


def infer(model: YOLO, img: np.ndarray, imgsz: int,
          half: bool, device: str) -> float:
    """Schnelle Inferenz nur für Timing (Detektionsinfo wird verworfen)."""
    t, _ = infer_with_detections(model, img, imgsz, half, device, CONF_THRESH)
    return t


def measure_frames(model: YOLO, img: np.ndarray, imgsz: int,
                   n: int, half: bool, device: str) -> list[float]:
    """Führt n Inferenzen durch und gibt Zeiten in ms zurück."""
    return [infer(model, img, imgsz, half, device) for _ in range(n)]


def print_stats(label: str, times: list[float]) -> None:
    avg  = statistics.mean(times)
    std  = statistics.stdev(times) if len(times) > 1 else 0.0
    mn   = min(times)
    mx   = max(times)
    fps  = 1000.0 / avg
    p95  = sorted(times)[int(0.95 * len(times))]
    print(f"  {label}")
    print(f"    Avg: {avg:7.2f} ms  |  Std: {std:5.2f} ms  |  "
          f"Min: {mn:6.2f}  Max: {mx:6.2f}  P95: {p95:6.2f}  |  {fps:5.1f} FPS")


def sep(char: str = "─", n: int = 72) -> None:
    print(char * n)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    sep("═")
    print("  YOLO imgsz Switching Benchmark")
    sep("═")

    # System-Info
    print(f"\n  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        mem_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {mem_gb:.1f} GB")
    print(f"  cuDNN  : DEAKTIVIERT (Orin Pure-CUDA-Kernel, entspricht App-Verhalten)")
    print(f"  Frames : {N_FRAMES} pro Stufe  |  Warmup: {N_WARMUP}")
    print(f"  Conf   : {CONF_THRESH}  (Mindest-Konfidenz für Detektions-Check)")

    # Modell laden
    sep()
    if not MODEL_PATH.exists():
        print(f"\n  FEHLER: Modell nicht gefunden: {MODEL_PATH}")
        print("  → Externe Festplatte eingesteckt?")
        return

    print(f"\n  Lade Modell: {MODEL_PATH.name} ...")
    model = YOLO(str(MODEL_PATH))
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameter  : {total_params:,}")
    print(f"  Klassen    : {model.names}")

    # Bild laden
    img = load_image(IMAGE_PATH)
    print(f"\n  Testbild   : {IMAGE_PATH.name}")
    print(f"  Bildgröße  : {img.shape[1]}×{img.shape[0]} px")

    # ── TEIL 1: Steady-State + Detektion pro Stufe ────────────────────────────
    sep()
    print(f"\n  TEIL 1 — Steady-State ({N_FRAMES} Frames pro Stufe, nach {N_WARMUP} Warmup-Frames)\n")
    sep()

    steady_results: dict[int, list[float]] = {}
    detection_results: dict[int, list[dict]] = {}

    for label, imgsz in STAGES:
        print(f"\n  imgsz={imgsz:<5}  {label}")

        # Warmup
        print(f"    Warmup ({N_WARMUP} Frames) ...", end=" ", flush=True)
        measure_frames(model, img, imgsz, N_WARMUP, HALF, DEVICE)
        print("fertig")

        # Detektions-Check: 1 Inferenz mit Detektion lesen
        _, dets = infer_with_detections(model, img, imgsz, HALF, DEVICE, CONF_THRESH)
        detection_results[imgsz] = dets
        if dets:
            cls_names = [model.names[d["cls"]] for d in dets]
            confs     = [f"{d['conf']:.2f}" for d in dets]
            sizes     = [f"{d['w_px']:.0f}×{d['h_px']:.0f}px" for d in dets]
            det_str   = "  |  ".join(
                f"{n} conf={c} bbox={s}"
                for n, c, s in zip(cls_names, confs, sizes)
            )
            print(f"    Detektion  : {len(dets)}× gefunden  →  {det_str}")
        else:
            print(f"    Detektion  : NICHT ERKANNT (conf < {CONF_THRESH})")

        # Timing-Messung
        print(f"    Messe {N_FRAMES} Frames ...", end=" ", flush=True)
        times = measure_frames(model, img, imgsz, N_FRAMES, HALF, DEVICE)
        print("fertig")

        steady_results[imgsz] = times
        print_stats(f"imgsz={imgsz}", times)

    # ── TEIL 2: Switch-Cost ───────────────────────────────────────────────────
    sep()
    print("\n  TEIL 2 — Switch-Cost\n")
    print("  Misst: wie lange dauert Frame 1 nach einem imgsz-Wechsel?")
    print(f"  Sequenz: {' → '.join(str(s) for s in SWITCH_SEQUENCE)}\n")
    sep()

    switch_costs: list[dict] = []
    prev_imgsz = None

    for imgsz in SWITCH_SEQUENCE:
        if prev_imgsz is not None and imgsz != prev_imgsz:
            switch_ms = infer(model, img, imgsz, HALF, DEVICE)
            follow_times = measure_frames(model, img, imgsz, 5, HALF, DEVICE)
            steady_ref   = statistics.mean(steady_results.get(imgsz, follow_times))
            overhead = switch_ms - steady_ref
            tag = "RECOMPILE" if overhead > 20 else "ok"
            print(f"  {prev_imgsz:>4} → {imgsz:<4}  "
                  f"Frame1: {switch_ms:6.2f} ms  "
                  f"Ref: {steady_ref:6.2f} ms  "
                  f"Overhead: {overhead:+6.2f} ms  [{tag}]")
            switch_costs.append({
                "from": prev_imgsz, "to": imgsz,
                "switch_ms": switch_ms, "steady_ms": steady_ref,
                "overhead_ms": overhead,
            })
        else:
            measure_frames(model, img, imgsz, N_WARMUP, HALF, DEVICE)
        prev_imgsz = imgsz

    # ── ZUSAMMENFASSUNG ───────────────────────────────────────────────────────
    sep("═")
    print("\n  ZUSAMMENFASSUNG\n")
    sep("═")
    print(f"\n  {'imgsz':>6}  {'Avg ms':>8}  {'FPS':>6}  {'P95 ms':>8}  "
          f"{'Std ms':>7}  {'Min':>7}  {'Max':>7}  {'Detektion'}")
    sep()
    for label, imgsz in STAGES:
        ts   = steady_results[imgsz]
        avg  = statistics.mean(ts)
        fps  = 1000.0 / avg
        p95  = sorted(ts)[int(0.95 * len(ts))]
        std  = statistics.stdev(ts)
        mn   = min(ts)
        mx   = max(ts)
        dets = detection_results.get(imgsz, [])
        if dets:
            best = max(dets, key=lambda d: d["conf"])
            det_str = f"OK  {model.names[best['cls']]} {best['conf']:.2f}  {best['w_px']:.0f}x{best['h_px']:.0f}px"
        else:
            det_str = "NICHT ERKANNT"
        print(f"  {imgsz:>6}  {avg:>8.2f}  {fps:>6.1f}  {p95:>8.2f}  "
              f"{std:>7.2f}  {mn:>7.2f}  {mx:>7.2f}  {det_str}")

    sep()
    print(f"\n  Switch-Costs (Frame 1 nach Wechsel):\n")
    print(f"  {'Wechsel':<14} {'Frame1 ms':>10}  {'Ref ms':>10}  {'Overhead':>10}")
    sep()
    for sc in switch_costs:
        tag = "RECOMPILE" if sc["overhead_ms"] > 20 else "ok"
        print(f"  {sc['from']:>4} → {sc['to']:<4}   "
              f"{sc['switch_ms']:>10.2f}  {sc['steady_ms']:>10.2f}  "
              f"{sc['overhead_ms']:>+10.2f}  [{tag}]")

    sep("═")
    print("\n  Interpretation:")
    print("  • Overhead < 5ms  → Wechsel kostenlos, frame-by-frame schaltbar")
    print("  • Overhead 5–50ms → 1 Frame Jitter, akzeptabel")
    print("  • Overhead >50ms  → CUDA Graph Recompile, Hysterese empfohlen")
    sep("═")
    print()


if __name__ == "__main__":
    main()
