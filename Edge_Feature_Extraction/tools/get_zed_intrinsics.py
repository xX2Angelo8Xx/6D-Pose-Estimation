#!/usr/bin/env python3
"""
Liest die kalibrierten Kamera-Intrinsics der ZED 2i aus und gibt sie für
den Pose Optimizer aus (fx, fy, cx, cy).

Aufruf:
    python3 tools/get_zed_intrinsics.py
    python3 tools/get_zed_intrinsics.py --resolution HD720
    python3 tools/get_zed_intrinsics.py --save-json intrinsics.json

Ohne ZED-Kamera (kein SDK vorhanden): gibt Schätzwerte basierend auf
Stereolabs-Spezifikation aus.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path


def estimate_from_spec(resolution: str) -> dict:
    """
    Schätzt Kamera-Intrinsics aus Stereolabs-Spezifikation.
    ZED 2i: HFOV ≈ 110° (diagonal), VFOV ≈ 69°.
    Bei HD720 (1280×720) entspricht das fx=fy≈521 px.
    (Übereinstimmung: VFOV = 2*arctan(360/521) = 69.2°  ✓)
    """
    specs = {
        #  Resolution    W      H     fx      fy    cx    cy   HFOV   VFOV
        # ZED 2i 4mm, S/N 34754237 — SDK-kalibrierte Werte
        "HD2K":        (2208, 1242, 1475,  1475, 1104,  621,  73.2, 45.0),
        "HD1080":      (1920, 1080, 1283,  1283,  960,  540,  73.2, 45.0),
        "HD720":       (1280,  720,  951,   951,  639,  348,  67.8, 40.2),  # measured S/N 34754237
        "VGA":         ( 672,  376,  498,   498,  336,  188,  67.0, 40.0),
    }
    if resolution not in specs:
        raise ValueError(f"Unbekannte Auflösung: {resolution}. Wähle aus {list(specs.keys())}")
    w, h, fx, fy, cx, cy, hfov, vfov = specs[resolution]
    return dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy,
                hfov_deg=hfov, vfov_deg=vfov, source="spec_estimate")


def read_from_sdk(resolution: str) -> dict:
    """Liest kalibrierte Intrinsics direkt vom ZED SDK."""
    try:
        import pyzed.sl as sl
    except ImportError:
        raise RuntimeError("pyzed (ZED SDK Python) nicht installiert.")

    res_map = {
        "HD2K":   sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720":  sl.RESOLUTION.HD720,
        "VGA":    sl.RESOLUTION.VGA,
    }
    if resolution not in res_map:
        raise ValueError(f"Unbekannte Auflösung: {resolution}")

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = res_map[resolution]
    init.depth_mode = sl.DEPTH_MODE.NONE

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED konnte nicht geöffnet werden: {err}")

    info = zed.get_camera_information()
    cal  = info.camera_configuration.calibration_parameters.left_cam

    result = dict(
        width  = info.camera_configuration.resolution.width,
        height = info.camera_configuration.resolution.height,
        fx     = cal.fx,
        fy     = cal.fy,
        cx     = cal.cx,
        cy     = cal.cy,
        hfov_deg = 2 * math.degrees(math.atan(cal.cx / cal.fx)),
        vfov_deg = 2 * math.degrees(math.atan(cal.cy / cal.fy)),
        serial_number = info.serial_number,
        firmware_version = info.camera_configuration.firmware_version,
        source = "zed_sdk_calibration",
    )
    zed.close()
    return result


def main():
    ap = argparse.ArgumentParser(description="ZED 2i Kamera-Intrinsics lesen")
    ap.add_argument("--resolution", default="HD720",
                    choices=["HD2K", "HD1080", "HD720", "VGA"],
                    help="Auflösung / Sensor-Modus (default: HD720)")
    ap.add_argument("--save-json", default=None, metavar="PATH",
                    help="Speichert Ergebnis als JSON")
    ap.add_argument("--estimate-only", action="store_true",
                    help="Keine Kamera nötig — benutzt Spezwerte")
    args = ap.parse_args()

    print(f"\n{'='*55}")
    print(f"ZED 2i Kamera-Intrinsics  [{args.resolution}]")
    print(f"{'='*55}")

    if args.estimate_only:
        print("Quelle: Stereolabs Spezifikation (Schätzung)\n")
        result = estimate_from_spec(args.resolution)
    else:
        try:
            print("Versuche ZED SDK ...\n")
            result = read_from_sdk(args.resolution)
        except RuntimeError as e:
            print(f"ZED SDK nicht verfügbar: {e}")
            print("Fallback auf Spezifikations-Schätzwerte.\n")
            result = estimate_from_spec(args.resolution)

    print(f"  Auflösung : {result['width']} × {result['height']} px")
    print(f"  fx        : {result['fx']:.2f} px")
    print(f"  fy        : {result['fy']:.2f} px")
    print(f"  cx        : {result['cx']:.2f} px")
    print(f"  cy        : {result['cy']:.2f} px")
    print(f"  HFOV      : {result['hfov_deg']:.1f}°")
    print(f"  VFOV      : {result['vfov_deg']:.1f}°")
    print(f"  Quelle    : {result['source']}")

    print(f"\nFür Pose Optimizer GUI / run_on_real_silhouette.py:")
    print(f"  --fx {result['fx']:.1f} --fy {result['fy']:.1f} --cx {result['cx']:.1f} --cy {result['cy']:.1f}")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nGespeichert: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
