#!/usr/bin/env python3
"""
LUT Quality Deep Inspect
========================
Renders selected azimuths across all elevations to create a full az-sweep
diagnostic panel. Highlights the two identified failure modes:
  A) az≈88-94° / 268-274°  — large angular gap in side-view silhouette
  B) az≈110-116° / 244-250° — sparse rear-diagonal with low point count

Output:
  lut_quality_sweep_<label>.png   — full el-sweep for each suspicious az
  lut_quality_failuremodes.png    — side-by-side comparison panel
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE    = Path(__file__).resolve().parent.parent
LUT_NPZ  = BASE / "CAD-Edge-Generation/data/silhouette_lut_roll.npz"
LUT_JSON = BASE / "CAD-Edge-Generation/data/silhouette_lut_roll.lookup.json"
OUT_DIR  = Path(__file__).resolve().parent / "lut_quality_output"

LUT_FX = LUT_FY = 800.0
LUT_W, LUT_H = 1280, 720
LUT_CX, LUT_CY = 640.0, 360.0

THUMB_W, THUMB_H = 213, 120   # 16:9


def project(pts_cam: np.ndarray) -> np.ndarray:
    valid = pts_cam[:, 2] > 0.001
    p = pts_cam[valid]
    u = LUT_FX * p[:, 0] / p[:, 2] + LUT_CX
    v = LUT_FY * p[:, 1] / p[:, 2] + LUT_CY
    return np.column_stack([u, v])


def render(pts_cam: np.ndarray, w=LUT_W, h=LUT_H, thickness=3) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    if len(pts_cam) == 0:
        return img
    px = project(pts_cam)
    xi = np.clip(np.round(px[:, 0]).astype(int), 0, w-1)
    yi = np.clip(np.round(px[:, 1]).astype(int), 0, h-1)
    img[yi, xi] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    return cv2.dilate(img, k)


def gap_fraction(pts_cam: np.ndarray) -> float:
    if len(pts_cam) < 4:
        return 1.0
    px = project(pts_cam)
    cx = px[:, 0].mean(); cy = px[:, 1].mean()
    ang = np.sort(np.arctan2(px[:, 1]-cy, px[:, 0]-cx))
    d   = np.diff(ang); wrap = 2*math.pi + ang[0] - ang[-1]
    return float(np.append(d, wrap).max()) / (2*math.pi)


def az_sweep_panel(lut, poses: dict, az_target: float,
                   el_range=(-20, 90, 2), roll=0.0) -> np.ndarray:
    """Render one row of thumbnails at fixed az, sweep elevation."""
    els = [e for e in np.arange(el_range[0], el_range[1]+1, el_range[2])]
    thumbs = []
    for el in els:
        key = f"az{az_target:.2f}_el{el:.2f}_roll{roll:.2f}"
        if key not in poses:
            thumb = np.zeros((THUMB_H, THUMB_W, 3), np.uint8)
            cv2.putText(thumb, f"N/A el={el:.0f}", (4, THUMB_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)
        else:
            if key in lut.files:
                pts = lut[key].copy()
                img = render(pts, THUMB_W, THUMB_H, thickness=2)
                gf  = gap_fraction(pts)
                n   = poses[key]['n_points']
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # Red border for large gap, orange for low n_points
                if gf > 0.15:
                    cv2.rectangle(rgb, (0,0), (THUMB_W-1,THUMB_H-1), (0,0,255), 3)
                elif n < 2000:
                    cv2.rectangle(rgb, (0,0), (THUMB_W-1,THUMB_H-1), (0,165,255), 3)
                cv2.putText(rgb, f"el={el:.0f} n={n} gap={gf:.2f}",
                            (3, THUMB_H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (240,240,0), 1, cv2.LINE_AA)
                thumb = rgb
            else:
                thumb = np.full((THUMB_H, THUMB_W, 3), 40, np.uint8)
                cv2.putText(thumb, f"el={el:.0f}", (4, THUMB_H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)
        thumbs.append(thumb)

    # Add label column
    label = np.zeros((THUMB_H, 60, 3), np.uint8)
    cv2.putText(label, f"az={az_target:.0f}", (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
    return np.hstack([label] + thumbs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading JSON …")
    with open(LUT_JSON) as f:
        idx = json.load(f)
    poses = idx['poses']

    print("Loading NPZ (mmap) …")
    lut = np.load(LUT_NPZ, mmap_mode='r')
    available = set(lut.files)
    print(f"  {len(available)} arrays in NPZ")

    # ── Select azimuths to sweep ─────────────────────────────────────────────
    # Cluster A: pure side-view failure  (large gap)
    cluster_a = [86, 88, 90, 92, 94,   266, 268, 270, 272, 274]
    # Cluster B: rear-diagonal failure   (low n_points)
    cluster_b = [108, 110, 112, 114, 116,  242, 244, 246, 248, 250]
    # Reference (healthy)
    ref_azs   = [0, 45, 135, 180, 315]

    all_azs = cluster_a + cluster_b + ref_azs
    el_range = (-10, 20, 2)   # focus on problematic low-elevation band

    rows = []
    total = len(all_azs)
    for i, az in enumerate(all_azs):
        print(f"  Rendering az={az}°  ({i+1}/{total}) …")
        row = az_sweep_panel(lut, poses, float(az), el_range=el_range)
        rows.append(row)

    # Divider rows between clusters
    divider = np.full((8, rows[0].shape[1], 3), 60, np.uint8)

    panel_a = np.vstack(rows[:10])
    panel_b = np.vstack(rows[10:20])
    panel_r = np.vstack(rows[20:])

    def add_header(img, text):
        hdr = np.full((28, img.shape[1], 3), 20, np.uint8)
        cv2.putText(hdr, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (100, 220, 100), 1, cv2.LINE_AA)
        return np.vstack([hdr, img])

    panel_a = add_header(panel_a, "CLUSTER A — Pure side views (az≈88-94° / 268-274°)  [large angular gap in silhouette]")
    panel_b = add_header(panel_b, "CLUSTER B — Rear-diagonal views (az≈110-116° / 242-250°)  [low n_points / tail holes]")
    panel_r = add_header(panel_r, "REFERENCE — Healthy views (az=0° / 45° / 135° / 180° / 315°)")

    full = np.vstack([panel_a, divider, panel_b, divider, panel_r])
    out_path = OUT_DIR / "lut_quality_sweep_detail.png"
    cv2.imwrite(str(out_path), full)
    print(f"\n✓ Detail sweep saved → {out_path}")

    # ── Statistics summary ───────────────────────────────────────────────────
    print("\nCluster statistics (roll=0, el -10…+20°):")
    print(f"  {'az':>6}  {'n_min':>6}  {'n_mean':>7}  {'gap_max':>8}  {'flag'}")
    for az in cluster_a + cluster_b + [0, 180]:
        az_keys = [k for k,v in poses.items()
                   if abs(v['azimuth']-az) < 0.5 and v['roll']==0.0
                   and -10 <= v['elevation'] <= 20]
        if not az_keys:
            continue
        ns    = [poses[k]['n_points'] for k in az_keys]
        gaps  = []
        for k in az_keys:
            if k in available:
                pts = lut[k].copy()
                gaps.append(gap_fraction(pts))
        cluster = "A" if az in cluster_a else ("B" if az in cluster_b else "REF")
        gap_max = max(gaps) if gaps else float('nan')
        print(f"  {az:6.0f}°  {min(ns):6d}  {np.mean(ns):7.1f}  {gap_max:8.3f}  {cluster}")

    print(f"\n✓ Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
