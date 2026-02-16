#!/usr/bin/env python3
"""Diagnose projected edges: print stats and create debug overlay.

Usage:
  python diagnose_edges.py --cache CAD_Ranger.edges.npz --out edges_debug.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2


def load_cache(path: Path):
    data = np.load(path)
    return data


def project_points(Vt: np.ndarray, K: np.ndarray):
    X = Vt[:, 0]
    Y = Vt[:, 1]
    Z = Vt[:, 2]
    eps = 1e-8
    Zsafe = np.where(Z > eps, Z, eps)
    x = (K[0, 0] * X) / Zsafe + K[0, 2]
    y = (K[1, 1] * Y) / Zsafe + K[1, 2]
    return np.vstack([x, y, Z]).T


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', '-c', required=True)
    p.add_argument('--out', '-o', default='edges_debug.png')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fx', type=float, default=800.0)
    p.add_argument('--fy', type=float, default=800.0)
    p.add_argument('--cx', type=float, default=None)
    p.add_argument('--cy', type=float, default=None)
    p.add_argument('--tz', type=float, default=6.0)
    args = p.parse_args()

    cache = Path(args.cache)
    data = load_cache(cache)
    V = data['vertices']
    F = data['faces']
    E = data['edges']
    Efaces = data['edges_faces']
    crease = data['crease'].astype(bool)

    W = args.width
    H = args.height
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0
    K = np.array([[args.fx, 0.0, cx], [0.0, args.fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    print(f"Vertices: {V.shape}, Faces: {F.shape}, Edges: {E.shape}")
    print(f"Vertex bbox min/max:\n  min {V.min(axis=0)}\n  max {V.max(axis=0)}")
    print(f"Vertex centroid: {V.mean(axis=0)}")

    # Auto-scale like render_edges: if extents > 10 assume mm -> convert to meters
    extents = V.max(axis=0) - V.min(axis=0)
    max_extent = float(extents.max())
    scale = 1.0
    if max_extent > 10.0:
        scale = 1.0 / 1000.0
        print(f"Auto-scaling vertices by {scale} (assuming mesh in mm)")
    V = V.astype(np.float32) * scale

    R = np.eye(3, dtype=np.float32)
    t = np.array([0.0, 0.0, args.tz], dtype=np.float32)
    Vt = (R @ V.T).T + t

    print(f"After transform centroid: {Vt.mean(axis=0)}")
    print(f"Z range after transform: min={Vt[:,2].min():.6f}, max={Vt[:,2].max():.6f}")

    proj = project_points(Vt, K)
    xs = proj[:,0]
    ys = proj[:,1]
    zs = proj[:,2]
    valid = zs > 1e-6
    if valid.any():
        print(f"Projected x range (valid): min={xs[valid].min():.1f}, max={xs[valid].max():.1f}")
        print(f"Projected y range (valid): min={ys[valid].min():.1f}, max={ys[valid].max():.1f}")
    else:
        print("No valid projected points (all Z<=0)")

    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H) & (zs > 0)
    print(f"Projected points inside image: {inside.sum()} / {len(xs)} ({inside.sum()/len(xs):.2%})")

    # Build overlay
    vis = np.zeros((H, W, 3), dtype=np.uint8)

    # draw projected vertices as colored by depth (closer=red)
    zmin, zmax = zs.min(), zs.max()
    zr = zmax - zmin if zmax > zmin else 1.0
    for xi, yi, zi in proj:
        if zi <= 0:
            continue
        x = int(round(xi))
        y = int(round(yi))
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        norm = (zi - zmin) / zr
        # color map: near red, far blue
        col = (int(255 * (1 - norm)), 0, int(255 * norm))
        cv2.circle(vis, (x, y), 1, col, -1)

    # draw edges: green used, red skipped
    skipped = 0
    drawn = 0
    for i, (v0, v1) in enumerate(E):
        p1 = proj[int(v0)]
        p2 = proj[int(v1)]
        if p1[2] <= 0 or p2[2] <= 0:
            skipped += 1
            continue
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        if (x1 < -50 or x1 > W + 50 or y1 < -50 or y1 > H + 50) and (
            x2 < -50 or x2 > W + 50 or y2 < -50 or y2 > H + 50
        ):
            skipped += 1
            continue
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        drawn += 1

    print(f"Edges drawn: {drawn}, skipped: {skipped}")

    outp = Path(args.out)
    cv2.imwrite(str(outp), vis)
    print(f"Wrote debug overlay: {outp}")


if __name__ == '__main__':
    main()
