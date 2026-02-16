#!/usr/bin/env python3
"""Generate silhouette contour images for multiple viewpoints.

Saves outer-contour-only PNGs into `views/` with filenames containing yaw/pitch/roll.

Usage:
  python generate_views.py --cache CAD_Ranger.edges.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2
import math


def load_cache(path: Path):
    data = np.load(path)
    return data


def euler_to_R(yaw: float, pitch: float, roll: float) -> np.ndarray:
    # yaw around Z, pitch around Y, roll around X (degrees)
    y = math.radians(yaw)
    p = math.radians(pitch)
    r = math.radians(roll)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]], dtype=np.float32)
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]], dtype=np.float32)
    # apply R = Rz * Ry * Rx (intrinsic rotations)
    return Rz @ Ry @ Rx


def project(Vt: np.ndarray, K: np.ndarray):
    X = Vt[:, 0]
    Y = Vt[:, 1]
    Z = Vt[:, 2]
    eps = 1e-8
    Zsafe = np.where(Z > eps, Z, eps)
    x = (K[0, 0] * X) / Zsafe + K[0, 2]
    y = (K[1, 1] * Y) / Zsafe + K[1, 2]
    return np.vstack([x, y, Z]).T


def render_silhouette(V, F, E, Efaces, yaw, pitch, roll, tx, ty, tz, K, W, H):
    R = euler_to_R(yaw, pitch, roll)
    t = np.array([tx, ty, tz], dtype=np.float32)
    Vt = (R @ V.T).T + t

    proj = project(Vt, K)

    # compute front-facing faces
    v0 = Vt[F[:, 0]]
    v1 = Vt[F[:, 1]]
    v2 = Vt[F[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms
    centroids = (v0 + v1 + v2) / 3.0
    view = -centroids
    dot = np.sum(normals * view, axis=1)
    front = dot > 0

    mask = np.zeros((H, W), dtype=np.uint8)
    for i, (a, b) in enumerate(E):
        f1, f2 = int(Efaces[i, 0]), int(Efaces[i, 1])
        use = False
        if f1 >= 0 and f2 >= 0:
            if front[f1] != front[f2]:
                use = True
        else:
            use = True
        if not use:
            continue
        p1 = proj[int(a)]
        p2 = proj[int(b)]
        if p1[2] <= 0 or p2[2] <= 0:
            continue
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        # simple bounds check
        if (x1 < -50 or x1 > W + 50 or y1 < -50 or y1 > H + 50) and (x2 < -50 or x2 > W + 50 or y2 < -50 or y2 > H + 50):
            continue
        cv2.line(mask, (x1, y1), (x2, y2), 255, 1)

    # close small gaps and fill
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.dilate(mask, kernel, iterations=2)
    closed = cv2.erode(closed, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    outer_only = np.zeros_like(mask)
    cv2.drawContours(outer_only, [largest], -1, 255, thickness=1)
    return outer_only, filled


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', '-c', required=True)
    p.add_argument('--outdir', '-o', default='CAD-Edge-Generation/views')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fx', type=float, default=800.0)
    p.add_argument('--fy', type=float, default=800.0)
    p.add_argument('--cx', type=float, default=None)
    p.add_argument('--cy', type=float, default=None)
    p.add_argument('--tz', type=float, default=6.0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_cache(Path(args.cache))
    V = data['vertices'].astype(np.float32)
    F = data['faces']
    E = data['edges']
    Efaces = data['edges_faces']

    # auto-scale mm->m
    extents = V.max(axis=0) - V.min(axis=0)
    if float(extents.max()) > 10.0:
        V = V * (1.0 / 1000.0)

    W = args.width
    H = args.height
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0
    K = np.array([[args.fx, 0.0, cx], [0.0, args.fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    # define candidate views (yaw, pitch, roll)
    yaws = [-90, -45, 0, 45, 90]
    pitches = [0, -10, 10]
    rolls = [0]

    for yaw in yaws:
        for pitch in pitches:
            for roll in rolls:
                outer, filled = render_silhouette(V, F, E, Efaces, yaw, pitch, roll, 0.0, 0.0, args.tz, K, W, H)
                name = f"silhouette_y{yaw:+03d}_p{pitch:+03d}_r{roll:+03d}.png"
                if outer is None:
                    print(f"No silhouette for {name}")
                    continue
                cv2.imwrite(str(outdir / name), outer)
                print(f"Wrote {outdir / name}")


if __name__ == '__main__':
    main()
