#!/usr/bin/env python3
"""Measure rendering time for a single silhouette pose using the precomputed cache.

Outputs timings for: cache load, render loop only, and total (load+render+write).
"""
from __future__ import annotations

import time
import argparse
from pathlib import Path
import numpy as np
import cv2
import math


def project_points(Vt: np.ndarray, K: np.ndarray):
    X = Vt[:, 0]
    Y = Vt[:, 1]
    Z = Vt[:, 2]
    eps = 1e-8
    Zsafe = np.where(Z > eps, Z, eps)
    x = (K[0, 0] * X) / Zsafe + K[0, 2]
    y = (K[1, 1] * Y) / Zsafe + K[1, 2]
    return np.vstack([x, y, Z]).T


def compute_camera_transform(cam_pos: np.ndarray, target: np.ndarray, up=np.array([0.0,0.0,1.0])):
    f = (target - cam_pos)
    f = f / (np.linalg.norm(f) + 1e-12)
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(r, f)
    u = u / (np.linalg.norm(u) + 1e-12)
    R = np.vstack([r, u, f])
    return R


def render_mask(V_world, F, E, Efaces, cam_pos, target, K, W, H):
    R_cam = compute_camera_transform(cam_pos, target)
    V_cam = (R_cam @ (V_world - cam_pos).T).T
    proj = project_points(V_cam, K)

    # front-facing
    v0 = V_cam[F[:, 0]]
    v1 = V_cam[F[:, 1]]
    v2 = V_cam[F[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms
    centroids = (v0 + v1 + v2) / 3.0
    view = -centroids
    dot = np.sum(normals * view, axis=1)
    front = dot > 0

    mask = np.zeros((H, W), dtype=np.uint8)
    # render loop
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
        if (x1 < -50 or x1 > W + 50 or y1 < -50 or y1 > H + 50) and (x2 < -50 or x2 > W + 50 or y2 < -50 or y2 > H + 50):
            continue
        cv2.line(mask, (x1, y1), (x2, y2), 255, 1)

    # closing/fill (postproc)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.dilate(mask, kernel, iterations=2)
    closed = cv2.erode(closed, kernel, iterations=2)
    return closed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', '-c', required=True)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fx', type=float, default=800.0)
    p.add_argument('--fy', type=float, default=800.0)
    p.add_argument('--tx', type=float, default=0.0)
    p.add_argument('--ty', type=float, default=0.0)
    p.add_argument('--tz', type=float, default=6.0)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    start_total = time.perf_counter()
    t0 = time.perf_counter()
    data = np.load(args.cache)
    load_time = time.perf_counter() - t0

    V = data['vertices'].astype(np.float32)
    F = data['faces']
    E = data['edges']
    Efaces = data['edges_faces']

    # autoscale
    extents = V.max(axis=0) - V.min(axis=0)
    if float(extents.max()) > 10.0:
        V = V * (1.0 / 1000.0)

    centroid = V.mean(axis=0)
    cam_pos = np.array([centroid[0] + args.tx, centroid[1] + args.ty, centroid[2] + args.tz], dtype=np.float32)
    target = centroid.copy()

    W = args.width
    H = args.height
    K = np.array([[args.fx, 0.0, W / 2.0], [0.0, args.fy, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    # measure render loop only
    t1 = time.perf_counter()
    mask = render_mask(V, F, E, Efaces, cam_pos, target, K, W, H)
    render_time = time.perf_counter() - t1

    # write image if requested and measure I/O
    io_time = 0.0
    if args.out:
        t2 = time.perf_counter()
        cv2.imwrite(args.out, mask)
        io_time = time.perf_counter() - t2

    total_time = time.perf_counter() - start_total

    print(f"Load time: {load_time*1000:.2f} ms")
    print(f"Render loop time: {render_time*1000:.2f} ms")
    print(f"I/O time (write): {io_time*1000:.2f} ms")
    print(f"Total time (load+render+io): {total_time*1000:.2f} ms")


if __name__ == '__main__':
    main()
