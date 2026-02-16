#!/usr/bin/env python3
"""Compute silhouette (outer contour) and estimate model orientation from cache.

Outputs:
 - silhouette_only.png : rasterized outer contour (thin lines)
 - silhouette_filled.png : filled silhouette area
 - silhouette_annotated.png : annotated with principal axis and nose arrow

Prints estimated yaw angle (degrees) in image coordinates (0 = right, 90 = down).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2
from sklearn.decomposition import PCA


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
    p.add_argument('--outdir', '-o', default='CAD-Edge-Generation')
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
    crease = data['crease'].astype(bool)

    W = args.width
    H = args.height
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0
    K = np.array([[args.fx, 0.0, cx], [0.0, args.fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    # auto-scale mm->m
    extents = V.max(axis=0) - V.min(axis=0)
    if float(extents.max()) > 10.0:
        V = V * (1.0 / 1000.0)

    # pose: identity rotation and translation tz
    R = np.eye(3, dtype=np.float32)
    t = np.array([0.0, 0.0, args.tz], dtype=np.float32)
    Vt = (R @ V.T).T + t

    proj = project_points(Vt, K)

    # Determine silhouette edges using face front/back test
    # front-facing per face
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
        cv2.line(mask, (x1, y1), (x2, y2), 255, 1)

    # Fill small gaps then find largest contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.dilate(mask, kernel, iterations=2)
    closed = cv2.erode(closed, kernel, iterations=2)

    # Fill interior
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('No contours found')
        return
    largest = max(contours, key=cv2.contourArea)

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)

    # Extract outer contour points
    outer_pts = largest.reshape(-1, 2).astype(np.float32)

    # PCA to find major axis
    pca = PCA(n_components=2)
    pca.fit(outer_pts)
    center = outer_pts.mean(axis=0)
    major_vec = pca.components_[0]

    # Project contour points onto major axis to find extremes
    proj_vals = (outer_pts - center) @ major_vec
    min_idx = np.argmin(proj_vals)
    max_idx = np.argmax(proj_vals)
    end1 = outer_pts[min_idx]
    end2 = outer_pts[max_idx]

    # Determine nose end by local width: sample points near each end and compute spread orthogonal
    perp = np.array([-major_vec[1], major_vec[0]])
    def width_at(pt):
        # consider points within 20% of range around projected value
        val = (pt - center) @ major_vec
        rng = proj_vals.max() - proj_vals.min()
        mask_local = np.abs(proj_vals - val) < 0.2 * rng
        pts_local = outer_pts[mask_local]
        if len(pts_local) < 5:
            return np.inf
        dists = np.abs((pts_local - pt) @ perp)
        return dists.mean()

    w1 = width_at(end1)
    w2 = width_at(end2)

    # nose is the narrower end
    if w1 < w2:
        nose = end1
        nose_side = 'min'
        nose_vec = end1 - center
    else:
        nose = end2
        nose_side = 'max'
        nose_vec = end2 - center

    # angle of major_vec in image coords: 0 = right, positive clockwise (y down)
    angle_rad = np.arctan2(major_vec[1], major_vec[0])
    angle_deg = np.degrees(angle_rad)

    # If nose is at negative proj (min side), direction is -major_vec
    if nose_side == 'min':
        angle_deg = (angle_deg + 180.0) % 360.0

    print(f"Estimated orientation (image yaw) = {angle_deg:.2f} deg (0=right, 90=down)")

    # save outputs
    cv2.imwrite(str(outdir / 'silhouette_only.png'), mask)
    cv2.imwrite(str(outdir / 'silhouette_filled.png'), filled)

    # Create image with only the outer contour (no internal edges)
    outer_only = np.zeros_like(mask)
    cv2.drawContours(outer_only, [largest], -1, 255, thickness=1)
    cv2.imwrite(str(outdir / 'silhouette_contour_only.png'), outer_only)

    # annotated
    vis = cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)
    cxy = tuple(center.astype(int))
    cv2.circle(vis, cxy, 3, (0, 0, 255), -1)
    e1 = tuple(end1.astype(int))
    e2 = tuple(end2.astype(int))
    cv2.line(vis, e1, e2, (255, 0, 0), 2)
    nose_pt = tuple(nose.astype(int))
    cv2.arrowedLine(vis, (int(center[0]), int(center[1])), (int(nose[0]), int(nose[1])), (0,255,0), 2, tipLength=0.1)
    cv2.putText(vis, f"yaw={angle_deg:.1f}deg", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
    cv2.imwrite(str(outdir / 'silhouette_annotated.png'), vis)


if __name__ == '__main__':
    main()
