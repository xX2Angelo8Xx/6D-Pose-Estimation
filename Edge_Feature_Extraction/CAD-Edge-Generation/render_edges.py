#!/usr/bin/env python3
"""Render precomputed mesh edges for a given pose to a 2D edge mask.

Usage example:
  python render_edges.py --cache model.edges.npz --out edges.png --width 1280 --height 720 --tz 6.0

This script loads the `.npz` cache produced by `precompute_edges.py`, applies a
pose (here: translation along camera z), selects silhouette and crease edges,
projects them with a simple pinhole camera and rasterizes lines to an image.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2


def load_cache(path: Path):
    data = np.load(path)
    vertices = data['vertices']  # (N,3)
    edges = data['edges']  # (M,2)
    edges_faces = data['edges_faces']  # (M,2)
    crease = data['crease'].astype(bool)
    faces = data['faces']
    return vertices, faces, edges, edges_faces, crease


def transform_vertices(V: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    # V: (N,3), R: (3,3), t: (3,)
    return (R @ V.T).T + t


def face_normals_and_centroids(Vt: np.ndarray, faces: np.ndarray):
    v0 = Vt[faces[:, 0]]
    v1 = Vt[faces[:, 1]]
    v2 = Vt[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    # normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms
    centroids = (v0 + v1 + v2) / 3.0
    return normals, centroids


def is_front_facing(normals: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # camera at origin; vector from centroid to camera = -centroid
    view = -centroids
    dot = np.sum(normals * view, axis=1)
    return dot > 0


def project_points(Vt: np.ndarray, K: np.ndarray):
    # simple pinhole: Vt in camera coords, K 3x3
    X = Vt[:, 0]
    Y = Vt[:, 1]
    Z = Vt[:, 2]
    # compute projections but keep Z values; caller must skip Z<=0
    x = (K[0, 0] * X) / Z + K[0, 2]
    y = (K[1, 1] * Y) / Z + K[1, 2]
    return np.vstack([x, y, Z]).T


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', '-c', required=True, help='Path to .edges.npz cache')
    p.add_argument('--out', '-o', default='edges.png', help='Output image path')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fx', type=float, default=800.0)
    p.add_argument('--fy', type=float, default=800.0)
    p.add_argument('--cx', type=float, default=None)
    p.add_argument('--cy', type=float, default=None)
    p.add_argument('--tz', type=float, default=6.0, help='Camera-to-model distance along z (m)')
    p.add_argument('--tx', type=float, default=0.0)
    p.add_argument('--ty', type=float, default=0.0)
    p.add_argument('--thickness', type=int, default=1)
    p.add_argument('--include-crease', action='store_true')
    args = p.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit(f"Cache not found: {cache}")

    W = args.width
    H = args.height
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0
    K = np.array([[args.fx, 0.0, cx], [0.0, args.fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    V, F, E, Efaces, crease = load_cache(cache)

    # Auto-detect units/scale: if mesh extents are large (>10), assume units in mm and scale to meters
    extents = V.max(axis=0) - V.min(axis=0)
    max_extent = float(extents.max())
    scale = 1.0
    if max_extent > 10.0:
        # likely in millimeters -> convert to meters
        scale = 1.0 / 1000.0
        print(f"Auto-scaling vertices by {scale} (assuming mesh in mm)")
    V = V.astype(np.float32) * scale

    # Simple pose: identity rotation, translate model to z = tz (i.e. model centered at (0,0,tz))
    R = np.eye(3, dtype=np.float32)
    t = np.array([args.tx, args.ty, args.tz], dtype=np.float32)

    Vt = transform_vertices(V, R, t)

    normals, centroids = face_normals_and_centroids(Vt, F)
    front = is_front_facing(normals, centroids)

    # Determine silhouette edges
    silhouette_mask = np.zeros((H, W), dtype=np.uint8)

    projected = project_points(Vt, K)  # (N,3) x,y,Z

    def clip_and_draw(p1, p2):
        # both p's are [x,y,Z]
        if p1[2] <= 0 or p2[2] <= 0:
            return
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        # optional bounds check
        if (x1 < -50 or x1 > W + 50 or y1 < -50 or y1 > H + 50) and (
            x2 < -50 or x2 > W + 50 or y2 < -50 or y2 > H + 50
        ):
            return
        cv2.line(silhouette_mask, (x1, y1), (x2, y2), color=255, thickness=args.thickness)

    for i, (v0i, v1i) in enumerate(E):
        f1, f2 = int(Efaces[i, 0]), int(Efaces[i, 1])
        use = False
        if f1 >= 0 and f2 >= 0:
            if front[f1] != front[f2]:
                use = True
        else:
            # boundary edge is silhouette
            use = True

        if not use and args.include_crease:
            if crease[i]:
                # include crease if any adjacent face is front-facing
                cond = False
                if f1 >= 0 and front[f1]:
                    cond = True
                if f2 >= 0 and front[f2]:
                    cond = True
                use = cond

        if not use:
            continue

        p1 = projected[v0i]
        p2 = projected[v1i]
        clip_and_draw(p1, p2)

    out_path = Path(args.out)
    cv2.imwrite(str(out_path), silhouette_mask)
    print(f"Wrote edge image: {out_path}")


if __name__ == '__main__':
    main()
