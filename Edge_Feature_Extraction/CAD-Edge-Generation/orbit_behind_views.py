#!/usr/bin/env python3
"""Render silhouettes from camera positions behind the aircraft by orbiting around Z.

Camera is placed on a circle of radius `r` around the model centroid at height `cz`.
For each azimuth angle the camera looks at the model centroid and we render the outer contour.
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


def compute_camera_transform(cam_pos: np.ndarray, target: np.ndarray, up=np.array([0.0,0.0,1.0])):
    # returns rotation matrix R (3x3) such that V_cam = R @ (V_world - cam_pos)
    f = (target - cam_pos)
    f = f / (np.linalg.norm(f) + 1e-12)
    # right vector
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-12)
    # recompute up_cam to ensure orthogonality
    u = np.cross(r, f)
    u = u / (np.linalg.norm(u) + 1e-12)
    R = np.vstack([r, u, f])
    return R


def project_points(V_cam: np.ndarray, K: np.ndarray):
    X = V_cam[:,0]
    Y = V_cam[:,1]
    Z = V_cam[:,2]
    eps = 1e-8
    Zsafe = np.where(Z > eps, Z, eps)
    x = (K[0,0] * X) / Zsafe + K[0,2]
    y = (K[1,1] * Y) / Zsafe + K[1,2]
    return np.vstack([x,y,Z]).T


def render_one(V_world, F, E, Efaces, cam_pos, target, K, W, H):
    R_cam = compute_camera_transform(cam_pos, target)
    V_cam = (R_cam @ (V_world - cam_pos).T).T

    proj = project_points(V_cam, K)

    # front-facing test in camera frame: faces with normal dot view > 0
    v0 = V_cam[F[:,0]]
    v1 = V_cam[F[:,1]]
    v2 = V_cam[F[:,2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms
    centroids = (v0 + v1 + v2) / 3.0
    view = -centroids
    dot = np.sum(normals * view, axis=1)
    front = dot > 0

    mask = np.zeros((H, W), dtype=np.uint8)
    for i, (a,b) in enumerate(E):
        f1, f2 = int(Efaces[i,0]), int(Efaces[i,1])
        use = False
        if f1 >=0 and f2 >=0:
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
        x1,y1 = int(round(p1[0])), int(round(p1[1]))
        x2,y2 = int(round(p2[0])), int(round(p2[1]))
        if (x1 < -50 or x1 > W+50 or y1 < -50 or y1 > H+50) and (x2 < -50 or x2 > W+50 or y2 < -50 or y2 > H+50):
            continue
        cv2.line(mask, (x1,y1), (x2,y2), 255, 1)

    # close gaps and extract largest contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.dilate(mask, kernel, iterations=2)
    closed = cv2.erode(closed, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    outer = np.zeros_like(mask)
    cv2.drawContours(outer, [largest], -1, 255, thickness=1)
    return outer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', '-c', required=True)
    p.add_argument('--outdir', '-o', default='CAD-Edge-Generation/views/behind')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fx', type=float, default=800.0)
    p.add_argument('--fy', type=float, default=800.0)
    p.add_argument('--cx', type=float, default=None)
    p.add_argument('--cy', type=float, default=None)
    p.add_argument('--radius', type=float, default=6.0, help='orbit radius (m)')
    p.add_argument('--steps', type=int, default=12, help='number of azimuth steps')
    p.add_argument('--cz-offset', type=float, default=0.0, help='camera height offset from model centroid')
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

    centroid = V.mean(axis=0)
    target = centroid.copy()
    target[2] += 0.0

    W = args.width
    H = args.height
    cx = args.cx if args.cx is not None else W/2.0
    cy = args.cy if args.cy is not None else H/2.0
    K = np.array([[args.fx, 0.0, cx],[0.0, args.fy, cy],[0.0,0.0,1.0]], dtype=np.float32)

    # Generate cameras on circle behind the aircraft: ensure we start at +X (behind) and orbit around Z
    for i in range(args.steps):
        theta = 2.0 * math.pi * i / args.steps
        cam_x = centroid[0] + args.radius * math.cos(theta)
        cam_y = centroid[1] + args.radius * math.sin(theta)
        cam_z = centroid[2] + args.cz_offset
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        outer = render_one(V, F, E, Efaces, cam_pos, target, K, W, H)
        name = f"behind_az{int(math.degrees(theta)):+03d}.png"
        if outer is None:
            print(f"No silhouette for {name}")
            continue
        cv2.imwrite(str(outdir / name), outer)
        print(f"Wrote {outdir / name}")


if __name__ == '__main__':
    main()
