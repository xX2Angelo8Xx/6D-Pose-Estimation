#!/usr/bin/env python3
"""Precompute mesh edge adjacency and crease flags and save to NPZ.

Usage:
    python precompute_edges.py --mesh path/to/model.stl --out path/to/cache.npz

This script loads a triangulated mesh (STL/OBJ/PLY), computes the unique edges,
their adjacent faces and a crease flag (dihedral angle > threshold). The output
is a compact `.npz` file with arrays needed for fast runtime silhouette extraction.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh


def compute_crease_flags(mesh: trimesh.Trimesh, angle_deg: float = 30.0):
    # Build unique edges and adjacent faces robustly (works across trimesh versions)
    faces = mesh.faces
    edge_map = {}  # (v_min, v_max) -> list of face indices
    for fi, face in enumerate(faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_map.setdefault(key, []).append(fi)

    edges = np.array(list(edge_map.keys()), dtype=np.int32)
    edges_faces = -np.ones((len(edges), 2), dtype=np.int32)
    for i, key in enumerate(list(edge_map.keys())):
        adj = edge_map[key]
        if len(adj) >= 1:
            edges_faces[i, 0] = adj[0]
        if len(adj) >= 2:
            edges_faces[i, 1] = adj[1]

    face_normals = mesh.face_normals  # (n_faces,3)

    crease = np.zeros(len(edges), dtype=np.bool_)

    angle_rad_thresh = math.radians(angle_deg)

    for i in range(len(edges)):
        f1, f2 = int(edges_faces[i, 0]), int(edges_faces[i, 1])
        if f1 < 0 or f2 < 0:
            # boundary edge -> treat as crease
            crease[i] = True
            continue
        n1 = face_normals[f1]
        n2 = face_normals[f2]
        # clamp dot
        dot = float(np.dot(n1, n2))
        dot = max(-1.0, min(1.0, dot))
        ang = math.acos(dot)
        if ang >= angle_rad_thresh:
            crease[i] = True

    return edges, edges_faces, crease


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', '-m', required=True, help='Path to mesh (stl/obj/ply)')
    p.add_argument('--out', '-o', default=None, help='Output .npz cache path')
    p.add_argument('--crease-deg', type=float, default=30.0, help='Dihedral angle threshold')
    args = p.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        raise SystemExit(f"Mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        # some formats produce Scene; try to merge
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.dump()))

    # Ensure triangular faces (most STLs are already triangles)
    if mesh.faces.shape[1] != 3:
        # Attempt to triangulate / fallback
        mesh = mesh.convex_hull.triangulate()

    edges, edges_faces, crease = compute_crease_flags(mesh, args.crease_deg)

    out_path = Path(args.out) if args.out else mesh_path.with_suffix('.edges.npz')

    np.savez_compressed(
        out_path,
        vertices=mesh.vertices.astype(np.float32),
        faces=mesh.faces.astype(np.int32),
        edges=edges.astype(np.int32),
        edges_faces=edges_faces.astype(np.int32),
        crease=crease.astype(np.uint8),
    )

    print(f"Wrote edge cache: {out_path} (edges={len(edges)})")


if __name__ == '__main__':
    main()
