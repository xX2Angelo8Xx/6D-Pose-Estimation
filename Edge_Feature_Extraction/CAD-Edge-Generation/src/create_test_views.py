#!/usr/bin/env python3
"""
Create test views for orientation verification.
Generates 10 diverse camera poses to validate coordinate system and
silhouette pipeline, then saves PNG renders via visualize_debug.

Usage:
    python src/create_test_views.py \
        --cache  data/CAD_Ranger_simplified.edges.npz \
        --outdir results/smoke_test \
        [--supersample 4] [--distance 5.0]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from extract_silhouette import (
    load_cache,
    extract_silhouette_contour,
    spherical_to_cartesian_aircraft,
)
from visualize_debug import visualize_npz

# ── Test-pose catalogue ────────────────────────────────────────────────────
# Azimuth convention:  0=rear(-X), 90=right(+Y), 180=front(+X), 270=left(-Y)
TEST_VIEWS = [
    # name                   az   el   description
    ("rear_level",            0,   0,  "Rear  — level         (primary tracking)"),
    ("rear_above_30",         0,  30,  "Rear  — 30° above"),
    ("rear_above_60",         0,  60,  "Rear  — 60° above"),
    ("rear_right_diagonal",  45,  15,  "Rear-right diagonal   — 15° above"),
    ("rear_left_diagonal",  315,  15,  "Rear-left diagonal    — 15° above"),
    ("front_level",         180,   0,  "Front — level         (nose toward cam)"),
    ("front_above_45",      180,  45,  "Front — 45° above"),
    ("right_side",           90,   0,  "Right side (starboard) — level"),
    ("left_side",           270,   0,  "Left side (port)       — level"),
    ("top_view",              0,  80,  "Top view              — 80° above"),
]


def main():
    ap = argparse.ArgumentParser(description="Render smoke-test views from the silhouette pipeline")
    ap.add_argument('--cache',       required=True,  help='edges.npz mesh cache')
    ap.add_argument('--outdir',      default='results/smoke_test')
    ap.add_argument('--supersample', type=int,   default=4)
    ap.add_argument('--distance',    type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    npz_dir = out_dir / 'npz'
    png_dir = out_dir / 'renders'
    npz_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    # ── load mesh ────────────────────────────────────────────────────────
    print("Loading mesh …")
    V, F, _, _ = load_cache(Path(args.cache))
    if np.max(V.max(0) - V.min(0)) > 100:
        V = V * 0.001
    target = V.mean(axis=0)
    print(f"  {len(V):,} verts  {len(F):,} faces")

    # ── JIT warmup ───────────────────────────────────────────────────────
    print("Warming up JIT …")
    _cam = target + spherical_to_cartesian_aircraft(0, 0, args.distance)
    extract_silhouette_contour(V, F, _cam, target, supersample=1)
    if args.supersample > 1:
        extract_silhouette_contour(V, F, _cam, target, supersample=args.supersample)
    print("  ✓ done\n")

    # ── extract & save ───────────────────────────────────────────────────
    sep = "=" * 70
    print(sep)
    print(f"  SMOKE TEST — {len(TEST_VIEWS)} poses  "
          f"(supersample={args.supersample}×  dist={args.distance}m)")
    print(sep)
    print(f"{'View':<28} {'az':>4} {'el':>4}  {'pts':>6}  {'ms':>7}  description")
    print("─" * 70)

    successes = 0
    npz_paths = []

    for name, az, el, desc in TEST_VIEWS:
        cam_pos = target + spherical_to_cartesian_aircraft(az, el, args.distance)
        t0 = time.perf_counter()
        pts = extract_silhouette_contour(
            V, F, cam_pos, target,
            supersample=args.supersample,
        )
        ms = (time.perf_counter() - t0) * 1000.0

        ok = len(pts) > 0
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name:<26} {az:>4} {el:>4}  {len(pts):>6,}  {ms:>7.1f}  {desc}")

        if ok:
            npz_path = npz_dir / f"{name}.npz"
            np.savez_compressed(npz_path,
                                points_3d=pts,
                                cam_pos=cam_pos,
                                target=target,
                                azimuth=np.float32(az),
                                elevation=np.float32(el))
            npz_paths.append(npz_path)
            successes += 1

    print("─" * 70)
    print(f"\n  {successes}/{len(TEST_VIEWS)} views extracted successfully")

    # ── visualize ────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Rendering PNGs …")
    rendered = 0
    for npz_path in npz_paths:
        try:
            visualize_npz(npz_path, png_dir)
            rendered += 1
        except Exception as e:
            print(f"    ⚠  {npz_path.stem}: {e}")

    print(f"  {rendered}/{len(npz_paths)} renders saved → {png_dir}/")
    print(f"\n{sep}")
    if successes == len(TEST_VIEWS):
        print("  ✅  All views OK — pipeline is healthy")
    else:
        print(f"  ⚠   {len(TEST_VIEWS) - successes} views failed — check output above")
    print(sep)


if __name__ == '__main__':
    main()
