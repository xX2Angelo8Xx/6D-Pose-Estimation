#!/usr/bin/env python3
"""
Build a ROLL-EXTENDED lookup table for silhouette-based 6-D pose estimation.

This script loads an existing azimuth/elevation LUT and generates 5 roll slices
(-20°, -10°, 0°, +10°, +20°) for each (az, el) grid point. The output is a new
NPZ file with keys of the form "pose_{idx:05d}_roll_{roll_deg:+03d}".

Roll is applied by rotating the mesh around its +X (fuselage) axis before
projection. This captures bank/roll asymmetry for most viewing angles.

Output structure:
  • <output>.npz          — all contours: (N,3) float32 per key
  • <output>.lookup.json  — extended index with roll_slices list
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from extract_silhouette import (
    load_cache,
    extract_silhouette_contour,
    spherical_to_cartesian_aircraft,
    compute_camera_transform,
)


# Roll slices to precompute (degrees)
ROLL_SLICES = [-20.0, -10.0, 0.0, +10.0, +20.0]


def apply_roll_to_mesh(V: np.ndarray, target: np.ndarray, roll_deg: float) -> np.ndarray:
    """Rotate vertices around the +X axis through `target` by `roll_deg`."""
    r = float(np.radians(roll_deg))
    cr, sr = float(np.cos(r)), float(np.sin(r))
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr,  cr],
    ], dtype=np.float32)
    return (Rx @ (V - target).T).T + target


# ─────────────────────────────────────────────────────────────────────────────
# Worker pool
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init(cache_path: str, supersample: int):
    """Load mesh once per worker, warmup JIT, pin Numba to single thread."""
    import numba
    numba.set_num_threads(1)

    global _V, _F, _target, _supersample
    V, F, _, _ = load_cache(Path(cache_path))
    if np.max(V.max(0) - V.min(0)) > 100:
        V = V * 0.001
    _V = V
    _F = F
    _target = V.mean(axis=0)
    _supersample = supersample

    # warmup
    cam_pos = _target + spherical_to_cartesian_aircraft(0, 0, 5)
    extract_silhouette_contour(_V, _F, cam_pos, _target, supersample=1)


def _worker_task(args):
    """Extract one (az, el, roll) pose."""
    idx, azimuth, elevation, distance, roll_deg = args

    # Roll the mesh first
    V_rolled = apply_roll_to_mesh(_V, _target, roll_deg)
    cam_pos = _target + spherical_to_cartesian_aircraft(azimuth, elevation, distance)

    t0 = time.perf_counter()
    points_3d = extract_silhouette_contour(
        V_rolled, _F, cam_pos, _target, supersample=_supersample)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pose_key = f"pose_{idx:05d}_roll_{roll_deg:+04.0f}"  # e.g. roll_+010
    meta = {
        "azimuth":   azimuth,
        "elevation": elevation,
        "distance":  distance,
        "roll":      roll_deg,
        "cam_pos":   cam_pos.tolist(),
        "n_points":  int(len(points_3d)),
        "time_ms":   round(elapsed_ms, 2),
    }
    return pose_key, points_3d, meta


# ─────────────────────────────────────────────────────────────────────────────
# Main precompute
# ─────────────────────────────────────────────────────────────────────────────

def precompute_roll_lut(cache_path: str,
                        base_lut_json: str,
                        output_path: str,
                        roll_slices: list = None,
                        supersample: int = 4,
                        n_workers: int = None):
    """Build LUT with roll slices for each (az, el) grid point."""
    roll_slices = roll_slices or ROLL_SLICES
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or max(1, mp.cpu_count() - 1)

    # Load base LUT metadata
    with open(base_lut_json) as f:
        base_lut = json.load(f)

    base_poses = base_lut["poses"]
    metadata = base_lut["metadata"]
    distance = metadata["distance"]

    # Extract (az, el) from each base pose
    az_el_pairs = []
    for pose_key, meta in base_poses.items():
        az_el_pairs.append((meta["azimuth"], meta["elevation"]))

    n_base = len(az_el_pairs)
    n_rolls = len(roll_slices)
    n_total = n_base * n_rolls

    print("\n" + "="*70)
    print("ROLL-EXTENDED SILHOUETTE LUT PRECOMPUTATION")
    print("="*70)
    print(f"  Base LUT     : {base_lut_json}")
    print(f"  Base poses   : {n_base:,}")
    print(f"  Roll slices  : {roll_slices}")
    print(f"  Total poses  : {n_total:,}  ({n_base} × {n_rolls})")
    print(f"  Cache        : {cache_path}")
    print(f"  Output       : {output_path}")
    print(f"  Supersample  : {supersample}x")
    print(f"  Workers      : {n_workers}")

    # Build work items
    work_items = []
    idx = 0
    for az, el in az_el_pairs:
        for roll_deg in roll_slices:
            work_items.append((idx, az, el, distance, roll_deg))
            idx += 1

    # Estimate time
    V_test, F_test, _, _ = load_cache(Path(cache_path))
    if np.max(V_test.max(0) - V_test.min(0)) > 100:
        V_test *= 0.001
    tgt = V_test.mean(0)
    cp  = tgt + spherical_to_cartesian_aircraft(0, 0, distance)
    extract_silhouette_contour(V_test, F_test, cp, tgt, supersample=1)
    t0 = time.perf_counter()
    for _ in range(3):
        extract_silhouette_contour(V_test, F_test, cp, tgt, supersample=supersample)
    t_per_pose = (time.perf_counter() - t0) / 3.0
    del V_test, F_test
    est_serial   = t_per_pose * n_total
    est_parallel = est_serial / n_workers
    print(f"\n  Per-pose    : {t_per_pose*1000:.0f} ms")
    print(f"  Serial ETA  : {est_serial/60:.1f} min")
    print(f"  Parallel ETA: {est_parallel/60:.1f} min  ({n_workers} workers)")

    # Run workers
    print(f"\n{'─'*70}")
    print("Precomputing …")

    t_start = time.perf_counter()
    results = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(str(cache_path), supersample),
    ) as pool:
        chunksize = max(1, n_total // (n_workers * 8))
        t_last = time.perf_counter()

        for i, res in enumerate(
            pool.imap_unordered(_worker_task, work_items, chunksize=chunksize)
        ):
            results.append(res)
            now = time.perf_counter()
            if now - t_last >= 2.0 or i == n_total - 1:
                done = i + 1
                elapsed = now - t_start
                rate = done / elapsed
                eta  = (n_total - done) / rate if rate > 0 else 0
                pct  = done / n_total * 100
                print(f"  {done:6d}/{n_total}  ({pct:5.1f}%)  "
                      f"{rate:5.1f} poses/s  ETA {eta:5.0f}s",
                      end="\r", flush=True)
                t_last = now

    print()
    t_total = time.perf_counter() - t_start
    print(f"Done in {t_total:.1f}s  ({t_total/60:.1f} min)  "
          f"avg {t_total/n_total*1000:.1f} ms/pose")

    # Sort and save
    results.sort(key=lambda r: r[0])

    print("\nSaving NPZ …")
    np.savez_compressed(output_path, **{key: pts for key, pts, _ in results})
    npz_size = output_path.stat().st_size
    print(f"  {output_path}  ({npz_size/1e6:.1f} MB)")

    # Build JSON index
    pose_metas = {key: meta for key, _, meta in results}

    # Group by (az, el) → list of pose_keys for each roll
    from collections import defaultdict
    az_el_to_keys = defaultdict(list)
    for key, _, meta in results:
        az_el_key = f"{meta['azimuth']:.4f}_{meta['elevation']:.4f}"
        az_el_to_keys[az_el_key].append((meta['roll'], key))

    # Sort roll slices within each group
    for az_el_key in az_el_to_keys:
        az_el_to_keys[az_el_key].sort(key=lambda x: x[0])

    lut = {
        "metadata": {
            **metadata,
            "roll_slices": roll_slices,
            "n_total_poses": n_total,
            "build_time_s": round(t_total, 1),
        },
        "angle_index_with_roll": dict(az_el_to_keys),  # {az_el -> [(roll, key), ...]}
        "poses": pose_metas,
    }

    json_path = output_path.with_suffix(".lookup.json")
    with open(json_path, "w") as f:
        json.dump(lut, f, separators=(",", ":"))
    json_size = json_path.stat().st_size
    print(f"  {json_path}  ({json_size/1e6:.1f} MB)")

    # Stats
    n_pts = [m["n_points"] for m in pose_metas.values()]
    print(f"\nPoint statistics:")
    print(f"  mean {np.mean(n_pts):.0f}  "
          f"min {np.min(n_pts)}  max {np.max(n_pts)}  "
          f"std {np.std(n_pts):.0f}")

    print(f"\n{'='*70}")
    print("✅  Roll-extended LUT ready.")
    print(f"    NPZ   : {output_path}")
    print(f"    Index : {json_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute roll-extended silhouette LUT")
    parser.add_argument("--cache", required=True, help="Mesh cache NPZ")
    parser.add_argument("--base-lut-json", required=True,
                        help="Base LUT lookup.json")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--roll-slices", type=str,
                        default="-20,-10,0,10,20",
                        help="Comma-separated roll angles (degrees)")
    parser.add_argument("--supersample", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()
    roll_slices = [float(x.strip()) for x in args.roll_slices.split(",")]

    precompute_roll_lut(
        cache_path=args.cache,
        base_lut_json=args.base_lut_json,
        output_path=args.output,
        roll_slices=roll_slices,
        supersample=args.supersample,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
