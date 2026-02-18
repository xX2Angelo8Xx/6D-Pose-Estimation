#!/usr/bin/env python3
"""
Precompute silhouette contour lookup table for 6D pose estimation.

New approach (Approach C):
  • Rasterise filled mask + depth buffer (Numba parallel prange)
  • cv2.findContours -> outer contour only, zero interior debris
  • 4x supersampling for smooth contours in the stored templates

Parallelism strategy:
  • Option 1: Numba parallel prange within each rasterisation (multi-core per pose)
  • Option 3: multiprocessing.Pool across poses (all cores for the batch)

Lookup table format:
  • <output>.npz         — compressed NumPy, one (N,3) float32 array per pose key
  • <output>.lookup.json — JSON index with:
        metadata:    grid parameters, distance, supersample factor
        angle_index: {"az_el" -> pose_key}  for O(1) tracker nearest-neighbor query
        poses:       {pose_key -> {azimuth, elevation, cam_pos, n_points, time_ms}}
"""

import argparse
import json
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np

# ── Worker must import at top level so it is picklable for multiprocessing ───
from extract_silhouette import (
    load_cache,
    extract_silhouette_contour,
    spherical_to_cartesian_aircraft,
)


# ─────────────────────────────────────────────────────────────────────────────
# Grid generation
# ─────────────────────────────────────────────────────────────────────────────

def is_rear_hemisphere(azimuth: float) -> bool:
    """True when azimuth is in the primary tracking hemisphere (90°–270°)."""
    return 90.0 <= azimuth <= 270.0


def generate_pose_grid(distance: float,
                       az_step_rear: float, el_step_rear: float,
                       az_step_front: float, el_step_front: float,
                       el_min: float = -30.0, el_max: float = 90.0):
    """Return list of (azimuth, elevation, distance) tuples covering the sphere.

    Rear hemisphere (az 90°–270°) uses finer resolution; front uses coarser.

    The grid is built as three explicit az-ranges so that every node always
    lands on an exact multiple of the relevant step size:
      Front-A : az  0°  …  <90°  in  az_step_front  increments
      Rear    : az 90°  … 270°   in  az_step_rear   increments
      Front-B : az 270° …<360°   in  az_step_front  increments
                starting at (270 + az_step_front) rounded to the step boundary
    This prevents the misalignment that occurred when the old single-while-loop
    continued from Rear's last node (e.g. 272° with rear-step=2) into the
    Front-B range at az_step_front=5°, producing 272, 277, 282 … instead of
    the expected 275, 280, 285 …
    """
    import math

    def _el_range(el_step):
        elevations = []
        el = el_min
        while el <= el_max + 1e-6:
            elevations.append(round(el, 4))
            el += el_step
        return elevations

    el_front = _el_range(el_step_front)
    el_rear  = _el_range(el_step_rear)

    poses = []

    # ── Front-A: [0°, 90°) ─────────────────────────────────────────────────
    az = 0.0
    while az < 90.0 - 1e-6:
        for el in el_front:
            poses.append((round(az, 4), el, distance))
        az = round(az + az_step_front, 10)

    # ── Rear: [90°, 270°] ──────────────────────────────────────────────────
    az = 90.0
    while az <= 270.0 + 1e-6:
        for el in el_rear:
            poses.append((round(az, 4), el, distance))
        az = round(az + az_step_rear, 10)

    # ── Front-B: (270°, 360°) — start at the first clean multiple of
    #    az_step_front that is strictly greater than 270° ──────────────────
    first_front_b = math.ceil((270.0 + 1e-9) / az_step_front) * az_step_front
    az = first_front_b
    while az < 360.0 - 1e-6:
        for el in el_front:
            poses.append((round(az, 4), el, distance))
        az = round(az + az_step_front, 10)

    return poses


# ─────────────────────────────────────────────────────────────────────────────
# Per-pose worker (runs in a subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init(cache_path: str, supersample: int):
    """Subprocess initialiser: load mesh once per worker and JIT-warmup.

    Pins Numba to a single thread per worker so that N workers across N cores
    is efficient (no OMP over-subscription when using multiprocessing.Pool).
    """
    import numba
    numba.set_num_threads(1)

    global _V, _F, _target, _supersample
    V, F, _, _ = load_cache(Path(cache_path))
    extents = V.max(axis=0) - V.min(axis=0)
    if np.max(extents) > 100:
        V = V * 0.001
    _V = V
    _F = F
    _target = V.mean(axis=0)
    _supersample = supersample
    # Warmup Numba (loads .nbc cache from disk, ~0.3 s per worker)
    cam_pos = _target + spherical_to_cartesian_aircraft(0.0, 0.0, 5.0)
    extract_silhouette_contour(_V, _F, cam_pos, _target, supersample=1)


def _worker_extract(args):
    """Extract one pose.  Returns (pose_key, points_3d, meta_dict)."""
    idx, azimuth, elevation, distance = args
    cam_pos = _target + spherical_to_cartesian_aircraft(azimuth, elevation, distance)
    t0 = time.perf_counter()
    points_3d = extract_silhouette_contour(
        _V, _F, cam_pos, _target,
        supersample=_supersample,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pose_key = f"pose_{idx:05d}"
    meta = {
        "azimuth":   azimuth,
        "elevation": elevation,
        "distance":  distance,
        "cam_pos":   cam_pos.tolist(),
        "n_points":  int(len(points_3d)),
        "time_ms":   round(elapsed_ms, 2),
    }
    return pose_key, points_3d, meta


# ─────────────────────────────────────────────────────────────────────────────
# O(1) angle index helpers
# ─────────────────────────────────────────────────────────────────────────────

def _angle_key(az: float, el: float) -> str:
    return f"{az:.4f}_{el:.4f}"


def build_angle_index(poses, pose_keys):
    """Map every (azimuth, elevation) pair to its pose key.

    At runtime, round the query angle to the nearest grid node and do a dict
    lookup — O(1) regardless of LUT size.
    """
    return {
        _angle_key(az, el): key
        for (az, el, _), key in zip(poses, pose_keys)
    }


# Module-level cache: id(angle_index) -> (az_arr, el_arr, keys_arr)
_INDEX_CACHE: dict = {}


def _build_index_arrays(angle_index: dict):
    """Parse angle_index keys into numpy arrays (cached per-index-object)."""
    idx_id = id(angle_index)
    if idx_id not in _INDEX_CACHE:
        keys = list(angle_index.keys())
        az_vals = np.empty(len(keys), np.float32)
        el_vals = np.empty(len(keys), np.float32)
        for i, k in enumerate(keys):
            az_s, el_s = k.split('_')
            az_vals[i] = float(az_s)
            el_vals[i] = float(el_s)
        vals = list(angle_index.values())
        _INDEX_CACHE[idx_id] = (az_vals, el_vals, vals)
    return _INDEX_CACHE[idx_id]


def nearest_pose_key(angle_index: dict,
                     az_query: float, el_query: float,
                     az_step_rear: float, el_step_rear: float,
                     az_step_front: float, el_step_front: float) -> str:
    """Nearest-pose lookup for use by the tracker at runtime.

    Fast path: arithmetic snap to the grid node (O(1) dict lookup).
    At grid-hemisphere boundaries the arithmetic snap may not land on an actual
    node (the front hemisphere on the 270-360° side starts at 272°, not 270°).
    Fallback: numpy nearest-neighbour over all index keys (~0.1 ms, cached).
    """
    az_q = az_query % 360.0
    az_step = az_step_rear if is_rear_hemisphere(az_q) else az_step_front
    el_step = el_step_rear if is_rear_hemisphere(az_q) else el_step_front

    az_r = round(az_q   / az_step) * az_step % 360.0
    el_r = max(-30.0, min(90.0, round(el_query / el_step) * el_step))

    key = _angle_key(round(az_r, 4), round(el_r, 4))
    result = angle_index.get(key)
    if result is not None:
        return result

    # ── fallback: brute-force nearest neighbour (boundary / off-grid cases) ──
    az_arr, el_arr, pose_keys = _build_index_arrays(angle_index)
    # Angular distance with wrap-around on azimuth
    d_az = np.abs(az_arr - az_q)
    d_az = np.minimum(d_az, 360.0 - d_az)          # wrap
    d_el = np.abs(el_arr - el_query)
    dist  = d_az ** 2 + d_el ** 2
    return pose_keys[int(np.argmin(dist))]


# ─────────────────────────────────────────────────────────────────────────────
# Main precomputation driver
# ─────────────────────────────────────────────────────────────────────────────

def precompute(cache_path, output_path,
               distance=5.0,
               az_step_rear=2.0,  el_step_rear=2.0,
               az_step_front=10.0, el_step_front=10.0,
               supersample=4,
               n_workers=None):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_workers = n_workers or max(1, mp.cpu_count() - 1)

    print("\n" + "="*70)
    print("SILHOUETTE LOOKUP TABLE PRECOMPUTATION")
    print("="*70)
    print(f"  Cache       : {cache_path}")
    print(f"  Output      : {output_path}")
    print(f"  Supersample : {supersample}x  "
          f"(~{6.25/supersample:.1f} mm/step @ 5 m, f=800)")
    print(f"  Rear grid   : {az_step_rear}° az × {el_step_rear}° el")
    print(f"  Front grid  : {az_step_front}° az × {el_step_front}° el")
    print(f"  Workers     : {n_workers}")

    poses = generate_pose_grid(distance,
                               az_step_rear, el_step_rear,
                               az_step_front, el_step_front)
    n_poses = len(poses)
    n_rear  = sum(1 for az, _, _ in poses if is_rear_hemisphere(az))
    print(f"\n  Poses total : {n_poses:,}  "
          f"(rear {n_rear:,} / front {n_poses-n_rear:,})")

    # ── estimate time ──────────────────────────────────────────────────────
    # Quick single-pose timing at supersample=1 to calibrate estimate
    V_test, F_test, _, _ = load_cache(Path(cache_path))
    if np.max(V_test.max(0) - V_test.min(0)) > 100:
        V_test *= 0.001
    tgt = V_test.mean(0)
    cp  = tgt + spherical_to_cartesian_aircraft(0, 0, distance)
    extract_silhouette_contour(V_test, F_test, cp, tgt, supersample=1)  # warmup
    t0 = time.perf_counter()
    for _ in range(3):
        extract_silhouette_contour(V_test, F_test, cp, tgt,
                                   supersample=supersample)
    t_per_pose = (time.perf_counter() - t0) / 3.0
    del V_test, F_test
    est_serial   = t_per_pose * n_poses
    est_parallel = est_serial / n_workers
    print(f"\n  Per-pose    : {t_per_pose*1000:.0f} ms")
    print(f"  Serial ETA  : {est_serial/60:.1f} min")
    print(f"  Parallel ETA: {est_parallel/60:.1f} min  ({n_workers} workers)")

    # ── worker pool ────────────────────────────────────────────────────────
    work_items = [(idx, az, el, dist)
                  for idx, (az, el, dist) in enumerate(poses)]

    t_start = time.perf_counter()
    results = []   # list of (pose_key, points_3d, meta)

    print(f"\n{'─'*70}")
    print("Precomputing …")

    ctx = mp.get_context("spawn")   # spawn is safer with Numba across platforms
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(str(cache_path), supersample),
    ) as pool:
        chunksize = max(1, n_poses // (n_workers * 8))
        t_last = time.perf_counter()

        for i, res in enumerate(
            pool.imap_unordered(_worker_extract, work_items,
                                chunksize=chunksize)
        ):
            results.append(res)
            now = time.perf_counter()
            if now - t_last >= 2.0 or i == n_poses - 1:
                done = i + 1
                elapsed = now - t_start
                rate = done / elapsed
                eta  = (n_poses - done) / rate if rate > 0 else 0
                pct  = done / n_poses * 100
                print(f"  {done:5d}/{n_poses}  ({pct:5.1f}%)  "
                      f"{rate:5.1f} poses/s  ETA {eta:4.0f}s",
                      end="\r", flush=True)
                t_last = now

    print()  # newline after \r
    t_total = time.perf_counter() - t_start
    print(f"Done in {t_total:.1f}s  ({t_total/60:.1f} min)  "
          f"avg {t_total/n_poses*1000:.1f} ms/pose (wall-clock per pose)")

    # ── sort results back into pose index order ────────────────────────────
    results.sort(key=lambda r: r[0])   # sort by pose_key string

    # ── build and save NPZ ────────────────────────────────────────────────
    print("\nSaving NPZ …")
    np.savez_compressed(
        output_path,
        **{key: pts for key, pts, _ in results}
    )
    npz_size = output_path.stat().st_size
    print(f"  {output_path}  ({npz_size/1e6:.1f} MB)")

    # ── build and save JSON index ─────────────────────────────────────────
    pose_keys  = [key  for key, _, _    in results]
    pose_metas = {key: meta for key, _, meta in results}
    angle_idx  = build_angle_index(poses, pose_keys)

    lut = {
        "metadata": {
            "distance":      distance,
            "supersample":   supersample,
            "az_step_rear":  az_step_rear,
            "el_step_rear":  el_step_rear,
            "az_step_front": az_step_front,
            "el_step_front": el_step_front,
            "el_min":        -30.0,
            "el_max":        90.0,
            "n_poses":       n_poses,
            "build_time_s":  round(t_total, 1),
        },
        "angle_index": angle_idx,   # O(1) nearest-pose lookup for tracker
        "poses": pose_metas,
    }

    json_path = output_path.with_suffix(".lookup.json")
    with open(json_path, "w") as f:
        json.dump(lut, f, separators=(",", ":"))   # compact, no indent
    json_size = json_path.stat().st_size
    print(f"  {json_path}  ({json_size/1e6:.1f} MB)")

    # ── statistics ────────────────────────────────────────────────────────
    n_pts = [m["n_points"] for m in pose_metas.values()]
    print(f"\nPoint statistics:")
    print(f"  mean {np.mean(n_pts):.0f}  "
          f"min {np.min(n_pts)}  max {np.max(n_pts)}  "
          f"std {np.std(n_pts):.0f}")

    print(f"\n{'='*70}")
    print("✅  Lookup table ready.")
    print(f"    NPZ   : {output_path}")
    print(f"    Index : {json_path}")
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Precompute silhouette lookup table (parallel, supersampled)"
    )
    parser.add_argument("--cache",    required=True,  help="Mesh cache NPZ")
    parser.add_argument("--output",   required=True,  help="Output NPZ path")
    parser.add_argument("--distance", type=float, default=5.0)
    parser.add_argument("--az-rear",  type=float, default=2.0,
                        help="Azimuth step (rear hemisphere, degrees)")
    parser.add_argument("--el-rear",  type=float, default=2.0,
                        help="Elevation step (rear hemisphere, degrees)")
    parser.add_argument("--az-front", type=float, default=10.0,
                        help="Azimuth step (front hemisphere, degrees)")
    parser.add_argument("--el-front", type=float, default=10.0,
                        help="Elevation step (front hemisphere, degrees)")
    parser.add_argument("--supersample", type=int, default=4,
                        help="Supersampling factor (4 = ~1.6 mm/step at 5 m)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes (default: CPU count − 1)")

    args = parser.parse_args()

    precompute(
        cache_path=args.cache,
        output_path=args.output,
        distance=args.distance,
        az_step_rear=args.az_rear,
        el_step_rear=args.el_rear,
        az_step_front=args.az_front,
        el_step_front=args.el_front,
        supersample=args.supersample,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
