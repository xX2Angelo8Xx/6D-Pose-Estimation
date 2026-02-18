#!/usr/bin/env python3
"""
LUT Evaluation — live render vs stored template comparison.

For a set of diverse test poses this script:
  1. Renders a live silhouette contour (the ground-truth reference)
  2. Times the O(1) angle-index lookup into the LUT
  3. Loads the stored template from the NPZ
  4. Computes quality metrics (Chamfer distance, point-count ratio)
  5. Saves side-by-side PNG comparison plots

Usage:
    python src/evaluate_lut.py \
        --cache  data/CAD_Ranger_simplified.edges.npz \
        --lut    data/silhouette_lut_4x.npz \
        --index  data/silhouette_lut_4x.lookup.json \
        --outdir results/lut_eval \
        [--supersample 4] [--lookup-reps 10000]
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from extract_silhouette import (
    load_cache,
    extract_silhouette_contour,
    spherical_to_cartesian_aircraft,
)
from precompute_visibility import nearest_pose_key, is_rear_hemisphere


# ─────────────────────────────────────────────────────────────────────────────
# Test-pose catalogue  —  diverse spread over the sphere
# ─────────────────────────────────────────────────────────────────────────────

TEST_POSES = [
    # label                        az      el    notes
    # ── exact grid nodes ──────────────────────────────────────────────────
    ("rear_level",                  0.0,    0.0),  # az=0 front, el=0 (on grid)
    ("rear_90_level",              90.0,    0.0),  # rear boundary, on grid
    ("rear_180_level",            180.0,    0.0),  # on grid
    ("rear_270_level",            270.0,    0.0),  # rear boundary, on grid
    ("rear_90_30up",               90.0,   30.0),  # rear, on grid
    ("rear_180_45up",             180.0,   45.0),  # rear, on grid
    ("rear_180_minus30",          180.0,  -30.0),  # rear, el min, on grid
    ("top_down_rear",               0.0,   90.0),  # el max, on grid
    ("top_down_right",             90.0,   90.0),  # el max, on grid
    # ── off-grid: inside rear hemisphere (snap ≤1°) ───────────────────────
    ("rear_offgrid_91az_1el",      91.0,    1.0),  # snaps to (92,2) — 1° off
    ("rear_offgrid_135az_15el",   135.0,   15.0),  # snaps to (136,16) — 1.4° off
    ("rear_offgrid_225az_33el",   225.0,   33.0),  # snaps to (226,34) — 1.4° off
    ("rear_offgrid_269az_0el",    269.0,    0.0),  # snaps to (270,0) — 1° off
    # ── off-grid: inside front hemisphere (snap ≤5°) ─────────────────────
    ("front_offgrid_5az_5el",       5.0,    5.0),  # snaps to (10,10) — 7° off
    ("front_offgrid_355az_15el",  355.0,   15.0),  # snaps to (0,20) or (360=0)
    ("front_offgrid_45az_10el",    45.0,   10.0),  # snaps to (50,10) — 5° off
    # ── boundary transitions (rear→front crossover) ───────────────────────
    ("boundary_272az_0el",        272.0,    0.0),  # first front node after 270
    ("boundary_315az_30el",       315.0,   30.0),  # between 312 and 322 (fallback)
    ("boundary_88az_0el",          88.0,    0.0),  # near 90° front/rear transition
]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def chamfer_distance(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """One-way Chamfer distances A→B and B→A (mean nearest-neighbour dist, mm)."""
    if len(A) == 0 or len(B) == 0:
        return float('nan'), float('nan')
    tree_B = cKDTree(B)
    tree_A = cKDTree(A)
    d_AB, _ = tree_B.query(A, workers=1)
    d_BA, _ = tree_A.query(B, workers=1)
    return float(d_AB.mean()) * 1000.0, float(d_BA.mean()) * 1000.0   # → mm


def coverage_ratio(live: np.ndarray, lut: np.ndarray, thr_m: float = 0.005) -> float:
    """Fraction of live points that have a LUT neighbour within thr_m (5 mm)."""
    if len(live) == 0 or len(lut) == 0:
        return float('nan')
    tree = cKDTree(lut)
    d, _ = tree.query(live, workers=1)
    return float((d < thr_m).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Projection helper (for 2-D plots)
# ─────────────────────────────────────────────────────────────────────────────

def project_cam_pts(pts_cam: np.ndarray,
                    fx: float = 800.0, fy: float = 800.0,
                    cx: float = 640.0, cy: float = 360.0
                    ) -> np.ndarray:
    """Pinhole-project camera-space (X, Y, Z) points to pixel coordinates.

    extract_silhouette_contour already returns points in camera space,
    so no additional R/t transform is needed.
    """
    if len(pts_cam) == 0:
        return np.empty((0, 2))
    pts_cam = np.atleast_2d(pts_cam)
    valid = pts_cam[:, 2] > 0
    uv = np.full((len(pts_cam), 2), np.nan)
    d = pts_cam[valid, 2]
    uv[valid, 0] = fx * pts_cam[valid, 0] / d + cx
    uv[valid, 1] = fy * pts_cam[valid, 1] / d + cy
    return uv


# ─────────────────────────────────────────────────────────────────────────────
# Single-pose comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_plot(label: str,
                         az: float, el: float,
                         snapped_az: float, snapped_el: float,
                         live_pts: np.ndarray,
                         lut_pts:  np.ndarray,
                         ch_live2lut: float, ch_lut2live: float,
                         coverage: float,
                         lookup_us: float,
                         out_path: Path,
                         width: int = 1280, height: int = 720,
                         fx: float = 800.0, fy: float = 800.0):
    dpi = 100
    fig, axes = plt.subplots(1, 3, figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')

    cx, cy = width / 2, height / 2

    titles = ["Live render (reference)", "LUT template", "Overlay (live=cyan, LUT=orange)"]
    for ax, title in zip(axes, titles):
        ax.set_facecolor('#0f0f1a')
        ax.set_title(title, color='white', fontsize=9, pad=4)
        ax.set_xlim(0, width);  ax.set_ylim(height, 0)
        ax.set_aspect('equal')
        ax.tick_params(colors='#666', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    uv_live = project_cam_pts(live_pts, fx, fy, cx, cy)
    uv_lut  = project_cam_pts(lut_pts,  fx, fy, cx, cy)

    def _scatter(ax, uv, color, lbl, s=4, alpha=0.7):
        m = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) if len(uv) else np.array([], bool)
        if m.any():
            ax.scatter(uv[m, 0], uv[m, 1], c=color, s=s, alpha=alpha,
                       linewidths=0, label=lbl, rasterized=True)

    _scatter(axes[0], uv_live, 'cyan',   f'live  ({len(live_pts)} pts)')
    _scatter(axes[1], uv_lut,  'orange', f'LUT   ({len(lut_pts)} pts)')
    _scatter(axes[2], uv_live, 'cyan',   f'live')
    _scatter(axes[2], uv_lut,  'orange', f'LUT', alpha=0.5)

    for ax in axes:
        ax.legend(fontsize=7, framealpha=0.3,
                  labelcolor='white', facecolor='#111')

    # ── global title ──────────────────────────────────────────────────────
    snap_str = (f"  →  snapped az={snapped_az:.1f}° el={snapped_el:.1f}°"
                if (abs(az - snapped_az) > 0.01 or abs(el - snapped_el) > 0.01)
                else "  (on-grid)")
    title = (f"{label}   az={az:.1f}° el={el:.1f}°{snap_str}\n"
             f"Chamfer live→LUT={ch_live2lut:.2f} mm  LUT→live={ch_lut2live:.2f} mm"
             f"   coverage@5mm={coverage*100:.1f}%   lookup={lookup_us:.1f} µs")
    fig.suptitle(title, color='white', fontsize=9, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate LUT quality vs live renders")
    ap.add_argument('--cache',        required=True,  help='edges.npz mesh cache')
    ap.add_argument('--lut',          required=True,  help='LUT NPZ file')
    ap.add_argument('--index',        required=True,  help='lookup.json file')
    ap.add_argument('--outdir',       default='results/lut_eval')
    ap.add_argument('--supersample',  type=int,   default=4)
    ap.add_argument('--distance',     type=float, default=5.0)
    ap.add_argument('--lookup-reps',  type=int,   default=10_000,
                    help='Repeat lookup N times to get stable µs timing')
    ap.add_argument('--az-rear',      type=float, default=2.0)
    ap.add_argument('--el-rear',      type=float, default=2.0)
    ap.add_argument('--az-front',     type=float, default=10.0)
    ap.add_argument('--el-front',     type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load mesh ─────────────────────────────────────────────────────────
    print("Loading mesh …")
    V, F, _, _ = load_cache(Path(args.cache))
    if np.max(V.max(0) - V.min(0)) > 100:
        V = V * 0.001
    target = V.mean(axis=0)
    print(f"  {len(V):,} verts  {len(F):,} faces  target={target}")

    # ── JIT warmup ────────────────────────────────────────────────────────
    print("Warming up JIT …")
    _cam = target + spherical_to_cartesian_aircraft(0, 0, args.distance)
    extract_silhouette_contour(V, F, _cam, target, supersample=1)
    extract_silhouette_contour(V, F, _cam, target, supersample=args.supersample)
    print("  ✓ done")

    # ── load LUT ──────────────────────────────────────────────────────────
    print(f"Loading LUT …  {args.lut}")
    lut_npz = np.load(args.lut)
    with open(args.index, 'r') as f:
        lut_json = json.load(f)
    angle_index = lut_json['angle_index']
    meta        = lut_json.get('metadata', {})
    print(f"  LUT poses: {len(angle_index):,}  "
          f"supersample={meta.get('supersample','?')}x  "
          f"distance={meta.get('distance','?')}m")

    # ── run evaluation ────────────────────────────────────────────────────
    rows = []
    print()
    header = (f"{'Pose':<26} {'az':>6} {'el':>5} "
              f"{'live_pts':>9} {'lut_pts':>9} "
              f"{'ch_l2t mm':>10} {'ch_t2l mm':>10} "
              f"{'cov%':>6} {'render ms':>10} {'lookup µs':>10}")
    sep    = "─" * len(header)
    print(header)
    print(sep)

    for label, az, el in TEST_POSES:
        cam_pos = target + spherical_to_cartesian_aircraft(az, el, args.distance)

        # — live render ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        live_pts = extract_silhouette_contour(
            V, F, cam_pos, target,
            supersample=args.supersample,
        )
        render_ms = (time.perf_counter() - t0) * 1000.0

        # — O(1) LUT lookup ───────────────────────────────────────────────
        # warm timing run first
        nearest_pose_key(angle_index, az, el,
                         args.az_rear, args.el_rear,
                         args.az_front, args.el_front)
        reps = args.lookup_reps
        t0 = time.perf_counter()
        for _ in range(reps):
            pkey = nearest_pose_key(angle_index, az, el,
                                    args.az_rear, args.el_rear,
                                    args.az_front, args.el_front)
        lookup_us = (time.perf_counter() - t0) / reps * 1e6

        if pkey is None:
            print(f"  ⚠  {label}: no LUT key found for az={az} el={el}")
            continue

        lut_pts = lut_npz[pkey].astype(np.float32)

        # recover snapped angles from JSON
        pose_meta  = lut_json['poses'].get(pkey, {})
        snapped_az = pose_meta.get('azimuth', az)
        snapped_el = pose_meta.get('elevation', el)

        # — quality metrics ───────────────────────────────────────────────
        ch_l2t, ch_t2l = chamfer_distance(live_pts, lut_pts)
        coverage        = coverage_ratio(live_pts, lut_pts, thr_m=0.005)

        # — log row ───────────────────────────────────────────────────────
        print(f"{label:<26} {az:>6.1f} {el:>5.1f} "
              f"{len(live_pts):>9,} {len(lut_pts):>9,} "
              f"{ch_l2t:>10.2f} {ch_t2l:>10.2f} "
              f"{coverage*100:>6.1f} {render_ms:>10.1f} {lookup_us:>10.2f}")

        rows.append(dict(label=label, az=az, el=el,
                         snapped_az=snapped_az, snapped_el=snapped_el,
                         pkey=pkey,
                         n_live=len(live_pts), n_lut=len(lut_pts),
                         ch_l2t=ch_l2t, ch_t2l=ch_t2l, coverage=coverage,
                         render_ms=render_ms, lookup_us=lookup_us))

        # — comparison plot ───────────────────────────────────────────────
        plot_path = out_dir / f"{label}.png"
        make_comparison_plot(
            label, az, el, snapped_az, snapped_el,
            live_pts, lut_pts,
            ch_l2t, ch_t2l, coverage, lookup_us,
            plot_path,
        )

    print(sep)

    if not rows:
        print("No results — check LUT and index paths.")
        return

    # ── aggregate stats ───────────────────────────────────────────────────
    ch_l2t_vals   = [r['ch_l2t']   for r in rows if np.isfinite(r['ch_l2t'])]
    ch_t2l_vals   = [r['ch_t2l']   for r in rows if np.isfinite(r['ch_t2l'])]
    cov_vals      = [r['coverage'] for r in rows if np.isfinite(r['coverage'])]
    render_vals   = [r['render_ms']  for r in rows]
    lookup_vals   = [r['lookup_us']  for r in rows]

    print(f"\n{'SUMMARY':─<{len(sep)}}")
    print(f"  Poses evaluated   : {len(rows)}")
    print(f"  Live render time  : mean {np.mean(render_vals):.1f} ms  "
          f"std {np.std(render_vals):.1f}  min {np.min(render_vals):.1f}  "
          f"max {np.max(render_vals):.1f}")
    print(f"  LUT lookup time   : mean {np.mean(lookup_vals):.2f} µs  "
          f"(×{args.lookup_reps:,} reps each)")
    print(f"  Chamfer live→LUT  : mean {np.mean(ch_l2t_vals):.2f} mm  "
          f"max {np.max(ch_l2t_vals):.2f} mm")
    print(f"  Chamfer LUT→live  : mean {np.mean(ch_t2l_vals):.2f} mm  "
          f"max {np.max(ch_t2l_vals):.2f} mm")
    print(f"  Coverage @5 mm    : mean {np.mean(cov_vals)*100:.1f}%  "
          f"min {np.min(cov_vals)*100:.1f}%")

    # ── summary figure  ───────────────────────────────────────────────────
    _make_summary_figure(rows, out_dir)
    print(f"\nPlots saved to  {out_dir}/")
    print(f"Summary figure: {out_dir}/00_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary scatter / bar figure
# ─────────────────────────────────────────────────────────────────────────────

def _make_summary_figure(rows: list, out_dir: Path):
    dpi = 100
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')

    labels  = [r['label']    for r in rows]
    ch_l2t  = [r['ch_l2t']   for r in rows]
    ch_t2l  = [r['ch_t2l']   for r in rows]
    cov     = [r['coverage'] * 100 for r in rows]
    renders = [r['render_ms']  for r in rows]
    n_live  = [r['n_live']     for r in rows]
    n_lut   = [r['n_lut']      for r in rows]

    x = np.arange(len(rows))

    for ax in axes.flat:
        ax.set_facecolor('#0f0f1a')
        ax.tick_params(colors='#aaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    def bar_ax(ax, vals, color, title, ylabel, ideal_line=None):
        ax.bar(x, vals, color=color, alpha=0.8, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6.5,
                           color='#ccc')
        ax.set_title(title, color='white', fontsize=10)
        ax.set_ylabel(ylabel, color='#ccc', fontsize=8)
        ax.yaxis.label.set_color('#ccc')
        if ideal_line is not None:
            ax.axhline(ideal_line, color='white', lw=1, ls='--', alpha=0.5,
                       label=f'target {ideal_line}')
            ax.legend(fontsize=7, labelcolor='white', facecolor='#222',
                      framealpha=0.4)

    bar_ax(axes[0, 0], ch_l2t, '#00bfff', 'Chamfer live → LUT (mm)',
           'mm', ideal_line=5.0)
    bar_ax(axes[0, 1], cov,    '#00e676', 'Coverage @5 mm (%)',
           '%',  ideal_line=90.0)
    bar_ax(axes[1, 0], renders,'#ff9100', 'Live render time (ms)',
           'ms')
    bar_ax(axes[1, 1], n_live, '#e040fb', 'Point count',
           'points')
    bar_ax_twin = axes[1, 1].twinx()
    bar_ax_twin.plot(x, n_lut, 'o-', color='orange', ms=5, lw=1.5,
                     label='LUT pts')
    bar_ax_twin.tick_params(colors='#aaa', labelsize=7)
    bar_ax_twin.legend(fontsize=7, labelcolor='white', facecolor='#222',
                       framealpha=0.4)
    axes[1, 1].get_legend_handles_labels()  # consumed by twinx

    fig.suptitle('LUT Evaluation Summary', color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / '00_summary.png', dpi=dpi,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
