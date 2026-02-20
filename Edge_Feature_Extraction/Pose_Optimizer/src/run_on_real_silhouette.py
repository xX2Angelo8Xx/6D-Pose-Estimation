#!/usr/bin/env python3
"""
Pose Optimizer — Real Silhouette Mode (No Initial Guess)
=========================================================

Loads an exported silhouette NPZ (produced by the Edge Detection GUI) and
finds the best-matching pose through a two-stage pipeline:

  Stage 1 — Global coarse search
      Projects every LUT template (8 296 poses × N distance candidates) to
      pixel space and computes the one-way Chamfer distance against the
      observed edge pixels.  No initial guess is required.

  Stage 2 — Nelder-Mead refinement
      The top-K coarse candidates are refined jointly over (az, el, dist,
      roll) with scipy.optimize.minimize('Nelder-Mead').

Output: JSON result + 3-panel visualisation PNG.

Usage
-----
    python3 Pose_Optimizer/src/run_on_real_silhouette.py \\
        --silhouette  Edge_Detection/data/exported_silhouettes/silhouette_*.npz \\
        [--lut        CAD-Edge-Generation/data/silhouette_lut_4x_final.npz] \\
        [--index      CAD-Edge-Generation/data/silhouette_lut_4x_final.lookup.json] \\
        [--fx 800] [--fy 800] [--cx 640] [--cy 360] \\
        [--dist-min 2.0] [--dist-max 15.0] [--dist-steps 7] \\
        [--top-k 20] [--lut-ref-dist 10.0] \\
        [--outdir results_real]

Camera intrinsic note
---------------------
The synthetic optimizer uses FX=FY=800, CX=640, CY=360 for a 1280×720
sensor.  If your real camera has different intrinsics, pass --fx --fy --cx
--cy to compensate.  The observed edge pixel coordinates (u, v) must be in
the same coordinate frame as the intrinsics you supply.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# ── import shared helpers from CAD-Edge-Generation ───────────────────────────
_SRC = Path(__file__).parent.parent.parent / "CAD-Edge-Generation" / "src"
sys.path.insert(0, str(_SRC))

from extract_silhouette import (
    spherical_to_cartesian_aircraft,
    compute_camera_transform,
)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def project_world_to_pixels(
    pts_world: np.ndarray,
    cam_pos: np.ndarray,
    R_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Project N world-space points to (u, v) pixel coords.  Returns (K, 2)."""
    pts_cam = (R_cam @ (pts_world - cam_pos).T).T
    valid   = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return np.empty((0, 2), np.float32)
    z   = pts_cam[:, 2]
    uv  = np.empty((len(pts_cam), 2), np.float32)
    uv[:, 0] = fx * pts_cam[:, 0] / z + cx
    uv[:, 1] = fy * pts_cam[:, 1] / z + cy
    return uv


def cam_space_to_world(
    pts_cam: np.ndarray, R_cam: np.ndarray, cam_pos: np.ndarray
) -> np.ndarray:
    """Inverse projection: camera-space → world-space."""
    return (R_cam.T @ pts_cam.T).T + cam_pos


def apply_roll_to_world(
    pts_world: np.ndarray, target: np.ndarray, roll_deg: float
) -> np.ndarray:
    """Rotate model around fuselage +X axis through `target` by roll_deg."""
    r  = float(np.radians(roll_deg))
    cr, sr = float(np.cos(r)), float(np.sin(r))
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cr,  -sr ],
                   [0.0, sr,   cr ]], dtype=np.float32)
    return (Rx @ (pts_world - target).T).T + target


def chamfer_obs_to_model(observed_px: np.ndarray, model_px: np.ndarray) -> float:
    """Mean nearest-neighbour distance observed → model  (correct direction).

    For each observed edge pixel find its nearest model pixel.
    A pose where the model covers few pixels is heavily penalised because most
    observed pixels are far from the tiny model projection.
    """
    if len(model_px) < 3:
        return 1e6
    model_tree = cKDTree(model_px)
    d, _ = model_tree.query(observed_px, workers=1)
    return float(d.mean())


def chamfer_symmetric(observed_px: np.ndarray,
                      model_px: np.ndarray,
                      obs_tree: cKDTree) -> float:
    """Symmetric Chamfer = 0.5 * (obs→model + model→obs).

    obs_tree is pre-built from observed_px for speed (reuse across calls).
    """
    if len(model_px) < 3:
        return 1e6
    model_tree = cKDTree(model_px)
    d_o2m, _ = model_tree.query(observed_px, workers=1)
    d_m2o, _ = obs_tree.query(model_px, workers=1)
    return 0.5 * (float(d_o2m.mean()) + float(d_m2o.mean()))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Global coarse search
# ─────────────────────────────────────────────────────────────────────────────

def global_coarse_search(
    observed_px:    np.ndarray,
    lut_npz,
    angle_index:    dict,
    target:         np.ndarray,
    dist_candidates: list,
    lut_ref_dist:   float,
    fx: float, fy: float, cx: float, cy: float,
    roll: float = 0.0,
    verbose: bool = True,
) -> list[tuple]:
    """
    Evaluate every LUT entry at every candidate distance.

    Cost metric: obs→model Chamfer (each observed pixel to nearest model pixel).
    A degenerate pose that projects only a few pixels is penalised because most
    observed pixels have no nearby model pixel.

    Speed trick — projection scaling:
    For a fixed (az, el), the projected pixel at distance d relates to the
    reference projection at d_ref by:
        (u - cx) * d = (u_ref - cx) * d_ref
    So we project once at d_ref, then for each distance d we scale the
    *observed* pixels by d/d_ref around the principal point and query the
    fixed model KD-tree.  One tree build per LUT entry instead of per
    (entry × distance).

    Returns a list of (cost, az, el, dist, model_world) sorted ascending.
    """
    ai = angle_index.get("angle_index", angle_index)   # unwrap JSON wrapper

    pk_to_az_el: dict[str, tuple[float, float]] = {}
    for az_el_key, pkey in ai.items():
        az_s, el_s = az_el_key.split("_", 1)
        pk_to_az_el[pkey] = (float(az_s), float(el_s))

    principal  = np.array([cx, cy], dtype=np.float32)
    # Observed pixels relative to principal point
    obs_c      = (observed_px - principal).astype(np.float32)   # (N, 2)
    dist_arr   = np.array(dist_candidates, dtype=np.float32)    # (D,)
    # Scaling factors: at distance d the pixel offset scales by d_ref/d.
    # To compare obs vs model_ref both at reference scale we scale obs by d/d_ref.
    inv_scales  = dist_arr / lut_ref_dist                        # (D,)  = d / d_ref
    # The raw Chamfer in scaled space is multiplied by d_ref/d to recover pixel units.
    px_scales   = lut_ref_dist / dist_arr                        # (D,)  = d_ref / d

    candidates: list[tuple] = []
    n_total    = len(pk_to_az_el) * len(dist_candidates)
    t0         = time.perf_counter()

    if verbose:
        print(f"  {len(pk_to_az_el)} LUT entries × {len(dist_candidates)} distances "
              f"= {n_total:,} evaluations …", flush=True)

    for pkey, (az_s, el_s) in pk_to_az_el.items():
        pts_lut_cam = lut_npz[pkey].astype(np.float32)

        cam_pos_lut = target + spherical_to_cartesian_aircraft(az_s, el_s, lut_ref_dist)
        R_lut       = compute_camera_transform(cam_pos_lut, target)
        model_world = cam_space_to_world(pts_lut_cam, R_lut, cam_pos_lut)

        if abs(roll) > 0.01:
            model_world = apply_roll_to_world(model_world, target, roll)

        # Project once at reference distance
        model_px_ref = project_world_to_pixels(
            model_world, cam_pos_lut, R_lut, fx, fy, cx, cy)
        if len(model_px_ref) < 5:
            continue

        # Model pixels centred around principal point (in ref-scale frame)
        model_c = (model_px_ref - principal).astype(np.float32)   # (M, 2)
        model_tree = cKDTree(model_c)                              # build once

        for i, dist in enumerate(dist_candidates):
            # Scale observed pixels into ref-scale frame, then query fixed tree
            obs_scaled = obs_c * float(inv_scales[i])              # (N, 2)
            d_raw, _   = model_tree.query(obs_scaled, workers=1)   # (N,)
            # Recover pixel-space distances
            cost = float(d_raw.mean()) * float(px_scales[i])
            candidates.append((cost, az_s, el_s, dist, model_world.copy()))

    elapsed = time.perf_counter() - t0
    candidates.sort(key=lambda x: x[0])

    if verbose:
        best = candidates[0]
        print(f"  Done in {elapsed:.1f} s — best: "
              f"az={best[1]:.1f}°  el={best[2]:.1f}°  d={best[3]:.1f} m  "
              f"cost={best[0]:.2f} px", flush=True)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Nelder-Mead refinement
# ─────────────────────────────────────────────────────────────────────────────

class PoseCostFunctionReal:
    """Cost function closure for scipy.optimize.minimize.

    The model_world points are refreshed from the LUT whenever the optimizer
    crosses a LUT grid cell boundary (az/el changes to a different nearest
    entry).  This ensures visibility-correct silhouettes throughout the whole
    optimisation, not just at the warm-start pose.

    History tuples:
        store_frames=False : (az, el, dist, roll, cost)
        store_frames=True  : (az, el, dist, roll, cost, model_px_copy)
    """

    def __init__(
        self,
        observed_px:  np.ndarray,
        model_world:  np.ndarray,
        target:       np.ndarray,
        fx: float, fy: float, cx: float, cy: float,
        store_frames: bool = False,
        lut_npz=None,
        angle_index: dict | None = None,
        lut_ref_dist: float = 10.0,
    ):
        self.observed_px  = observed_px
        self.model_world  = model_world
        self.target       = target
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.store_frames = store_frames
        self.obs_tree     = cKDTree(observed_px)
        self.n_evals      = 0
        self.history: list[tuple] = []

        # ── LUT-refresh support ───────────────────────────────────────────
        # Build a unit-vector KD-tree over all (az, el) grid entries so that
        # each optimizer step can quickly find its nearest LUT key and reload
        # fresh, visibility-correct edge points when the cell changes.
        self._lut_tree    = None  # disabled unless LUT is provided
        if lut_npz is not None and angle_index is not None:
            ai = angle_index.get("angle_index", angle_index)
            units, pkey_list = [], []
            for az_el_key, pkey in ai.items():
                az_s, el_s = az_el_key.split("_", 1)
                ar = math.radians(float(az_s))
                er = math.radians(float(el_s))
                units.append([math.cos(er)*math.cos(ar),
                               math.cos(er)*math.sin(ar),
                               math.sin(er)])
                pkey_list.append((pkey, float(az_s), float(el_s)))
            self._lut_tree     = cKDTree(np.array(units, dtype=np.float32))
            self._lut_pkeys    = pkey_list
            self._lut_npz      = lut_npz
            self._lut_ref_dist = lut_ref_dist
            self._cached_key   = None  # last loaded pkey

    def __call__(self, params: np.ndarray) -> float:
        az, el, dist, roll = params
        az   = az % 360.0
        el   = float(np.clip(el,   -89.0, 89.0))
        dist = float(np.clip(dist,   0.5, 50.0))

        # ── LUT refresh: reload model_world when optimizer enters new cell ──
        # The LUT stores visibility-filtered points for each discrete pose.
        # Keeping the initial frozen world points causes ghost edges / holes
        # when the camera moves far from the warm-start LUT pose.
        if self._lut_tree is not None:
            ar = math.radians(az)
            er = math.radians(el)
            unit = [math.cos(er)*math.cos(ar),
                    math.cos(er)*math.sin(ar),
                    math.sin(er)]
            _, idx = self._lut_tree.query(unit)
            pkey, snap_az, snap_el = self._lut_pkeys[idx]
            if pkey != self._cached_key:
                pts_cam = self._lut_npz[pkey].astype(np.float32)
                snap_cp = self.target + spherical_to_cartesian_aircraft(
                    snap_az, snap_el, self._lut_ref_dist)
                snap_R  = compute_camera_transform(snap_cp, self.target)
                self.model_world = cam_space_to_world(pts_cam, snap_R, snap_cp)
                self._cached_key = pkey

        cam_pos   = self.target + spherical_to_cartesian_aircraft(az, el, dist)
        R_cam     = compute_camera_transform(cam_pos, self.target)
        mw_rolled = apply_roll_to_world(self.model_world, self.target, roll)
        model_px  = project_world_to_pixels(
            mw_rolled, cam_pos, R_cam, self.fx, self.fy, self.cx, self.cy
        )
        cost = chamfer_symmetric(self.observed_px, model_px, self.obs_tree)

        self.n_evals += 1
        if self.store_frames:
            self.history.append((az, el, dist, roll, cost, model_px.copy()))
        else:
            self.history.append((az, el, dist, roll, cost))
        return cost


def refine_candidate(
    observed_px:  np.ndarray,
    candidate:    tuple,    # (cost, az, el, dist, model_world)
    target:       np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    store_frames: bool = False,
    lut_npz=None,
    angle_index: dict | None = None,
    lut_ref_dist: float = 10.0,
) -> dict:
    """Run Nelder-Mead on a single coarse candidate.  Returns result dict."""
    cost0, az0, el0, dist0, model_world = candidate

    cost_fn = PoseCostFunctionReal(
        observed_px, model_world, target, fx, fy, cx, cy,
        store_frames=store_frames,
        lut_npz=lut_npz, angle_index=angle_index, lut_ref_dist=lut_ref_dist,
    )
    x0      = np.array([az0, el0, dist0, 0.0])           # roll warm-start = 0
    delta   = np.array([5.0, 5.0, dist0 * 0.2, 5.0])
    isimplex = np.vstack([x0, x0 + np.diag(delta)])

    result = minimize(
        cost_fn, x0, method="Nelder-Mead",
        options=dict(xatol=0.1, fatol=0.5, maxiter=1000,
                     initial_simplex=isimplex),
    )

    opt_az, opt_el, opt_dist, opt_roll = result.x
    opt_az   = opt_az % 360.0
    opt_el   = float(np.clip(opt_el,   -89, 89))
    opt_dist = float(np.clip(opt_dist,  0.5, 50))

    return {
        "az":         round(opt_az, 2),
        "el":         round(opt_el, 2),
        "dist":       round(opt_dist, 3),
        "roll":       round(float(opt_roll), 2),
        "cost_final": round(result.fun, 3),
        "cost_coarse": round(cost0, 3),
        "converged":  bool(result.success),
        "n_evals":    cost_fn.n_evals,
        "init_az":    round(az0, 2),
        "init_el":    round(el0, 2),
        "init_dist":  round(dist0, 3),
        # internal — stripped before JSON serialisation
        "_model_world": model_world,
        "_history":     cost_fn.history,
        "_store_frames": store_frames,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convergence GIF
# ─────────────────────────────────────────────────────────────────────────────

def _az_el_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    """Spherical (az, el) → unit cartesian for 3-D sphere plot."""
    ar, er = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(er)*np.cos(ar), np.cos(er)*np.sin(ar), np.sin(er)])


def render_convergence_gif(
    best: dict,
    observed_px: np.ndarray,
    img_w: int, img_h: int,
    out_path: Path,
    max_frames: int = 80,
    fps: int = 12,
) -> None:
    """Render an animated GIF of the Nelder-Mead refinement for `best`.

    Layout (dark, 16:9)
    ┌─────────────────────────┬──────────────────┐
    │  2-D: observed (cyan)   │  3-D sphere traj │
    │        model (orange)   │                  │
    │                         ├──────────────────┤
    │                         │  cost curve      │
    └─────────────────────────┴──────────────────┘

    Requires store_frames=True during refinement (history tuples length 6).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    hist = best["_history"]
    if not hist or len(hist[0]) < 6:
        print("  GIF: no frame data — skipping (run with --visualize)")
        return

    n_hist = len(hist)
    # Sub-sample to max_frames, always keep first + last
    if n_hist <= max_frames:
        indices = list(range(n_hist))
    else:
        indices = sorted(set(
            [0]
            + list(np.linspace(1, n_hist - 2, max_frames - 2, dtype=int))
            + [n_hist - 1]
        ))
    frames     = [hist[i] for i in indices]
    costs_all  = [h[4] for h in hist]
    evals_all  = np.arange(n_hist)
    traj_xyz   = np.array([_az_el_to_unit(h[0], h[1]) for h in hist])
    init_xyz   = _az_el_to_unit(best["init_az"], best["init_el"])
    final_xyz  = _az_el_to_unit(best["az"], best["el"])

    BG, PAD_C = '#0f0f1a', '#1a1a2e'
    fig = plt.figure(figsize=(16, 9), dpi=90, facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.5, 1], height_ratios=[1.4, 1],
        hspace=0.35, wspace=0.28,
        left=0.04, right=0.97, top=0.92, bottom=0.06,
    )
    ax_2d   = fig.add_subplot(gs[:, 0])
    ax_3d   = fig.add_subplot(gs[0, 1], projection='3d')
    ax_cost = fig.add_subplot(gs[1, 1])

    for ax in (ax_2d, ax_cost):
        ax.set_facecolor(PAD_C)
        ax.tick_params(colors='#888', labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor('#333')
    ax_3d.set_facecolor(BG)
    ax_3d.xaxis.pane.fill = ax_3d.yaxis.pane.fill = ax_3d.zaxis.pane.fill = False

    # ── static elements ───────────────────────────────────────────────────
    scat_obs = ax_2d.scatter(
        observed_px[:, 0], observed_px[:, 1],
        s=0.8, c='#4da6ff', alpha=0.6, linewidths=0, label='observed')
    mpx0 = frames[0][5]
    scat_mod = ax_2d.scatter(
        mpx0[:, 0] if len(mpx0) else [],
        mpx0[:, 1] if len(mpx0) else [],
        s=0.8, c='#ff7c3a', alpha=0.75, linewidths=0, label='model')
    ax_2d.set_xlim(0, img_w)
    ax_2d.set_ylim(img_h, 0)   # y flipped — image coords
    ax_2d.set_aspect('equal')
    ax_2d.legend(loc='lower left', fontsize=7, labelcolor='white',
                 facecolor='#111', framealpha=0.6, markerscale=4)
    title_2d = ax_2d.set_title('', color='white', fontsize=8, pad=6)

    # 3-D hemisphere wireframe
    u_w = np.linspace(0, 2*np.pi, 36)
    v_w = np.linspace(-np.pi/6, np.pi/2, 18)
    xs  = np.outer(np.cos(v_w), np.cos(u_w))
    ys  = np.outer(np.cos(v_w), np.sin(u_w))
    zs  = np.outer(np.sin(v_w), np.ones_like(u_w))
    ax_3d.plot_wireframe(xs, ys, zs, color='#333', lw=0.3, alpha=0.4)
    # Faded full trajectory
    ax_3d.scatter(traj_xyz[:,0], traj_xyz[:,1], traj_xyz[:,2],
                  s=4, c='#555', alpha=0.35, depthshade=False)
    # Coarse init (fixed cyan diamond)
    ax_3d.scatter(*init_xyz, s=100, c='#00e5ff', marker='D', zorder=10,
                  label=f'coarse init\naz={best["init_az"]:.0f}° el={best["init_el"]:.0f}°',
                  depthshade=False)
    # Final refined (fixed green star)
    ax_3d.scatter(*final_xyz, s=120, c='#39ff14', marker='*', zorder=11,
                  label=f'refined\naz={best["az"]:.0f}° el={best["el"]:.0f}°',
                  depthshade=False)
    # Animated current (red dot)
    cur0 = _az_el_to_unit(frames[0][0], frames[0][1])
    scat_cur = ax_3d.scatter(*cur0, s=60, c='#ff4040', zorder=12, depthshade=False)
    ax_3d.set_xlim(-1.1, 1.1); ax_3d.set_ylim(-1.1, 1.1); ax_3d.set_zlim(-0.2, 1.15)
    ax_3d.set_xlabel('X', color='#888', fontsize=7, labelpad=1)
    ax_3d.set_ylabel('Y', color='#888', fontsize=7, labelpad=1)
    ax_3d.set_zlabel('Z', color='#888', fontsize=7, labelpad=1)
    ax_3d.tick_params(colors='#555', labelsize=6)
    ax_3d.legend(fontsize=6, labelcolor='white', facecolor='#111',
                 framealpha=0.5, loc='upper left')
    title_3d = ax_3d.set_title('', color='white', fontsize=8)

    # Cost curve
    ax_cost.plot(evals_all, costs_all, color='#555', lw=0.8, alpha=0.6)
    vline    = ax_cost.axvline(x=0, color='#ff7c3a', lw=1.2, alpha=0.9)
    dot_cost = ax_cost.scatter([0], [costs_all[0]], s=25, c='#ff7c3a', zorder=5)
    ax_cost.set_xlim(-1, n_hist)
    valid_costs = [c for c in costs_all if c < 1e5]
    ax_cost.set_ylim(0, (max(valid_costs) if valid_costs else 200) * 1.15)
    ax_cost.set_xlabel('Optimizer eval', color='#888', fontsize=7)
    ax_cost.set_ylabel('Sym. Chamfer (px)', color='#888', fontsize=7)
    ax_cost.set_title('Cost convergence', color='white', fontsize=8)

    fig.suptitle(
        f"Pose Optimizer — Real Silhouette   "
        f"coarse: az={best['init_az']:.0f}°  el={best['init_el']:.0f}°  d={best['init_dist']:.1f} m   "
        f"refined: az={best['az']:.0f}°  el={best['el']:.0f}°  d={best['dist']:.1f} m  "
        f"roll={best['roll']:.0f}°",
        color='white', fontsize=9, y=0.975,
    )

    def update(fi):
        az, el, dist, roll, cost, mpx = frames[fi]
        global_idx = indices[fi]

        if len(mpx) > 0:
            scat_mod.set_offsets(mpx)
        else:
            scat_mod.set_offsets(np.empty((0, 2)))
        title_2d.set_text(
            f"Eval {global_idx+1}/{n_hist}   "
            f"az={az:.1f}°  el={el:.1f}°  roll={roll:.1f}°  dist={dist:.2f} m   "
            f"cost={cost:.2f} px")

        xyz = _az_el_to_unit(az, el)
        scat_cur._offsets3d = ([xyz[0]], [xyz[1]], [xyz[2]])
        title_3d.set_text(f"az={az:.1f}°  el={el:.1f}°")

        vline.set_xdata([global_idx, global_idx])
        dot_cost.set_offsets([[global_idx, cost]])
        return scat_mod, scat_cur, title_2d, title_3d, vline, dot_cost

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=max(40, 1000 // fps),
        blit=False,
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=90)
    plt.close(fig)
    print(f"  Convergence GIF: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_visualization(
    observed_px: np.ndarray,
    best: dict,
    img_w: int, img_h: int,
    target: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    out_path: Path,
) -> None:
    """Three-panel PNG: observed | best LUT match | optimised."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(18, 6))
        gs  = GridSpec(1, 3, figure=fig, wspace=0.04)

        uc   = observed_px[:, 0].astype(int)
        vc   = observed_px[:, 1].astype(int)
        valid_obs = (uc >= 0) & (uc < img_w) & (vc >= 0) & (vc < img_h)

        def render_model(az, el, dist, roll, mw) -> np.ndarray:
            cp   = target + spherical_to_cartesian_aircraft(az, el, dist)
            Rc   = compute_camera_transform(cp, target)
            mwR  = apply_roll_to_world(mw, target, roll)
            mpx  = project_world_to_pixels(mwR, cp, Rc, fx, fy, cx, cy)
            canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            if len(mpx) > 0:
                um = mpx[:, 0].astype(int); vm = mpx[:, 1].astype(int)
                ok = (um >= 0) & (um < img_w) & (vm >= 0) & (vm < img_h)
                canvas[vm[ok], um[ok]] = [255, 128, 0]          # orange = model
            canvas[vc[valid_obs], uc[valid_obs]] = [0, 200, 255]  # cyan  = observed
            return canvas

        mw = best["_model_world"]

        # Panel 1 — observed silhouette only
        ax1 = fig.add_subplot(gs[0])
        obs_gray = np.zeros((img_h, img_w), dtype=np.uint8)
        obs_gray[vc[valid_obs], uc[valid_obs]] = 255
        ax1.imshow(obs_gray, cmap="gray", origin="upper")
        ax1.set_title(f"Observed silhouette\n({len(observed_px):,} edge pixels)", fontsize=9)
        ax1.axis("off")

        # Panel 2 — best LUT coarse match (roll=0)
        ax2 = fig.add_subplot(gs[1])
        c2  = render_model(best["init_az"], best["init_el"], best["init_dist"], 0.0, mw)
        ax2.imshow(c2, origin="upper")
        ax2.set_title(
            f"Best LUT match  (coarse)\n"
            f"az={best['init_az']:.1f}°  el={best['init_el']:.1f}°  "
            f"d={best['init_dist']:.1f} m\n"
            f"cost={best['cost_coarse']:.2f} px  [cyan=obs, orange=model]",
            fontsize=8,
        )
        ax2.axis("off")

        # Panel 3 — Nelder-Mead refined
        ax3 = fig.add_subplot(gs[2])
        c3  = render_model(best["az"], best["el"], best["dist"], best["roll"], mw)
        ax3.imshow(c3, origin="upper")
        ax3.set_title(
            f"Nelder-Mead refined\n"
            f"az={best['az']:.1f}°  el={best['el']:.1f}°  "
            f"roll={best['roll']:.1f}°  d={best['dist']:.2f} m\n"
            f"cost={best['cost_final']:.3f} px  evals={best['n_evals']}",
            fontsize=8,
        )
        ax3.axis("off")

        plt.suptitle(
            "Pose Estimation — Real Silhouette (No Initial Guess)", fontsize=11, y=1.01
        )
        fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Visualisation: {out_path}")

    except Exception as exc:
        print(f"  Visualisation failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    _HERE  = Path(__file__).parent
    _DATA  = _HERE.parent.parent / "CAD-Edge-Generation" / "data"

    parser = argparse.ArgumentParser(
        description="Pose optimizer on real silhouette export (no initial guess)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--silhouette", required=True,
        help="NPZ exported by Edge Detection GUI",
    )
    parser.add_argument(
        "--lut", default=str(_DATA / "silhouette_lut_4x_final.npz"),
        help="Base LUT NPZ",
    )
    parser.add_argument(
        "--index", default=str(_DATA / "silhouette_lut_4x_final.lookup.json"),
        help="LUT lookup JSON",
    )
    # Camera intrinsics
    # Camera intrinsics — ZED 2i S/N 34754237, 4 mm lens @ HD720 (1280×720),
    # read via ZED SDK (tools/get_zed_intrinsics.py).  HFOV=67.8°  VFOV=40.2°
    parser.add_argument("--fx", type=float, default=951.1,
                        help="Focal length X [px]  (ZED 2i S/N 34754237 @ HD720)")
    parser.add_argument("--fy", type=float, default=951.1,
                        help="Focal length Y [px]  (ZED 2i S/N 34754237 @ HD720)")
    parser.add_argument("--cx", type=float, default=638.9,
                        help="Principal point X [px]")
    parser.add_argument("--cy", type=float, default=348.0,
                        help="Principal point Y [px]")
    # Search space
    parser.add_argument("--dist-min",   type=float, default=3.0,  help="Min distance [m] — keep ≥3 to avoid camera-inside-model")
    parser.add_argument("--dist-max",   type=float, default=15.0, help="Max distance [m]")
    parser.add_argument("--dist-steps", type=int,   default=7,    help="Number of distance candidates")
    parser.add_argument("--top-k",      type=int,   default=20,   help="Top-K candidates to refine")
    parser.add_argument(
        "--lut-ref-dist", type=float, default=5.0,
        help="Reference distance at which the LUT was built (see lookup JSON metadata.distance)",
    )
    parser.add_argument("--outdir", default="results_real", help="Output directory")
    parser.add_argument(
        "--visualize", action="store_true",
        help="Render animated GIF of the best candidate's Nelder-Mead convergence",
    )
    parser.add_argument("--gif-fps", type=int, default=12, help="GIF frame rate")

    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load exported silhouette ──────────────────────────────────────────────
    sil_path = Path(args.silhouette)
    if not sil_path.exists():
        print(f"ERROR: silhouette file not found: {sil_path}")
        return 1

    sil_data     = np.load(str(sil_path))
    observed_px  = sil_data["edge_pixels"].astype(np.float32)   # (N, 2) u,v
    image_size   = sil_data["image_size"]
    img_w, img_h = int(image_size[0]), int(image_size[1])

    # ── Centre-correct observed pixels ──────────────────────────────────────
    # The synthetic camera ALWAYS looks at the aircraft (aircraft → image centre).
    # In reality the camera looks forward and the aircraft appears off-centre.
    # We compensate by shifting observed pixels so their centroid aligns with
    # the principal point (cx, cy).  This is equivalent to assuming the camera
    # look-direction points at the detected bounding-box centre.
    bbox = sil_data["bbox"] if "bbox" in sil_data.files else None
    if bbox is not None and bbox.sum() > 0:
        bx1, by1, bx2, by2 = bbox
        bbox_cx = 0.5 * (bx1 + bx2)
        bbox_cy = 0.5 * (by1 + by2)
    else:
        bbox_cx = float(observed_px[:, 0].mean())
        bbox_cy = float(observed_px[:, 1].mean())

    # Offset applied to every observed pixel
    obs_offset = np.array([args.cx - bbox_cx, args.cy - bbox_cy], dtype=np.float32)
    observed_px = observed_px + obs_offset   # centred around (cx, cy)

    # Optional metadata sidecar
    meta_path  = sil_path.with_suffix(".json")
    source_info = ""
    if meta_path.exists():
        with open(meta_path) as f:
            meta_json = json.load(f)
        src_img    = Path(meta_json.get("source_image", "")).name
        algo       = meta_json.get("algorithm", "")
        source_info = f"{src_img}  [{algo}]"

    # ── Header ───────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  Pose Optimizer — Real Silhouette Mode (No Initial Guess)")
    print("=" * 64)
    print(f"  Silhouette : {sil_path.name}")
    if source_info:
        print(f"  Source     : {source_info}")
    print(f"  Edge pixels: {len(observed_px):,}")
    print(f"  Image size : {img_w} × {img_h}")
    print(f"  Bbox centre: ({bbox_cx:.0f}, {bbox_cy:.0f})  →  centred to ({args.cx:.0f}, {args.cy:.0f})  "
          f"[offset {obs_offset[0]:+.0f}, {obs_offset[1]:+.0f} px]")
    print(f"  Camera     : fx={args.fx}  fy={args.fy}  cx={args.cx}  cy={args.cy}")
    print(f"  Dist search: {args.dist_min}–{args.dist_max} m  ({args.dist_steps} steps)")
    print(f"  Top-K      : {args.top_k}")
    print()

    if len(observed_px) < 20:
        print(f"ERROR: too few edge pixels ({len(observed_px)}).  Need ≥ 20.")
        return 1

    # ── Load LUT ─────────────────────────────────────────────────────────────
    print(f"Loading LUT …")
    lut_npz = np.load(args.lut)
    with open(args.index) as f:
        angle_index_full = json.load(f)
    angle_index = angle_index_full.get("angle_index", angle_index_full)
    print(f"  {len(lut_npz.files):,} poses loaded")

    # Aircraft at world origin
    target = np.zeros(3, dtype=np.float32)

    # ── Distance candidates ───────────────────────────────────────────────────
    dist_candidates = np.linspace(args.dist_min, args.dist_max, args.dist_steps).tolist()
    print(f"  Distances  : {[round(d, 2) for d in dist_candidates]}")
    print()

    # ── Stage 1: Global coarse search ─────────────────────────────────────────
    print("Stage 1 — Global coarse search")
    t_coarse = time.perf_counter()
    candidates = global_coarse_search(
        observed_px, lut_npz, angle_index, target,
        dist_candidates=dist_candidates,
        lut_ref_dist=args.lut_ref_dist,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        verbose=True,
    )
    t_coarse = time.perf_counter() - t_coarse
    print()

    # ── Stage 1b: Validate top candidates with actual projection ─────────────
    # The scaling-trick coarse score is approximate: it doesn't account for
    # Z-clipping. Re-project the top 3×top_k candidates at their actual
    # distance, drop those with < 10 model pixels, recompute real cost.
    print("Stage 1b — Validating coarse candidates (actual projection) …", flush=True)
    validated: list[tuple] = []
    scan_limit = min(len(candidates), args.top_k * 5)
    for cost_scaled, az0, el0, dist0, mw in candidates[:scan_limit]:
        cam_pos_v = target + spherical_to_cartesian_aircraft(az0, el0, dist0)
        R_v       = compute_camera_transform(cam_pos_v, target)
        mpx_v     = project_world_to_pixels(mw, cam_pos_v, R_v,
                                            args.fx, args.fy, args.cx, args.cy)
        if len(mpx_v) < 10:
            continue
        # Recompute obs→model cost at actual distance and projection
        real_cost = chamfer_obs_to_model(observed_px, mpx_v)
        validated.append((real_cost, az0, el0, dist0, mw))
        if len(validated) >= args.top_k * 3:
            break
    validated.sort(key=lambda x: x[0])
    if not validated:
        print("  WARNING: no valid candidates after projection check — using raw coarse list")
        validated = candidates
    else:
        print(f"  {len(validated)} valid candidates  |  best: "
              f"az={validated[0][1]:.1f}°  el={validated[0][2]:.1f}°  "
              f"d={validated[0][3]:.1f} m  cost={validated[0][0]:.2f} px")
    print()
    # ── Stage 2: Nelder-Mead refinement ──────────────────────────────────────
    top_k = min(args.top_k, len(validated))
    print(f"Stage 2 — Refining top-{top_k} candidates")
    t_refine = time.perf_counter()

    refined: list[dict] = []
    for i, cand in enumerate(validated[:top_k]):
        # Only store per-eval model_px for the first candidate (cheapest to render)
        # Full store activated after we know the best candidate below.
        r = refine_candidate(
            observed_px, cand, target, args.fx, args.fy, args.cx, args.cy,
            store_frames=False,
            lut_npz=lut_npz, angle_index=angle_index,
            lut_ref_dist=args.lut_ref_dist,
        )
        refined.append(r)
        if (i + 1) % 5 == 0 or i == top_k - 1:
            print(
                f"  [{i+1:2d}/{top_k}]  az={r['az']:6.1f}°  el={r['el']:5.1f}°  "
                f"d={r['dist']:5.2f}m  roll={r['roll']:5.1f}°  "
                f"cost={r['cost_final']:.3f} px",
                flush=True,
            )

    t_refine = time.perf_counter() - t_refine
    refined.sort(key=lambda x: x["cost_final"])
    best = refined[0]
    print()

    # ── Re-run the best candidate with store_frames=True for GIF ─────────────
    if args.visualize:
        print("Re-running best candidate with frame capture for GIF …", flush=True)
        best_cand = (
            best["cost_coarse"],
            best["init_az"], best["init_el"], best["init_dist"],
            best["_model_world"],
        )
        best = refine_candidate(
            observed_px, best_cand, target,
            args.fx, args.fy, args.cx, args.cy,
            store_frames=True,
            lut_npz=lut_npz, angle_index=angle_index,
            lut_ref_dist=args.lut_ref_dist,
        )
        print(f"  GIF run: cost={best['cost_final']:.3f} px  evals={best['n_evals']}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 64)
    print("  BEST ESTIMATE")
    print("=" * 64)
    print(f"  az   = {best['az']:.2f}°")
    print(f"  el   = {best['el']:.2f}°")
    print(f"  dist = {best['dist']:.3f} m")
    print(f"  roll = {best['roll']:.2f}°")
    print(f"  cost = {best['cost_final']:.3f} px  (Chamfer model→observed)")
    print(f"  evals (refinement) = {best['n_evals']}")
    print()
    print(f"  Timing: coarse {t_coarse:.1f} s  |  "
          f"refinement {t_refine:.1f} s  |  "
          f"total {t_coarse + t_refine:.1f} s")
    print("=" * 64)

    # ── Save JSON result ───────────────────────────────────────────────────────
    _INTERNAL = {"_model_world", "_history", "_store_frames"}
    result_dict = {k: v for k, v in best.items() if k not in _INTERNAL}
    result_dict["top_results"] = [
        {k: v for k, v in r.items() if k not in _INTERNAL}
        for r in refined[:10]
    ]
    result_dict["source_silhouette"]   = str(sil_path)
    result_dict["image_size"]          = [img_w, img_h]
    result_dict["camera_intrinsics"]   = {
        "fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy
    }
    result_dict["dist_candidates"]     = [round(d, 3) for d in dist_candidates]
    result_dict["lut_ref_dist"]        = args.lut_ref_dist
    result_dict["timing_s"]            = {
        "coarse": round(t_coarse, 2), "refinement": round(t_refine, 2)
    }

    stem        = sil_path.stem
    result_path = out_dir / f"{stem}_result.json"
    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n  Results  : {result_path}")

    # ── Save visualisation PNG ───────────────────────────────────────────────
    viz_path = out_dir / f"{stem}_visualization.png"
    save_visualization(
        observed_px, best, img_w, img_h, target,
        args.fx, args.fy, args.cx, args.cy, viz_path,
    )

    # ── Render convergence GIF ───────────────────────────────────────────────
    if args.visualize:
        gif_path = out_dir / f"{stem}_convergence.gif"
        render_convergence_gif(
            best, observed_px, img_w, img_h, gif_path,
            max_frames=80, fps=args.gif_fps,
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
