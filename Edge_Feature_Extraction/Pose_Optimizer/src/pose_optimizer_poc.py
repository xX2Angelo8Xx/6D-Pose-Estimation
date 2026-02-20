#!/usr/bin/env python3
"""
Pose Optimizer — Proof of Concept
==================================

Demonstrates how the silhouette LUT feeds a continuous optimizer to recover
the exact 6-DoF (here: 3-DoF spherical) pose of a fixed-wing UAV from a
simulated camera image.

Scenario
--------
1. A "true" pose (az, el, dist) is chosen.
2. The silhouette is rendered at that exact pose and projected to 2-D pixels
   — these simulate the *detected edge pixels* coming from the real camera.
3. The LUT is queried with the true pose ± up to 15° of noise, giving a
   coarse warm-start template.
4. A gradient-free optimizer (Nelder-Mead via scipy) minimises the 2-D
   Chamfer distance between the projected model and the observed pixels.
5. Results: initial error, converged error, n_evals, wall-clock time.

Key geometry
------------
  LUT templates are stored in *camera space* of the grid pose.
  To re-use them during optimization at varying estimated poses,
  we convert to world space once:

      pts_world = R_stored.T @ pts_cam + cam_pos_stored

  Then at each optimizer step we project from world to the new estimated
  camera frame:

      pts_cam_est = R_est @ (pts_world - cam_pos_est)
      u = fx * X/Z + cx
      v = fy * Y/Z + cy

  Cost = mean one-way Chamfer  (observed → model)  in pixels.

Usage
-----
    python src/pose_optimizer_poc.py \\
        --cache  ../CAD-Edge-Generation/data/CAD_Ranger_simplified.edges.npz \\
        --lut    ../CAD-Edge-Generation/data/silhouette_lut_4x_final.npz \\
        --index  ../CAD-Edge-Generation/data/silhouette_lut_4x_final.lookup.json \\
        [--noise 15] [--runs 8] [--seed 42] [--outdir results]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# ── import silhouette helpers from sibling project ──────────────────────────
_SRC = Path(__file__).parent.parent.parent / 'CAD-Edge-Generation' / 'src'
sys.path.insert(0, str(_SRC))

from extract_silhouette import (
    load_cache,
    extract_silhouette_contour,
    spherical_to_cartesian_aircraft,
    compute_camera_transform,
)
from precompute_visibility import nearest_pose_key


# ─────────────────────────────────────────────────────────────────────────────
# Camera / projection helpers
# ─────────────────────────────────────────────────────────────────────────────

FX, FY = 800.0, 800.0
CX, CY = 640.0, 360.0
IMG_W,  IMG_H  = 1280, 720


def project_world_to_pixels(pts_world: np.ndarray,
                             cam_pos: np.ndarray,
                             R_cam: np.ndarray) -> np.ndarray:
    """Project N world-space points to (u, v) pixel coords.

    Returns only points with Z > 0.01 m (in front of camera).
    Shape: (K, 2) float32.
    """
    pts_cam = (R_cam @ (pts_world - cam_pos).T).T          # (N, 3)
    valid   = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return np.empty((0, 2), np.float32)
    z   = pts_cam[:, 2]
    uv  = np.empty((len(pts_cam), 2), np.float32)
    uv[:, 0] = FX * pts_cam[:, 0] / z + CX
    uv[:, 1] = FY * pts_cam[:, 1] / z + CY
    return uv


def cam_space_to_world(pts_cam: np.ndarray,
                       R_cam: np.ndarray,
                       cam_pos: np.ndarray) -> np.ndarray:
    """Inverse: camera-space → world-space.  pts_cam shape (N, 3)."""
    return (R_cam.T @ pts_cam.T).T + cam_pos


def apply_roll_to_world(pts_world: np.ndarray,
                        target: np.ndarray,
                        roll_deg: float) -> np.ndarray:
    """Rotate points around the fuselage +X axis through `target` by `roll_deg` degrees.

    This is a rigid rotation in the model's local frame and returns new world points.
    """
    r = float(np.radians(roll_deg))
    cr, sr = float(np.cos(r)), float(np.sin(r))
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr,  cr],
    ], dtype=np.float32)
    return (Rx @ (pts_world - target).T).T + target


def chamfer_obs_to_model(obs_px: np.ndarray,
                         model_px: np.ndarray) -> float:
    """Mean nearest-neighbour distance obs → model (pixels)."""
    if len(obs_px) == 0 or len(model_px) == 0:
        return 1e6
    d, _ = cKDTree(model_px).query(obs_px, workers=1)
    return float(d.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer cost function
# ─────────────────────────────────────────────────────────────────────────────

class PoseCostFunction:
    """Closure over the fixed observed pixels and model world points.

        When store_frames=True every call also saves a copy of model_px so that
        the convergence animation can replay every evaluation step frame-by-frame.
        History tuples:
            store_frames=False : (az, el, dist, roll, cost)
            store_frames=True  : (az, el, dist, roll, cost, model_px_copy)
    """

    def __init__(self,
                 observed_px: np.ndarray,
                 model_world: np.ndarray,
                 target: np.ndarray,
                 store_frames: bool = False):
        self.observed_px   = observed_px
        self.model_world   = model_world
        self.target        = target
        self.store_frames  = store_frames
        self.n_evals       = 0
        self.history: list[tuple] = []   # see docstring for tuple layout

    def __call__(self, params: np.ndarray) -> float:
        az, el, dist, roll = params

        # Clamp to valid ranges
        az   = az % 360.0
        el   = float(np.clip(el, -89.0, 89.0))
        dist = float(np.clip(dist, 0.5, 50.0))

        cam_pos = self.target + spherical_to_cartesian_aircraft(az, el, dist)
        R_cam   = compute_camera_transform(cam_pos, self.target)

        # apply roll to the fixed world model before projecting
        model_world_rolled = apply_roll_to_world(self.model_world, self.target, roll)
        model_px = project_world_to_pixels(model_world_rolled, cam_pos, R_cam)
        cost     = chamfer_obs_to_model(self.observed_px, model_px)

        self.n_evals += 1
        if self.store_frames:
            self.history.append((az, el, dist, roll, cost, model_px.copy()))
        else:
            self.history.append((az, el, dist, roll, cost))
        return cost


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_single(V, F, target,
               lut_npz, angle_index,
               true_az, true_el, true_dist, true_roll,
               roll_lut_npz, roll_lut_index,
               noise_deg: float,
               rng: np.random.Generator,
               label: str = "",
               visualize: bool = False) -> dict:
    """
    1. Render observed silhouette at true pose → 2-D pixels.
    2. Query LUT with noisy estimate → warm-start model in world space.
    3. Optimize (az, el, dist, roll) with Nelder-Mead.
    4. Return metrics dict.
    """

    # ── 1. ground-truth "image" ───────────────────────────────────────────
    # Apply true roll to mesh, then render
    V_rolled_true = apply_roll_to_world(V, target, true_roll)
    cam_pos_true = target + spherical_to_cartesian_aircraft(
        true_az, true_el, true_dist)
    R_true   = compute_camera_transform(cam_pos_true, target)

    pts_cam_true = extract_silhouette_contour(
        V_rolled_true, F, cam_pos_true, target, supersample=1)   # 1× = fast
    observed_px  = project_world_to_pixels(
        cam_space_to_world(pts_cam_true, R_true, cam_pos_true),
        cam_pos_true, R_true)

    n_observed = len(observed_px)

    # ── 2. noisy initial estimate + LUT warm start ────────────────────────
    noise_az = rng.uniform(-noise_deg, noise_deg)
    noise_el = rng.uniform(-noise_deg, noise_deg)
    # distance noise: ±20 % or ±noise_deg/15 * 1 metre
    noise_dist = rng.uniform(-true_dist * 0.15, true_dist * 0.15)

    init_az   = (true_az  + noise_az)  % 360.0
    init_el   = float(np.clip(true_el  + noise_el,  -30.0, 89.0))
    init_dist = float(np.clip(true_dist + noise_dist, 1.0, 20.0))

    pkey = nearest_pose_key(angle_index,
                            init_az, init_el,
                            2.0, 2.0, 4.0, 2.0)
    if pkey is None:
        return None

    # LUT template: camera space of stored pose → world space (fixed model)
    lut_meta_az   = angle_index   # we'll decode from pkey
    # retrieve stored pose angles from pkey index (parse from angle_index reverse)
    pts_lut_cam   = lut_npz[pkey].astype(np.float32)
    # stored camera position for this LUT entry
    # derive from the snapped angles encoded in pkey via nearest_pose_key internals:
    # re-snap to get the exact stored angles
    from precompute_visibility import is_rear_hemisphere, _angle_key
    az_q   = init_az
    az_stp = 2.0 if is_rear_hemisphere(az_q) else 4.0
    el_stp = 2.0
    stored_az   = round(round(az_q   / az_stp) * az_stp % 360.0, 4)
    stored_el   = float(np.clip(round(init_el / el_stp) * el_stp, -30.0, 90.0))
    cam_pos_lut = target + spherical_to_cartesian_aircraft(
        stored_az, stored_el, true_dist)       # use true dist for LUT pose
    R_lut       = compute_camera_transform(cam_pos_lut, target)
    model_world = cam_space_to_world(pts_lut_cam, R_lut, cam_pos_lut)

    # ── Roll warm-start from roll LUT (if available) ──────────────────────
    init_roll = 0.0
    if roll_lut_npz is not None and roll_lut_index is not None:
        # Query roll LUT at the snapped (az, el)
        az_el_key = f"{stored_az:.4f}_{stored_el:.4f}"
        roll_slices = roll_lut_index.get("angle_index_with_roll", {}).get(az_el_key)
        if roll_slices:
            # roll_slices is [(roll_deg, pose_key), ...]
            # Pick closest roll from the available slices
            best_roll, best_key = min(roll_slices, key=lambda x: abs(x[0]))
            init_roll = best_roll

    # projected model at initial estimate (before any optimization)
    cam_pos_init = target + spherical_to_cartesian_aircraft(
        init_az, init_el, init_dist)
    R_init = compute_camera_transform(cam_pos_init, target)
    model_px_init = project_world_to_pixels(model_world, cam_pos_init, R_init)
    cost_init = chamfer_obs_to_model(observed_px, model_px_init)

    # angular error of initial estimate
    def angular_error(az1, el1, az2, el2):
        """Great-circle angle between two (az, el) directions (degrees)."""
        def to_cart(az, el):
            az_r, el_r = np.radians(az), np.radians(el)
            return np.array([
                np.cos(el_r) * np.cos(az_r),
                np.cos(el_r) * np.sin(az_r),
                np.sin(el_r)])
        v1, v2 = to_cart(az1, el1), to_cart(az2, el2)
        return float(np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1))))

    err_init_deg = angular_error(init_az, init_el, true_az, true_el)

    # ── 3. Nelder-Mead optimization ───────────────────────────────────────
    cost_fn = PoseCostFunction(observed_px, model_world, target,
                               store_frames=visualize)

    # include roll as a 4th variable (degrees). Warm-start from roll LUT or 0.
    x0 = np.array([init_az, init_el, init_dist, init_roll])

    # Initial simplex: perturb each dimension by ~1/3 of noise
    delta = np.array([noise_deg * 0.5, noise_deg * 0.5, true_dist * 0.1, noise_deg * 0.5])
    initial_simplex = np.vstack([x0, x0 + np.diag(delta)])

    t_start = time.perf_counter()
    result  = minimize(
        cost_fn,
        x0,
        method='Nelder-Mead',
        options=dict(
            xatol=0.05,          # convergence tolerance in degrees / metres
            fatol=0.3,           # convergence in pixels (Chamfer)
            maxiter=2000,
            initial_simplex=initial_simplex,
        ),
    )
    t_elapsed = (time.perf_counter() - t_start) * 1000.0

    opt_az, opt_el, opt_dist, opt_roll = result.x
    opt_az  = opt_az % 360.0
    cost_final = result.fun

    err_final_deg  = angular_error(opt_az, opt_el, true_az, true_el)
    err_dist_m     = abs(opt_dist - true_dist)
    # Roll error in [-180, 180] accounting for wrap
    err_roll_deg   = float(abs((opt_roll - true_roll + 180.0) % 360.0 - 180.0))

    return dict(
        label       = label,
        true_az=true_az,    true_el=true_el,    true_dist=true_dist,    true_roll=true_roll,
        noise_az=round(noise_az, 2),
        noise_el=round(noise_el, 2),
        init_az=round(init_az, 2),
        init_el=round(init_el, 2),
        init_dist=round(init_dist, 3),
        init_roll=round(init_roll, 2),
        opt_az=round(opt_az, 3),
        opt_el=round(opt_el, 3),
        opt_dist=round(opt_dist, 3),
        opt_roll=round(float(opt_roll), 3),
        err_init_deg=round(err_init_deg, 2),
        err_final_deg=round(err_final_deg, 3),
        err_roll_deg=round(err_roll_deg, 3),
        err_dist_m=round(err_dist_m, 4),
        cost_init=round(cost_init, 2),
        cost_final=round(cost_final, 3),
        n_evals=cost_fn.n_evals,
        time_ms=round(t_elapsed, 1),
        n_observed=n_observed,
        converged=result.success,
        history=cost_fn.history,
        # visualizer payloads (only populated when visualize=True)
        observed_px=observed_px if visualize else None,
        model_world=model_world if visualize else None,
        viz_target=target       if visualize else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-run animated convergence video
# ─────────────────────────────────────────────────────────────────────────────

def _az_el_to_unit(az_deg, el_deg):
    """Spherical (az, el) → unit cartesian for 3-D sphere plot."""
    az_r = np.radians(az_deg)
    el_r = np.radians(el_deg)
    return np.array([
        np.cos(el_r) * np.cos(az_r),
        np.cos(el_r) * np.sin(az_r),
        np.sin(el_r),
    ])


def render_convergence_video(result: dict, out_dir: Path,
                              max_frames: int = 80,
                              fps: int = 12) -> Path:
    """Render an animated GIF showing the optimizer converging step-by-step.

    Layout (dark background, 16×9)
    ┌──────────────────────────┬──────────────────┐
    │  2-D silhouette overlay  │  3-D sphere traj │
    │  blue = observed         │                  │
    │  orange = model@current  ├──────────────────┤
    │                          │  cost curve      │
    └──────────────────────────┴──────────────────┘

    Rendering time is NOT part of any latency measurement — this function
    is called after timing has already been recorded in `run_single`.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

    hist       = result['history']   # (az, el, dist, roll, cost, model_px)
    obs_px     = result['observed_px']    # (K, 2)
    true_az    = result['true_az']
    true_el    = result['true_el']
    label      = result['label']

    if obs_px is None or len(hist) == 0 or len(hist[0]) < 6:
        print(f"  ⚠  render_convergence_video: no frame data for '{label}' "
              f"— run with --visualize")
        return None

    n_hist = len(hist)

    # Sub-sample to at most max_frames while always keeping first + last
    if n_hist <= max_frames:
        indices = list(range(n_hist))
    else:
        indices = sorted(set(
            [0] +
            list(np.linspace(1, n_hist - 2, max_frames - 2, dtype=int)) +
            [n_hist - 1]
        ))

    frames   = [hist[i] for i in indices]   # each: (az, el, dist, roll, cost, model_px)
    costs_all = [h[4] for h in hist]        # full cost trace for background curve
    evals_all = np.arange(n_hist)

    # Pre-compute sphere trajectory coords for all history points
    traj_xyz = np.array([_az_el_to_unit(h[0], h[1]) for h in hist])  # (N, 3)
    true_xyz = _az_el_to_unit(true_az, true_el)

    # ── figure layout ──────────────────────────────────────────────────
    BG, PAD_C = '#0f0f1a', '#1a1a2e'
    fig = plt.figure(figsize=(16, 9), dpi=90, facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.5, 1],
        height_ratios=[1.4, 1],
        hspace=0.35, wspace=0.28,
        left=0.04, right=0.97, top=0.92, bottom=0.06,
    )
    ax_2d   = fig.add_subplot(gs[:, 0])          # left: full height
    ax_3d   = fig.add_subplot(gs[0, 1], projection='3d')
    ax_cost = fig.add_subplot(gs[1, 1])

    for ax in (ax_2d, ax_cost):
        ax.set_facecolor(PAD_C)
        ax.tick_params(colors='#888', labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor('#333')
    ax_3d.set_facecolor(BG)
    ax_3d.xaxis.pane.fill = ax_3d.yaxis.pane.fill = ax_3d.zaxis.pane.fill = False

    # Fixed elements ────────────────────────────────────────────────────
    # 2-D: observed pixels (static blue)
    scat_obs = ax_2d.scatter(
        obs_px[:, 0], obs_px[:, 1],
        s=0.8, c='#4da6ff', alpha=0.6, linewidths=0, label='observed')
    # 2-D: model pixels (animated orange) — start with frame 0
    scat_mod = ax_2d.scatter(
        frames[0][5][:, 0] if len(frames[0][5]) else [[]],
        frames[0][5][:, 1] if len(frames[0][5]) else [[]],
        s=0.8, c='#ff7c3a', alpha=0.75, linewidths=0, label='model')
    ax_2d.set_xlim(0, IMG_W)
    ax_2d.set_ylim(IMG_H, 0)           # image coords: y flipped
    ax_2d.set_aspect('equal')
    ax_2d.legend(loc='lower left', fontsize=7, labelcolor='white',
                 facecolor='#111', framealpha=0.6, markerscale=4)
    title_2d = ax_2d.set_title('', color='white', fontsize=9, pad=6)

    # 3-D: wireframe hemisphere
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(-np.pi / 6, np.pi / 2, 18)
    xs = np.outer(np.cos(v), np.cos(u))
    ys = np.outer(np.cos(v), np.sin(u))
    zs = np.outer(np.sin(v), np.ones_like(u))
    ax_3d.plot_wireframe(xs, ys, zs, color='#333', lw=0.3, alpha=0.4)
    # Full trajectory (faded)
    ax_3d.scatter(traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2],
                  s=4, c='#555', alpha=0.35, depthshade=False)
    # True pose (fixed gold star)
    ax_3d.scatter(*true_xyz, s=120, c='gold', marker='*', zorder=10,
                  label='true pose', depthshade=False)
    # Current pose (animated red dot)
    cur0 = _az_el_to_unit(frames[0][0], frames[0][1])
    scat_cur = ax_3d.scatter(*cur0, s=60, c='#ff4040', zorder=11,
                              depthshade=False)
    ax_3d.set_xlim(-1.1, 1.1); ax_3d.set_ylim(-1.1, 1.1); ax_3d.set_zlim(-0.2, 1.15)
    ax_3d.set_xlabel('X', color='#888', fontsize=7, labelpad=1)
    ax_3d.set_ylabel('Y', color='#888', fontsize=7, labelpad=1)
    ax_3d.set_zlabel('Z', color='#888', fontsize=7, labelpad=1)
    ax_3d.tick_params(colors='#555', labelsize=6)
    ax_3d.legend(fontsize=7, labelcolor='white', facecolor='#111',
                 framealpha=0.5, loc='upper left')
    title_3d = ax_3d.set_title('', color='white', fontsize=8)

    # Cost curve
    ax_cost.plot(evals_all, costs_all, color='#555', lw=0.8, alpha=0.6)
    vline = ax_cost.axvline(x=0, color='#ff7c3a', lw=1.2, alpha=0.9)
    dot_cost = ax_cost.scatter([0], [costs_all[0]], s=25, c='#ff7c3a',
                                zorder=5)
    ax_cost.set_xlim(-1, n_hist)
    ax_cost.set_ylim(0, max(costs_all) * 1.15)
    ax_cost.set_xlabel('Optimizer eval', color='#888', fontsize=7)
    ax_cost.set_ylabel('Chamfer (px)',   color='#888', fontsize=7)
    ax_cost.set_title('Cost convergence', color='white', fontsize=8)

    fig.suptitle(
        f"Pose Optimizer — '{label}'   "
        f"true ({true_az:.1f}°, {true_el:.1f}°)   "
        f"noise (±{result['noise_az']:.1f}°, ±{result['noise_el']:.1f}°)",
        color='white', fontsize=10, y=0.975)

    # ── animation update ───────────────────────────────────────────────
    def _ang_err(az, el):
        v1 = _az_el_to_unit(az, el)
        v2 = true_xyz
        return float(np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1))))

    def update(fi):
        frame         = frames[fi]
        az, el, dist, roll, cost, mpx = frame
        global_idx    = indices[fi]   # position in full history

        # 2-D overlay
        if len(mpx) > 0:
            scat_mod.set_offsets(mpx)
        else:
            scat_mod.set_offsets(np.empty((0, 2)))
        ang = _ang_err(az, el)
        title_2d.set_text(
            f"Eval {global_idx+1}/{n_hist}   "
            f"az={az:.1f}°  el={el:.1f}°  roll={roll:.1f}°  dist={dist:.2f}m   "
            f"cost={cost:.2f} px   ang_err={ang:.2f}°")

        # 3-D sphere
        xyz = _az_el_to_unit(az, el)
        scat_cur._offsets3d = ([xyz[0]], [xyz[1]], [xyz[2]])
        title_3d.set_text(f"az={az:.1f}°  el={el:.1f}°")

        # Cost curve
        vline.set_xdata([global_idx, global_idx])
        dot_cost.set_offsets([[global_idx, cost]])

        return scat_mod, scat_cur, title_2d, title_3d, vline, dot_cost

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=max(40, 1000 // fps),
        blit=False,
    )

    out_path = out_dir / f"{label}_convergence.gif"
    writer   = animation.PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=90)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Convergence plot
# ─────────────────────────────────────────────────────────────────────────────

def save_convergence_plots(results: list, out_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpi = 100
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')

    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(results)))

    for ax in axes:
        ax.set_facecolor('#0f0f1a')
        ax.tick_params(colors='#aaa', labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor('#333')

    for r, col in zip(results, colors):
        hist  = r['history']
        iters = np.arange(len(hist))
        costs = [h[4] for h in hist]

        # angular error at each eval
        def ae(az, el):
            def tc(a, e):
                ar, er = np.radians(a), np.radians(e)
                return np.array([np.cos(er)*np.cos(ar),
                                 np.cos(er)*np.sin(ar), np.sin(er)])
            v1, v2 = tc(az, el), tc(r['true_az'], r['true_el'])
            return float(np.degrees(np.arccos(np.clip(v1@v2, -1, 1))))

        ang_errs = [ae(h[0], h[1]) for h in hist]

        axes[0].plot(iters, costs,     color=col, lw=1.2, alpha=0.85,
                     label=r['label'])
        axes[1].plot(iters, ang_errs,  color=col, lw=1.2, alpha=0.85)

    axes[0].set_title('Chamfer cost (pixels)',  color='white', fontsize=10)
    axes[1].set_title('Angular error (°)',       color='white', fontsize=10)
    axes[0].set_xlabel('Optimizer evaluations',  color='#ccc',  fontsize=8)
    axes[1].set_xlabel('Optimizer evaluations',  color='#ccc',  fontsize=8)
    axes[0].set_ylabel('pixels', color='#ccc', fontsize=8)
    axes[1].set_ylabel('degrees',color='#ccc', fontsize=8)
    axes[0].legend(fontsize=6.5, labelcolor='white',
                   facecolor='#111', framealpha=0.5)

    # bar: time per run
    labels = [r['label'] for r in results]
    times  = [r['time_ms'] for r in results]
    x = np.arange(len(results))
    bars = axes[2].bar(x, times, color=colors, alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=40, ha='right',
                             fontsize=7, color='#ccc')
    axes[2].set_title('Optimization wall time (ms)', color='white', fontsize=10)
    axes[2].set_ylabel('ms', color='#ccc', fontsize=8)
    for bar, r in zip(bars, results):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 2,
                     f"{r['err_final_deg']:.2f}°",
                     ha='center', va='bottom', fontsize=7, color='white')

    fig.suptitle('Pose Optimizer PoC — Nelder-Mead on 2-D Chamfer',
                 color='white', fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = out_dir / 'convergence.png'
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# Diverse test poses — spread across hemisphere, on and off grid deliberately
# Now including roll to test 4-DOF recovery
DEFAULT_POSES = [
    # label                  az      el    dist   roll
    ("rear_level",          0.0,    0.0,   5.0,    0.0),
    ("rear_offgrid_1",     93.5,   17.3,   5.0,  -12.0),
    ("rear_offgrid_2",    181.4,   33.7,   5.0,   +8.0),
    ("rear_high",         135.0,   62.0,   5.0,   +5.0),
    ("left_side",         270.0,    5.2,   5.0,  -15.0),
    ("right_diag",         67.8,   28.9,   5.0,  +18.0),
    ("front_low",         340.5,  -12.0,   5.0,   -6.0),
    ("top_down",            0.0,   85.0,   5.0,    0.0),
]


def main():
    ap = argparse.ArgumentParser(
        description="6-DoF pose optimizer PoC using silhouette LUT warm start")
    ap.add_argument('--cache', required=True, help='edges.npz mesh cache')
    ap.add_argument('--lut',   required=True, help='LUT NPZ file')
    ap.add_argument('--index', required=True, help='lookup.json file')
    ap.add_argument('--roll-lut',   default=None, help='Roll LUT NPZ (optional)')
    ap.add_argument('--roll-index', default=None, help='Roll LUT lookup.json (optional)')
    ap.add_argument('--outdir',   default='results')
    ap.add_argument('--noise',    type=float, default=15.0,
                    help='Max angular noise ±° applied to true pose')
    ap.add_argument('--seed',     type=int,   default=42)
    ap.add_argument('--runs',     type=int,   default=len(DEFAULT_POSES),
                    help='How many of the default poses to run')
    ap.add_argument('--visualize', action='store_true',
                    help='Render per-pose animated GIF showing step-by-step convergence')
    ap.add_argument('--gif-fps',  type=int,   default=12,
                    help='Frame-rate for the convergence GIFs (default 12)')
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── load assets ───────────────────────────────────────────────────────
    print("Loading mesh …")
    V, F, _, _ = load_cache(Path(args.cache))
    if np.max(V.max(0) - V.min(0)) > 100:
        V = V * 0.001
    target = V.mean(axis=0)
    print(f"  {len(V):,} verts  {len(F):,} faces  target={target.round(3)}")

    print("Warming up JIT …")
    _cp = target + spherical_to_cartesian_aircraft(0, 0, 5)
    extract_silhouette_contour(V, F, _cp, target, supersample=1)
    print("  ✓")

    print(f"Loading LUT …  {args.lut}")
    lut_npz = np.load(args.lut)
    with open(args.index) as f:
        lut_json = json.load(f)
    angle_index = lut_json['angle_index']
    print(f"  {len(angle_index):,} poses loaded")

    # Load roll LUT if provided
    roll_lut_npz = None
    roll_lut_index = None
    if args.roll_lut and args.roll_index:
        print(f"Loading roll LUT …  {args.roll_lut}")
        roll_lut_npz = np.load(args.roll_lut)
        with open(args.roll_index) as f:
            roll_lut_index = json.load(f)
        n_roll = roll_lut_index['metadata']['n_total_poses']
        print(f"  {n_roll:,} roll-extended poses loaded")
    print()

    # ── run ───────────────────────────────────────────────────────────────
    sep  = "═" * 100
    sep2 = "─" * 100
    print(sep)
    print(f"  POSE OPTIMIZER PoC   noise=±{args.noise}°   seed={args.seed}")
    print(sep)
    print(f"{'Pose':<22} {'true az/el/roll':>17} "
          f"{'noise az/el':>12} "
          f"{'init err°':>10} "
          f"{'final err°':>11} "
          f"{'roll err°':>10} "
          f"{'Δdist m':>8} "
          f"{'px_init':>8} {'px_fin':>7} "
          f"{'evals':>6} "
          f"{'ms':>7}  conv")
    print(sep2)

    all_results = []
    for label, az, el, dist, roll in DEFAULT_POSES[:args.runs]:
        r = run_single(V, F, target,
                       lut_npz, angle_index,
                       az, el, dist, roll,
                       roll_lut_npz, roll_lut_index,
                       noise_deg=args.noise,
                       rng=rng,
                       label=label,
                       visualize=args.visualize)
        if r is None:
            print(f"  ⚠  {label}: LUT miss — skipped")
            continue

        conv = "✓" if r['converged'] else "⚠"
        print(f"  {r['label']:<20} "
              f"  {r['true_az']:6.1f}/{r['true_el']:5.1f}/{r['true_roll']:+5.1f}"
              f"  {r['noise_az']:+6.1f}/{r['noise_el']:+5.1f}"
              f"  {r['err_init_deg']:>10.2f}"
              f"  {r['err_final_deg']:>11.3f}"
              f"  {r['err_roll_deg']:>10.3f}"
              f"  {r['err_dist_m']:>8.4f}"
              f"  {r['cost_init']:>8.1f}"
              f"  {r['cost_final']:>7.2f}"
              f"  {r['n_evals']:>6}"
              f"  {r['time_ms']:>7.1f}  {conv}")
        all_results.append(r)

    print(sep2)

    if not all_results:
        print("No results.")
        return

    # ── aggregate ─────────────────────────────────────────────────────────
    init_errs  = [r['err_init_deg']  for r in all_results]
    final_errs = [r['err_final_deg'] for r in all_results]
    roll_errs  = [r['err_roll_deg']  for r in all_results]
    dist_errs  = [r['err_dist_m']    for r in all_results]
    times_ms   = [r['time_ms']       for r in all_results]
    evals      = [r['n_evals']       for r in all_results]
    px_init    = [r['cost_init']     for r in all_results]
    px_final   = [r['cost_final']    for r in all_results]

    print(f"\n{'SUMMARY':{'═'}<{100}}")
    print(f"  Runs                : {len(all_results)}")
    print(f"  Initial angular err : mean {np.mean(init_errs):.2f}°  "
          f"max {np.max(init_errs):.2f}°   (LUT warm start)")
    print(f"  Final  angular err  : mean {np.mean(final_errs):.3f}°  "
          f"max {np.max(final_errs):.3f}°  "
          f"min {np.min(final_errs):.3f}°")
    print(f"  Final  roll error   : mean {np.mean(roll_errs):.3f}°  "
          f"max {np.max(roll_errs):.3f}°  "
          f"min {np.min(roll_errs):.3f}°")
    print(f"  Chamfer init→final  : {np.mean(px_init):.1f} px  →  "
          f"{np.mean(px_final):.2f} px  (mean)")
    print(f"  Distance error      : mean {np.mean(dist_errs)*100:.1f} cm  "
          f"max {np.max(dist_errs)*100:.1f} cm")
    print(f"  Optimizer evals     : mean {np.mean(evals):.0f}  "
          f"max {np.max(evals):.0f}")
    print(f"  Wall time           : mean {np.mean(times_ms):.1f} ms  "
          f"total {np.sum(times_ms):.0f} ms")
    print(f"  Converged           : "
          f"{sum(r['converged'] for r in all_results)}/{len(all_results)}")

    # ── save results JSON ─────────────────────────────────────────────────
    _SKIP = {'history', 'observed_px', 'model_world', 'viz_target'}
    json_out = out_dir / 'results.json'
    safe = [{k: v for k, v in r.items() if k not in _SKIP}
            for r in all_results]
    with open(json_out, 'w') as f:
        json.dump(safe, f, indent=2)
    print(f"\n  Results JSON → {json_out}")

    # ── convergence plots ─────────────────────────────────────────────────
    plot_path = save_convergence_plots(all_results, out_dir)
    print(f"  Convergence plot  → {plot_path}")

    # ── per-pose animated GIFs ────────────────────────────────────────────
    if args.visualize:
        print("\n  Rendering convergence GIFs (excluded from timing) …")
        for r in all_results:
            gif_path = render_convergence_video(
                r, out_dir, max_frames=80, fps=args.gif_fps)
            if gif_path:
                print(f"    {r['label']:<22} → {gif_path}")

    print("═" * 100)


if __name__ == '__main__':
    main()
