#!/usr/bin/env python3
"""
Pose Optimizer GUI
==================
Interactive PySide6 application for 6-DOF pose estimation from silhouette NPZ
files.

Search modes (determined automatically from filled fields):
  ● az + el + dist + roll  →  direct single Nelder-Mead refinement
  ● az + el + dist         →  coarse roll search   (e.g. every 18°) + refine
  ● az + el                →  dist × roll grid search + refine
  ●  (nothing)             →  full LUT coarse search  (Stage 1 + Stage 2)

Live display: every optimizer evaluation updates the 2-D overlay in real-time.
After completion: GIF player with frame-by-frame animation + play/pause/slider.
"""

from __future__ import annotations

import sys
import time
import json
import traceback
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# ── PySide6 ───────────────────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QLabel, QPushButton, QLineEdit, QFileDialog,
    QTextEdit, QSlider, QSplitter, QGroupBox, QDoubleSpinBox,
    QSpinBox, QFrame, QSizePolicy, QScrollArea, QCheckBox,
    QProgressBar,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QMutex, QMutexLocker,
)
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QPalette, QColor, QIntValidator,
)

# ── import shared helpers ─────────────────────────────────────────────────────
_HERE  = Path(__file__).parent
_CAD   = _HERE.parent.parent / "CAD-Edge-Generation" / "src"

for _p in (_HERE, _CAD):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from extract_silhouette import (
    spherical_to_cartesian_aircraft,
    compute_camera_transform,
)
from run_on_real_silhouette import (
    project_world_to_pixels,
    cam_space_to_world,
    apply_roll_to_world,
    chamfer_obs_to_model,
    chamfer_symmetric,
    global_coarse_search,
    refine_candidate,
    PoseCostFunctionReal,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme palette
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK  = "#0f0f1a"
BG_MID   = "#1a1a2e"
BG_PANEL = "#16213e"
ACCENT   = "#00e5ff"
ORANGE   = "#ff7c3a"
GREEN    = "#39ff14"
TEXT     = "#e0e0e0"
MUTED    = "#666"


DARK_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {BG_MID};
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    margin-top: 6px;
    padding: 8px 6px 8px 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    color: {ACCENT};
    font-weight: bold;
    font-size: 12px;
}}
QLabel {{
    color: {TEXT};
    background: transparent;
}}
QLabel#hint {{
    color: {MUTED};
    font-size: 11px;
}}
QLineEdit, QDoubleSpinBox, QSpinBox {{
    background-color: #0d0d1a;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 3px 6px;
    color: {TEXT};
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QPushButton {{
    background-color: #1e1e3a;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 5px 12px;
    color: {TEXT};
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: #2a2a5a;
    border: 1px solid {ACCENT};
}}
QPushButton:pressed {{
    background-color: #111130;
}}
QPushButton#run_btn {{
    background-color: #0a3a2a;
    border: 1px solid {GREEN};
    color: {GREEN};
    font-weight: bold;
    font-size: 14px;
    padding: 8px 16px;
}}
QPushButton#run_btn:hover {{
    background-color: #0e5a3a;
}}
QPushButton#run_btn:disabled {{
    background-color: #1a1a2e;
    border: 1px solid #333;
    color: #444;
}}
QPushButton#stop_btn {{
    background-color: #3a0a0a;
    border: 1px solid #cc3333;
    color: #ff5555;
    font-weight: bold;
}}
QPushButton#stop_btn:hover {{
    background-color: #5a1010;
}}
QTextEdit {{
    background-color: #0a0a12;
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    color: #aabbcc;
    font-family: "Cascadia Code", "Fira Mono", monospace;
    font-size: 11px;
}}
QScrollBar:vertical {{
    background: #0f0f1a;
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: #3a3a5a;
    border-radius: 4px;
    min-height: 20px;
}}
QSlider::groove:horizontal {{
    background: #2a2a4a;
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QProgressBar {{
    background-color: #1a1a2e;
    border: 1px solid #333;
    border-radius: 3px;
    text-align: center;
    color: {TEXT};
    height: 10px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_overlay(
    observed_px: np.ndarray,
    model_px: Optional[np.ndarray],
    img_w: int,
    img_h: int,
) -> QImage:
    """Render a dark-background QImage with cyan observed and orange model pixels."""
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    if len(observed_px) > 0:
        u = observed_px[:, 0].astype(int)
        v = observed_px[:, 1].astype(int)
        mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        canvas[v[mask], u[mask]] = [0, 200, 255]          # cyan

    if model_px is not None and len(model_px) > 0:
        u = model_px[:, 0].astype(int)
        v = model_px[:, 1].astype(int)
        mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        canvas[v[mask], u[mask]] = [255, 128, 0]           # orange

    h, w, ch = canvas.shape
    return QImage(canvas.tobytes(), w, h, w * ch, QImage.Format.Format_RGB888)


def qimage_to_pixmap_scaled(img: QImage, max_w: int, max_h: int) -> QPixmap:
    pm = QPixmap.fromImage(img)
    return pm.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer worker (background QThread)
# ─────────────────────────────────────────────────────────────────────────────

class OptimizerWorker(QThread):
    """Runs pose estimation in background.  Emits signals for live preview."""

    # Emits (model_px np.ndarray, az, el, dist, roll, cost) at each eval
    frame_ready  = Signal(object, float, float, float, float, float)
    log_msg      = Signal(str)
    finished     = Signal(list, dict)   # (all_frames, best_result)
    error        = Signal(str)
    progress_val = Signal(int)          # 0-100

    def __init__(
        self,
        sil_path:    Path,
        lut_path:    Path,
        index_path:  Path,
        fx: float, fy: float, cx: float, cy: float,
        dist_min: float, dist_max: float, dist_steps: int,
        lut_ref_dist: float,
        top_k: int,
        # Initial guess — None means "not given"
        init_az:   Optional[float] = None,
        init_el:   Optional[float] = None,
        init_dist: Optional[float] = None,
        init_roll: Optional[float] = None,
        roll_step: float = 18.0,
        parent=None,
    ):
        super().__init__(parent)
        self.sil_path    = sil_path
        self.lut_path    = lut_path
        self.index_path  = index_path
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.dist_min    = dist_min
        self.dist_max    = dist_max
        self.dist_steps  = dist_steps
        self.lut_ref_dist = lut_ref_dist
        self.top_k       = top_k
        self.init_az     = init_az
        self.init_el     = init_el
        self.init_dist   = init_dist
        self.init_roll   = init_roll
        self.roll_step   = roll_step
        self._abort      = False
        self._mutex      = QMutex()

    def abort(self):
        with QMutexLocker(self._mutex):
            self._abort = True

    def _is_aborted(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._abort

    def _emit_frame(self, az: float, el: float, dist: float,
                    roll: float, cost: float, model_px: np.ndarray):
        self.frame_ready.emit(model_px.copy() if model_px is not None else np.empty((0, 2)),
                              az, el, dist, roll, cost)

    def _log(self, msg: str):
        self.log_msg.emit(msg)

    def run(self):
        try:
            self._run_impl()
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    def _run_impl(self):
        # ── Load silhouette ───────────────────────────────────────────────────
        sil_data    = np.load(str(self.sil_path))
        observed_px = sil_data["edge_pixels"].astype(np.float32)
        image_size  = sil_data["image_size"]
        img_w, img_h = int(image_size[0]), int(image_size[1])

        # Bbox centering
        bbox = sil_data["bbox"] if "bbox" in sil_data.files else None
        if bbox is not None and bbox.sum() > 0:
            bx1, by1, bx2, by2 = bbox
            bbox_cx = 0.5 * (bx1 + bx2)
            bbox_cy = 0.5 * (by1 + by2)
        else:
            bbox_cx = float(observed_px[:, 0].mean())
            bbox_cy = float(observed_px[:, 1].mean())

        obs_offset  = np.array([self.cx - bbox_cx, self.cy - bbox_cy], dtype=np.float32)
        observed_px = observed_px + obs_offset

        self._log(f"Silhouette: {self.sil_path.name}")
        self._log(f"Edge pixels: {len(observed_px):,}  |  Image: {img_w}×{img_h}")
        self._log(f"Bbox centre ({bbox_cx:.0f}, {bbox_cy:.0f}) → offset "
                  f"({obs_offset[0]:+.0f}, {obs_offset[1]:+.0f}) px")

        if len(observed_px) < 20:
            self.error.emit(f"Too few edge pixels ({len(observed_px)}). Need ≥ 20.")
            return

        target = np.zeros(3, dtype=np.float32)

        # ── Determine mode ────────────────────────────────────────────────────
        has_az   = self.init_az   is not None
        has_el   = self.init_el   is not None
        has_dist = self.init_dist is not None
        has_roll = self.init_roll is not None

        if has_az and has_el and has_dist and has_roll:
            mode = "DIRECT"
        elif has_az and has_el and has_dist:
            mode = "ROLL_SEARCH"
        elif has_az and has_el:
            mode = "DIST_ROLL_SEARCH"
        else:
            mode = "GLOBAL"

        self._log(f"Mode: {mode}")
        self.progress_val.emit(2)

        # ── Load LUT (needed for all except DIRECT) ───────────────────────────
        lut_npz     = None
        angle_index = None

        if mode != "DIRECT":
            self._log("Loading LUT …")
            lut_npz = np.load(str(self.lut_path))
            with open(self.index_path) as f:
                ai_full = json.load(f)
            angle_index = ai_full.get("angle_index", ai_full)
            self._log(f"  {len(lut_npz.files):,} poses loaded")
            self.progress_val.emit(5)

        # ── Build initial candidates list ─────────────────────────────────────
        candidates: list[tuple] = []   # (cost, az, el, dist, model_world)

        if mode == "DIRECT":
            # Build model_world from az/el at given dist, roll=0
            self._log(f"Direct refinement: az={self.init_az}° el={self.init_el}° "
                      f"dist={self.init_dist} m roll={self.init_roll}°")
            cam_pos = target + spherical_to_cartesian_aircraft(
                self.init_az, self.init_el, self.init_dist)
            R_cam = compute_camera_transform(cam_pos, target)
            # Find matching LUT key or synthesise model_world
            # We need model_world — load from LUT the nearest az/el or use the LUT
            # For DIRECT, still need to find the model_world from LUT.
            # Load LUT here to get model world.
            self._log("Loading LUT for model geometry …")
            lut_npz = np.load(str(self.lut_path))
            with open(self.index_path) as f:
                ai_full = json.load(f)
            angle_index = ai_full.get("angle_index", ai_full)
            pk_to_az_el = {}
            for az_el_key, pkey in angle_index.items():
                az_s, el_s = az_el_key.split("_", 1)
                pk_to_az_el[pkey] = (float(az_s), float(el_s))
            # Find nearest LUT entry to the initial guess
            best_lut_key = None
            best_lut_dist = float("inf")
            for pkey, (az_l, el_l) in pk_to_az_el.items():
                d = abs(az_l - self.init_az) + abs(el_l - self.init_el)
                if d < best_lut_dist:
                    best_lut_dist = d
                    best_lut_key  = pkey
                    best_lut_az   = az_l
                    best_lut_el   = el_l
            pts_cam_lut = lut_npz[best_lut_key].astype(np.float32)
            cam_lut = target + spherical_to_cartesian_aircraft(
                best_lut_az, best_lut_el, self.lut_ref_dist)
            R_lut   = compute_camera_transform(cam_lut, target)
            model_world = cam_space_to_world(pts_cam_lut, R_lut, cam_lut)
            self._log(f"  Nearest LUT entry: az={best_lut_az}°  el={best_lut_el}°")
            candidates = [(0.0, self.init_az, self.init_el, self.init_dist, model_world)]
            self.progress_val.emit(10)

        elif mode == "ROLL_SEARCH":
            # az, el, dist given → scan roll candidates
            self._log(f"Roll search: az={self.init_az}° el={self.init_el}° "
                      f"dist={self.init_dist} m  roll_step={self.roll_step}°")
            pk_to_az_el = {}
            for az_el_key, pkey in angle_index.items():
                az_s, el_s = az_el_key.split("_", 1)
                pk_to_az_el[pkey] = (float(az_s), float(el_s))
            # find best lut entry
            best_lut_key = None
            best_lut_dist_val = float("inf")
            for pkey, (az_l, el_l) in pk_to_az_el.items():
                d = abs(az_l - self.init_az) + abs(el_l - self.init_el)
                if d < best_lut_dist_val:
                    best_lut_dist_val = d
                    best_lut_key = pkey
                    best_lut_az  = az_l
                    best_lut_el  = el_l
            pts_cam_lut = lut_npz[best_lut_key].astype(np.float32)
            cam_lut = target + spherical_to_cartesian_aircraft(
                best_lut_az, best_lut_el, self.lut_ref_dist)
            R_lut   = compute_camera_transform(cam_lut, target)
            model_world = cam_space_to_world(pts_cam_lut, R_lut, cam_lut)

            roll_candidates = np.arange(-180.0, 180.0, self.roll_step)
            self._log(f"  {len(roll_candidates)} roll candidates: "
                      f"{roll_candidates[0]:.0f}° … {roll_candidates[-1]:.0f}°")

            cam_pos_d = target + spherical_to_cartesian_aircraft(
                self.init_az, self.init_el, self.init_dist)
            R_d = compute_camera_transform(cam_pos_d, target)

            obs_tree = cKDTree(observed_px)
            for roll_c in roll_candidates:
                if self._is_aborted():
                    return
                mw_rolled = apply_roll_to_world(model_world, target, roll_c)
                mpx = project_world_to_pixels(
                    mw_rolled, cam_pos_d, R_d,
                    self.fx, self.fy, self.cx, self.cy)
                cost = chamfer_obs_to_model(observed_px, mpx)
                # 6-element: (cost, az, el, dist, roll_init, model_world)
                candidates.append((cost, self.init_az, self.init_el, self.init_dist,
                                   roll_c, model_world))
                self._emit_frame(self.init_az, self.init_el, self.init_dist,
                                 roll_c, cost, mpx)
            candidates.sort(key=lambda x: x[0])
            best_roll = candidates[0][4]
            self._log(f"  Roll search done. Best roll={best_roll:.0f}°  "
                      f"cost={candidates[0][0]:.2f} px")
            self.progress_val.emit(20)

        elif mode == "DIST_ROLL_SEARCH":
            # az, el given → dist × roll grid
            dist_candidates = np.linspace(
                self.dist_min, self.dist_max, self.dist_steps).tolist()
            roll_candidates  = np.arange(-180.0, 180.0, self.roll_step)
            self._log(f"Dist×Roll grid: {len(dist_candidates)} dists × "
                      f"{len(roll_candidates)} rolls = "
                      f"{len(dist_candidates)*len(roll_candidates)} evals")

            # Find best lut entry for az/el
            pk_to_az_el = {}
            for az_el_key, pkey in angle_index.items():
                az_s, el_s = az_el_key.split("_", 1)
                pk_to_az_el[pkey] = (float(az_s), float(el_s))
            best_lut_key = None
            best_lut_dist_val = float("inf")
            for pkey, (az_l, el_l) in pk_to_az_el.items():
                d = abs(az_l - self.init_az) + abs(el_l - self.init_el)
                if d < best_lut_dist_val:
                    best_lut_dist_val = d
                    best_lut_key = pkey
                    best_lut_az = az_l
                    best_lut_el = el_l
            pts_cam_lut = lut_npz[best_lut_key].astype(np.float32)
            cam_lut = target + spherical_to_cartesian_aircraft(
                best_lut_az, best_lut_el, self.lut_ref_dist)
            R_lut = compute_camera_transform(cam_lut, target)
            model_world = cam_space_to_world(pts_cam_lut, R_lut, cam_lut)

            n_total = len(dist_candidates) * len(roll_candidates)
            n_done  = 0
            for d_c in dist_candidates:
                if self._is_aborted():
                    return
                cam_pos_d = target + spherical_to_cartesian_aircraft(
                    self.init_az, self.init_el, d_c)
                R_d = compute_camera_transform(cam_pos_d, target)
                for roll_c in roll_candidates:
                    if self._is_aborted():
                        return
                    mw_rolled = apply_roll_to_world(model_world, target, roll_c)
                    mpx  = project_world_to_pixels(
                        mw_rolled, cam_pos_d, R_d,
                        self.fx, self.fy, self.cx, self.cy)
                    cost = chamfer_obs_to_model(observed_px, mpx)
                    # 6-element tuple including roll_init
                    candidates.append((cost, self.init_az, self.init_el, d_c,
                                       roll_c, model_world))
                    self._emit_frame(self.init_az, self.init_el, d_c, roll_c, cost, mpx)
                    n_done += 1
                    self.progress_val.emit(5 + int(30 * n_done / n_total))
            candidates.sort(key=lambda x: x[0])
            self._log(f"  Grid done. Best cost={candidates[0][0]:.2f} px")

        elif mode == "GLOBAL":
            # Full LUT coarse search
            dist_candidates = np.linspace(
                self.dist_min, self.dist_max, self.dist_steps).tolist()
            self._log(f"Global coarse search: {len(angle_index):,} LUT entries × "
                      f"{len(dist_candidates)} distances …")
            t0 = time.perf_counter()
            candidates = global_coarse_search(
                observed_px, lut_npz, {"angle_index": angle_index}, target,
                dist_candidates=dist_candidates,
                lut_ref_dist=self.lut_ref_dist,
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            self._log(f"  Done in {elapsed:.1f} s — best: az={candidates[0][1]:.1f}°  "
                      f"el={candidates[0][2]:.1f}°  d={candidates[0][3]:.1f} m  "
                      f"cost={candidates[0][0]:.2f} px")
            # Emit the best coarse frame
            if candidates:
                cost0, az0, el0, d0, mw0 = candidates[0]
                cam0 = target + spherical_to_cartesian_aircraft(az0, el0, d0)
                R0   = compute_camera_transform(cam0, target)
                mpx0 = project_world_to_pixels(mw0, cam0, R0, self.fx, self.fy, self.cx, self.cy)
                self._emit_frame(az0, el0, d0, 0.0, cost0, mpx0)

            self.progress_val.emit(45)

            # Stage 1b validation
            self._log("Stage 1b: validating coarse candidates …")
            validated: list[tuple] = []
            scan_limit = min(len(candidates), self.top_k * 5)
            for cost_scaled, az0, el0, dist0, mw in candidates[:scan_limit]:
                if self._is_aborted():
                    return
                cam_v = target + spherical_to_cartesian_aircraft(az0, el0, dist0)
                R_v   = compute_camera_transform(cam_v, target)
                mpx_v = project_world_to_pixels(
                    mw, cam_v, R_v, self.fx, self.fy, self.cx, self.cy)
                if len(mpx_v) < 10:
                    continue
                real_cost = chamfer_obs_to_model(observed_px, mpx_v)
                validated.append((real_cost, az0, el0, dist0, mw))
                if len(validated) >= self.top_k * 3:
                    break
            validated.sort(key=lambda x: x[0])
            if not validated:
                self._log("  WARNING: no valid after Stage 1b — using raw coarse list")
                validated = candidates
            else:
                self._log(f"  {len(validated)} valid — best: az={validated[0][1]:.1f}°  "
                          f"el={validated[0][2]:.1f}°  d={validated[0][3]:.1f} m  "
                          f"cost={validated[0][0]:.2f} px")
            candidates = validated
            self.progress_val.emit(55)

        if self._is_aborted():
            return

        # ── Stage 2: Nelder-Mead refinement ──────────────────────────────────
        top_k = min(self.top_k, len(candidates))
        self._log(f"Refining top-{top_k} candidates …")

        # For DIRECT with roll given, set the warm-start roll
        if mode == "DIRECT" and has_roll:
            x0_roll = float(self.init_roll)
        else:
            x0_roll = 0.0

        all_frames: list[tuple] = []   # (az, el, dist, roll, cost, model_px)
        refined: list[dict] = []

        total_refine = top_k
        for i, cand in enumerate(candidates[:top_k]):
            if self._is_aborted():
                return

            # Support both 5-element (GLOBAL/DIRECT) and 6-element (ROLL/DIST_ROLL)
            if len(cand) == 6:
                cost0, az0, el0, dist0, cand_roll_init, model_world = cand
            else:
                cost0, az0, el0, dist0, model_world = cand
                cand_roll_init = None

            # Custom cost function with live emit callback
            is_last = (i == top_k - 1)
            cost_fn = PoseCostFunctionReal(
                observed_px, model_world, target,
                self.fx, self.fy, self.cx, self.cy,
                store_frames=True,
            )

            def _make_callback(cfn, az_start, el_start, dist_start):
                _last_emit_time = [0.0]
                def _cb(params):
                    az, el, dist, roll = params
                    nonlocal _last_emit_time
                    now = time.perf_counter()
                    if now - _last_emit_time[0] > 0.05 and len(cfn.history) > 0:
                        _last_emit_time[0] = now
                        h = cfn.history[-1]
                        if len(h) >= 6:
                            self._emit_frame(h[0], h[1], h[2], h[3], h[4], h[5])
                return _cb

            # Use scipy callback for live display
            callback_fn = _make_callback(cost_fn, az0, el0, dist0)

            # Determine initial roll for this candidate
            if cand_roll_init is not None:
                x0_roll_cand = float(cand_roll_init)
            else:
                x0_roll_cand = x0_roll

            x0      = np.array([az0, el0, dist0, x0_roll_cand])
            delta   = np.array([5.0, 5.0, dist0 * 0.2, 5.0])
            isimplex = np.vstack([x0, x0 + np.diag(delta)])

            result = minimize(
                cost_fn, x0, method="Nelder-Mead",
                callback=callback_fn,
                options=dict(xatol=0.1, fatol=0.5, maxiter=1000,
                             initial_simplex=isimplex),
            )

            opt_az, opt_el, opt_dist, opt_roll = result.x
            opt_az   = opt_az % 360.0
            opt_el   = float(np.clip(opt_el,   -89, 89))
            opt_dist = float(np.clip(opt_dist,  0.5, 50))

            r = {
                "az":          round(opt_az, 2),
                "el":          round(opt_el, 2),
                "dist":        round(opt_dist, 3),
                "roll":        round(float(opt_roll), 2),
                "cost_final":  round(result.fun, 3),
                "cost_coarse": round(cost0, 3),
                "converged":   bool(result.success),
                "n_evals":     cost_fn.n_evals,
                "init_az":     round(az0, 2),
                "init_el":     round(el0, 2),
                "init_dist":   round(dist0, 3),
                "_model_world": model_world,
                "_history":    cost_fn.history,
                "_store_frames": True,
            }
            refined.append(r)
            all_frames.extend(cost_fn.history)

            pct = 55 + int(40 * (i + 1) / total_refine)
            self.progress_val.emit(pct)
            self._log(f"  [{i+1}/{top_k}]  az={opt_az:6.1f}°  el={opt_el:5.1f}°  "
                      f"d={opt_dist:5.2f}m  roll={opt_roll:5.1f}°  "
                      f"cost={result.fun:.3f} px  evals={cost_fn.n_evals}")

        refined.sort(key=lambda x: x["cost_final"])
        best = refined[0]

        self._log("")
        self._log("=" * 52)
        self._log("  BEST ESTIMATE")
        self._log(f"  az   = {best['az']:.2f}°")
        self._log(f"  el   = {best['el']:.2f}°")
        self._log(f"  dist = {best['dist']:.3f} m")
        self._log(f"  roll = {best['roll']:.2f}°")
        self._log(f"  cost = {best['cost_final']:.3f} px")
        self._log("=" * 52)

        self.progress_val.emit(100)

        # Collect best-candidate frames for GIF/replay
        best_frames = [h for h in best["_history"] if len(h) >= 6]
        self.finished.emit(best_frames, best)


# ─────────────────────────────────────────────────────────────────────────────
# Live overlay display widget
# ─────────────────────────────────────────────────────────────────────────────

class OverlayDisplay(QLabel):
    """QLabel that shows the 2-D overlay image, auto-scaled to its size."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 220)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background: {BG_DARK}; border: 1px solid #2a2a4a; border-radius:4px;")
        self._qimage:  Optional[QImage]  = None
        self._qpixmap: Optional[QPixmap] = None   # raw PNG preview
        self.setText("Kein Bild – wähle eine NPZ-Silhouette und starte")
        self.setWordWrap(True)

    def set_image(self, img: QImage):
        """Show a QImage (optimizer overlay). Clears any PNG preview."""
        self._qpixmap = None
        self._qimage  = img
        self._refresh()

    def set_pixmap(self, pm: QPixmap):
        """Show a raw QPixmap (PNG preview). Cleared when optimizer starts."""
        self._qimage  = None
        self._qpixmap = pm
        self._refresh()

    def _refresh(self):
        pw = self.width()  - 4
        ph = self.height() - 4
        if pw <= 0 or ph <= 0:
            return
        if self._qimage is not None:
            pm = QPixmap.fromImage(self._qimage)
        elif self._qpixmap is not None:
            pm = self._qpixmap
        else:
            return
        self.setPixmap(pm.scaled(
            pw, ph,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Optional field helper
# ─────────────────────────────────────────────────────────────────────────────

def _try_float(text: str) -> Optional[float]:
    t = text.strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Settings panel
# ─────────────────────────────────────────────────────────────────────────────

class SettingsPanel(QWidget):
    """Left sidebar with all configuration inputs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Silhouette file ───────────────────────────────────────────────────
        grp_file = QGroupBox("Silhouette")
        fl = QVBoxLayout(grp_file)
        fl.setSpacing(4)
        row_f = QHBoxLayout()
        self.lbl_file = QLabel("Keine Datei gewählt")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setObjectName("hint")
        self.btn_browse = QPushButton("📂 Browse …")
        self.btn_browse.setFixedWidth(110)
        row_f.addWidget(self.lbl_file)
        row_f.addWidget(self.btn_browse)
        fl.addLayout(row_f)
        layout.addWidget(grp_file)

        # ── Initial guess ─────────────────────────────────────────────────────
        grp_guess = QGroupBox("Initial Guess  (leer = automatisch suchen)")
        gl = QFormLayout(grp_guess)
        gl.setSpacing(4)
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.inp_az   = QLineEdit(); self.inp_az.setPlaceholderText("z.B. 180")
        self.inp_el   = QLineEdit(); self.inp_el.setPlaceholderText("z.B. 20")
        self.inp_dist = QLineEdit(); self.inp_dist.setPlaceholderText("z.B. 11.4")
        self.inp_roll = QLineEdit(); self.inp_roll.setPlaceholderText(
            "leer → roll-Suche falls az/el/dist gegeben")

        for w in (self.inp_az, self.inp_el, self.inp_dist, self.inp_roll):
            w.setFixedWidth(150)

        gl.addRow("az [°]:",   self.inp_az)
        gl.addRow("el [°]:",   self.inp_el)
        gl.addRow("dist [m]:", self.inp_dist)
        gl.addRow("roll [°]:", self.inp_roll)

        hint = QLabel(
            "az + el + dist + roll → direktes Verfeinern\n"
            "az + el + dist         → roll-Grobsuche\n"
            "az + el                → dist × roll Suche\n"
            "(nichts)               → vollständige LUT-Suche"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        gl.addRow(hint)
        layout.addWidget(grp_guess)

        # ── Camera intrinsics ─────────────────────────────────────────────────
        grp_cam = QGroupBox("Kameraparameter")
        cl = QFormLayout(grp_cam)
        cl.setSpacing(4)
        cl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def dspin(val, lo=1.0, hi=5000.0, dec=1):
            w = QDoubleSpinBox()
            w.setRange(lo, hi); w.setDecimals(dec); w.setValue(val)
            w.setFixedWidth(100)
            return w

        self.spin_fx = dspin(951.1)   # ZED 2i S/N 34754237 @ HD720 — SDK-kalibriert
        self.spin_fy = dspin(951.1)   # HFOV=67.8°  VFOV=40.2°
        self.spin_cx = dspin(638.9, lo=0.0)
        self.spin_cy = dspin(348.0, lo=0.0)
        cl.addRow("fx [px]:", self.spin_fx)
        cl.addRow("fy [px]:", self.spin_fy)
        cl.addRow("cx [px]:", self.spin_cx)
        cl.addRow("cy [px]:", self.spin_cy)
        layout.addWidget(grp_cam)

        # ── Search params ─────────────────────────────────────────────────────
        grp_search = QGroupBox("Suchparameter")
        sl = QFormLayout(grp_search)
        sl.setSpacing(4)
        sl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def dspin2(val, lo, hi, dec=1):
            w = QDoubleSpinBox(); w.setRange(lo, hi); w.setDecimals(dec)
            w.setValue(val); w.setFixedWidth(100)
            return w

        def ispin(val, lo=1, hi=500):
            w = QSpinBox(); w.setRange(lo, hi); w.setValue(val)
            w.setFixedWidth(100)
            return w

        self.spin_dmin    = dspin2(3.0,   0.5, 100.0)
        self.spin_dmax    = dspin2(15.0,  0.5, 100.0)
        self.spin_dsteps  = ispin(7, 2, 50)
        self.spin_topk    = ispin(20, 1, 100)
        self.spin_roll_step = dspin2(18.0, 1.0, 90.0)
        self.spin_lut_ref = dspin2(5.0, 0.5, 100.0, dec=1)
        sl.addRow("dist min [m]:", self.spin_dmin)
        sl.addRow("dist max [m]:", self.spin_dmax)
        sl.addRow("dist steps:",   self.spin_dsteps)
        sl.addRow("top-K:",        self.spin_topk)
        sl.addRow("roll step [°]:", self.spin_roll_step)
        sl.addRow("LUT ref dist:", self.spin_lut_ref)
        layout.addWidget(grp_search)

        # ── Paths ─────────────────────────────────────────────────────────────
        grp_paths = QGroupBox("Dateipfade (LUT)")
        pl = QVBoxLayout(grp_paths)
        pl.setSpacing(4)

        _DATA = _HERE.parent.parent / "CAD-Edge-Generation" / "data"

        row_lut = QHBoxLayout()
        self.lbl_lut = QLabel(str(_DATA / "silhouette_lut_4x_final.npz"))
        self.lbl_lut.setObjectName("hint"); self.lbl_lut.setWordWrap(True)
        self.btn_lut = QPushButton("…"); self.btn_lut.setFixedWidth(28)
        row_lut.addWidget(QLabel("LUT:")); row_lut.addWidget(self.lbl_lut, 1)
        row_lut.addWidget(self.btn_lut)
        pl.addLayout(row_lut)

        row_idx = QHBoxLayout()
        self.lbl_idx = QLabel(str(_DATA / "silhouette_lut_4x_final.lookup.json"))
        self.lbl_idx.setObjectName("hint"); self.lbl_idx.setWordWrap(True)
        self.btn_idx = QPushButton("…"); self.btn_idx.setFixedWidth(28)
        row_idx.addWidget(QLabel("Idx:")); row_idx.addWidget(self.lbl_idx, 1)
        row_idx.addWidget(self.btn_idx)
        pl.addLayout(row_idx)

        layout.addWidget(grp_paths)

        layout.addStretch()


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class PoseOptimizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Optimizer — Real Silhouette")
        self.resize(1280, 820)

        self._worker:   Optional[OptimizerWorker] = None
        self._obs_px:   Optional[np.ndarray]      = None
        self._img_w:    int                        = 1280
        self._img_h:    int                        = 720
        self._all_frames: list = []       # (az, el, dist, roll, cost, model_px)
        self._best:     Optional[dict]             = None
        self._gif_timer = QTimer(self)
        self._gif_timer.timeout.connect(self._gif_next_frame)
        self._gif_frame_idx = 0
        self._last_frame_ms = 0

        self._build_ui()
        self._connect_signals()
        self.setStyleSheet(DARK_STYLESHEET)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Left sidebar ──────────────────────────────────────────────────────
        self.settings = SettingsPanel()

        scroll = QScrollArea()
        scroll.setWidget(self.settings)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # ── Right panel ───────────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        # Run / Stop buttons
        btn_row = QHBoxLayout()
        self.btn_run  = QPushButton("▶  Optimierung starten")
        self.btn_run.setObjectName("run_btn")
        self.btn_stop = QPushButton("■  Stopp")
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        rl.addLayout(btn_row)

        # Progress bar
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setFixedHeight(10)
        rl.addWidget(self.pbar)

        # Status label
        self.lbl_status = QLabel("Bereit.")
        self.lbl_status.setStyleSheet(f"color: {ACCENT}; font-size: 12px;")
        rl.addWidget(self.lbl_status)

        # Live overlay
        self.overlay = OverlayDisplay()
        rl.addWidget(self.overlay, 3)

        # GIF player controls (hidden until frames available)
        self.gif_controls = QWidget()
        gc_l = QHBoxLayout(self.gif_controls)
        gc_l.setContentsMargins(0, 0, 0, 0)
        gc_l.setSpacing(6)
        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setFixedWidth(80)
        self.btn_save_gif   = QPushButton("💾 GIF speichern")
        self.lbl_frame_info = QLabel("Frame 0/0")
        self.lbl_frame_info.setObjectName("hint")
        self.slider_frames  = QSlider(Qt.Orientation.Horizontal)
        self.slider_frames.setRange(0, 0)

        gc_l.addWidget(self.btn_play_pause)
        gc_l.addWidget(self.slider_frames, 1)
        gc_l.addWidget(self.lbl_frame_info)
        gc_l.addWidget(self.btn_save_gif)
        self.gif_controls.setVisible(False)
        rl.addWidget(self.gif_controls)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(170)
        rl.addWidget(self.log)

        root.addWidget(scroll, 0)
        root.addWidget(right, 1)

    def _connect_signals(self):
        s = self.settings
        s.btn_browse.clicked.connect(self._browse_silhouette)
        s.btn_lut.clicked.connect(self._browse_lut)
        s.btn_idx.clicked.connect(self._browse_idx)
        self.btn_run.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_play_pause.clicked.connect(self._toggle_gif_play)
        self.btn_save_gif.clicked.connect(self._save_gif)
        self.slider_frames.valueChanged.connect(self._slider_changed)

    # ── File pickers ──────────────────────────────────────────────────────────

    def _browse_silhouette(self):
        _DEF = str(_HERE.parent.parent / "Edge_Detection" / "data" / "exported_silhouettes")
        path, _ = QFileDialog.getOpenFileName(
            self, "Silhouette NPZ wählen", _DEF, "NPZ files (*.npz)")
        if path:
            self.settings.lbl_file.setText(Path(path).name)
            self.settings.lbl_file.setToolTip(path)
            self._sil_path = Path(path)
            # Try to load metadata for info
            self._load_sil_meta(Path(path))

    def _load_sil_meta(self, p: Path):
        try:
            d = np.load(str(p))
            w, h = int(d["image_size"][0]), int(d["image_size"][1])
            n    = len(d["edge_pixels"])
            self._log_msg(f"Geladen: {p.name}  ({n} Pixel, {w}×{h})")
        except Exception:
            pass

        # ── Load preview PNG ─────────────────────────────────────────
        # Convention: the Edge Detection GUI creates <stem>_preview.png
        # alongside the NPZ. Show it so the user can estimate the initial pose.
        preview_path = p.parent / (p.stem + "_preview.png")
        if preview_path.exists():
            pm = QPixmap(str(preview_path))
            if not pm.isNull():
                self.overlay.set_pixmap(pm)
                self._log_msg(f"Vorschau: {preview_path.name}")
        else:
            # No PNG: render cyan dot silhouette from edge pixels
            try:
                d      = np.load(str(p))
                obs    = d["edge_pixels"].astype(float)
                iw, ih = int(d["image_size"][0]), int(d["image_size"][1])
                img    = render_overlay(obs, None, iw, ih)
                self.overlay.set_image(img)
            except Exception:
                pass

    def _browse_lut(self):
        path, _ = QFileDialog.getOpenFileName(self, "LUT NPZ wählen", "", "NPZ (*.npz)")
        if path:
            self.settings.lbl_lut.setText(path)
            self.settings.lbl_lut.setToolTip(path)

    def _browse_idx(self):
        path, _ = QFileDialog.getOpenFileName(self, "Index JSON wählen", "", "JSON (*.json)")
        if path:
            self.settings.lbl_idx.setText(path)
            self.settings.lbl_idx.setToolTip(path)

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def _start(self):
        if not hasattr(self, "_sil_path"):
            self._log_msg("⚠ Bitte erst eine Silhouette-Datei wählen!")
            return

        s = self.settings
        init_az   = _try_float(s.inp_az.text())
        init_el   = _try_float(s.inp_el.text())
        init_dist = _try_float(s.inp_dist.text())
        init_roll = _try_float(s.inp_roll.text())

        lut_path   = Path(s.lbl_lut.toolTip() or s.lbl_lut.text())
        index_path = Path(s.lbl_idx.toolTip() or s.lbl_idx.text())

        if not lut_path.exists():
            self._log_msg(f"⚠ LUT nicht gefunden: {lut_path}")
            return
        if not index_path.exists():
            self._log_msg(f"⚠ Index nicht gefunden: {index_path}")
            return

        # Load observed pixels for display
        sil_data = np.load(str(self._sil_path))
        self._obs_px_raw = sil_data["edge_pixels"].astype(np.float32)
        img_size   = sil_data["image_size"]
        self._img_w = int(img_size[0])
        self._img_h = int(img_size[1])

        # Bbox centering for display preview
        bbox = sil_data["bbox"] if "bbox" in sil_data.files else None
        if bbox is not None and bbox.sum() > 0:
            bx1, by1, bx2, by2 = bbox
            self._bbox_cx = 0.5*(bx1+bx2); self._bbox_cy = 0.5*(by1+by2)
        else:
            self._bbox_cx = float(self._obs_px_raw[:,0].mean())
            self._bbox_cy = float(self._obs_px_raw[:,1].mean())

        cx = s.spin_cx.value(); cy = s.spin_cy.value()
        obs_offset = np.array([cx - self._bbox_cx, cy - self._bbox_cy], dtype=np.float32)
        self._obs_px = self._obs_px_raw + obs_offset

        # Show initial observed-only overlay
        img = render_overlay(self._obs_px, None, self._img_w, self._img_h)
        self.overlay.set_image(img)

        self._gif_timer.stop()
        self._all_frames = []
        self._best = None
        self.gif_controls.setVisible(False)
        self.pbar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._worker = OptimizerWorker(
            sil_path=self._sil_path,
            lut_path=lut_path,
            index_path=index_path,
            fx=s.spin_fx.value(), fy=s.spin_fy.value(),
            cx=cx, cy=cy,
            dist_min=s.spin_dmin.value(), dist_max=s.spin_dmax.value(),
            dist_steps=s.spin_dsteps.value(),
            lut_ref_dist=s.spin_lut_ref.value(),
            top_k=s.spin_topk.value(),
            init_az=init_az, init_el=init_el,
            init_dist=init_dist, init_roll=init_roll,
            roll_step=s.spin_roll_step.value(),
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.log_msg.connect(self._log_msg)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress_val.connect(self.pbar.setValue)

        self._last_frame_ms = 0
        self._worker.start()
        self.lbl_status.setText("Läuft …")

    def _stop(self):
        if self._worker:
            self._worker.abort()
            self._log_msg("⏸ Gestoppt.")
        self._set_running(False)

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        if not running:
            self.lbl_status.setText("Gestoppt.")

    # ── Worker signals ────────────────────────────────────────────────────────

    def _on_frame(self, model_px: np.ndarray,
                  az: float, el: float, dist: float, roll: float, cost: float):
        # Throttle display to ~ 20 fps
        now_ms = int(time.perf_counter() * 1000)
        if now_ms - self._last_frame_ms < 50:
            return
        self._last_frame_ms = now_ms

        if self._obs_px is not None:
            img = render_overlay(self._obs_px, model_px, self._img_w, self._img_h)
            self.overlay.set_image(img)

        self.lbl_status.setText(
            f"az={az:.1f}°  el={el:.1f}°  dist={dist:.2f} m  "
            f"roll={roll:.1f}°  cost={cost:.3f} px")

    def _on_finished(self, frames: list, best: dict):
        self._all_frames = frames
        self._best = best
        self._set_running(False)
        self.lbl_status.setText(
            f"✅  az={best['az']:.2f}°  el={best['el']:.2f}°  "
            f"dist={best['dist']:.3f} m  roll={best['roll']:.2f}°  "
            f"cost={best['cost_final']:.3f} px")
        self._log_msg("Fertig!")

        # Show final frame
        if frames:
            *_, final_mpx = frames[-1]
            img = render_overlay(self._obs_px, final_mpx, self._img_w, self._img_h)
            self.overlay.set_image(img)

        # Activate GIF player
        if frames:
            self.slider_frames.setRange(0, len(frames) - 1)
            self.slider_frames.setValue(0)
            self.gif_controls.setVisible(True)
            self._gif_frame_idx = 0
            self._show_gif_frame(0)

        self.pbar.setValue(100)

    def _on_error(self, msg: str):
        self._set_running(False)
        self._log_msg(f"❌ FEHLER:\n{msg}")
        self.lbl_status.setText("Fehler — siehe Log.")

    def _log_msg(self, msg: str):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    # ── GIF player ────────────────────────────────────────────────────────────

    def _show_gif_frame(self, idx: int):
        if not self._all_frames or idx >= len(self._all_frames):
            return
        h = self._all_frames[idx]
        az, el, dist, roll, cost, model_px = h
        if self._obs_px is not None:
            img = render_overlay(self._obs_px, model_px, self._img_w, self._img_h)
            self.overlay.set_image(img)
        n = len(self._all_frames)
        self.lbl_frame_info.setText(
            f"Frame {idx+1}/{n}  az={az:.1f}°  el={el:.1f}°  "
            f"d={dist:.2f}m  roll={roll:.1f}°  cost={cost:.3f} px")
        self.slider_frames.blockSignals(True)
        self.slider_frames.setValue(idx)
        self.slider_frames.blockSignals(False)

    def _gif_next_frame(self):
        n = len(self._all_frames)
        if n == 0:
            return
        self._gif_frame_idx = (self._gif_frame_idx + 1) % n
        self._show_gif_frame(self._gif_frame_idx)

    def _toggle_gif_play(self):
        if self._gif_timer.isActive():
            self._gif_timer.stop()
            self.btn_play_pause.setText("▶ Play")
        else:
            self._gif_timer.start(80)   # ~12 fps
            self.btn_play_pause.setText("⏸ Pause")

    def _slider_changed(self, val: int):
        self._gif_frame_idx = val
        self._show_gif_frame(val)

    # ── Save GIF ──────────────────────────────────────────────────────────────

    def _save_gif(self):
        if not self._all_frames or self._best is None:
            self._log_msg("⚠ Keine Frames vorhanden.")
            return

        default_name = "convergence.gif"
        if hasattr(self, "_sil_path"):
            default_name = self._sil_path.stem + "_convergence.gif"
        default_dir = str(_HERE.parent / "results_real")

        out_path, _ = QFileDialog.getSaveFileName(
            self, "GIF speichern", str(Path(default_dir) / default_name),
            "GIF (*.gif)")
        if not out_path:
            return

        self._log_msg(f"Rendere GIF → {out_path} …")
        QApplication.processEvents()

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import matplotlib.gridspec as gridspec
            from mpl_toolkits.mplot3d import Axes3D  # noqa

            # Delegate to the CLI renderer with our collected history
            from run_on_real_silhouette import (
                render_convergence_gif, _az_el_to_unit,
            )

            best_for_gif = dict(self._best)
            best_for_gif["_history"] = [list(h) for h in self._all_frames]
            best_for_gif["_store_frames"] = True

            render_convergence_gif(
                best_for_gif,
                self._obs_px,
                self._img_w, self._img_h,
                Path(out_path),
                max_frames=80,
                fps=12,
            )
            self._log_msg(f"✅ GIF gespeichert: {out_path}")
        except Exception as exc:
            self._log_msg(f"❌ GIF-Render Fehler: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Pose Optimizer GUI")
    win = PoseOptimizerApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
