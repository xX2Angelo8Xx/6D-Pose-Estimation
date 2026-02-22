#!/usr/bin/env python3
"""
LUT Quality Checker
===================
Systematic quality scan of silhouette_lut_roll.npz to detect poses with
holes, sparse coverage, or broken silhouettes.

Phase 1 (JSON only, ~instant):
  - Load n_points from lookup.json for every pose
  - Build (az × el) heatmap coloured by n_points (roll=0 slice)
  - Flag outliers: global p10 + per-elevation-band z-score

Phase 2 (NPZ lazy-load, only suspicious poses):
  - Project camera-space points → 2D pixels using LUT intrinsics
  - Render contour to binary image
  - Compute convex-hull fill-ratio and largest-gap metric

Phase 3 (Output):
  - lut_quality_heatmap.png       — az×el n_points map + flagged cells
  - lut_quality_report.csv        — all poses ranked by quality
  - lut_quality_bad_renders.png   — contact sheet of worst pose renders
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------—
# Paths (relative to this file → repo root)
BASE = Path(__file__).resolve().parent.parent
LUT_NPZ  = BASE / "CAD-Edge-Generation/data/silhouette_lut_roll.npz"
LUT_JSON = BASE / "CAD-Edge-Generation/data/silhouette_lut_roll.lookup.json"
OUT_DIR  = Path(__file__).resolve().parent

# LUT generation intrinsics  (extract_silhouette_contour defaults)
LUT_FX, LUT_FY = 800.0, 800.0
LUT_W,  LUT_H  = 1280, 720          # rendered image size
LUT_CX, LUT_CY = LUT_W / 2, LUT_H / 2   # 640, 360

# Thresholds
BAND_ZSCORE_THRESH = -2.5   # flag if (n - band_mean) / band_std < this
GLOBAL_P_THRESH    = 10      # also flag global bottom percentile
FILL_RATIO_THRESH  = 0.40   # hull fill <40% → poor coverage
GAP_FRAC_THRESH    = 0.12   # largest contour gap > 12% of perimeter → hole
MAX_BAD_RENDERS    = 80     # max poses shown in contact sheet


# ===========================================================================
# Helpers
# ===========================================================================