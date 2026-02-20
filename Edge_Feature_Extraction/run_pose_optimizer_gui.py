#!/usr/bin/env python3
"""
Launcher for the Pose Optimizer GUI.

Usage (from workspace root):
    python3 run_pose_optimizer_gui.py
"""
import sys
from pathlib import Path

# ── Ensure all source paths are on sys.path ───────────────────────────────────
_ROOT  = Path(__file__).parent
_SRC   = _ROOT / "Pose_Optimizer" / "src"
_CAD   = _ROOT / "CAD-Edge-Generation" / "src"

for p in (_ROOT, _SRC, _CAD):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from pose_optimizer_gui import main  # noqa: E402

if __name__ == "__main__":
    main()
