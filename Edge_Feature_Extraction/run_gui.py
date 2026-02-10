#!/usr/bin/env python3
"""
Launcher script for the Edge Detection GUI.

Usage:
    python3 run_gui.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gui.edge_detection_gui import main

if __name__ == '__main__':
    main()
