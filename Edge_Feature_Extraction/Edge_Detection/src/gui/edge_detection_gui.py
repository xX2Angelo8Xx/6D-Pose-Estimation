#!/usr/bin/env python3
"""
Interactive GUI for testing edge detection algorithms.

Features:
- Load images from data/debug_images/debug/
- Select and apply algorithms within YOLO bounding boxes
- Zoomable image viewer (zoom at mouse cursor position)
- Display runtime (ms) and theoretical FPS
- Session-wise saving to output folders per algorithm
"""

from __future__ import annotations

import sys
import re
import shutil
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QMessageBox,
    QSlider,
    QGroupBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QProgressBar,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QWheelEvent

# Import algorithms (add parent to path)
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.preproc import apply_clahe
from algorithms.canny_pipeline import canny_edge_detection
from algorithms.contour_silhouette import extract_silhouette_from_contours
from algorithms.hough_ransac import hough_lines_probabilistic

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. SVO Mode will not be available.")

# ZED SDK imports (only available on Jetson / machines with ZED SDK installed)
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("Warning: pyzed not installed. SVO2 direct-read mode not available.")


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view with zoom support centered on mouse cursor."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.zoom_factor = 1.15
        self._zoom_level = 1.0
        
    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out on mouse wheel."""
        if event.angleDelta().y() > 0:
            # Zoom in
            self.scale(self.zoom_factor, self.zoom_factor)
            self._zoom_level *= self.zoom_factor
        else:
            # Zoom out
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)
            self._zoom_level /= self.zoom_factor
    
    def get_zoom_level(self) -> float:
        """Return current zoom level."""
        return self._zoom_level
    
    def reset_zoom(self):
        """Reset zoom to 1.0."""
        self.resetTransform()
        self._zoom_level = 1.0
    
    def set_zoom_level(self, level: float):
        """Set zoom to specific level."""
        self.resetTransform()
        self.scale(level, level)
        self._zoom_level = level


class ExtractionWorker(QThread):
    """Worker thread for copying images from YoloTrainingImagesV1 to originals/ + labels/."""
    progress = Signal(int)        # 0-100
    status = Signal(str)          # status message
    finished_ok = Signal(int, int)  # (copied_count, skipped_count)
    error = Signal(str)

    def __init__(
        self,
        source_dir: Path,
        originals_dir: Path,
        labels_dir: Path,
        max_depth: float,
        parent=None,
    ):
        super().__init__(parent)
        self.source_dir = source_dir
        self.originals_dir = originals_dir
        self.labels_dir = labels_dir
        self.max_depth = max_depth
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            # ---- collect matching files ----
            self.status.emit("Suche nach Bilddateien …")
            candidates: List[Tuple[Path, Path]] = []  # (jpg, txt)

            for folder in sorted(self.source_dir.iterdir()):
                if not re.match(r'^[0-8]_', folder.name):
                    continue
                for jpg in sorted(folder.rglob("*.jpg")):
                    m = re.search(r'depth-([\d.]+)m', jpg.stem)
                    if not m:
                        continue
                    if float(m.group(1)) > self.max_depth:
                        continue
                    txt = jpg.with_suffix('.txt')
                    if not txt.exists():
                        continue
                    candidates.append((jpg, txt))

            if not candidates:
                self.error.emit(
                    f"Keine passenden Bilder gefunden in:\n{self.source_dir}\n"
                    f"(max depth = {self.max_depth} m)"
                )
                return

            total = len(candidates)
            self.status.emit(f"{total} Bilder gefunden. Löschen alter Daten …")

            # ---- clear existing contents ----
            self.originals_dir.mkdir(parents=True, exist_ok=True)
            self.labels_dir.mkdir(parents=True, exist_ok=True)

            for f in self.originals_dir.iterdir():
                if f.is_file():
                    f.unlink()
            for f in self.labels_dir.iterdir():
                if f.is_file():
                    f.unlink()

            # ---- copy files ----
            copied = 0
            skipped = 0
            for i, (jpg, txt) in enumerate(candidates):
                if self._abort:
                    self.status.emit("Abgebrochen.")
                    return

                dest_jpg = self.originals_dir / jpg.name
                dest_txt = self.labels_dir / txt.name

                # skip if name collision (different source folders may have same filename)
                if dest_jpg.exists():
                    skipped += 1
                else:
                    shutil.copy2(jpg, dest_jpg)
                    shutil.copy2(txt, dest_txt)
                    copied += 1

                pct = int((i + 1) / total * 100)
                self.progress.emit(pct)
                if i % 50 == 0:
                    self.status.emit(f"Kopiere … {i+1}/{total}")

            # ---- write metadata.csv ----
            meta_path = self.originals_dir.parent / "metadata.csv"
            with open(meta_path, "w") as f:
                f.write("source,originals,labels\n")
                for jpg, txt in candidates:
                    dest_jpg = self.originals_dir / jpg.name
                    dest_txt = self.labels_dir / txt.name
                    f.write(f"{jpg},{dest_jpg},{dest_txt}\n")

            self.finished_ok.emit(copied, skipped)

        except Exception as exc:
            self.error.emit(str(exc))


class EdgeDetectionGUI(QMainWindow):
    """Main GUI for edge detection algorithm testing."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge Detection Algorithm Tester")
        self.setGeometry(100, 100, 1400, 900)
        
        # Paths - Debug Mode
        self.project_root = Path(__file__).parent.parent.parent
        self.debug_images_dir = self.project_root / "data" / "debug_images" / "originals"
        self.labels_dir = self.project_root / "data" / "debug_images" / "labels"
        self.output_dir = self.project_root / "data" / "algorithm_outputs"
        # Source for extraction (YoloTrainingImages)
        self.extraction_source_dir = Path("/media/angelo/DRONE_DATA1/YoloTrainingImagesV1")
        
        # Paths - SVO Mode
        self.svo_base_dir = Path("/media/angelo/DRONE_DATA1/SVO2_Frame_Export")
        self.yolo_model_path = Path("/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt")
        self.yolo_model = None
        
        # Mode selection
        self.current_mode = "debug"  # "debug" or "svo"
        self.svo_folders = []
        self.current_svo_folder = None
        self.current_depth_data = None

        # SVO2 Mode — direct ZED SDK access
        self.svo2_file_path: Optional[Path] = None
        self.zed_camera = None          # sl.Camera() instance while open
        self.svo2_total_frames: int = 0
        self.svo2_current_frame: int = 0
        self.svo2_raw_frame: Optional[np.ndarray] = None    # BGR full-res left image
        self.svo2_depth_map: Optional[np.ndarray] = None    # float32 depth map (m)
        self.svo2_depth_mode: str = "NEURAL"                # depth mode name
        self.svo2_yolo_input_size: int = 1280               # YOLO inference width
        self.svo2_depth_opacity: float = 0.6                # depth overlay opacity
        self.svo2_bbox: Optional[Tuple[int, int, int, int]] = None       # YOLO bbox – left frame
        self.svo2_right_frame: Optional[np.ndarray] = None               # BGR right camera image
        self.svo2_bbox_right: Optional[Tuple[int, int, int, int]] = None  # YOLO bbox – right frame
        self.svo2_show_depth_overlay: bool = True           # toggle overlay rendering
        # Default model directory (scanned at startup for .pt/.onnx/.engine files)
        self.yolo_model_dir = Path(
            "/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models"
        )
        
        # Session management
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.output_dir / "sessions.json"
        self.load_or_create_session()
        
        # Current state
        self.image_files: List[Path] = []
        self.current_image_idx: int = 0
        self.current_image: Optional[np.ndarray] = None
        self._clean_image:  Optional[np.ndarray] = None   # immutable original, never drawn on
        self.current_result: Optional[np.ndarray] = None
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_runtime_ms: float = 0.0
        self.yolo_inference_time: float = 0.0
        self.edge_detection_time: float = 0.0
        self.current_edge_pixels_abs: Optional[np.ndarray] = None  # (N, 2) = (col, row) pixel coords
        self.export_dir = self.project_root / "data" / "exported_silhouettes"
        
        # Algorithm parameters
        self.params = {
            'canny_low': 50,
            'canny_high': 150,
            'canny_auto': True,
            'gaussian_sigma': 1.0,
            'clahe_clip': 2.0,
            'bbox_padding': 5,  # ROI expansion: px outside bbox that are still processed
            'morph_kernel': 5,
            'snake_alpha': 0.01,  # Active contour elasticity
            'snake_beta': 0.1,    # Active contour stiffness
            'snake_iterations': 100,
            'depth_radius': 5,    # Depth filter: search radius for valid depth
            'use_depth_filter': False,  # Enable/disable depth filtering
        }
        
        # Default parameter values (for reset)
        self.default_params = self.params.copy()
        
        # Available algorithms with their used parameters
        self.algorithms = {
            "Canny (Auto)": {
                'func': self.apply_canny_auto,
                'params': ['gaussian_sigma']
            },
            "CLAHE + Canny": {
                'func': self.apply_clahe_canny,
                'params': ['clahe_clip', 'gaussian_sigma']
            },
            "Canny (Manual)": {
                'func': self.apply_canny_manual,
                'params': ['canny_low', 'canny_high', 'gaussian_sigma']
            },
            "Sobel Magnitude": {
                'func': self.apply_sobel,
                'params': ['gaussian_sigma']
            },
            "Laplacian of Gaussian": {
                'func': self.apply_log,
                'params': ['gaussian_sigma']
            },
            "Multi-scale Canny": {
                'func': self.apply_multiscale_canny,
                'params': []
            },
            "Structured Edges": {
                'func': self.apply_structured_edges,
                'params': ['gaussian_sigma']
            },
            "Active Contours (Snake)": {
                'func': self.apply_active_contours,
                'params': ['gaussian_sigma', 'snake_alpha', 'snake_beta', 'snake_iterations']
            },
            "Canny + Contour Silhouette": {
                'func': self.apply_canny_contour,
                'params': ['gaussian_sigma', 'morph_kernel']
            },
            "Canny + Hough Lines": {
                'func': self.apply_canny_hough,
                'params': ['gaussian_sigma']
            },
        }
        
        self.init_ui()
        # Don't load images yet - wait for mode initialization
        # self.load_image_list()
        # self.load_first_image()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Mode selection panel
        mode_layout = QHBoxLayout()
        mode_group = QGroupBox("Operation Mode")
        mode_group_layout = QHBoxLayout()
        
        self.mode_button_group = QButtonGroup()
        self.radio_debug_mode = QRadioButton("Debug Mode (YOLO Labels)")
        self.radio_svo_mode = QRadioButton("SVO2 Mode (Live ZED · YOLO · Depth)")
        self.radio_debug_mode.setChecked(True)
        
        self.mode_button_group.addButton(self.radio_debug_mode)
        self.mode_button_group.addButton(self.radio_svo_mode)
        
        self.radio_debug_mode.toggled.connect(self.on_mode_changed)
        self.radio_svo_mode.toggled.connect(self.on_mode_changed)
        
        mode_group_layout.addWidget(self.radio_debug_mode)
        mode_group_layout.addWidget(self.radio_svo_mode)

        # (legacy folder widgets — kept as hidden stubs so on_mode_changed can reference them)
        self.lbl_svo_folder = QLabel("SVO Folder:")
        self.combo_svo_folder = QComboBox()
        self.lbl_svo_folder.setVisible(False)
        self.combo_svo_folder.setVisible(False)

        mode_group_layout.addStretch()

        # Extraction button (only visible in Debug mode)
        self.btn_extract_images = QPushButton("📥 Bilder extrahieren")
        self.btn_extract_images.setToolTip(
            "Bilder aus YoloTrainingImagesV1 filtern und nach originals/ + labels/ kopieren"
        )
        self.btn_extract_images.clicked.connect(self.open_extraction_dialog)
        mode_group_layout.addWidget(self.btn_extract_images)

        mode_group.setLayout(mode_group_layout)
        main_layout.addWidget(mode_group)

        # ── SVO2 Configuration Panel (hidden until SVO2 mode is selected) ──
        self.svo2_group = QGroupBox("SVO2 Configuration")
        svo2_main = QVBoxLayout()

        # Row 1 — file/model/depth-mode pickers
        svo2_row1 = QHBoxLayout()

        # SVO2 file picker
        svo2_row1.addWidget(QLabel("SVO2 File:"))
        self.lbl_svo2_file = QLabel("(no file selected)")
        self.lbl_svo2_file.setMinimumWidth(220)
        svo2_row1.addWidget(self.lbl_svo2_file)
        self.btn_svo2_pick_file = QPushButton("Browse…")
        self.btn_svo2_pick_file.clicked.connect(self.on_svo2_file_pick)
        svo2_row1.addWidget(self.btn_svo2_pick_file)

        svo2_row1.addSpacing(12)

        # YOLO model picker
        svo2_row1.addWidget(QLabel("Model:"))
        self.combo_svo2_model = QComboBox()
        self.combo_svo2_model.setMinimumWidth(180)
        self.combo_svo2_model.currentTextChanged.connect(self.on_svo2_model_changed)
        svo2_row1.addWidget(self.combo_svo2_model)
        self.btn_svo2_pick_model_dir = QPushButton("…")
        self.btn_svo2_pick_model_dir.setFixedWidth(28)
        self.btn_svo2_pick_model_dir.setToolTip("Browse for model folder")
        self.btn_svo2_pick_model_dir.clicked.connect(self.on_svo2_pick_model_dir)
        svo2_row1.addWidget(self.btn_svo2_pick_model_dir)

        svo2_row1.addSpacing(12)

        # Depth mode
        svo2_row1.addWidget(QLabel("Depth Mode:"))
        self.combo_svo2_depth_mode = QComboBox()
        # Only current (non-legacy) ZED SDK v5 depth modes – see docs/guides/zed_depth_modes.md
        self.combo_svo2_depth_mode.addItems(["NEURAL_PLUS", "NEURAL", "NEURAL_LIGHT", "NONE"])
        self.combo_svo2_depth_mode.setCurrentText("NEURAL")
        self.combo_svo2_depth_mode.currentTextChanged.connect(self.on_svo2_depth_mode_changed)
        svo2_row1.addWidget(self.combo_svo2_depth_mode)

        svo2_row1.addStretch()
        svo2_main.addLayout(svo2_row1)

        # Row 2 — navigation + frame info
        svo2_row2 = QHBoxLayout()

        self.btn_svo2_m5 = QPushButton("◀◀ -5")
        self.btn_svo2_m5.clicked.connect(lambda: self.svo2_navigate(-5))
        svo2_row2.addWidget(self.btn_svo2_m5)

        self.btn_svo2_m1 = QPushButton("◀ -1")
        self.btn_svo2_m1.clicked.connect(lambda: self.svo2_navigate(-1))
        svo2_row2.addWidget(self.btn_svo2_m1)

        self.btn_svo2_p1 = QPushButton("+1 ▶")
        self.btn_svo2_p1.clicked.connect(lambda: self.svo2_navigate(+1))
        svo2_row2.addWidget(self.btn_svo2_p1)

        self.btn_svo2_p5 = QPushButton("+5 ▶▶")
        self.btn_svo2_p5.clicked.connect(lambda: self.svo2_navigate(+5))
        svo2_row2.addWidget(self.btn_svo2_p5)

        self.lbl_svo2_frame = QLabel("Frame: -- / --")
        svo2_row2.addWidget(self.lbl_svo2_frame)

        svo2_row2.addStretch()

        # YOLO controls
        svo2_row2.addWidget(QLabel("YOLO input:"))
        self.svo2_yolo_size_group = QButtonGroup()
        self.radio_svo2_1280 = QRadioButton("1280")
        self.radio_svo2_640 = QRadioButton("640")
        self.radio_svo2_1280.setChecked(True)
        self.svo2_yolo_size_group.addButton(self.radio_svo2_1280)
        self.svo2_yolo_size_group.addButton(self.radio_svo2_640)
        self.radio_svo2_1280.toggled.connect(
            lambda checked: self._set_svo2_yolo_size(1280) if checked else None
        )
        self.radio_svo2_640.toggled.connect(
            lambda checked: self._set_svo2_yolo_size(640) if checked else None
        )
        svo2_row2.addWidget(self.radio_svo2_1280)
        svo2_row2.addWidget(self.radio_svo2_640)

        self.btn_svo2_run_yolo = QPushButton("▶ Run YOLO")
        self.btn_svo2_run_yolo.clicked.connect(self.run_yolo_svo2)
        svo2_row2.addWidget(self.btn_svo2_run_yolo)

        self.lbl_svo2_timing = QLabel("YOLO: --ms")
        svo2_row2.addWidget(self.lbl_svo2_timing)

        svo2_row2.addSpacing(16)

        # Depth overlay controls
        self.chk_svo2_show_depth = QCheckBox("Show Depth Overlay")
        self.chk_svo2_show_depth.setChecked(True)
        self.chk_svo2_show_depth.stateChanged.connect(self.on_svo2_depth_overlay_toggled)
        svo2_row2.addWidget(self.chk_svo2_show_depth)

        svo2_row2.addWidget(QLabel("Opacity:"))
        self.slider_svo2_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_svo2_opacity.setRange(0, 100)
        self.slider_svo2_opacity.setValue(60)
        self.slider_svo2_opacity.setFixedWidth(90)
        self.slider_svo2_opacity.valueChanged.connect(self.on_svo2_opacity_changed)
        svo2_row2.addWidget(self.slider_svo2_opacity)
        self.lbl_svo2_opacity = QLabel("60%")
        svo2_row2.addWidget(self.lbl_svo2_opacity)

        svo2_main.addLayout(svo2_row2)
        self.svo2_group.setLayout(svo2_main)
        self.svo2_group.setVisible(False)
        main_layout.addWidget(self.svo2_group)
        # ── end SVO2 Configuration Panel ──
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Image navigation
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.clicked.connect(self.prev_image)
        control_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.next_image)
        control_layout.addWidget(self.btn_next)
        
        self.lbl_image_info = QLabel("No image loaded")
        control_layout.addWidget(self.lbl_image_info)
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)
        
        # Algorithm selection and execution
        algo_layout = QHBoxLayout()
        
        algo_layout.addWidget(QLabel("Algorithm:"))
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems(list(self.algorithms.keys()))
        self.combo_algorithm.currentTextChanged.connect(self.on_algorithm_changed)
        algo_layout.addWidget(self.combo_algorithm)
        
        self.btn_apply = QPushButton("Apply Algorithm")
        self.btn_apply.clicked.connect(self.apply_algorithm)
        algo_layout.addWidget(self.btn_apply)
        
        self.lbl_runtime = QLabel("Runtime: -- ms | FPS: --")
        algo_layout.addWidget(self.lbl_runtime)
        
        # Detailed timing (only visible in SVO mode)
        self.lbl_detailed_timing = QLabel("")
        algo_layout.addWidget(self.lbl_detailed_timing)
        self.lbl_detailed_timing.setVisible(False)
        
        algo_layout.addStretch()
        
        self.btn_save = QPushButton("💾 Save Result")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)
        algo_layout.addWidget(self.btn_save)

        self.btn_export_pose = QPushButton("📐 Export Silhouette")
        self.btn_export_pose.setToolTip("Export detected edge pixels as NPZ for Pose Optimizer (no initial guess)")
        self.btn_export_pose.clicked.connect(self.export_silhouette_for_pose_optimizer)
        self.btn_export_pose.setEnabled(False)
        algo_layout.addWidget(self.btn_export_pose)
        
        main_layout.addLayout(algo_layout)
        
        # Parameter panel
        param_group = QGroupBox("Algorithm Parameters")
        param_main_layout = QVBoxLayout()
        param_layout = QHBoxLayout()
        
        # Store all parameter layouts for enable/disable
        self.param_widgets = {}
        
        # Canny Low threshold
        self.canny_low_layout = QVBoxLayout()
        lbl = QLabel("Canny Low:")
        self.canny_low_layout.addWidget(lbl)
        self.slider_canny_low = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny_low.setRange(10, 200)
        self.slider_canny_low.setValue(self.params['canny_low'])
        self.slider_canny_low.valueChanged.connect(lambda v: self.update_param('canny_low', v))
        self.lbl_canny_low = QLabel(str(self.params['canny_low']))
        self.canny_low_layout.addWidget(self.slider_canny_low)
        self.canny_low_layout.addWidget(self.lbl_canny_low)
        param_layout.addLayout(self.canny_low_layout)
        self.param_widgets['canny_low'] = [lbl, self.slider_canny_low, self.lbl_canny_low]
        
        # Canny High threshold
        self.canny_high_layout = QVBoxLayout()
        lbl = QLabel("Canny High:")
        self.canny_high_layout.addWidget(lbl)
        self.slider_canny_high = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny_high.setRange(50, 300)
        self.slider_canny_high.setValue(self.params['canny_high'])
        self.slider_canny_high.valueChanged.connect(lambda v: self.update_param('canny_high', v))
        self.lbl_canny_high = QLabel(str(self.params['canny_high']))
        self.canny_high_layout.addWidget(self.slider_canny_high)
        self.canny_high_layout.addWidget(self.lbl_canny_high)
        param_layout.addLayout(self.canny_high_layout)
        self.param_widgets['canny_high'] = [lbl, self.slider_canny_high, self.lbl_canny_high]
        
        # Gaussian sigma
        self.sigma_layout = QVBoxLayout()
        lbl = QLabel("Gaussian σ:")
        self.sigma_layout.addWidget(lbl)
        self.slider_sigma = QSlider(Qt.Orientation.Horizontal)
        self.slider_sigma.setRange(1, 50)
        self.slider_sigma.setValue(int(self.params['gaussian_sigma'] * 10))
        self.slider_sigma.valueChanged.connect(lambda v: self.update_param('gaussian_sigma', v / 10.0))
        self.lbl_sigma = QLabel(f"{self.params['gaussian_sigma']:.1f}")
        self.sigma_layout.addWidget(self.slider_sigma)
        self.sigma_layout.addWidget(self.lbl_sigma)
        param_layout.addLayout(self.sigma_layout)
        self.param_widgets['gaussian_sigma'] = [lbl, self.slider_sigma, self.lbl_sigma]
        
        # CLAHE clip limit
        self.clahe_layout = QVBoxLayout()
        lbl = QLabel("CLAHE Clip:")
        self.clahe_layout.addWidget(lbl)
        self.slider_clahe = QSlider(Qt.Orientation.Horizontal)
        self.slider_clahe.setRange(10, 100)
        self.slider_clahe.setValue(int(self.params['clahe_clip'] * 10))
        self.slider_clahe.valueChanged.connect(lambda v: self.update_param('clahe_clip', v / 10.0))
        self.lbl_clahe = QLabel(f"{self.params['clahe_clip']:.1f}")
        self.clahe_layout.addWidget(self.slider_clahe)
        self.clahe_layout.addWidget(self.lbl_clahe)
        param_layout.addLayout(self.clahe_layout)
        self.param_widgets['clahe_clip'] = [lbl, self.slider_clahe, self.lbl_clahe]
        
        # Morph kernel
        self.morph_layout = QVBoxLayout()
        lbl = QLabel("Morph Kernel:")
        self.morph_layout.addWidget(lbl)
        self.slider_morph = QSlider(Qt.Orientation.Horizontal)
        self.slider_morph.setRange(3, 15)
        self.slider_morph.setValue(self.params['morph_kernel'])
        self.slider_morph.valueChanged.connect(lambda v: self.update_param('morph_kernel', v))
        self.lbl_morph = QLabel(str(self.params['morph_kernel']))
        self.morph_layout.addWidget(self.slider_morph)
        self.morph_layout.addWidget(self.lbl_morph)
        param_layout.addLayout(self.morph_layout)
        self.param_widgets['morph_kernel'] = [lbl, self.slider_morph, self.lbl_morph]
        
        # Snake parameters (second row)
        param_layout2 = QHBoxLayout()
        
        # Snake Alpha
        self.snake_alpha_layout = QVBoxLayout()
        lbl = QLabel("Snake Alpha:")
        self.snake_alpha_layout.addWidget(lbl)
        self.slider_snake_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_snake_alpha.setRange(1, 100)
        self.slider_snake_alpha.setValue(int(self.params['snake_alpha'] * 1000))
        self.slider_snake_alpha.valueChanged.connect(lambda v: self.update_param('snake_alpha', v / 1000.0))
        self.lbl_snake_alpha = QLabel(f"{self.params['snake_alpha']:.3f}")
        self.snake_alpha_layout.addWidget(self.slider_snake_alpha)
        self.snake_alpha_layout.addWidget(self.lbl_snake_alpha)
        param_layout2.addLayout(self.snake_alpha_layout)
        self.param_widgets['snake_alpha'] = [lbl, self.slider_snake_alpha, self.lbl_snake_alpha]
        
        # Snake Beta
        self.snake_beta_layout = QVBoxLayout()
        lbl = QLabel("Snake Beta:")
        self.snake_beta_layout.addWidget(lbl)
        self.slider_snake_beta = QSlider(Qt.Orientation.Horizontal)
        self.slider_snake_beta.setRange(1, 500)
        self.slider_snake_beta.setValue(int(self.params['snake_beta'] * 1000))
        self.slider_snake_beta.valueChanged.connect(lambda v: self.update_param('snake_beta', v / 1000.0))
        self.lbl_snake_beta = QLabel(f"{self.params['snake_beta']:.3f}")
        self.snake_beta_layout.addWidget(self.slider_snake_beta)
        self.snake_beta_layout.addWidget(self.lbl_snake_beta)
        param_layout2.addLayout(self.snake_beta_layout)
        self.param_widgets['snake_beta'] = [lbl, self.slider_snake_beta, self.lbl_snake_beta]
        
        # Snake Iterations
        self.snake_iter_layout = QVBoxLayout()
        lbl = QLabel("Snake Iterations:")
        self.snake_iter_layout.addWidget(lbl)
        self.slider_snake_iter = QSlider(Qt.Orientation.Horizontal)
        self.slider_snake_iter.setRange(10, 500)
        self.slider_snake_iter.setValue(self.params['snake_iterations'])
        self.slider_snake_iter.valueChanged.connect(lambda v: self.update_param('snake_iterations', v))
        self.lbl_snake_iter = QLabel(str(self.params['snake_iterations']))
        self.snake_iter_layout.addWidget(self.slider_snake_iter)
        self.snake_iter_layout.addWidget(self.lbl_snake_iter)
        param_layout2.addLayout(self.snake_iter_layout)
        self.param_widgets['snake_iterations'] = [lbl, self.slider_snake_iter, self.lbl_snake_iter]
        
        # Depth Filter (SVO Mode only)
        self.depth_layout = QVBoxLayout()
        depth_control_layout = QHBoxLayout()
        
        self.chk_use_depth = QCheckBox("Enable Depth Filter")
        self.chk_use_depth.setChecked(self.params['use_depth_filter'])
        self.chk_use_depth.stateChanged.connect(lambda state: self.update_param('use_depth_filter', state == Qt.CheckState.Checked.value))
        depth_control_layout.addWidget(self.chk_use_depth)
        
        lbl_depth = QLabel("Depth Search Radius (px):")
        depth_control_layout.addWidget(lbl_depth)
        
        self.slider_depth_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_depth_radius.setRange(1, 50)
        self.slider_depth_radius.setValue(self.params['depth_radius'])
        self.slider_depth_radius.valueChanged.connect(lambda v: self.update_param('depth_radius', v))
        depth_control_layout.addWidget(self.slider_depth_radius)
        
        self.lbl_depth_radius = QLabel(str(self.params['depth_radius']))
        depth_control_layout.addWidget(self.lbl_depth_radius)
        
        self.depth_layout.addLayout(depth_control_layout)
        param_layout2.addLayout(self.depth_layout)
        self.param_widgets['depth_radius'] = [self.chk_use_depth, lbl_depth, self.slider_depth_radius, self.lbl_depth_radius]
        
        # Hide depth filter initially (only for SVO mode)
        for widget in self.param_widgets['depth_radius']:
            widget.setVisible(False)
        
        # Add layouts to main param layout
        param_main_layout.addLayout(param_layout)
        param_main_layout.addLayout(param_layout2)
        
        # Reset button
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        self.btn_reset_params = QPushButton("🔄 Reset Parameters")
        self.btn_reset_params.clicked.connect(self.reset_parameters)
        reset_layout.addWidget(self.btn_reset_params)
        param_main_layout.addLayout(reset_layout)
        
        param_group.setLayout(param_main_layout)
        main_layout.addWidget(param_group)
        
        # Image viewer
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        main_layout.addWidget(self.view)
        
        # Initialize parameter visibility
        self.on_algorithm_changed(self.combo_algorithm.currentText())
        
        # Initialize SVO mode components
        self.init_svo_mode()
    
    def init_svo_mode(self):
        """Initialize SVO2 mode components (called once at GUI startup)."""
        # Scan default model directory for available model files
        self.svo2_scan_model_files()

        # Load YOLO model if a model file is already selected
        # (lazy-loaded on first use in SVO2 mode)

        # Load debug images (Debug Mode is default at startup)
        self.load_image_list()
        self.load_first_image()
    
    def on_mode_changed(self, checked: bool):
        """Handle mode switch between Debug and SVO2 mode."""
        if not checked:
            return

        if self.radio_debug_mode.isChecked():
            self.current_mode = "debug"
            # Close ZED camera if it was open
            self.svo2_close_camera()
            # Show / hide panels
            self.svo2_group.setVisible(False)
            self.lbl_detailed_timing.setVisible(False)
            self.btn_extract_images.setVisible(True)
            for widget in self.param_widgets['depth_radius']:
                widget.setVisible(False)
            # Load debug images
            self.load_image_list()
            self.load_first_image()

        elif self.radio_svo_mode.isChecked():
            self.current_mode = "svo"
            # Show / hide panels
            self.svo2_group.setVisible(True)
            self.lbl_detailed_timing.setVisible(True)
            self.btn_extract_images.setVisible(False)
            for widget in self.param_widgets['depth_radius']:
                widget.setVisible(False)  # depth filter not used in SVO2 direct mode

            if not YOLO_AVAILABLE:
                QMessageBox.warning(
                    self, "YOLO Not Available",
                    "ultralytics is not installed — YOLO inference unavailable.\n"
                    "Install with: pip install ultralytics"
                )
            if not ZED_AVAILABLE:
                QMessageBox.warning(
                    self, "ZED SDK Not Available",
                    "pyzed is not installed — SVO2 direct-read unavailable.\n"
                    "SVO2 mode requires the ZED SDK on this machine."
                )

            # Clear any previously loaded debug-mode image list
            self.image_files = []
            self.current_image = None
            self._clean_image = None
            self.current_bbox = None
            self.lbl_image_info.setText("No SVO2 file open")
    
    def on_svo_folder_changed(self, folder_name: str):
        """Legacy stub — no longer used in SVO2 direct mode."""
        pass

    def load_svo_images(self):
        """Legacy stub — no longer used in SVO2 direct mode."""
        pass

    # ==================================================================
    # SVO2 direct-read methods (ZED SDK)
    # ==================================================================

    def svo2_scan_model_files(self):
        """Scan yolo_model_dir for .pt / .onnx / .engine files and populate combo."""
        self.combo_svo2_model.blockSignals(True)
        self.combo_svo2_model.clear()
        if self.yolo_model_dir.exists():
            model_files = sorted(
                f.name for f in self.yolo_model_dir.iterdir()
                if f.suffix in ('.pt', '.onnx', '.engine')
            )
            self.combo_svo2_model.addItems(model_files)
            if model_files:
                # Prefer .pt as default
                pt_files = [f for f in model_files if f.endswith('.pt')]
                default = pt_files[0] if pt_files else model_files[0]
                self.combo_svo2_model.setCurrentText(default)
        self.combo_svo2_model.blockSignals(False)

    def on_svo2_pick_model_dir(self):
        """Browse for a different model directory."""
        d = QFileDialog.getExistingDirectory(
            self, "Select YOLO model folder", str(self.yolo_model_dir)
        )
        if d:
            self.yolo_model_dir = Path(d)
            self.svo2_scan_model_files()
            # Unload currently loaded model — will be reloaded on next YOLO run
            self.yolo_model = None

    def on_svo2_model_changed(self, model_name: str):
        """Unload cached model when selection changes (will reload on next run)."""
        self.yolo_model = None

    def on_svo2_file_pick(self):
        """Open a file dialog to select an SVO2 file, then open it."""
        start_dir = "/media/angelo/DRONE_DATA1" if Path("/media/angelo/DRONE_DATA1").exists() else str(Path.home())
        svo2_path, _ = QFileDialog.getOpenFileName(
            self, "Select SVO2 File", start_dir, "SVO2 Files (*.svo2);;All Files (*)"
        )
        if not svo2_path:
            return
        self.svo2_file_path = Path(svo2_path)
        self.lbl_svo2_file.setText(self.svo2_file_path.name)
        self.svo2_open_camera()

    def on_svo2_depth_mode_changed(self, mode_name: str):
        """Depth mode changed — reopen camera if already open."""
        self.svo2_depth_mode = mode_name
        if self.zed_camera is not None:
            # Reopen with new depth mode, restoring frame position
            saved_frame = self.svo2_current_frame
            self.svo2_open_camera()
            if self.zed_camera is not None:
                self.svo2_current_frame = min(saved_frame, max(0, self.svo2_total_frames - 1))
                self.svo2_grab_frame()

    def svo2_open_camera(self):
        """Open the selected SVO2 file with the ZED SDK."""
        if not ZED_AVAILABLE:
            QMessageBox.critical(self, "ZED SDK Not Available",
                                 "pyzed is not installed — cannot open SVO2 file.")
            return
        if self.svo2_file_path is None or not self.svo2_file_path.exists():
            QMessageBox.warning(self, "File Not Found",
                                f"SVO2 file not found:\n{self.svo2_file_path}")
            return

        # Close any previously open camera
        self.svo2_close_camera()

        # Map depth-mode name to SDK enum (ZED SDK v5 – AI modes only; legacy removed)
        # See docs/guides/zed_depth_modes.md for details
        depth_mode_map = {
            "NEURAL_PLUS":  sl.DEPTH_MODE.NEURAL_PLUS,
            "NEURAL":       sl.DEPTH_MODE.NEURAL,
            "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
            "NONE":         sl.DEPTH_MODE.NONE,
        }
        depth_mode = depth_mode_map.get(self.svo2_depth_mode, sl.DEPTH_MODE.NEURAL)

        cam = sl.Camera()
        init = sl.InitParameters()
        init.set_from_svo_file(str(self.svo2_file_path))
        init.svo_real_time_mode = False
        init.depth_mode = depth_mode
        init.coordinate_units = sl.UNIT.METER

        err = cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            QMessageBox.critical(self, "ZED Open Error",
                                 f"Could not open SVO2 file:\n{err}")
            return

        self.zed_camera = cam
        self.svo2_total_frames = cam.get_svo_number_of_frames()
        self.svo2_current_frame = 0
        print(f"[SVO2] Opened: {self.svo2_file_path.name}  |  "
              f"{self.svo2_total_frames} frames  |  depth={self.svo2_depth_mode}")
        self.svo2_grab_frame()

    def svo2_close_camera(self):
        """Close the ZED camera if open."""
        if self.zed_camera is not None:
            try:
                self.zed_camera.close()
            except Exception:
                pass
            self.zed_camera = None
        self.svo2_raw_frame = None
        self.svo2_right_frame = None
        self.svo2_depth_map = None
        self.svo2_bbox = None
        self.svo2_bbox_right = None
        self.current_bbox = None

    def svo2_navigate(self, delta: int):
        """Navigate forward/backward by delta frames."""
        if self.zed_camera is None:
            QMessageBox.information(self, "No SVO2 Open", "Please open an SVO2 file first.")
            return
        new_frame = self.svo2_current_frame + delta
        new_frame = max(0, min(self.svo2_total_frames - 1, new_frame))
        if new_frame == self.svo2_current_frame and delta != 0:
            return  # already at boundary
        self.svo2_current_frame = new_frame
        # Set exact position (handles both forward and backward seek)
        self.zed_camera.set_svo_position(self.svo2_current_frame)
        self.svo2_grab_frame()

    def svo2_grab_frame(self):
        """Grab the current frame from the ZED camera and update display."""
        if self.zed_camera is None:
            return

        runtime = sl.RuntimeParameters()
        err = self.zed_camera.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[SVO2] grab() returned: {err}")
            return

        # Retrieve left camera image (BGRA → BGR)
        left_mat = sl.Mat()
        self.zed_camera.retrieve_image(left_mat, sl.VIEW.LEFT)
        frame_bgra = left_mat.get_data()
        self.svo2_raw_frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        # Retrieve right camera image (BGRA → BGR)
        right_mat = sl.Mat()
        self.zed_camera.retrieve_image(right_mat, sl.VIEW.RIGHT)
        right_bgra = right_mat.get_data()
        self.svo2_right_frame = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2BGR)

        # Retrieve depth map (float32 metres; NaN / inf = invalid)
        # Depth is aligned to the LEFT camera coordinate frame.
        if self.svo2_depth_mode != "NONE":
            depth_mat = sl.Mat()
            self.zed_camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            self.svo2_depth_map = depth_mat.get_data().copy()  # float32 (H, W)
        else:
            self.svo2_depth_map = None

        # Update frame counter label
        self.lbl_svo2_frame.setText(
            f"Frame: {self.svo2_current_frame + 1} / {self.svo2_total_frames}"
        )

        # Reset previous YOLO result when navigating
        self.svo2_bbox = None
        self.svo2_bbox_right = None
        self.current_bbox = None
        # _clean_image = left frame (algorithms always operate on left frame)
        self._clean_image = self.svo2_raw_frame.copy()
        self.current_image = self.svo2_raw_frame.copy()

        # Show both cameras side by side (no bbox yet)
        self.display_image(self._svo2_build_display_image(), draw_bbox=False)
        self.lbl_image_info.setText(
            f"SVO2: {self.svo2_file_path.name if self.svo2_file_path else '?'}  "
            f"— frame {self.svo2_current_frame + 1}/{self.svo2_total_frames}"
        )
        self.lbl_svo2_timing.setText("YOLO: --ms")
        self.current_result = None
        self.current_edge_pixels_abs = None
        self.btn_save.setEnabled(False)
        self.btn_export_pose.setEnabled(False)

    def _ensure_svo2_yolo_loaded(self) -> bool:
        """Load YOLO model if not already loaded. Returns True on success."""
        if not YOLO_AVAILABLE:
            QMessageBox.critical(self, "YOLO unavailable",
                                 "ultralytics not installed.")
            return False
        if self.yolo_model is not None:
            return True
        model_name = self.combo_svo2_model.currentText()
        if not model_name:
            QMessageBox.warning(self, "No Model Selected",
                                "Please select a YOLO model file.")
            return False
        model_path = self.yolo_model_dir / model_name
        if not model_path.exists():
            QMessageBox.warning(self, "Model Not Found",
                                f"Model file not found:\n{model_path}")
            return False
        try:
            import torch
            # Disable cuDNN to avoid 'GET was unable to find an engine' on Jetson
            if torch.cuda.is_available():
                torch.backends.cudnn.enabled = False
            print(f"[SVO2] Loading YOLO model: {model_path}")
            self.yolo_model = YOLO(str(model_path))
            print("[SVO2] YOLO model loaded.")
            return True
        except Exception as e:
            QMessageBox.critical(self, "YOLO Load Error", str(e))
            return False

    def _set_svo2_yolo_size(self, size: int):
        """Update YOLO inference size."""
        self.svo2_yolo_input_size = size

    def _yolo_detect_bbox(
        self,
        frame: np.ndarray,
        input_size: int,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Run YOLO on a single frame and return (bbox, elapsed_ms).
        Timing includes the downscale step. Returns (None, ms) if no detection.
        """
        h, w = frame.shape[:2]
        if input_size < w:
            scale = input_size / w
            resized = cv2.resize(
                frame,
                (input_size, int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = frame
            scale = 1.0

        t0 = time.perf_counter()
        results = self.yolo_model(resized, imgsz=input_size, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if not results or len(results[0].boxes) == 0:
            return None, elapsed

        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        if scale != 1.0:
            x1, x2 = x1 / scale, x2 / scale
            y1, y2 = y1 / scale, y2 / scale
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        return (x1, y1, x2, y2), elapsed

    def run_yolo_svo2(self):
        """Run YOLO separately on the left and right SVO2 frames."""
        if self.svo2_raw_frame is None:
            QMessageBox.information(self, "No Frame", "Navigate to a frame first.")
            return
        if not self._ensure_svo2_yolo_loaded():
            return

        input_size = self.svo2_yolo_input_size

        # ── Left frame ──────────────────────────────────────────────────────
        try:
            bbox_left, ms_left = self._yolo_detect_bbox(self.svo2_raw_frame, input_size)
        except Exception as e:
            QMessageBox.warning(self, "YOLO Error (left)", str(e))
            return

        # ── Right frame ─────────────────────────────────────────────────────
        bbox_right, ms_right = None, 0.0
        if self.svo2_right_frame is not None:
            try:
                bbox_right, ms_right = self._yolo_detect_bbox(self.svo2_right_frame, input_size)
            except Exception as e:
                print(f"[SVO2] YOLO right-frame error: {e}")

        total_ms = ms_left + ms_right
        det_left  = "✓" if bbox_left  is not None else "–"
        det_right = "✓" if bbox_right is not None else "–"
        self.lbl_svo2_timing.setText(
            f"YOLO L{det_left} R{det_right}: {total_ms:.1f} ms"
        )

        if bbox_left is None and bbox_right is None:
            QMessageBox.information(self, "No Detection",
                                    "YOLO found no object in left or right frame.")
            self.svo2_bbox = None
            self.svo2_bbox_right = None
            self.current_bbox = None
            self.display_image(self._svo2_build_display_image(), draw_bbox=False)
            return

        self.svo2_bbox = bbox_left
        self.svo2_bbox_right = bbox_right
        self.current_bbox = self.svo2_bbox  # left bbox is the primary (for pose export)

        self.display_image(self._svo2_build_display_image(), draw_bbox=False)
        self.btn_save.setEnabled(False)
        self.btn_export_pose.setEnabled(False)

    def _svo2_build_display_image(self) -> np.ndarray:
        """
        Build the side-by-side display image:
          LEFT panel  │  RIGHT panel
        Both panels receive their independent YOLO bbox and depth overlay.
        Depth is computed from the left stereo frame (ZED convention) and applied
        to both panels at the same pixel coordinates – a small sub-pixel offset
        on the right panel is acceptable at operational drone distances (> 2 m).
        """
        if self.svo2_raw_frame is None:
            return np.zeros((720, 2560, 3), dtype=np.uint8)

        left_panel  = self.svo2_raw_frame.copy()
        right_panel = (
            self.svo2_right_frame.copy()
            if self.svo2_right_frame is not None
            else np.zeros_like(left_panel)
        )

        # ── Left panel: YOLO bbox + depth overlay ───────────────────────────
        if self.svo2_bbox is not None:
            x1, y1, x2, y2 = self.svo2_bbox
            cv2.rectangle(left_panel, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(left_panel, "YOLO-L", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            if self.svo2_show_depth_overlay and self.svo2_depth_map is not None:
                left_panel = self._svo2_apply_depth_overlay(left_panel, self.svo2_bbox)

        # ── Right panel: YOLO bbox + depth overlay ──────────────────────────
        if self.svo2_bbox_right is not None:
            x1, y1, x2, y2 = self.svo2_bbox_right
            cv2.rectangle(right_panel, (x1, y1), (x2, y2), (0, 210, 0), 2)
            cv2.putText(right_panel, "YOLO-R", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 0), 1)
            if self.svo2_show_depth_overlay and self.svo2_depth_map is not None:
                right_panel = self._svo2_apply_depth_overlay(right_panel, self.svo2_bbox_right)

        # ── Camera labels ────────────────────────────────────────────────────
        cv2.putText(left_panel,  "LEFT",  (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right_panel, "RIGHT", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return np.hstack([left_panel, right_panel])

    def _svo2_apply_depth_overlay(
        self,
        panel: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Overlay a TURBO colormap depth map inside *bbox* on *panel*.
        panel  — single camera image (H×W×3 BGR), modified in-place copy.
        bbox   — (x1,y1,x2,y2) in panel coordinates.
        Depth map is always from the left camera; applying it to the right panel
        introduces a small parallax offset that is visually acceptable at > 2 m.
        """
        x1, y1, x2, y2 = bbox
        display = panel
        depth_crop = self.svo2_depth_map[y1:y2, x1:x2].copy()

        # Mask: only finite, positive depth values
        valid = np.isfinite(depth_crop) & (depth_crop > 0)
        if not valid.any():
            return display

        d_min = float(depth_crop[valid].min())
        d_max = float(depth_crop[valid].max())
        if d_max - d_min < 1e-4:
            return display

        # Normalise to 0-255
        norm = np.zeros_like(depth_crop, dtype=np.uint8)
        norm[valid] = ((depth_crop[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        # Apply colormap; black out invalid pixels
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        colored[~valid] = 0

        # Blend with original image
        alpha = self.svo2_depth_opacity
        roi = display[y1:y2, x1:x2].astype(np.float32)
        col_f = colored.astype(np.float32)
        # Only blend where depth is valid
        blended = roi.copy()
        mask3 = np.stack([valid, valid, valid], axis=-1)
        blended[mask3] = (roi * (1.0 - alpha) + col_f * alpha)[mask3]
        display[y1:y2, x1:x2] = blended.astype(np.uint8)

        # Depth range label inside bbox corner
        cv2.putText(
            display,
            f"{d_min:.1f}m – {d_max:.1f}m",
            (x1 + 4, y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

        # Simple vertical colour legend on right edge of bbox (3 px wide strip)
        legend_x = min(x2 + 4, display.shape[1] - 4)
        legend_h = y2 - y1
        if legend_h > 20 and legend_x + 12 < display.shape[1]:
            strip = np.arange(255, -1, -256 / legend_h, dtype=np.float32)[:legend_h]
            strip = strip.astype(np.uint8).reshape(-1, 1)
            strip_colored = cv2.applyColorMap(strip, cv2.COLORMAP_TURBO)  # (H, 1, 3)
            strip_colored = np.repeat(strip_colored, 10, axis=1)           # (H, 10, 3)
            r_y1, r_y2 = y1, y1 + legend_h
            r_x1, r_x2 = legend_x, legend_x + 10
            if r_x2 <= display.shape[1] and r_y2 <= display.shape[0]:
                display[r_y1:r_y2, r_x1:r_x2] = strip_colored
                # labels
                cv2.putText(display, f"{d_min:.1f}m", (r_x1, r_y2 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(display, f"{d_max:.1f}m", (r_x1, r_y1 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return display

    def on_svo2_depth_overlay_toggled(self, state: int):
        """Toggle depth overlay visibility."""
        self.svo2_show_depth_overlay = (state == Qt.CheckState.Checked.value)
        if self.svo2_raw_frame is not None:
            self.display_image(self._svo2_build_display_image(), draw_bbox=False)

    def on_svo2_opacity_changed(self, value: int):
        """Update depth overlay opacity from slider."""
        self.svo2_depth_opacity = value / 100.0
        self.lbl_svo2_opacity.setText(f"{value}%")
        if self.svo2_raw_frame is not None:
            self.display_image(self._svo2_build_display_image(), draw_bbox=False)

    # ------------------------------------------------------------------
    # Extraction dialog
    # ------------------------------------------------------------------
    def open_extraction_dialog(self):
        """Open dialog to configure and run image extraction."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Bilder extrahieren")
        dlg.setMinimumWidth(520)
        layout = QVBoxLayout(dlg)

        # Source folder
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Quellordner:"))
        self._extract_src_label = QLabel(str(self.extraction_source_dir))
        self._extract_src_label.setWordWrap(True)
        src_row.addWidget(self._extract_src_label, 1)
        btn_pick_src = QPushButton("…")
        btn_pick_src.setFixedWidth(32)

        def _pick_src():
            d = QFileDialog.getExistingDirectory(
                dlg, "Quellordner wählen", str(self.extraction_source_dir)
            )
            if d:
                self.extraction_source_dir = Path(d)
                self._extract_src_label.setText(d)

        btn_pick_src.clicked.connect(_pick_src)
        src_row.addWidget(btn_pick_src)
        layout.addLayout(src_row)

        # Max depth
        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Max. Tiefe (m):"))
        spin_depth = QDoubleSpinBox()
        spin_depth.setRange(0.1, 200.0)
        spin_depth.setSingleStep(0.5)
        spin_depth.setDecimals(2)
        spin_depth.setValue(15.0)
        depth_row.addWidget(spin_depth)
        depth_row.addStretch()
        layout.addLayout(depth_row)

        # Progress bar + status
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        layout.addWidget(progress_bar)

        lbl_status = QLabel("Bereit.")
        lbl_status.setWordWrap(True)
        layout.addWidget(lbl_status)

        # Buttons
        btn_box = QDialogButtonBox()
        btn_start = QPushButton("▶ Starten")
        btn_abort = QPushButton("✖ Abbrechen")
        btn_close = QPushButton("Schließen")
        btn_abort.setEnabled(False)
        btn_box.addButton(btn_start, QDialogButtonBox.ButtonRole.ActionRole)
        btn_box.addButton(btn_abort, QDialogButtonBox.ButtonRole.ActionRole)
        btn_box.addButton(btn_close, QDialogButtonBox.ButtonRole.RejectRole)
        btn_close.clicked.connect(dlg.reject)
        layout.addWidget(btn_box)

        self._extraction_worker: Optional[ExtractionWorker] = None

        def _start():
            btn_start.setEnabled(False)
            btn_abort.setEnabled(True)
            progress_bar.setValue(0)
            lbl_status.setText("Starte …")

            self._extraction_worker = ExtractionWorker(
                source_dir=self.extraction_source_dir,
                originals_dir=self.project_root / "data" / "debug_images" / "originals",
                labels_dir=self.project_root / "data" / "debug_images" / "labels",
                max_depth=spin_depth.value(),
            )
            w = self._extraction_worker
            w.progress.connect(progress_bar.setValue)
            w.status.connect(lbl_status.setText)
            w.finished_ok.connect(lambda c, s: _on_done(c, s))
            w.error.connect(lambda msg: _on_error(msg))
            w.start()

        def _abort():
            if self._extraction_worker:
                self._extraction_worker.abort()
            btn_abort.setEnabled(False)
            btn_start.setEnabled(True)

        def _on_done(copied: int, skipped: int):
            btn_abort.setEnabled(False)
            btn_start.setEnabled(True)
            lbl_status.setText(
                f"Fertig! {copied} Bilder kopiert, {skipped} übersprungen (Namenskollision)."
            )
            progress_bar.setValue(100)
            # Reload image list so GUI reflects new files
            self.load_image_list()
            self.load_first_image()

        def _on_error(msg: str):
            btn_abort.setEnabled(False)
            btn_start.setEnabled(True)
            lbl_status.setText(f"Fehler: {msg}")
            QMessageBox.critical(dlg, "Fehler", msg)

        btn_start.clicked.connect(_start)
        btn_abort.clicked.connect(_abort)

        dlg.exec()

    # ------------------------------------------------------------------
    # (end extraction dialog)
    # ------------------------------------------------------------------

    def update_param(self, key: str, value):
        """Update parameter and refresh label."""
        self.params[key] = value
        if key == 'canny_low':
            self.lbl_canny_low.setText(str(value))
        elif key == 'canny_high':
            self.lbl_canny_high.setText(str(value))
        elif key == 'gaussian_sigma':
            self.lbl_sigma.setText(f"{value:.1f}")
        elif key == 'clahe_clip':
            self.lbl_clahe.setText(f"{value:.1f}")
        elif key == 'morph_kernel':
            self.lbl_morph.setText(str(value))
        elif key == 'snake_alpha':
            self.lbl_snake_alpha.setText(f"{value:.3f}")
        elif key == 'snake_beta':
            self.lbl_snake_beta.setText(f"{value:.3f}")
        elif key == 'snake_iterations':
            self.lbl_snake_iter.setText(str(value))
        elif key == 'depth_radius':
            self.lbl_depth_radius.setText(str(value))
        elif key == 'use_depth_filter':
            # Boolean value, no label update needed
            pass
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.params = self.default_params.copy()
        # Update sliders
        self.slider_canny_low.setValue(self.params['canny_low'])
        self.slider_canny_high.setValue(self.params['canny_high'])
        self.slider_sigma.setValue(int(self.params['gaussian_sigma'] * 10))
        self.slider_clahe.setValue(int(self.params['clahe_clip'] * 10))
        self.slider_morph.setValue(self.params['morph_kernel'])
        self.slider_snake_alpha.setValue(int(self.params['snake_alpha'] * 1000))
        self.slider_snake_beta.setValue(int(self.params['snake_beta'] * 1000))
        self.slider_snake_iter.setValue(self.params['snake_iterations'])
        self.slider_depth_radius.setValue(self.params['depth_radius'])
        self.chk_use_depth.setChecked(self.params['use_depth_filter'])
    
    def on_algorithm_changed(self, algo_name: str):
        """Enable/disable parameters based on selected algorithm."""
        if algo_name not in self.algorithms:
            return
        
        used_params = self.algorithms[algo_name]['params']
        
        # Enable/disable each parameter widget
        for param_key, widgets in self.param_widgets.items():
            enabled = param_key in used_params
            for widget in widgets:
                widget.setEnabled(enabled)
        
    def load_or_create_session(self):
        """Load existing sessions or create new session entry."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                sessions = json.load(f)
        else:
            sessions = {}
        
        # Find next session number for each algorithm
        self.session_numbers = {}
        for algo_name in ["Canny (Auto)", "Canny + Contour Silhouette", "CLAHE + Canny", "Canny + Hough Lines"]:
            algo_key = self.sanitize_algo_name(algo_name)
            if algo_key in sessions:
                self.session_numbers[algo_key] = max(sessions[algo_key]) + 1 if sessions[algo_key] else 1
            else:
                self.session_numbers[algo_key] = 1
                sessions[algo_key] = []
        
        # Save updated sessions
        with open(self.session_file, 'w') as f:
            json.dump(sessions, f, indent=2)
            
    def sanitize_algo_name(self, name: str) -> str:
        """Convert algorithm name to filesystem-safe string."""
        return name.replace(" ", "_").replace("+", "").replace("(", "").replace(")", "").lower()
    
    def load_image_list(self):
        """Load list of images from debug directory."""
        if not self.debug_images_dir.exists():
            QMessageBox.warning(self, "Warning", f"Debug images directory not found:\n{self.debug_images_dir}")
            return
        
        self.image_files = sorted([
            f for f in self.debug_images_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ])
        
        if not self.image_files:
            QMessageBox.warning(self, "Warning", "No images found in debug directory.")
    
    def load_first_image(self):
        """Load the first image."""
        if self.image_files:
            self.current_image_idx = 0
            self.load_current_image()
    
    def load_current_image(self):
        """Load current image and its YOLO labels or run YOLO inference."""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_image_idx]
        # Read-only: Never modify original files
        self.current_image = cv2.imread(str(img_path))
        self._clean_image  = self.current_image.copy()   # frozen original — never modify

        if self.current_image is None:
            QMessageBox.warning(self, "Error", f"Failed to load image:\n{img_path}")
            return
        
        # Mode-specific loading
        if self.current_mode == "debug":
            # Load YOLO labels from file
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            self.current_bbox = self.load_yolo_bbox(label_path, self.current_image.shape)
            self.current_depth_data = None
            
        elif self.current_mode == "svo":
            # Run YOLO inference + load depth data
            self.current_bbox = None
            self.current_depth_data = None
            
            # Load depth data (.npy file)
            depth_path = img_path.with_suffix('.npy')
            if depth_path.exists():
                try:
                    self.current_depth_data = np.load(str(depth_path))
                except Exception as e:
                    print(f"Failed to load depth data: {e}")
                    self.current_depth_data = None
            
            # Run YOLO inference
            if self.yolo_model:
                try:
                    start_yolo = time.perf_counter()
                    results = self.yolo_model(self.current_image, verbose=False)
                    end_yolo = time.perf_counter()
                    self.yolo_inference_time = (end_yolo - start_yolo) * 1000.0
                    
                    # Extract first detection bbox
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        box = results[0].boxes[0]  # First detection
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        self.current_bbox = (x1, y1, x2, y2)
                    else:
                        QMessageBox.information(self, "No Detection", "YOLO did not detect any objects in this image.")
                except Exception as e:
                    QMessageBox.warning(self, "YOLO Error", f"YOLO inference failed: {e}")
                    self.yolo_inference_time = 0.0
        
        # Update UI
        self.lbl_image_info.setText(f"Image {self.current_image_idx + 1}/{len(self.image_files)}: {img_path.name}")
        # Display image WITHOUT bbox initially (bbox will be drawn after algorithm is applied)
        self.display_image(self.current_image, draw_bbox=True)
        self.current_result = None
        self.current_edge_pixels_abs = None
        self.btn_save.setEnabled(False)
        self.btn_export_pose.setEnabled(False)
        self.lbl_runtime.setText("Runtime: -- ms | FPS: --")
        self.lbl_detailed_timing.setText("")
    
    def load_yolo_bbox(self, label_path: Path, img_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Load first bounding box from YOLO label file."""
        if not label_path.exists():
            return None
        
        h, w = img_shape[:2]
        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    return None
                parts = line.split()
                if len(parts) < 5:
                    return None
                
                x_c, y_c, box_w, box_h = map(float, parts[1:5])
                x1 = int((x_c - box_w / 2) * w)
                y1 = int((y_c - box_h / 2) * h)
                x2 = int((x_c + box_w / 2) * w)
                y2 = int((y_c + box_h / 2) * h)
                
                # Clamp to image bounds
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))
                
                return (x1, y1, x2, y2)
        except Exception as e:
            print(f"Error loading bbox from {label_path}: {e}")
            return None
    
    def display_image(self, img: np.ndarray, draw_bbox: bool = True):
        """Display image in the viewer, preserving zoom level.
        
        Args:
            img: Image to display
            draw_bbox: If True, draw the bounding box on the image
        """
        # Save current zoom level
        current_zoom = self.view.get_zoom_level()
        
        # Draw bounding box if requested and available
        display_img = img.copy()
        if draw_bbox and self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange, thicker
            cv2.putText(display_img, "YOLO BBox", (x1, max(15, y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        # Convert BGR to RGB for Qt
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.pixmap_item.setPixmap(pixmap)
        
        # Restore zoom level or fit to view with better initial size
        if current_zoom > 1.01:  # Only restore if significantly zoomed
            self.view.set_zoom_level(current_zoom)
        else:
            # Fit to 80% of view to give better initial display
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.view.scale(0.95, 0.95)  # Slightly zoom in for better visibility
    
    def prev_image(self):
        """Load previous image."""
        if self.image_files and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_current_image()
    
    def next_image(self):
        """Load next image."""
        if self.image_files and self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self.load_current_image()
    
    def get_roi_with_padding(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract ROI from image based on bbox, expanded by bbox_padding pixels.
        Returns (roi_image, expanded_bbox_coordinates)

        The padding ensures the crop boundary lies OUTSIDE the object so that
        edge-detection algorithms do not mistake the hard crop border for an
        object edge.  The returned (x1,y1,x2,y2) already include the padding
        and are clamped to the image dimensions.
        """
        if self.current_bbox is None or self._clean_image is None:
            return self.current_image, None

        pad = int(self.params.get('bbox_padding', 5))
        bx1, by1, bx2, by2 = self.current_bbox
        # Use _clean_image for bounds (in SVO2 mode current_image may be composite)
        h, w = self._clean_image.shape[:2]

        # Expand by padding and clamp to image bounds
        x1 = max(0, bx1 - pad)
        y1 = max(0, by1 - pad)
        x2 = min(w, bx2 + pad)
        y2 = min(h, by2 + pad)

        # Extract padded ROI from the CLEAN original (never draw on this)
        roi = self._clean_image[y1:y2, x1:x2].copy()

        # Return ROI and PADDED bbox (in original image coordinates)
        return roi, (x1, y1, x2, y2)
    
    def has_valid_depth(self, contour, bbox_offset: Tuple[int, int]) -> bool:
        """
        Check if a contour has valid depth data within search radius.
        
        Args:
            contour: OpenCV contour in ROI coordinates
            bbox_offset: (x_offset, y_offset) to convert ROI coords to image coords
        
        Returns:
            True if valid depth found within radius, False otherwise
        """
        if not self.params['use_depth_filter'] or self.current_depth_data is None:
            return True  # No filtering if disabled or no depth data
        
        x_offset, y_offset = bbox_offset
        radius = self.params['depth_radius']
        depth_h, depth_w = self.current_depth_data.shape
        
        # Sample points along the contour
        num_samples = min(len(contour), 20)  # Sample up to 20 points
        indices = np.linspace(0, len(contour) - 1, num_samples, dtype=int)
        
        for idx in indices:
            pt = contour[idx][0]  # contour point in ROI coords
            # Convert to image coords
            x_img = pt[0] + x_offset
            y_img = pt[1] + y_offset
            
            # Search in neighborhood
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    check_x = x_img + dx
                    check_y = y_img + dy
                    
                    # Bounds check
                    if 0 <= check_y < depth_h and 0 <= check_x < depth_w:
                        depth_val = self.current_depth_data[check_y, check_x]
                        # Valid depth is non-zero and not NaN/Inf
                        if depth_val > 0 and np.isfinite(depth_val):
                            return True
        
        return False  # No valid depth found
    
    def apply_algorithm(self):
        """Apply selected algorithm to current image."""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        if self.current_bbox is None:
            QMessageBox.warning(self, "Warning", "No bounding box found for this image.")
            return

        algo_name = self.combo_algorithm.currentText()
        algo_func = self.algorithms[algo_name]['func']

        # ── Left / single-frame run ──────────────────────────────────────────
        start_edge = time.perf_counter()
        result_left = algo_func()   # operates on self._clean_image + self.current_bbox
        t_left = time.perf_counter() - start_edge

        # ── SVO2 mode: also run on right frame ───────────────────────────────
        if self.current_mode == "svo" and self.svo2_right_frame is not None:
            # Save left context
            _clean_left = self._clean_image
            _bbox_left  = self.current_bbox
            fw_single   = _clean_left.shape[1]   # single-frame pixel width

            # Switch to right frame context
            self._clean_image = self.svo2_right_frame.copy()
            self.current_bbox = self.svo2_bbox_right  # may be None → algo will warn

            if self.current_bbox is not None:
                t_right0 = time.perf_counter()
                result_right = algo_func()
                t_right = time.perf_counter() - t_right0
            else:
                result_right = self.svo2_right_frame.copy()
                cv2.putText(result_right, "No YOLO bbox (right)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
                t_right = 0.0

            # Restore left context
            self._clean_image = _clean_left
            self.current_bbox = _bbox_left

            # Composite left | right
            composite = np.hstack([result_left, result_right])

            # Draw both YOLO bboxes on the composite (algorithms don't draw them)
            if _bbox_left is not None:
                bx1, by1, bx2, by2 = _bbox_left
                cv2.rectangle(composite, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
            if self.svo2_bbox_right is not None:
                bx1, by1, bx2, by2 = self.svo2_bbox_right
                cv2.rectangle(composite,
                              (bx1 + fw_single, by1),
                              (bx2 + fw_single, by2),
                              (0, 210, 0), 2)

            # Camera labels on composite
            cv2.putText(composite, "LEFT",  (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(composite, "RIGHT", (fw_single + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            edge_total_ms = (t_left + t_right) * 1000.0
            result = composite
        else:
            # Debug mode or no right frame
            edge_total_ms = t_left * 1000.0
            result = result_left

        self.edge_detection_time = edge_total_ms
        self.last_runtime_ms     = edge_total_ms
        fps = 1000.0 / self.last_runtime_ms if self.last_runtime_ms > 0 else 0

        # Update timing display
        self.lbl_runtime.setText(f"Runtime: {self.last_runtime_ms:.2f} ms | FPS: {fps:.1f}")
        if self.current_mode == "svo":
            self.lbl_detailed_timing.setText(
                f"YOLO: {self.yolo_inference_time:.1f} ms | "
                f"Edge L+R: {edge_total_ms:.1f} ms"
            )

        self.current_result = result

        # Extract edge pixels by diffing LEFT result vs frozen LEFT clean image
        # (pose optimizer always uses the left camera frame)
        if self.current_mode == "svo" and self.svo2_right_frame is not None:
            left_only = result[:, :_clean_left.shape[1]]
            ref_image = _clean_left
        else:
            left_only = result
            ref_image = self._clean_image

        diff = np.abs(left_only.astype(np.int16) - ref_image.astype(np.int16))
        changed = diff.max(axis=2) > 20
        rows_ch, cols_ch = np.where(changed)
        if len(rows_ch) > 0:
            self.current_edge_pixels_abs = np.column_stack([cols_ch, rows_ch]).astype(np.int32)
        else:
            self.current_edge_pixels_abs = np.empty((0, 2), dtype=np.int32)
        self.btn_export_pose.setEnabled(len(self.current_edge_pixels_abs) > 10)

        # Display result (bboxes already embedded for SVO2 composite)
        draw_bb = (self.current_mode != "svo" or self.svo2_right_frame is None)
        self.display_image(result, draw_bbox=draw_bb)
        self.btn_save.setEnabled(True)
    
    def apply_canny_auto(self) -> np.ndarray:
        """Apply auto-threshold Canny within bbox (with padding)."""
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        
        # Convert to grayscale for threshold computation
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Apply Gaussian blur if sigma > 0
        if self.params['gaussian_sigma'] > 0:
            ksize = int(2 * round(3 * self.params['gaussian_sigma']) + 1)
            gray = cv2.GaussianBlur(gray, (ksize, ksize), self.params['gaussian_sigma'])
        
        # Compute auto thresholds
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        # Apply Canny
        edges = cv2.Canny(gray, lower, upper)
        
        # Update sliders with computed thresholds
        self.slider_canny_low.blockSignals(True)
        self.slider_canny_high.blockSignals(True)
        self.slider_canny_low.setValue(lower)
        self.slider_canny_high.setValue(upper)
        self.slider_canny_low.blockSignals(False)
        self.slider_canny_high.blockSignals(False)
        
        # Update parameter display labels
        self.lbl_canny_low.setText(f"Canny Low: {lower}")
        self.lbl_canny_high.setText(f"Canny High: {upper}")
        
        # Overlay on original
        result = self._clean_image.copy()
        result[y1:y2, x1:x2][edges > 0] = [0, 255, 0]
        
        return result
    
    def apply_canny_manual(self) -> np.ndarray:
        """Apply manual-threshold Canny within bbox (with padding)."""
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        
        edges = canny_edge_detection(
            roi, 
            auto_threshold=False, 
            low_threshold=self.params['canny_low'],
            high_threshold=self.params['canny_high'],
            blur_sigma=self.params['gaussian_sigma']
        )
        
        # Overlay on original
        result = self._clean_image.copy()
        result[y1:y2, x1:x2][edges > 0] = [0, 255, 0]
        
        return result
    
    def apply_sobel(self) -> np.ndarray:
        """Apply Sobel magnitude within bbox (with padding)."""
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        ksize = int(2 * round(3 * self.params['gaussian_sigma']) + 1)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), self.params['gaussian_sigma'])
        
        # Sobel gradients
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(255 * magnitude / magnitude.max())
        
        # Threshold
        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        
        # Overlay on original
        result = self._clean_image.copy()
        result[y1:y2, x1:x2][edges > 0] = [255, 0, 255]  # Magenta
        
        return result
    
    def apply_log(self) -> np.ndarray:
        """Apply Laplacian of Gaussian within bbox (with padding)."""
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        ksize = int(2 * round(3 * self.params['gaussian_sigma']) + 1)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), self.params['gaussian_sigma'])
        
        # Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Threshold
        _, edges = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
        
        # Overlay on original
        result = self._clean_image.copy()
        result[y1:y2, x1:x2][edges > 0] = [255, 255, 0]  # Yellow
        
        return result
    
    def apply_multiscale_canny(self) -> np.ndarray:
        """Apply multi-scale Canny and merge results within bbox (with padding)."""
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        
        # Multiple scales
        sigmas = [0.5, 1.0, 2.0]
        all_edges = []
        
        for sigma in sigmas:
            edges = canny_edge_detection(roi, auto_threshold=True, blur_sigma=sigma)
            all_edges.append(edges)
        
        # Combine edges (logical OR)
        combined = np.zeros_like(all_edges[0])
        for edges in all_edges:
            combined = np.logical_or(combined, edges > 0)
        
        combined = combined.astype(np.uint8) * 255
        
        # Overlay on original
        result = self._clean_image.copy()
        result[y1:y2, x1:x2][combined > 0] = [0, 128, 255]  # Orange
        
        return result
    
    def apply_canny_contour(self) -> np.ndarray:
        """Apply Canny + contour silhouette within bbox (with depth filtering in SVO mode)."""
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        
        edges = canny_edge_detection(roi, auto_threshold=True, blur_sigma=self.params['gaussian_sigma'])
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply depth filtering if in SVO mode
        if self.current_mode == "svo" and self.params['use_depth_filter']:
            filtered_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Minimum area threshold
                    if self.has_valid_depth(contour, (x1, y1)):
                        filtered_contours.append(contour)
            contours = filtered_contours
        
        # Extract silhouette from filtered contours
        silhouette = np.zeros(roi.shape[:2], dtype=np.uint8)
        if contours:
            cv2.drawContours(silhouette, contours, -1, 255, thickness=cv2.FILLED)
            
            # Apply morphology
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.params['morph_kernel'], self.params['morph_kernel'])
            )
            silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel)
        
        # Overlay silhouette on original
        result = self._clean_image.copy()
        roi_overlay = roi.copy()
        roi_overlay[silhouette > 0] = [0, 255, 255]  # Cyan for silhouette
        result[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.6, roi_overlay, 0.4, 0)
        
        return result
    
    def apply_clahe_canny(self) -> np.ndarray:
        """Apply CLAHE preprocessing + Canny within bbox (with padding)."""
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        
        clahe_roi = apply_clahe(roi, clip_limit=self.params['clahe_clip'], tile_grid_size=(8, 8))
        edges = canny_edge_detection(clahe_roi, auto_threshold=True, blur_sigma=self.params['gaussian_sigma'])
        
        # Overlay on original
        result = self._clean_image.copy()
        result[y1:y2, x1:x2][edges > 0] = [255, 0, 0]  # Blue edges
        
        return result
    
    def apply_canny_hough(self) -> np.ndarray:
        """Apply Canny + Hough line detection within bbox (with padding)."""
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        
        edges = canny_edge_detection(roi, auto_threshold=True, blur_sigma=self.params['gaussian_sigma'])
        
        # Detect lines
        roi_h, roi_w = roi.shape[:2]
        min_line_len = max(50, int(0.15 * max(roi_w, roi_h)))
        lines = hough_lines_probabilistic(edges, threshold=30, min_line_length=min_line_len, max_line_gap=15)
        
        # Draw lines on ROI
        roi_with_lines = roi.copy()
        for x1_l, y1_l, x2_l, y2_l in lines:
            cv2.line(roi_with_lines, (x1_l, y1_l), (x2_l, y2_l), (0, 0, 255), 2)
        
        # Place back in full image
        result = self._clean_image.copy()
        result[y1:y2, x1:x2] = roi_with_lines
        
        return result
    
    def apply_structured_edges(self) -> np.ndarray:
        """Apply structured edge detection within bbox (with padding).
        
        Uses multi-scale gradient fusion as a proxy for structured forests,
        since the structured forests model requires pre-trained weights.
        """
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        ksize = int(2 * round(3 * self.params['gaussian_sigma']) + 1)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), self.params['gaussian_sigma'])
        
        # Multi-scale gradient computation (3 scales)
        scales = [1, 2, 3]
        gradient_maps = []
        
        for scale in scales:
            # Compute gradients at this scale
            ksize_scale = 2 * scale + 1
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=ksize_scale)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=ksize_scale)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_maps.append(magnitude)
        
        # Normalize and fuse gradients
        fused = np.zeros_like(gradient_maps[0])
        for gmap in gradient_maps:
            normalized = gmap / (gmap.max() + 1e-8)
            fused += normalized
        
        fused = fused / len(scales)
        fused = np.uint8(255 * fused)
        
        # Apply adaptive threshold for edge extraction
        edges = cv2.adaptiveThreshold(
            fused, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert (we want edges as white)
        edges = 255 - edges
        
        # Overlay on original
        result = self._clean_image.copy()
        result[y1:y2, x1:x2][edges > 0] = [128, 0, 128]  # Purple
        
        return result
    
    def apply_active_contours(self) -> np.ndarray:
        """Apply Active Contours (Snake) within bbox.
        
        Initializes a contour slightly inside the bbox boundary and evolves it toward edges.
        """
        from skimage.segmentation import active_contour
        from skimage.filters import gaussian
        
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian smoothing
        smoothed = gaussian(gray, self.params['gaussian_sigma'])
        
        # Initialize rectangular contour slightly inside ROI boundaries
        roi_h, roi_w = gray.shape
        margin = 10  # Pixels from edge
        
        # Create rectangular initialization (clockwise from top-left)
        # Top edge
        top_points = np.linspace(margin, roi_w - margin, 25)
        top_y = np.full_like(top_points, margin)
        
        # Right edge
        right_y = np.linspace(margin, roi_h - margin, 25)
        right_x = np.full_like(right_y, roi_w - margin)
        
        # Bottom edge (reverse order for continuous path)
        bottom_points = np.linspace(roi_w - margin, margin, 25)
        bottom_y = np.full_like(bottom_points, roi_h - margin)
        
        # Left edge (reverse order)
        left_y = np.linspace(roi_h - margin, margin, 25)
        left_x = np.full_like(left_y, margin)
        
        # Combine all edges into one contour
        init_x = np.concatenate([top_points, right_x, bottom_points, left_x])
        init_y = np.concatenate([top_y, right_y, bottom_y, left_y])
        init_contour = np.array([init_x, init_y]).T
        
        # Evolve the contour
        try:
            snake = active_contour(
                smoothed,
                init_contour,
                alpha=self.params['snake_alpha'],      # Continuity (elasticity)
                beta=self.params['snake_beta'],        # Smoothness (stiffness)
                gamma=0.001,                           # Step size (smaller = more stable)
                max_num_iter=self.params['snake_iterations'],  # Correct parameter name
                w_line=0,                              # Don't attract to lines
                w_edge=1,                              # Attract to edges
                convergence=0.1                        # Convergence criterion
            )
            
            # Draw the evolved snake on ROI
            roi_result = roi.copy()
            snake_int = snake.astype(np.int32)
            cv2.polylines(roi_result, [snake_int], isClosed=True, color=(255, 128, 0), thickness=2)
            
            # Place back in full image
            result = self._clean_image.copy()
            result[y1:y2, x1:x2] = roi_result
            
        except Exception as e:
            # If snake fails, fallback to showing initialization
            result = self._clean_image.copy()
            roi_result = roi.copy()
            init_int = init_contour.astype(np.int32)
            cv2.polylines(roi_result, [init_int], isClosed=True, color=(128, 128, 128), thickness=1)
            cv2.putText(roi_result, f"Snake failed: {str(e)[:50]}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            result[y1:y2, x1:x2] = roi_result
            print(f"Active contour failed: {e}")
        
        return result
    
    def export_silhouette_for_pose_optimizer(self):
        """Export detected edge pixels as NPZ + JSON + preview PNG for the Pose Optimizer."""
        if self.current_edge_pixels_abs is None or len(self.current_edge_pixels_abs) < 10:
            QMessageBox.warning(self, "Export Error",
                                "No edge pixels available. Apply an algorithm first.")
            return
        if self.current_image is None:
            QMessageBox.warning(self, "Export Error", "No image loaded.")
            return

        # Create output directory
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Build filename from timestamp + source image + algorithm
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algo_name = self.combo_algorithm.currentText()
        algo_key = self.sanitize_algo_name(algo_name)
        img_stem = self.image_files[self.current_image_idx].stem if self.image_files else "unknown"
        base_name = f"silhouette_{timestamp}_{img_stem}_{algo_key}"

        npz_path     = self.export_dir / f"{base_name}.npz"
        meta_path    = self.export_dir / f"{base_name}.json"
        preview_path = self.export_dir / f"{base_name}_preview.png"

        h, w = self.current_image.shape[:2]

        # ── Save NPZ ─────────────────────────────────────────────────────────
        np.savez_compressed(
            str(npz_path),
            edge_pixels=self.current_edge_pixels_abs,        # (N, 2)  col, row
            image_size=np.array([w, h], dtype=np.int32),     # [width, height]
            bbox=np.array(self.current_bbox, dtype=np.int32)
                 if self.current_bbox else np.zeros(4, dtype=np.int32),
        )

        # ── Save metadata JSON sidecar ────────────────────────────────────────
        meta = {
            "source_image": str(self.image_files[self.current_image_idx])
                            if self.image_files else "",
            "algorithm": algo_name,
            "image_size": [w, h],
            "bbox": list(self.current_bbox) if self.current_bbox else None,
            "n_edge_pixels": int(len(self.current_edge_pixels_abs)),
            "runtime_ms": round(self.last_runtime_ms, 2),
            "export_timestamp": datetime.now().isoformat(),
            "mode": self.current_mode,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # ── Save preview PNG (edge pixels on black background) ───────────────
        preview = np.zeros((h, w, 3), dtype=np.uint8)
        cols_ep = self.current_edge_pixels_abs[:, 0]
        rows_ep = self.current_edge_pixels_abs[:, 1]
        valid   = (cols_ep >= 0) & (cols_ep < w) & (rows_ep >= 0) & (rows_ep < h)
        preview[rows_ep[valid], cols_ep[valid]] = [0, 255, 0]
        if self.current_bbox is not None:
            bx1, by1, bx2, by2 = self.current_bbox
            cv2.rectangle(preview, (bx1, by1), (bx2, by2), (0, 128, 255), 1)
        cv2.putText(preview, f"Algorithm: {algo_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(preview, f"Edges: {len(self.current_edge_pixels_abs)} px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(preview, img_stem, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imwrite(str(preview_path), preview)

        # ── Build run command hint ────────────────────────────────────────────
        pose_script = (
            self.project_root.parent / "Pose_Optimizer" / "src" / "run_on_real_silhouette.py"
        )
        run_hint = (
            f"python3 {pose_script} \\\n"
            f"  --silhouette \"{npz_path}\""
        )

        QMessageBox.information(
            self, "Export Successful",
            f"Silhouette saved:\n{npz_path.name}\n\n"
            f"Edge pixels: {len(self.current_edge_pixels_abs)}\n"
            f"Preview: {preview_path.name}\n\n"
            f"Run pose optimizer:\n{run_hint}"
        )

    def save_result(self):
        """Save current result to session folder."""
        if self.current_result is None:
            QMessageBox.warning(self, "Warning", "No result to save.")
            return
        
        algo_name = self.combo_algorithm.currentText()
        algo_key = self.sanitize_algo_name(algo_name)
        session_num = self.session_numbers[algo_key]
        
        # Create output folder
        algo_output_dir = self.output_dir / algo_key / f"session_{session_num:03d}"
        algo_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Derive a filename (SVO2 mode has no image_files list)
        if self.image_files:
            img_name = self.image_files[self.current_image_idx].name
            source_image_str = str(self.image_files[self.current_image_idx])
        else:
            img_name = (
                f"svo2_frame{self.svo2_current_frame:05d}_{datetime.now().strftime('%H%M%S')}.jpg"
            )
            source_image_str = (
                f"{self.svo2_file_path}:frame{self.svo2_current_frame}" if self.svo2_file_path else ""
            )
        output_path = algo_output_dir / img_name
        cv2.imwrite(str(output_path), self.current_result)
        
        # Save metadata
        meta_path = algo_output_dir / f"{img_name.rsplit('.', 1)[0]}_meta.json"
        metadata = {
            "original_image": source_image_str,
            "algorithm": algo_name,
            "runtime_ms": self.last_runtime_ms,
            "fps": 1000.0 / self.last_runtime_ms if self.last_runtime_ms > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "bbox": self.current_bbox,
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        QMessageBox.information(self, "Success", f"Result saved to:\n{output_path}")
        
        # Update session counter in sessions file
        with open(self.session_file, 'r') as f:
            sessions = json.load(f)
        
        if session_num not in sessions[algo_key]:
            sessions[algo_key].append(session_num)
        
        with open(self.session_file, 'w') as f:
            json.dump(sessions, f, indent=2)

    def closeEvent(self, event):
        """Ensure ZED camera is closed cleanly on application exit."""
        self.svo2_close_camera()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = EdgeDetectionGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
