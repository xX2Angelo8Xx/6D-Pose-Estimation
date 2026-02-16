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
)
from PySide6.QtCore import Qt
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


class EdgeDetectionGUI(QMainWindow):
    """Main GUI for edge detection algorithm testing."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge Detection Algorithm Tester")
        self.setGeometry(100, 100, 1400, 900)
        
        # Paths - Debug Mode
        self.project_root = Path(__file__).parent.parent.parent
        self.debug_images_dir = self.project_root / "data" / "debug_images" / "debug"
        self.labels_dir = self.project_root / "data" / "debug_images" / "labels"
        self.output_dir = self.project_root / "data" / "algorithm_outputs"
        
        # Paths - SVO Mode
        self.svo_base_dir = Path("/media/angelo/DRONE_DATA1/SVO2_Frame_Export")
        self.yolo_model_path = Path("/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt")
        self.yolo_model = None
        
        # Mode selection
        self.current_mode = "debug"  # "debug" or "svo"
        self.svo_folders = []
        self.current_svo_folder = None
        self.current_depth_data = None
        
        # Session management
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.output_dir / "sessions.json"
        self.load_or_create_session()
        
        # Current state
        self.image_files: List[Path] = []
        self.current_image_idx: int = 0
        self.current_image: Optional[np.ndarray] = None
        self.current_result: Optional[np.ndarray] = None
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_runtime_ms: float = 0.0
        self.yolo_inference_time: float = 0.0
        self.edge_detection_time: float = 0.0
        
        # Algorithm parameters
        self.params = {
            'canny_low': 50,
            'canny_high': 150,
            'canny_auto': True,
            'gaussian_sigma': 1.0,
            'clahe_clip': 2.0,
            'bbox_padding': 5,  # Padding to avoid bbox edge artifacts
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
        self.radio_svo_mode = QRadioButton("SVO Mode (YOLO Inference + Depth)")
        self.radio_debug_mode.setChecked(True)
        
        self.mode_button_group.addButton(self.radio_debug_mode)
        self.mode_button_group.addButton(self.radio_svo_mode)
        
        self.radio_debug_mode.toggled.connect(self.on_mode_changed)
        
        mode_group_layout.addWidget(self.radio_debug_mode)
        mode_group_layout.addWidget(self.radio_svo_mode)
        
        # SVO folder selection (only visible in SVO mode)
        self.lbl_svo_folder = QLabel("SVO Folder:")
        self.combo_svo_folder = QComboBox()
        # Don't connect signal yet - will be connected after folders are loaded
        # self.combo_svo_folder.currentTextChanged.connect(self.on_svo_folder_changed)
        mode_group_layout.addWidget(self.lbl_svo_folder)
        mode_group_layout.addWidget(self.combo_svo_folder)
        
        mode_group.setLayout(mode_group_layout)
        main_layout.addWidget(mode_group)
        
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
        """Initialize SVO mode components."""
        # Load available SVO folders
        if self.svo_base_dir.exists():
            self.svo_folders = sorted([
                f for f in self.svo_base_dir.iterdir()
                if f.is_dir() and f.name[0].isdigit()  # Folders like 1.1, 1.2, etc.
            ])
            # Add items without triggering signals
            for folder in self.svo_folders:
                self.combo_svo_folder.addItem(folder.name)
        
        # Now connect the signal after items are added
        self.combo_svo_folder.currentTextChanged.connect(self.on_svo_folder_changed)
        
        # Hide SVO controls initially (debug mode is default)
        self.lbl_svo_folder.setVisible(False)
        self.combo_svo_folder.setVisible(False)
        
        # Load YOLO model if available
        if YOLO_AVAILABLE and self.yolo_model_path.exists():
            try:
                self.yolo_model = YOLO(str(self.yolo_model_path))
                print(f"YOLO model loaded: {self.yolo_model_path}")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.yolo_model = None
        else:
            if not YOLO_AVAILABLE:
                print("YOLO not available: ultralytics not installed")
            elif not self.yolo_model_path.exists():
                print(f"YOLO model not found: {self.yolo_model_path}")
        
        # Load initial images (Debug Mode is default)
        self.load_image_list()
        self.load_first_image()
    
    def on_mode_changed(self, checked: bool):
        """Handle mode switch between Debug and SVO mode."""
        if not checked:
            return
        
        if self.radio_debug_mode.isChecked():
            self.current_mode = "debug"
            # Show debug mode controls, hide SVO controls
            self.lbl_svo_folder.setVisible(False)
            self.combo_svo_folder.setVisible(False)
            self.lbl_detailed_timing.setVisible(False)
            for widget in self.param_widgets['depth_radius']:
                widget.setVisible(False)
            
            # Load debug images
            self.load_image_list()
            self.load_first_image()
            
        elif self.radio_svo_mode.isChecked():
            self.current_mode = "svo"
            # Show SVO mode controls
            self.lbl_svo_folder.setVisible(True)
            self.combo_svo_folder.setVisible(True)
            self.lbl_detailed_timing.setVisible(True)
            for widget in self.param_widgets['depth_radius']:
                widget.setVisible(True)
            
            # Check if YOLO is available
            if not self.yolo_model:
                QMessageBox.warning(
                    self,
                    "YOLO Not Available",
                    "YOLO model is not available. Please install ultralytics or check model path."
                )
                self.radio_debug_mode.setChecked(True)
                return
            
            # Load SVO images
            if self.combo_svo_folder.count() > 0:
                self.on_svo_folder_changed(self.combo_svo_folder.currentText())
    
    def on_svo_folder_changed(self, folder_name: str):
        """Handle SVO folder selection change."""
        if not folder_name:
            return
        
        # Only load if we're in SVO mode OR if this is being called during mode initialization
        if self.current_mode != "svo":
            return
        
        self.current_svo_folder = self.svo_base_dir / folder_name
        self.load_svo_images()
        if self.image_files:  # Only load if images were found
            self.load_first_image()
    
    def load_svo_images(self):
        """Load images from selected SVO folder."""
        if not self.current_svo_folder or not self.current_svo_folder.exists():
            return
        
        # Find all .jpg files (read-only, never modified)
        self.image_files = sorted([
            f for f in self.current_svo_folder.iterdir()
            if f.suffix.lower() == '.jpg'
        ])
        
        if not self.image_files:
            QMessageBox.warning(self, "Warning", f"No images found in {self.current_svo_folder}")
    
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
        self.btn_save.setEnabled(False)
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
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # thickness 1
            cv2.putText(display_img, "YOLO BBox", (x1, max(15, y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # smaller font, thickness 1
        
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
        Extract ROI from image based on bbox.
        Returns (roi_image, bbox_coordinates)
        
        Note: No padding is applied since bbox is drawn AFTER algorithm processing.
        """
        if self.current_bbox is None or self.current_image is None:
            return self.current_image, None
        
        x1, y1, x2, y2 = self.current_bbox
        h, w = self.current_image.shape[:2]
        
        # Clamp to image bounds
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        
        # Extract ROI
        roi = self.current_image[y1:y2, x1:x2].copy()
        
        # Return ROI and bbox (in original image coordinates)
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
        
        # Measure edge detection runtime
        start_edge = time.perf_counter()
        result = algo_func()
        end_edge = time.perf_counter()
        
        self.edge_detection_time = (end_edge - start_edge) * 1000.0
        self.last_runtime_ms = self.edge_detection_time
        fps = 1000.0 / self.last_runtime_ms if self.last_runtime_ms > 0 else 0
        
        # Update timing display
        self.lbl_runtime.setText(f"Runtime: {self.last_runtime_ms:.2f} ms | FPS: {fps:.1f}")
        
        # Show detailed timing in SVO mode
        if self.current_mode == "svo":
            total_time = self.yolo_inference_time + self.edge_detection_time
            timing_text = (
                f"YOLO: {self.yolo_inference_time:.2f} ms | "
                f"Edge: {self.edge_detection_time:.2f} ms | "
                f"Total: {total_time:.2f} ms"
            )
            self.lbl_detailed_timing.setText(timing_text)
        
        self.current_result = result
        # Draw bbox AFTER algorithm processing
        self.display_image(result, draw_bbox=True)
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
        result = self.current_image.copy()
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
        result = self.current_image.copy()
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
        result = self.current_image.copy()
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
        result = self.current_image.copy()
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
        result = self.current_image.copy()
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
        result = self.current_image.copy()
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
        result = self.current_image.copy()
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
        result = self.current_image.copy()
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
        result = self.current_image.copy()
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
            result = self.current_image.copy()
            result[y1:y2, x1:x2] = roi_result
            
        except Exception as e:
            # If snake fails, fallback to showing initialization
            result = self.current_image.copy()
            roi_result = roi.copy()
            init_int = init_contour.astype(np.int32)
            cv2.polylines(roi_result, [init_int], isClosed=True, color=(128, 128, 128), thickness=1)
            cv2.putText(roi_result, f"Snake failed: {str(e)[:50]}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            result[y1:y2, x1:x2] = roi_result
            print(f"Active contour failed: {e}")
        
        return result
    
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
        
        # Save image
        img_name = self.image_files[self.current_image_idx].name
        output_path = algo_output_dir / img_name
        cv2.imwrite(str(output_path), self.current_result)
        
        # Save metadata
        meta_path = algo_output_dir / f"{img_name.rsplit('.', 1)[0]}_meta.json"
        metadata = {
            "original_image": str(self.image_files[self.current_image_idx]),
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


def main():
    app = QApplication(sys.argv)
    window = EdgeDetectionGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
