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
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QWheelEvent

# Import algorithms (add parent to path)
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.preproc import apply_clahe
from algorithms.canny_pipeline import canny_edge_detection
from algorithms.contour_silhouette import extract_silhouette_from_contours
from algorithms.hough_ransac import hough_lines_probabilistic


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
        
        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.debug_images_dir = self.project_root / "data" / "debug_images" / "debug"
        self.labels_dir = self.project_root / "data" / "debug_images" / "labels"
        self.output_dir = self.project_root / "data" / "algorithm_outputs"
        
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
        }
        
        # Default parameter values (for reset)
        self.default_params = self.params.copy()
        
        # Available algorithms with their used parameters
        self.algorithms = {
            "Canny (Auto)": {
                'func': self.apply_canny_auto,
                'params': ['gaussian_sigma', 'bbox_padding']
            },
            "CLAHE + Canny": {
                'func': self.apply_clahe_canny,
                'params': ['clahe_clip', 'gaussian_sigma', 'bbox_padding']
            },
            "Canny (Manual)": {
                'func': self.apply_canny_manual,
                'params': ['canny_low', 'canny_high', 'gaussian_sigma', 'bbox_padding']
            },
            "Sobel Magnitude": {
                'func': self.apply_sobel,
                'params': ['gaussian_sigma', 'bbox_padding']
            },
            "Laplacian of Gaussian": {
                'func': self.apply_log,
                'params': ['gaussian_sigma', 'bbox_padding']
            },
            "Multi-scale Canny": {
                'func': self.apply_multiscale_canny,
                'params': ['bbox_padding']
            },
            "Structured Edges": {
                'func': self.apply_structured_edges,
                'params': ['bbox_padding']
            },
            "Active Contours (Snake)": {
                'func': self.apply_active_contours,
                'params': ['gaussian_sigma', 'snake_alpha', 'snake_beta', 'snake_iterations', 'bbox_padding']
            },
            "Canny + Contour Silhouette": {
                'func': self.apply_canny_contour,
                'params': ['gaussian_sigma', 'morph_kernel', 'bbox_padding']
            },
            "Canny + Hough Lines": {
                'func': self.apply_canny_hough,
                'params': ['gaussian_sigma', 'bbox_padding']
            },
        }
        
        self.init_ui()
        self.load_image_list()
        self.load_first_image()
        
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
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
        
        # BBox padding
        self.padding_layout = QVBoxLayout()
        lbl = QLabel("BBox Padding:")
        self.padding_layout.addWidget(lbl)
        self.slider_padding = QSlider(Qt.Orientation.Horizontal)
        self.slider_padding.setRange(0, 20)
        self.slider_padding.setValue(self.params['bbox_padding'])
        self.slider_padding.valueChanged.connect(lambda v: self.update_param('bbox_padding', v))
        self.lbl_padding = QLabel(str(self.params['bbox_padding']))
        self.padding_layout.addWidget(self.slider_padding)
        self.padding_layout.addWidget(self.lbl_padding)
        param_layout.addLayout(self.padding_layout)
        self.param_widgets['bbox_padding'] = [lbl, self.slider_padding, self.lbl_padding]
        
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
        elif key == 'bbox_padding':
            self.lbl_padding.setText(str(value))
        elif key == 'morph_kernel':
            self.lbl_morph.setText(str(value))
        elif key == 'snake_alpha':
            self.lbl_snake_alpha.setText(f"{value:.3f}")
        elif key == 'snake_beta':
            self.lbl_snake_beta.setText(f"{value:.3f}")
        elif key == 'snake_iterations':
            self.lbl_snake_iter.setText(str(value))
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.params = self.default_params.copy()
        # Update sliders
        self.slider_canny_low.setValue(self.params['canny_low'])
        self.slider_canny_high.setValue(self.params['canny_high'])
        self.slider_sigma.setValue(int(self.params['gaussian_sigma'] * 10))
        self.slider_clahe.setValue(int(self.params['clahe_clip'] * 10))
        self.slider_padding.setValue(self.params['bbox_padding'])
        self.slider_morph.setValue(self.params['morph_kernel'])
        self.slider_snake_alpha.setValue(int(self.params['snake_alpha'] * 1000))
        self.slider_snake_beta.setValue(int(self.params['snake_beta'] * 1000))
        self.slider_snake_iter.setValue(self.params['snake_iterations'])
    
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
        """Load current image and its YOLO labels."""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_image_idx]
        self.current_image = cv2.imread(str(img_path))
        
        if self.current_image is None:
            QMessageBox.warning(self, "Error", f"Failed to load image:\n{img_path}")
            return
        
        # Load YOLO labels
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        self.current_bbox = self.load_yolo_bbox(label_path, self.current_image.shape)
        
        # Update UI
        self.lbl_image_info.setText(f"Image {self.current_image_idx + 1}/{len(self.image_files)}: {img_path.name}")
        self.display_image(self.current_image)
        self.current_result = None
        self.btn_save.setEnabled(False)
        self.lbl_runtime.setText("Runtime: -- ms | FPS: --")
    
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
    
    def display_image(self, img: np.ndarray):
        """Display image in the viewer, preserving zoom level."""
        # Save current zoom level
        current_zoom = self.view.get_zoom_level()
        
        # Draw bounding box if available (thinner lines)
        display_img = img.copy()
        if self.current_bbox is not None:
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
        Extract ROI from image with padding to avoid bbox edge artifacts.
        Returns (roi_image, adjusted_bbox_in_roi)
        """
        if self.current_bbox is None or self.current_image is None:
            return self.current_image, None
        
        x1, y1, x2, y2 = self.current_bbox
        h, w = self.current_image.shape[:2]
        pad = self.params['bbox_padding']
        
        # Apply padding (inward to avoid bbox edges)
        x1_pad = min(x1 + pad, x2 - 1)
        y1_pad = min(y1 + pad, y2 - 1)
        x2_pad = max(x2 - pad, x1 + 1)
        y2_pad = max(y2 - pad, y1 + 1)
        
        # Clamp to image bounds
        x1_pad = max(0, x1_pad)
        y1_pad = max(0, y1_pad)
        x2_pad = min(w, x2_pad)
        y2_pad = min(h, y2_pad)
        
        # Extract ROI
        roi = self.current_image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
        
        # Return ROI and adjusted bbox (in original image coordinates)
        return roi, (x1_pad, y1_pad, x2_pad, y2_pad)
    
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
        
        # Measure runtime
        start = time.perf_counter()
        result = algo_func()
        end = time.perf_counter()
        
        self.last_runtime_ms = (end - start) * 1000.0
        fps = 1000.0 / self.last_runtime_ms if self.last_runtime_ms > 0 else 0
        
        self.lbl_runtime.setText(f"Runtime: {self.last_runtime_ms:.2f} ms | FPS: {fps:.1f}")
        
        self.current_result = result
        self.display_image(result)
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
        """Apply Canny + contour silhouette within bbox (with padding)."""
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        
        edges = canny_edge_detection(roi, auto_threshold=True, blur_sigma=self.params['gaussian_sigma'])
        silhouette = extract_silhouette_from_contours(
            edges, 
            roi.shape, 
            morph_kernel_size=self.params['morph_kernel'], 
            min_area_ratio=0.01
        )
        
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
        """Apply Active Contours (Snake) within bbox (with padding).
        
        Initializes a contour at the bbox boundary and evolves it toward edges.
        """
        from skimage.segmentation import active_contour
        from skimage.filters import gaussian
        
        roi, (x1, y1, x2, y2) = self.get_roi_with_padding()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian smoothing
        smoothed = gaussian(gray, self.params['gaussian_sigma'])
        
        # Initialize circular contour in center of ROI
        roi_h, roi_w = gray.shape
        center_y, center_x = roi_h // 2, roi_w // 2
        radius = min(roi_h, roi_w) // 3
        
        # Create circular initialization
        theta = np.linspace(0, 2 * np.pi, 100)
        init_x = center_x + radius * np.cos(theta)
        init_y = center_y + radius * np.sin(theta)
        init_contour = np.array([init_x, init_y]).T
        
        # Evolve the contour
        try:
            snake = active_contour(
                smoothed,
                init_contour,
                alpha=self.params['snake_alpha'],
                beta=self.params['snake_beta'],
                gamma=0.01,
                max_iterations=self.params['snake_iterations'],
                w_line=-1,  # Attract to dark lines (edges)
                w_edge=2    # Strong edge attraction
            )
            
            # Draw the snake on ROI
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
            cv2.polylines(roi_result, [init_int], isClosed=True, color=(255, 128, 0), thickness=2)
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
