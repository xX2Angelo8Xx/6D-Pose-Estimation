#!/usr/bin/env python3
"""
SVO Mode Edge Detection GUI - Simplified standalone version.

Features:
- Load frames from SVO2_Frame_Export folders
- YOLO inference for object detection
- Depth data loading from .npy files
- Edge detection algorithms within detected bounding boxes
- Depth filtering with configurable radius
- Detailed performance measurement (YOLO + Edge)
- Read-only access to original data
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QMessageBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QSlider, QGroupBox, QCheckBox,
    QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QPainter

# Add parent to path for algorithm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.canny_pipeline import canny_edge_detection
from algorithms.preproc import apply_clahe
from algorithms.adaptive_pipeline import AdaptiveEdgePipeline

# YOLO import with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view with mouse wheel zoom capability."""
    
    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._zoom_level = 1.0
    
    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with mouse wheel."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self._zoom_level *= zoom_factor
        else:
            zoom_factor = zoom_out_factor
            self._zoom_level *= zoom_factor
        
        self.scale(zoom_factor, zoom_factor)
    
    def get_zoom_level(self) -> float:
        return self._zoom_level
    
    def set_zoom_level(self, level: float):
        self.resetTransform()
        self.scale(level, level)
        self._zoom_level = level


class SVOEdgeDetectionGUI(QMainWindow):
    """Simplified GUI for SVO frame edge detection."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SVO Edge Detection - YOLO + Depth")
        self.setGeometry(100, 100, 1400, 900)
        
        # Paths
        self.svo_base_dir = Path("/media/angelo/DRONE_DATA1/SVO2_Frame_Export")
        self.yolo_model_path = Path("/media/angelo/DRONE_DATA1/JetsonExport/svo_model_20251204_112724_1280/models/best.pt")
        
        # State
        self.current_folder: Optional[Path] = None
        self.image_files: List[Path] = []
        self.current_image_idx: int = 0
        self.current_image: Optional[np.ndarray] = None
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.current_depth_data: Optional[np.ndarray] = None
        self.current_result: Optional[np.ndarray] = None
        
        # Timing
        self.yolo_time: float = 0.0
        self.edge_time: float = 0.0
        
        # Parameters
        self.params = {
            'gaussian_sigma': 1.0,
            'clahe_clip': 2.0,
            'depth_radius': 5,  # Search radius for valid depth
            'depth_avg_radius': 3,  # Averaging radius for smooth depth colors
            'use_depth_filter': False,
            'show_filtered_pixels': False,  # Show what was removed by depth filter
            'color_by_depth': True,  # Color contours by depth value
            'use_half_precision': False,  # FP32 works best (FP16 has cuDNN issues)
            'inference_size': 320,  # YOLO 320x320 = ~50ms GPU, ~130ms CPU (Original-Auflösung für Edge!)
            'use_original_for_edges': True,  # Edge detection auf Original-Auflösung für Details
        }
        
        # Storage for filtered contours (for visualization)
        self.filtered_out_contours: List = []
        self.contour_depths: List[float] = []  # Store depth for each contour
        
        # Adaptive pipeline (deterministic edge detection)
        self.adaptive_pipeline: Optional[AdaptiveEdgePipeline] = None
        self.pipeline_debug_mode = False  # Toggle for debug visualization
        self.pipeline_stages = []  # Store intermediate stages
        self.debug_stage_index = 0  # Current stage being viewed in debug mode
        
        # Load YOLO model
        self.yolo_model = None
        if not YOLO_AVAILABLE:
            QMessageBox.critical(self, "Error", "YOLO not available. Install ultralytics first.")
            sys.exit(1)
        
        if not self.yolo_model_path.exists():
            QMessageBox.critical(self, "Error", f"YOLO model not found: {self.yolo_model_path}")
            sys.exit(1)
        
        # Detect CUDA availability
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                print(f"✅ CUDA is available! Device: {torch.cuda.get_device_name(0)}")
                
                # WORKAROUND: Disable cuDNN to avoid "GET was unable to find an engine" error
                # This happens because PyTorch was compiled for CUDA 11.x but Jetson has CUDA 12.2
                # GPU is still used, just without cuDNN optimizations (still 2-3x faster than CPU!)
                torch.backends.cudnn.enabled = False
                print("⚙️  cuDNN disabled (workaround for CUDA version mismatch)")
                print("   GPU is still used with native CUDA kernels (2-3x faster than CPU)")
                
                self.device = 'cuda:0'
            else:
                print("CUDA not available, using CPU")
                self.device = 'cpu'
        except ImportError:
            print("PyTorch not available, defaulting to CPU")
            self.cuda_available = False
            self.device = 'cpu'
        
        try:
            print(f"🔄 Loading YOLO model from {self.yolo_model_path}...")
            self.yolo_model = YOLO(str(self.yolo_model_path))
            print(f"✅ YOLO model loaded successfully! Using device: {self.device}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load YOLO model: {e}")
            sys.exit(1)
        
        # Initialize UI
        self.init_ui()
        
        # Load available folders
        self.load_folder_list()
        
        # Setup timer for debug keyboard handling
        self.debug_timer = QTimer()
        self.debug_timer.timeout.connect(self.on_debug_key_event)
        self.debug_timer.start(50)  # Check every 50ms
    
    def init_ui(self):
        """Initialize user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Folder selection
        folder_group = QGroupBox("SVO Folder Selection")
        folder_layout = QHBoxLayout()
        
        folder_layout.addWidget(QLabel("Folder:"))
        self.combo_folder = QComboBox()
        self.combo_folder.currentTextChanged.connect(self.on_folder_changed)
        folder_layout.addWidget(self.combo_folder)
        
        self.lbl_folder_info = QLabel("No folder selected")
        folder_layout.addWidget(self.lbl_folder_info)
        folder_layout.addStretch()
        
        # Device selection (CPU/GPU)
        folder_layout.addWidget(QLabel("Device:"))
        self.btn_group_device = QButtonGroup()
        self.radio_cpu = QRadioButton("CPU")
        self.radio_gpu = QRadioButton("GPU")
        self.btn_group_device.addButton(self.radio_cpu)
        self.btn_group_device.addButton(self.radio_gpu)
        
        # Set default based on CUDA availability
        if hasattr(self, 'cuda_available') and self.cuda_available:
            self.radio_gpu.setChecked(True)
        else:
            self.radio_cpu.setChecked(True)
            self.radio_gpu.setEnabled(False)
        
        self.radio_cpu.toggled.connect(self.on_device_changed)
        folder_layout.addWidget(self.radio_cpu)
        folder_layout.addWidget(self.radio_gpu)
        
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)
        
        # Navigation and algorithm
        control_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.clicked.connect(self.prev_image)
        control_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.next_image)
        control_layout.addWidget(self.btn_next)
        
        self.lbl_image_info = QLabel("No image loaded")
        control_layout.addWidget(self.lbl_image_info)
        
        control_layout.addStretch()
        
        control_layout.addWidget(QLabel("Algorithm:"))
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems([
            "Canny (Auto)", 
            "CLAHE + Canny", 
            "Canny + Contours", 
            "🔧 Adaptive Pipeline",
            "⚡ CLAHE + Direct Canny (Simple)"
        ])
        control_layout.addWidget(self.combo_algorithm)
        
        self.btn_apply = QPushButton("Apply Algorithm")
        self.btn_apply.clicked.connect(self.apply_algorithm)
        control_layout.addWidget(self.btn_apply)
        
        # Debug mode toggle for adaptive pipeline
        self.chk_debug_pipeline = QCheckBox("🔍 Debug Mode")
        self.chk_debug_pipeline.setToolTip("Show all intermediate pipeline steps in separate window")
        self.chk_debug_pipeline.setChecked(False)
        self.chk_debug_pipeline.stateChanged.connect(self.on_debug_mode_changed)
        control_layout.addWidget(self.chk_debug_pipeline)
        
        # Show depth map button
        self.btn_show_depth = QPushButton("📊 Show Depth Map")
        self.btn_show_depth.setToolTip("Display depth map with colorbar in separate window")
        self.btn_show_depth.clicked.connect(self.show_depth_map)
        control_layout.addWidget(self.btn_show_depth)
        
        main_layout.addLayout(control_layout)
        
        # Timing display
        timing_layout = QHBoxLayout()
        self.lbl_timing = QLabel("YOLO: -- ms | Edge: -- ms | Total: -- ms")
        timing_layout.addWidget(self.lbl_timing)
        timing_layout.addStretch()
        main_layout.addLayout(timing_layout)
        
        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QHBoxLayout()
        
        # Gaussian Sigma
        param_layout.addWidget(QLabel("Gaussian σ:"))
        self.slider_sigma = QSlider(Qt.Orientation.Horizontal)
        self.slider_sigma.setRange(1, 50)
        self.slider_sigma.setValue(int(self.params['gaussian_sigma'] * 10))
        self.slider_sigma.valueChanged.connect(lambda v: self.update_param('gaussian_sigma', v / 10.0))
        param_layout.addWidget(self.slider_sigma)
        self.lbl_sigma = QLabel(f"{self.params['gaussian_sigma']:.1f}")
        param_layout.addWidget(self.lbl_sigma)
        
        # CLAHE
        param_layout.addWidget(QLabel("CLAHE Clip:"))
        self.slider_clahe = QSlider(Qt.Orientation.Horizontal)
        self.slider_clahe.setRange(10, 50)
        self.slider_clahe.setValue(int(self.params['clahe_clip'] * 10))
        self.slider_clahe.valueChanged.connect(lambda v: self.update_param('clahe_clip', v / 10.0))
        param_layout.addWidget(self.slider_clahe)
        self.lbl_clahe = QLabel(f"{self.params['clahe_clip']:.1f}")
        param_layout.addWidget(self.lbl_clahe)
        
        # Depth Filter
        self.chk_depth = QCheckBox("Depth Filter")
        self.chk_depth.setChecked(self.params['use_depth_filter'])
        self.chk_depth.stateChanged.connect(lambda s: self.update_param('use_depth_filter', s == Qt.CheckState.Checked.value))
        param_layout.addWidget(self.chk_depth)
        
        param_layout.addWidget(QLabel("Search:"))
        self.slider_depth = QSlider(Qt.Orientation.Horizontal)
        self.slider_depth.setRange(1, 50)
        self.slider_depth.setValue(self.params['depth_radius'])
        self.slider_depth.setToolTip("Search radius for valid depth pixels")
        self.slider_depth.valueChanged.connect(lambda v: self.update_param('depth_radius', v))
        param_layout.addWidget(self.slider_depth)
        self.lbl_depth = QLabel(str(self.params['depth_radius']))
        param_layout.addWidget(self.lbl_depth)
        
        param_layout.addWidget(QLabel("Avg:"))
        self.slider_depth_avg = QSlider(Qt.Orientation.Horizontal)
        self.slider_depth_avg.setRange(1, 20)
        self.slider_depth_avg.setValue(self.params['depth_avg_radius'])
        self.slider_depth_avg.setToolTip("Averaging radius for smooth depth colors")
        self.slider_depth_avg.valueChanged.connect(lambda v: self.update_param('depth_avg_radius', v))
        param_layout.addWidget(self.slider_depth_avg)
        self.lbl_depth_avg = QLabel(str(self.params['depth_avg_radius']))
        param_layout.addWidget(self.lbl_depth_avg)
        
        # Show filtered pixels toggle
        self.chk_show_filtered = QCheckBox("Show Filtered")
        self.chk_show_filtered.setChecked(self.params['show_filtered_pixels'])
        self.chk_show_filtered.setToolTip("Show pixels removed by depth filter in red")
        self.chk_show_filtered.stateChanged.connect(lambda s: self.update_param('show_filtered_pixels', s == Qt.CheckState.Checked.value))
        param_layout.addWidget(self.chk_show_filtered)
        
        # Color by depth toggle
        self.chk_color_depth = QCheckBox("Color by Depth")
        self.chk_color_depth.setChecked(self.params['color_by_depth'])
        self.chk_color_depth.setToolTip("Color contours by depth value (viridis)")
        self.chk_color_depth.stateChanged.connect(lambda s: self.update_param('color_by_depth', s == Qt.CheckState.Checked.value))
        param_layout.addWidget(self.chk_color_depth)
        
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)
        
        # Performance optimization group
        perf_group = QGroupBox("Performance Optimization")
        perf_layout = QHBoxLayout()
        
        # FP16 toggle
        self.chk_fp16 = QCheckBox("FP16 (Half Precision)")
        self.chk_fp16.setChecked(self.params['use_half_precision'])
        self.chk_fp16.setToolTip("FP16 has cuDNN issues - use FP32 (unchecked) for stability")
        self.chk_fp16.setEnabled(hasattr(self, 'cuda_available') and self.cuda_available)
        self.chk_fp16.stateChanged.connect(lambda s: self.update_param('use_half_precision', s == Qt.CheckState.Checked.value))
        perf_layout.addWidget(self.chk_fp16)
        
        # Inference size
        perf_layout.addWidget(QLabel("YOLO Input:"))
        self.combo_imgsz = QComboBox()
        self.combo_imgsz.addItems(["320", "416", "512", "640", "800", "1024", "1280"])
        self.combo_imgsz.setCurrentText(str(self.params['inference_size']))
        self.combo_imgsz.setToolTip("YOLO Größe. 1280=Trainings-Auflösung (beste Qualität). Klein=schneller.")
        self.combo_imgsz.currentTextChanged.connect(lambda v: self.update_param('inference_size', int(v)))
        perf_layout.addWidget(self.combo_imgsz)
        
        # Info label
        lbl_info = QLabel("ℹ️ Edge Detection auf Original-Auflösung")
        lbl_info.setStyleSheet("color: #00AA00; font-size: 10px;")
        lbl_info.setToolTip("YOLO findet BBox schnell auf kleiner Auflösung.\nEdge Detection arbeitet dann auf Original 1280x720 für maximale Details!")
        perf_layout.addWidget(lbl_info)
        
        # Export TensorRT button
        self.btn_export_trt = QPushButton("🚀 Export TensorRT")
        self.btn_export_trt.setToolTip("Export model to TensorRT for 10-20x speedup (takes ~5 min)")
        self.btn_export_trt.clicked.connect(self.export_tensorrt)
        self.btn_export_trt.setEnabled(hasattr(self, 'cuda_available') and self.cuda_available)
        perf_layout.addWidget(self.btn_export_trt)
        
        perf_layout.addStretch()
        
        perf_group.setLayout(perf_layout)
        main_layout.addWidget(perf_group)
        
        # Image viewer
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        main_layout.addWidget(self.view)
    
    def update_param(self, key: str, value):
        """Update parameter."""
        self.params[key] = value
        if key == 'gaussian_sigma':
            self.lbl_sigma.setText(f"{value:.1f}")
        elif key == 'clahe_clip':
            self.lbl_clahe.setText(f"{value:.1f}")
        elif key == 'depth_radius':
            self.lbl_depth.setText(str(value))
        elif key == 'depth_avg_radius':
            self.lbl_depth_avg.setText(str(value))
        elif key in ['show_filtered_pixels', 'color_by_depth', 'depth_avg_radius']:
            # Re-apply algorithm to update visualization
            if self.current_result is not None:
                self.apply_algorithm()
        elif key == 'use_half_precision' or key == 'inference_size':
            # These require reloading current image
            if self.current_image is not None:
                print(f"Parameter {key} changed to {value}, will affect next inference")
    
    def on_device_changed(self):
        """Handle device (CPU/GPU) change."""
        if self.radio_cpu.isChecked():
            self.device = 'cpu'
            print("Switched to CPU")
        else:
            self.device = 'cuda:0'
            print("Switched to GPU (CUDA)")
        
        # Reload current image with new device
        if self.current_image is not None:
            print("Reloading image with new device...")
            self.load_current_image()
    
    def on_debug_mode_changed(self, state):
        """Toggle debug mode for adaptive pipeline."""
        self.pipeline_debug_mode = (state == Qt.CheckState.Checked.value)
        print(f"🔍 Pipeline debug mode: {'ON' if self.pipeline_debug_mode else 'OFF'}")
        
        # Reinitialize pipeline if it exists
        if self.adaptive_pipeline is not None:
            self.adaptive_pipeline.debug_mode = self.pipeline_debug_mode
            print("   Pipeline updated with new debug setting")
    
    def export_tensorrt(self):
        """Export YOLO model to TensorRT for faster inference."""
        reply = QMessageBox.question(
            self,
            "Export to TensorRT",
            f"This will export the model to TensorRT Engine format.\n\n"
            f"Settings:\n"
            f"- Input size: {self.params['inference_size']}\n"
            f"- Half precision: {self.params['use_half_precision']}\n\n"
            f"This takes ~5 minutes but will speed up inference 10-20x.\n\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                print("=" * 60)
                print("Exporting to TensorRT Engine...")
                print(f"Settings: imgsz={self.params['inference_size']}, half={self.params['use_half_precision']}")
                print("This may take several minutes...")
                
                # Export
                self.yolo_model.export(
                    format='engine',
                    half=self.params['use_half_precision'],
                    imgsz=self.params['inference_size'],
                    device=0
                )
                
                print("✓ TensorRT export completed!")
                print("=" * 60)
                
                QMessageBox.information(
                    self,
                    "Export Complete",
                    "TensorRT engine created successfully!\n\n"
                    "Restart the GUI to use the optimized engine."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export TensorRT engine:\n{e}"
                )
                print(f"TensorRT export failed: {e}")
    
    def load_folder_list(self):
        """Load available SVO folders."""
        if not self.svo_base_dir.exists():
            QMessageBox.critical(self, "Error", f"SVO directory not found: {self.svo_base_dir}")
            return
        
        folders = sorted([
            f for f in self.svo_base_dir.iterdir()
            if f.is_dir() and f.name[0].isdigit()
        ])
        
        for folder in folders:
            self.combo_folder.addItem(folder.name)
        
        print(f"Found {len(folders)} SVO folders")
    
    def on_folder_changed(self, folder_name: str):
        """Handle folder selection change."""
        if not folder_name:
            return
        
        self.current_folder = self.svo_base_dir / folder_name
        print(f"Loading folder: {self.current_folder}")
        
        # Reset temporal tracking when changing folders
        if self.adaptive_pipeline is not None:
            self.adaptive_pipeline.reset_temporal()
            print("🔄 Temporal tracking reset (new folder)")
        
        if not self.current_folder.exists():
            QMessageBox.warning(self, "Warning", f"Folder not found: {self.current_folder}")
            return
        
        # Load image list
        self.image_files = sorted([
            f for f in self.current_folder.iterdir()
            if f.suffix.lower() == '.jpg'
        ])
        
        print(f"Found {len(self.image_files)} images in {folder_name}")
        
        if not self.image_files:
            QMessageBox.warning(self, "Warning", f"No images found in {folder_name}")
            self.lbl_folder_info.setText(f"{folder_name}: No images")
            return
        
        self.lbl_folder_info.setText(f"{folder_name}: {len(self.image_files)} images")
        self.current_image_idx = 0
        self.load_current_image()
    
    def load_current_image(self):
        """Load current image with YOLO inference and depth data."""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_image_idx]
        print(f"Loading image: {img_path.name}")
        
        # Read image (READ-ONLY)
        self.current_image = cv2.imread(str(img_path))
        if self.current_image is None:
            QMessageBox.warning(self, "Error", f"Failed to load: {img_path.name}")
            return
        
        # Load depth data
        depth_path = img_path.with_suffix('.npy')
        self.current_depth_data = None
        if depth_path.exists():
            try:
                self.current_depth_data = np.load(str(depth_path))
                print(f"Loaded depth data: {self.current_depth_data.shape}")
            except Exception as e:
                print(f"Failed to load depth: {e}")
        
        # Run YOLO inference
        print(f"Running YOLO inference on {self.device}...")
        start_yolo = time.perf_counter()
        
        # Store original image dimensions for scaling BBox
        orig_h, orig_w = self.current_image.shape[:2]
        
        # Inference with optimization parameters
        results = self.yolo_model(
            self.current_image,
            verbose=False,
            device=self.device,
            half=self.params['use_half_precision'] and self.cuda_available,  # FP16 only on GPU
            imgsz=self.params['inference_size']  # Input size (smaller = faster)
        )
        
        end_yolo = time.perf_counter()
        self.yolo_time = (end_yolo - start_yolo) * 1000.0
        print(f"YOLO inference took {self.yolo_time:.2f} ms")
        
        # Extract bbox and scale to original image dimensions
        self.current_bbox = None
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xyxy = box.xyxy[0].cpu().numpy()
            
            # IMPORTANT: BBox coordinates are already in original image space!
            # YOLO automatically scales them regardless of imgsz parameter
            x1, y1, x2, y2 = map(int, xyxy)
            
            self.current_bbox = (x1, y1, x2, y2)
            conf = box.conf[0].item()
            print(f"Detected bbox: {self.current_bbox}, confidence: {conf:.3f}")
            print(f"💡 BBox auf Original-Auflösung ({orig_w}x{orig_h})")
        else:
            print("No detection found")
            QMessageBox.information(self, "No Detection", f"YOLO did not detect any objects in {img_path.name}")
        
        # Update UI
        self.lbl_image_info.setText(f"Image {self.current_image_idx + 1}/{len(self.image_files)}: {img_path.name}")
        self.display_image(self.current_image)
        self.current_result = None
        
        # Update timing
        self.lbl_timing.setText(f"YOLO: {self.yolo_time:.2f} ms | Edge: -- ms | Total: -- ms")
        
        # Auto-apply algorithm if one was previously selected (für Tracking über Zeit)
        if self.combo_algorithm.currentText() and self.current_bbox:
            print(f"🔄 Auto-reapplying: {self.combo_algorithm.currentText()}")
            self.apply_algorithm()
    
    def display_image(self, img: np.ndarray):
        """Display image with bbox overlay."""
        current_zoom = self.view.get_zoom_level()
        
        display_img = img.copy()
        if self.current_bbox:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_img, "YOLO", (x1, max(20, y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to Qt format
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.pixmap_item.setPixmap(pixmap)
        
        if current_zoom > 1.01:
            self.view.set_zoom_level(current_zoom)
        else:
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.view.scale(0.95, 0.95)
    
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
    
    def apply_algorithm(self):
        """Apply selected edge detection algorithm."""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        if self.current_bbox is None:
            QMessageBox.warning(self, "Warning", "No bounding box detected")
            return
        
        algo_name = self.combo_algorithm.currentText()
        
        # Measure edge detection time
        start_edge = time.perf_counter()
        
        if algo_name == "Canny (Auto)":
            result = self.apply_canny_auto()
        elif algo_name == "CLAHE + Canny":
            result = self.apply_clahe_canny()
        elif algo_name == "Canny + Contours":
            result = self.apply_canny_contours()
        elif algo_name == "🔧 Adaptive Pipeline":
            result = self.apply_adaptive_pipeline()
        elif algo_name == "⚡ CLAHE + Direct Canny (Simple)":
            result = self.apply_clahe_direct_canny()
        else:
            result = self.current_image.copy()
        
        end_edge = time.perf_counter()
        self.edge_time = (end_edge - start_edge) * 1000.0
        
        # Update timing
        total_time = self.yolo_time + self.edge_time
        self.lbl_timing.setText(
            f"YOLO: {self.yolo_time:.2f} ms | Edge: {self.edge_time:.2f} ms | Total: {total_time:.2f} ms"
        )
        
        self.current_result = result
        self.display_image(result)
    
    def get_roi(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract ROI from bbox."""
        x1, y1, x2, y2 = self.current_bbox
        h, w = self.current_image.shape[:2]
        
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        
        roi = self.current_image[y1:y2, x1:x2].copy()
        return roi, (x1, y1, x2, y2)
    
    def apply_canny_auto(self) -> np.ndarray:
        """Apply auto-threshold Canny."""
        roi, (x1, y1, x2, y2) = self.get_roi()
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.params['gaussian_sigma'] > 0:
            ksize = int(2 * round(3 * self.params['gaussian_sigma']) + 1)
            gray = cv2.GaussianBlur(gray, (ksize, ksize), self.params['gaussian_sigma'])
        
        v = np.median(gray)
        lower = int(max(0, 0.67 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray, lower, upper)
        
        result = self.current_image.copy()
        result[y1:y2, x1:x2][edges > 0] = [0, 255, 0]
        return result
    
    def apply_clahe_canny(self) -> np.ndarray:
        """Apply CLAHE + Canny."""
        roi, (x1, y1, x2, y2) = self.get_roi()
        
        clahe_roi = apply_clahe(roi, clip_limit=self.params['clahe_clip'])
        edges = canny_edge_detection(clahe_roi, auto_threshold=True, blur_sigma=self.params['gaussian_sigma'])
        
        result = self.current_image.copy()
        result[y1:y2, x1:x2][edges > 0] = [255, 0, 0]
        return result
    
    def apply_canny_contours(self) -> np.ndarray:
        """Apply Canny + contour detection with optional depth filtering."""
        roi, (x1, y1, x2, y2) = self.get_roi()
        
        edges = canny_edge_detection(roi, auto_threshold=True, blur_sigma=self.params['gaussian_sigma'])
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset filtered contours storage
        self.filtered_out_contours = []
        self.contour_depths = []
        
        # Apply depth filter if enabled
        if self.params['use_depth_filter'] and self.current_depth_data is not None:
            valid_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    if self.has_valid_depth(contour, (x1, y1)):
                        valid_contours.append(contour)
                    else:
                        # Store filtered out contours for visualization
                        self.filtered_out_contours.append(contour)
            
            print(f"Depth filtering: {len(valid_contours)} kept, {len(self.filtered_out_contours)} filtered (no valid depth)")
            contours = valid_contours
        else:
            # Filter by area only
            contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        # Get depth values for coloring if enabled
        if self.params['color_by_depth'] and self.current_depth_data is not None:
            for contour in contours:
                depth = self.get_contour_depth(contour, (x1, y1))
                self.contour_depths.append(depth)
            
            # Get depth range for colormap
            valid_depths = [d for d in self.contour_depths if d is not None]
            if valid_depths:
                min_depth = min(valid_depths)
                max_depth = max(valid_depths)
                print(f"🎨 Contour depth range: {min_depth:.2f}m - {max_depth:.2f}m")
            else:
                min_depth = max_depth = 0
        else:
            min_depth = max_depth = 0
        
        # Draw contours
        result = self.current_image.copy()
        roi_result = roi.copy()
        
        # Draw valid contours with depth-based colors
        for i, contour in enumerate(contours):
            if self.params['color_by_depth'] and i < len(self.contour_depths) and self.contour_depths[i] is not None:
                color = self.depth_to_bgr(self.contour_depths[i], min_depth, max_depth)
            else:
                color = (0, 255, 255)  # Default cyan
            
            cv2.drawContours(roi_result, [contour], -1, color, 2)
        
        # Draw filtered out contours in red if enabled
        if self.params['show_filtered_pixels'] and len(self.filtered_out_contours) > 0:
            cv2.drawContours(roi_result, self.filtered_out_contours, -1, (0, 0, 255), 2)
            print(f"Showing {len(self.filtered_out_contours)} filtered contours in red")
        
        result[y1:y2, x1:x2] = roi_result
        
        return result
    
    def has_valid_depth(self, contour, bbox_offset: Tuple[int, int]) -> bool:
        """Check if contour has valid depth data within radius."""
        if self.current_depth_data is None:
            return True
        
        x_offset, y_offset = bbox_offset
        radius = self.params['depth_radius']
        depth_h, depth_w = self.current_depth_data.shape
        
        # Sample contour points
        num_samples = min(len(contour), 20)
        indices = np.linspace(0, len(contour) - 1, num_samples, dtype=int)
        
        for idx in indices:
            pt = contour[idx][0]
            x_img = pt[0] + x_offset
            y_img = pt[1] + y_offset
            
            # Search in neighborhood
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    check_x = x_img + dx
                    check_y = y_img + dy
                    
                    if 0 <= check_y < depth_h and 0 <= check_x < depth_w:
                        depth_val = self.current_depth_data[check_y, check_x]
                        # EXCLUDE black/zero pixels!
                        if depth_val > 0 and np.isfinite(depth_val):
                            return True
        
        return False
    
    def apply_adaptive_pipeline(self) -> np.ndarray:
        """
        Apply adaptive edge detection pipeline with temporal filtering.
        
        This is a deterministic algorithm that:
        - Auto-computes optimal sigma based on ROI size, contrast, depth variance
        - Auto-selects CLAHE vs standard preprocessing
        - Uses multi-threshold Canny for robustness
        - Validates contours by geometry + depth
        - Tracks contours across frames (temporal filtering)
        - Shows debug visualization if enabled
        """
        roi, (x1, y1, x2, y2) = self.get_roi()
        
        # Get depth ROI if available
        depth_roi = None
        if self.current_depth_data is not None:
            depth_h, depth_w = self.current_depth_data.shape
            depth_roi = self.current_depth_data[y1:y2, x1:x2].copy()
        
        # Initialize pipeline on first use
        if self.adaptive_pipeline is None:
            self.adaptive_pipeline = AdaptiveEdgePipeline(
                temporal_history=5,
                temporal_decay=0.7,
                min_temporal_score=0.3,
                debug_mode=self.pipeline_debug_mode,
                minimal_preprocessing=False  # Always use intelligent adaptive mode
            )
            print("🔧 Adaptive Pipeline initialized (intelligent CLAHE + Bilateral decisions)")
        
        # Process with pipeline
        contour_infos, stages = self.adaptive_pipeline.process(
            roi, depth_roi, bbox_offset=(x1, y1)
        )
        
        # Store stages for visualization
        self.pipeline_stages = stages
        
        # Draw results
        result = self.current_image.copy()
        roi_result = roi.copy()
        
        # Compute depth range for coloring
        if self.params['color_by_depth'] and depth_roi is not None:
            depths = [info.depth_mean for info in contour_infos if info.depth_mean is not None]
            if depths:
                min_depth = min(depths)
                max_depth = max(depths)
            else:
                min_depth = max_depth = 0
        else:
            min_depth = max_depth = 0
        
        # Draw contours with depth-based colors and temporal scores
        for info in contour_infos:
            # Color by depth if enabled
            if self.params['color_by_depth'] and info.depth_mean is not None and max_depth > min_depth:
                color = self.depth_to_bgr(info.depth_mean, min_depth, max_depth)
            else:
                # Color by temporal score (green = high, yellow = low)
                score_norm = min(1.0, info.score / 2.0)  # normalize to 0-1
                r = int(255 * (1 - score_norm))
                g = 255
                b = 0
                color = (b, g, r)
            
            # Thickness based on score (persistent contours = thicker)
            thickness = max(1, int(2 * min(1.0, info.score / 1.5)))
            cv2.drawContours(roi_result, [info.contour], -1, color, thickness)
        
        result[y1:y2, x1:x2] = roi_result
        
        # Show debug visualization if enabled
        if self.pipeline_debug_mode and len(stages) > 0:
            self.show_pipeline_debug()
        
        # Print summary
        print(f"🔧 Adaptive Pipeline: {len(contour_infos)} contours")
        if contour_infos:
            avg_score = np.mean([i.score for i in contour_infos])
            print(f"   Avg temporal score: {avg_score:.2f}")
            if depth_roi is not None and depths:
                print(f"   Depth range: {min_depth:.2f}m - {max_depth:.2f}m")
        
        return result
    
    def apply_clahe_direct_canny(self) -> np.ndarray:
        """
        Simple, direct pipeline: CLAHE + Canny + Contours.
        No Bilateral, no Multi-Threshold, no excessive smoothing.
        
        Oft besser als komplexe Pipeline, weil:
        - Weniger Preprocessing → weniger Artefakte
        - Direkte Kanten → keine Verschmierung
        - Schneller → weniger Operationen
        """
        roi, (x1, y1, x2, y2) = self.get_roi()
        
        # Step 1: Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Step 2: CLAHE (immer anwenden, kein intelligentes Skip)
        # Clip limit niedrig halten für weniger Artefakte
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 3: Sehr leichtes Gaussian (nur Rausch-Reduktion, kein starkes Smoothing)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        
        # Step 4: Single-Threshold Canny (Standard Auto-Threshold)
        v = np.median(blurred)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, lower, upper)
        
        # Step 5: Find Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 6: Basic filtering (nur Area, keine komplexe Validation)
        min_area = max(10, 0.0005 * (roi.shape[0] * roi.shape[1]))
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        # Draw results
        result = self.current_image.copy()
        roi_result = roi.copy()
        
        # Color by depth if available
        if self.params['color_by_depth'] and self.current_depth_data is not None:
            depth_h, depth_w = self.current_depth_data.shape
            depth_roi = self.current_depth_data[y1:y2, x1:x2].copy()
            
            # Calculate depth range
            depths = []
            for contour in valid_contours:
                mask = np.zeros(depth_roi.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                contour_depths = depth_roi[mask > 0]
                valid_depths = contour_depths[(contour_depths > 0) & np.isfinite(contour_depths)]
                if valid_depths.size > 0:
                    depths.append(valid_depths.mean())
            
            if depths:
                min_depth, max_depth = min(depths), max(depths)
                
                # Draw with depth colors
                for i, contour in enumerate(valid_contours):
                    if i < len(depths):
                        color = self.depth_to_bgr(depths[i], min_depth, max_depth)
                    else:
                        color = (0, 255, 0)
                    cv2.drawContours(roi_result, [contour], -1, color, 2)
            else:
                # No depth, use green
                cv2.drawContours(roi_result, valid_contours, -1, (0, 255, 0), 2)
        else:
            # No depth mode, use green
            cv2.drawContours(roi_result, valid_contours, -1, (0, 255, 0), 2)
        
        result[y1:y2, x1:x2] = roi_result
        
        print(f"⚡ Simple Pipeline: {len(valid_contours)} contours (from {len(contours)} raw)")
        
        return result
    
    def show_pipeline_debug(self):
        """Show interactive debug visualization - navigate with arrow keys."""
        if not self.pipeline_stages:
            return
        
        # Start with first stage
        self.debug_stage_index = 0
        self.update_debug_window()
        
        print(f"🔍 Debug Mode: Use arrow keys ← → to navigate through {len(self.pipeline_stages)} stages")
        print("   Press 'q' to close debug window")
        print("   Press 'g' for grid view (all steps at once)")
    
    def update_debug_window(self):
        """Update debug window with current stage - ALWAYS horizontal layout."""
        if not self.pipeline_stages or self.debug_stage_index >= len(self.pipeline_stages):
            return
        
        stage = self.pipeline_stages[self.debug_stage_index]
        
        # Get original image
        img = stage.image.copy()
        h, w = img.shape[:2]
        
        # === ALWAYS HORIZONTAL LAYOUT: Text links, Bild rechts ===
        # This avoids text being cut off in narrow top bars
        
        # Scale image to reasonable size (max 800px height, max 1200px width)
        max_img_h = 800
        max_img_w = 1200
        scale = min(max_img_w / w, max_img_h / h, 1.0)  # Don't upscale
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w
        
        # Text panel on the left (500px wide for more space)
        text_panel_width = 500
        margin = 20
        total_width = text_panel_width + w + margin
        canvas_height = max(h + 40, 600)  # Min 600px height for text
        canvas = np.zeros((canvas_height, total_width, 3), dtype=np.uint8)
        
        # Place image on the right, centered vertically
        img_x = text_panel_width + margin
        img_y = (canvas_height - h) // 2
        canvas[img_y:img_y+h, img_x:img_x+w] = img
        
        # === TEXT PANEL (LEFT SIDE) ===
        text_x = 15
        y_pos = 40
        
        # Progress (large, bold)
        progress = f"Step {self.debug_stage_index + 1} / {len(self.pipeline_stages)}"
        cv2.putText(canvas, progress, (text_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 2, cv2.LINE_AA)
        y_pos += 60
        
        # Stage name (large, wrapped if needed)
        stage_name = stage.name
        if len(stage_name) > 25:
            # Wrap long names
            cv2.putText(canvas, stage_name[:25], (text_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            y_pos += 35
            cv2.putText(canvas, stage_name[25:], (text_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            y_pos += 50
        else:
            cv2.putText(canvas, stage_name, (text_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            y_pos += 55
        
        # Separator line
        cv2.line(canvas, (text_x, y_pos), (text_panel_width - 15, y_pos), (80, 80, 80), 2)
        y_pos += 25
        
        # Timing
        time_text = f"Time: {stage.time_ms:.2f} ms"
        cv2.putText(canvas, time_text, (text_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 255), 2, cv2.LINE_AA)
        y_pos += 45
        
        # ROI dimensions
        roi_info = f"Image: {w} x {h} px"
        cv2.putText(canvas, roi_info, (text_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1, cv2.LINE_AA)
        y_pos += 40
        
        # Separator line
        cv2.line(canvas, (text_x, y_pos), (text_panel_width - 15, y_pos), (60, 60, 60), 1)
        y_pos += 25
        
        # Metadata (multi-line, each item on new line)
        if stage.metadata:
            cv2.putText(canvas, "Metadata:", (text_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2, cv2.LINE_AA)
            y_pos += 35
            
            for k, v in stage.metadata.items():
                if isinstance(v, float):
                    meta_text = f"  {k}: {v:.2f}"
                elif isinstance(v, bool):
                    meta_text = f"  {k}: {'✓' if v else '✗'}"
                else:
                    meta_text = f"  {k}: {v}"
                
                # Word wrap long metadata
                if len(meta_text) > 45:
                    cv2.putText(canvas, meta_text[:45], (text_x, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
                    y_pos += 26
                    cv2.putText(canvas, "    " + meta_text[45:], (text_x, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
                else:
                    cv2.putText(canvas, meta_text, (text_x, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
                y_pos += 30
        
        # Navigation hints at bottom
        y_pos = canvas_height - 80
        cv2.line(canvas, (text_x, y_pos), (text_panel_width - 15, y_pos), (80, 80, 80), 1)
        y_pos += 30
        cv2.putText(canvas, "← → : Navigate Steps", (text_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
        y_pos += 28
        cv2.putText(canvas, "G : Grid View", (text_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
        
        # === NAVIGATION HINTS (centered at bottom) ===
        nav_text = "A/D: Navigate  |  G: Grid  |  Q: Quit"
        text_size = cv2.getTextSize(nav_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x_centered = (total_width - text_size[0]) // 2
        cv2.putText(canvas, nav_text, (text_x_centered, canvas_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
        
        # Show in named window (enables key capture)
        window_name = "Pipeline Debug - Interactive"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, canvas)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        self.handle_debug_key(key)
    
    def handle_debug_key(self, key):
        """Handle keyboard input for debug navigation."""
        if key == 255:  # No key pressed
            return
        
        # Arrow keys (platform-dependent, try multiple codes)
        # Left arrow: 81, 2, 63234
        # Right arrow: 83, 3, 63235
        if key in [81, 2, 63234]:  # Left arrow
            if self.debug_stage_index > 0:
                self.debug_stage_index -= 1
                self.update_debug_window()
                print(f"← Step {self.debug_stage_index + 1}/{len(self.pipeline_stages)}: {self.pipeline_stages[self.debug_stage_index].name}")
        
        elif key in [83, 3, 63235]:  # Right arrow
            if self.debug_stage_index < len(self.pipeline_stages) - 1:
                self.debug_stage_index += 1
                self.update_debug_window()
                print(f"→ Step {self.debug_stage_index + 1}/{len(self.pipeline_stages)}: {self.pipeline_stages[self.debug_stage_index].name}")
        
        elif key == ord('q') or key == 27:  # Q or ESC
            cv2.destroyWindow("Pipeline Debug - Interactive")
            cv2.destroyWindow("Pipeline Debug - Grid View")
            print("🔍 Debug window closed")
        
        # Also support A/D keys (easier on some keyboards)
        elif key == ord('a'):  # A = previous
            if self.debug_stage_index > 0:
                self.debug_stage_index -= 1
                self.update_debug_window()
                print(f"← Step {self.debug_stage_index + 1}/{len(self.pipeline_stages)}: {self.pipeline_stages[self.debug_stage_index].name}")
        
        elif key == ord('d'):  # D = next
            if self.debug_stage_index < len(self.pipeline_stages) - 1:
                self.debug_stage_index += 1
                self.update_debug_window()
                print(f"→ Step {self.debug_stage_index + 1}/{len(self.pipeline_stages)}: {self.pipeline_stages[self.debug_stage_index].name}")
        
        elif key == ord('g'):  # G = grid view
            self.show_grid_view()
        
        elif key == ord('h'):  # H = help
            print("\n🔍 Debug Navigation:")
            print("   ← / A  : Previous step")
            print("   → / D  : Next step")
            print("   G      : Grid view (all steps)")
            print("   Q / ESC: Close debug window")
            print("   H      : Show this help")
    
    def show_grid_view(self):
        """Show all pipeline stages in grid layout."""
        if not self.pipeline_stages:
            return
        
        # Calculate grid layout (try to make it roughly square)
        n_stages = len(self.pipeline_stages)
        cols = int(np.ceil(np.sqrt(n_stages)))
        rows = int(np.ceil(n_stages / cols))
        
        # Target size for each cell
        cell_width = 320
        cell_height = 280
        
        # Create canvas
        canvas_width = cols * cell_width
        canvas_height = rows * cell_height
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        for idx, stage in enumerate(self.pipeline_stages):
            row = idx // cols
            col = idx % cols
            
            # Resize stage image to fit cell
            img = stage.image.copy()
            h, w = img.shape[:2]
            
            # Scale to fit in cell (leave margin for text)
            max_w = cell_width - 10
            max_h = cell_height - 70  # Leave space for text
            
            scale = min(max_w / w, max_h / h)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Place in canvas
            y_start = row * cell_height + 5
            x_start = col * cell_width + 5
            y_end = y_start + img.shape[0]
            x_end = x_start + img.shape[1]
            
            canvas[y_start:y_end, x_start:x_end] = img
            
            # Add stage name
            text_y = row * cell_height + cell_height - 55
            text_x = col * cell_width + 10
            cv2.putText(canvas, stage.name, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add timing
            time_text = f"{stage.time_ms:.1f}ms"
            cv2.putText(canvas, time_text, (text_x, text_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Add metadata if available
            if stage.metadata:
                meta_items = []
                for k, v in list(stage.metadata.items())[:2]:
                    if isinstance(v, float):
                        meta_items.append(f"{k}:{v:.1f}")
                    else:
                        meta_items.append(f"{k}:{v}")
                meta_str = " ".join(meta_items)
                cv2.putText(canvas, meta_str, (text_x, text_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Show in resizable window
        cv2.namedWindow("Pipeline Debug - Grid View", cv2.WINDOW_NORMAL)
        cv2.imshow("Pipeline Debug - Grid View", canvas)
        cv2.waitKey(1)
        
        print("📊 Grid View: Press 'q' to close, or use interactive mode (arrow keys)")
    
    def on_debug_key_event(self):
        """Called periodically to check for key events in debug mode."""
        if self.pipeline_debug_mode and self.pipeline_stages:
            key = cv2.waitKey(1) & 0xFF
            self.handle_debug_key(key)

    
    def show_depth_map(self):
        """Display depth map ROI (within BBox) in separate OpenCV window with clean layout."""
        if self.current_depth_data is None:
            QMessageBox.information(self, "No Depth", "No depth data available for this image")
            return
        
        if self.current_bbox is None:
            QMessageBox.information(self, "No BBox", "No bounding box detected. Run YOLO first.")
            return
        
        # Extract ROI from depth data
        x1, y1, x2, y2 = self.current_bbox
        depth_h, depth_w = self.current_depth_data.shape
        
        # Clip to valid bounds
        x1 = max(0, min(depth_w - 1, x1))
        y1 = max(0, min(depth_h - 1, y1))
        x2 = max(0, min(depth_w - 1, x2))
        y2 = max(0, min(depth_h - 1, y2))
        
        # Extract depth ROI
        depth_roi = self.current_depth_data[y1:y2, x1:x2].copy()
        
        if depth_roi.size == 0:
            QMessageBox.warning(self, "Invalid ROI", "BBox region is empty")
            return
        
        # Filter valid depth (EXCLUDE black/zero!)
        valid_mask = (depth_roi > 0) & np.isfinite(depth_roi)
        
        if not valid_mask.any():
            QMessageBox.warning(self, "No Valid Depth", 
                              "All depth values in ROI are zero/invalid (black pixels)")
            return
        
        # Calculate statistics
        vmin = depth_roi[valid_mask].min()
        vmax = depth_roi[valid_mask].max()
        valid_percent = 100 * valid_mask.sum() / valid_mask.size
        mean_depth = depth_roi[valid_mask].mean()
        std_depth = depth_roi[valid_mask].std()
        
        # Normalize depth to 0-255 for colormap
        normalized = np.zeros_like(depth_roi, dtype=np.uint8)
        if vmax > vmin:
            normalized[valid_mask] = ((depth_roi[valid_mask] - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
        # Apply VIRIDIS colormap
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        colored[~valid_mask] = [0, 0, 0]  # Black for invalid pixels
        
        # --- RESIZE to minimum display size (avoid ultra zoom) ---
        MIN_DISPLAY_WIDTH = 500
        MIN_DISPLAY_HEIGHT = 300
        
        roi_h, roi_w = colored.shape[:2]
        scale_w = max(1.0, MIN_DISPLAY_WIDTH / roi_w)
        scale_h = max(1.0, MIN_DISPLAY_HEIGHT / roi_h)
        scale = min(scale_w, scale_h)  # Keep aspect ratio
        
        if scale > 1.0:
            new_w = int(roi_w * scale)
            new_h = int(roi_h * scale)
            colored_resized = cv2.resize(colored, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            colored_resized = colored
            new_h = roi_h
        
        # --- CREATE COLORBAR (fixed height matching display) ---
        colorbar_width = 60
        colorbar_height = new_h
        colorbar = np.linspace(255, 0, colorbar_height).astype(np.uint8).reshape(-1, 1)
        colorbar = np.repeat(colorbar, colorbar_width, axis=1)
        colorbar_colored = cv2.applyColorMap(colorbar, cv2.COLORMAP_VIRIDIS)
        
        # --- ADD TOP BAR for title (clean separation) ---
        top_bar_height = 80
        top_bar = np.zeros((top_bar_height, colored_resized.shape[1] + colorbar_width, 3), dtype=np.uint8)
        
        # Title
        title = f"Depth ROI [{x1}:{x2}, {y1}:{y2}]"
        cv2.putText(top_bar, title, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Range
        range_text = f"Range: {vmin:.2f}m - {vmax:.2f}m"
        cv2.putText(top_bar, range_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Stats
        stats = f"Valid: {valid_percent:.1f}% | Mean: {mean_depth:.2f}m (+/-{std_depth:.2f}m) | Size: {x2-x1}x{y2-y1}px"
        cv2.putText(top_bar, stats, (10, 72), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        # --- ADD COLORBAR LABELS (right side) ---
        label_positions = [30, colorbar_height//2, colorbar_height-20]
        label_values = [vmax, (vmax+vmin)/2, vmin]
        
        for pos, val in zip(label_positions, label_values):
            cv2.putText(colorbar_colored, f"{val:.2f}m", (5, pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # --- COMBINE: TOP BAR + (DEPTH ROI | COLORBAR) ---
        depth_and_colorbar = np.hstack([colored_resized, colorbar_colored])
        final = np.vstack([top_bar, depth_and_colorbar])
        
        # Show in resizable window
        cv2.namedWindow("Depth ROI Visualization", cv2.WINDOW_NORMAL)
        cv2.imshow("Depth ROI Visualization", final)
        cv2.waitKey(1)
        
        print(f"📊 Depth ROI [{x1}:{x2}, {y1}:{y2}]: {vmin:.2f}m - {vmax:.2f}m, {valid_percent:.1f}% valid")
    
    def get_contour_depth(self, contour, bbox_offset: Tuple[int, int]) -> Optional[float]:
        """Get average depth for contour within averaging radius."""
        if self.current_depth_data is None:
            return None
        
        x_offset, y_offset = bbox_offset
        radius = self.params['depth_avg_radius']
        depth_h, depth_w = self.current_depth_data.shape
        
        depth_values = []
        
        # Sample contour points
        num_samples = min(len(contour), 30)
        indices = np.linspace(0, len(contour) - 1, num_samples, dtype=int)
        
        for idx in indices:
            pt = contour[idx][0]
            x_img = pt[0] + x_offset
            y_img = pt[1] + y_offset
            
            # Average in neighborhood
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    check_x = x_img + dx
                    check_y = y_img + dy
                    
                    if 0 <= check_y < depth_h and 0 <= check_x < depth_w:
                        depth_val = self.current_depth_data[check_y, check_x]
                        # EXCLUDE black/zero pixels!
                        if depth_val > 0 and np.isfinite(depth_val):
                            depth_values.append(depth_val)
        
        if depth_values:
            return float(np.mean(depth_values))
        return None
    
    def depth_to_bgr(self, depth: float, min_d: float, max_d: float) -> Tuple[int, int, int]:
        """Convert depth to BGR color using viridis-like colormap."""
        if max_d == min_d:
            norm = 0.5
        else:
            norm = np.clip((depth - min_d) / (max_d - min_d), 0, 1)
        
        # Viridis approximation (similar to matplotlib)
        if norm < 0.25:
            r = int(68 + (norm / 0.25) * 155)
            g = int(1 + (norm / 0.25) * 97)
            b = int(84 + (norm / 0.25) * 26)
        elif norm < 0.5:
            t = (norm - 0.25) / 0.25
            r = int(33 + t * 15)
            g = int(145 + t * 15)
            b = int(140 + t * 20)
        elif norm < 0.75:
            t = (norm - 0.5) / 0.25
            r = int(53 + t * 79)
            g = int(183 + t * 20)
            b = int(121 - t * 121)
        else:
            t = (norm - 0.75) / 0.25
            r = 253
            g = int(231 - t * 108)
            b = 37
        
        return (b, g, r)  # BGR for OpenCV


def main():
    app = QApplication(sys.argv)
    window = SVOEdgeDetectionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
