#!/usr/bin/env python3
"""
SVO Mode Edge Detection GUI - Enhanced with Depth Visualization

New Features:
- Separate depth map visualization
- Contours colored by depth value
- Depth colorbar with meters
- Improved depth filtering (excludes black/zero pixels)
- Depth averaging radius slider
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
    QRadioButton, QButtonGroup, QSplitter
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QPainter

# Add parent to path for algorithm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.canny_pipeline import canny_edge_detection
from algorithms.preproc import apply_clahe

# YOLO import with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


class DepthVisualizationWidget(QWidget):
    """Widget for depth map visualization with colorbar."""
    
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.ax_depth = self.figure.add_subplot(111)
        self.depth_image = None
        self.colorbar = None
        
    def update_depth_map(self, depth_data: Optional[np.ndarray], bbox: Optional[Tuple[int, int, int, int]] = None):
        """Update depth map visualization."""
        self.ax_depth.clear()
        
        if depth_data is None:
            self.ax_depth.text(0.5, 0.5, 'No Depth Data', 
                              ha='center', va='center', fontsize=14, color='red')
            self.ax_depth.set_xlim([0, 1])
            self.ax_depth.set_ylim([0, 1])
        else:
            # Filter out zero/invalid depth values for better visualization
            valid_mask = (depth_data > 0) & np.isfinite(depth_data)
            
            if not valid_mask.any():
                self.ax_depth.text(0.5, 0.5, 'No Valid Depth Data\n(All Black/Zero)', 
                                  ha='center', va='center', fontsize=12, color='orange')
                self.ax_depth.set_xlim([0, 1])
                self.ax_depth.set_ylim([0, 1])
            else:
                # Use viridis colormap (good for depth)
                display_depth = depth_data.copy()
                display_depth[~valid_mask] = np.nan  # NaN for invalid pixels
                
                vmin = np.nanmin(display_depth)
                vmax = np.nanmax(display_depth)
                
                im = self.ax_depth.imshow(display_depth, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
                
                # Draw bbox if provided
                if bbox:
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        linewidth=2, edgecolor='red', facecolor='none')
                    self.ax_depth.add_patch(rect)
                
                # Colorbar with meters
                if self.colorbar:
                    self.colorbar.remove()
                self.colorbar = self.figure.colorbar(im, ax=self.ax_depth, label='Depth (m)')
                self.colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
                
                self.ax_depth.set_title(f'Depth Map ({vmin:.2f}m - {vmax:.2f}m)')
                
                # Stats
                mean_depth = np.nanmean(display_depth)
                std_depth = np.nanstd(display_depth)
                valid_pixels = valid_mask.sum()
                total_pixels = depth_data.size
                self.ax_depth.text(0.02, 0.98, 
                                  f'Valid: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)\n'
                                  f'Mean: {mean_depth:.2f}m ± {std_depth:.2f}m',
                                  transform=self.ax_depth.transAxes,
                                  verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                  fontsize=9)
        
        self.ax_depth.set_xlabel('X (pixels)')
        self.ax_depth.set_ylabel('Y (pixels)')
        self.figure.tight_layout()
        self.canvas.draw()


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
    """Enhanced GUI for SVO frame edge detection with depth visualization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SVO Edge Detection - YOLO + Depth + Visualization")
        self.setGeometry(50, 50, 1800, 1000)
        
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
            'depth_radius': 5,
            'depth_avg_radius': 3,  # NEW: for averaging depth around contour points
            'use_depth_filter': False,
            'show_filtered_pixels': False,
            'color_by_depth': True,  # NEW: color contours by depth
            'use_half_precision': False,
            'inference_size': 320,
            'use_original_for_edges': True,
        }
        
        # Storage for contour depth info
        self.contour_depths: List[float] = []
        self.filtered_out_contours: List = []
        
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
    
    def init_ui(self):
        """Initialize user interface with split view."""
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
        
        # Device selection
        folder_layout.addWidget(QLabel("Device:"))
        self.btn_group_device = QButtonGroup()
        self.radio_cpu = QRadioButton("CPU")
        self.radio_gpu = QRadioButton("GPU")
        self.btn_group_device.addButton(self.radio_cpu)
        self.btn_group_device.addButton(self.radio_gpu)
        
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
        self.combo_algorithm.addItems(["Canny (Auto)", "CLAHE + Canny", "Canny + Contours (Depth-Colored)"])
        control_layout.addWidget(self.combo_algorithm)
        
        self.btn_apply = QPushButton("Apply Algorithm")
        self.btn_apply.clicked.connect(self.apply_algorithm)
        control_layout.addWidget(self.btn_apply)
        
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
        
        param_layout.addWidget(QLabel("Search Radius:"))
        self.slider_depth = QSlider(Qt.Orientation.Horizontal)
        self.slider_depth.setRange(1, 50)
        self.slider_depth.setValue(self.params['depth_radius'])
        self.slider_depth.setToolTip("Radius to search for valid depth around contour points")
        self.slider_depth.valueChanged.connect(lambda v: self.update_param('depth_radius', v))
        param_layout.addWidget(self.slider_depth)
        self.lbl_depth = QLabel(str(self.params['depth_radius']))
        param_layout.addWidget(self.lbl_depth)
        
        # NEW: Depth Averaging Radius
        param_layout.addWidget(QLabel("Avg Radius:"))
        self.slider_depth_avg = QSlider(Qt.Orientation.Horizontal)
        self.slider_depth_avg.setRange(1, 20)
        self.slider_depth_avg.setValue(self.params['depth_avg_radius'])
        self.slider_depth_avg.setToolTip("Radius for averaging depth values around contour (for smoother colors)")
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
        self.chk_color_depth.setToolTip("Color contours according to their depth value")
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
        
        perf_layout.addStretch()
        perf_group.setLayout(perf_layout)
        main_layout.addWidget(perf_group)
        
        # SPLIT VIEW: Image + Depth Map
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Image viewer
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.addWidget(QLabel("Edge Detection Result:"))
        
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        image_layout.addWidget(self.view)
        
        # Right: Depth visualization
        depth_widget = QWidget()
        depth_layout = QVBoxLayout(depth_widget)
        depth_layout.addWidget(QLabel("Depth Map:"))
        self.depth_viz = DepthVisualizationWidget()
        depth_layout.addWidget(self.depth_viz)
        
        splitter.addWidget(image_widget)
        splitter.addWidget(depth_widget)
        splitter.setSizes([1000, 800])  # Initial sizes
        
        main_layout.addWidget(splitter)
    
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
            if self.current_result is not None:
                self.apply_algorithm()
        elif key == 'use_half_precision' or key == 'inference_size':
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
        
        if self.current_image is not None:
            print("Reloading image with new device...")
            self.load_current_image()
    
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
        
        # Read image
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
        
        orig_h, orig_w = self.current_image.shape[:2]
        
        results = self.yolo_model(
            self.current_image,
            verbose=False,
            device=self.device,
            half=self.params['use_half_precision'] and self.cuda_available,
            imgsz=self.params['inference_size']
        )
        
        end_yolo = time.perf_counter()
        self.yolo_time = (end_yolo - start_yolo) * 1000.0
        print(f"YOLO inference took {self.yolo_time:.2f} ms")
        
        # Extract bbox
        self.current_bbox = None
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            self.current_bbox = (x1, y1, x2, y2)
            conf = box.conf[0].item()
            print(f"Detected bbox: {self.current_bbox}, confidence: {conf:.3f}")
        else:
            print("No detection found")
        
        # Update UI
        self.lbl_image_info.setText(f"Image {self.current_image_idx + 1}/{len(self.image_files)}: {img_path.name}")
        self.display_image(self.current_image)
        self.current_result = None
        
        # Update depth visualization
        self.depth_viz.update_depth_map(self.current_depth_data, self.current_bbox)
        
        # Update timing
        self.lbl_timing.setText(f"YOLO: {self.yolo_time:.2f} ms | Edge: -- ms | Total: -- ms")
    
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
        
        start_edge = time.perf_counter()
        
        if algo_name == "Canny (Auto)":
            result = self.apply_canny_auto()
        elif algo_name == "CLAHE + Canny":
            result = self.apply_clahe_canny()
        elif "Contours" in algo_name:
            result = self.apply_canny_contours_with_depth()
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
    
    def get_contour_depth(self, contour, bbox_offset: Tuple[int, int]) -> Optional[float]:
        """Get average depth value for contour within averaging radius."""
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
    
    def depth_to_color(self, depth: float, min_depth: float, max_depth: float) -> Tuple[int, int, int]:
        """Convert depth value to BGR color using viridis colormap."""
        if max_depth == min_depth:
            normalized = 0.5
        else:
            normalized = (depth - min_depth) / (max_depth - min_depth)
        normalized = np.clip(normalized, 0, 1)
        
        # Viridis colormap (approximation)
        cmap = plt.cm.get_cmap('viridis')
        rgba = cmap(normalized)
        # Convert RGBA (0-1) to BGR (0-255)
        b = int(rgba[2] * 255)
        g = int(rgba[1] * 255)
        r = int(rgba[0] * 255)
        return (b, g, r)
    
    def has_valid_depth(self, contour, bbox_offset: Tuple[int, int]) -> bool:
        """Check if contour has valid depth data within search radius."""
        if self.current_depth_data is None:
            return True  # No depth data = no filtering
        
        x_offset, y_offset = bbox_offset
        radius = self.params['depth_radius']
        depth_h, depth_w = self.current_depth_data.shape
        
        num_samples = min(len(contour), 20)
        indices = np.linspace(0, len(contour) - 1, num_samples, dtype=int)
        
        for idx in indices:
            pt = contour[idx][0]
            x_img = pt[0] + x_offset
            y_img = pt[1] + y_offset
            
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
    
    def apply_canny_contours_with_depth(self) -> np.ndarray:
        """Apply Canny + contours with depth filtering and depth-based coloring."""
        roi, (x1, y1, x2, y2) = self.get_roi()
        
        edges = canny_edge_detection(roi, auto_threshold=True, blur_sigma=self.params['gaussian_sigma'])
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.filtered_out_contours = []
        self.contour_depths = []
        
        # Filter by depth if enabled
        if self.params['use_depth_filter'] and self.current_depth_data is not None:
            valid_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    if self.has_valid_depth(contour, (x1, y1)):
                        valid_contours.append(contour)
                    else:
                        self.filtered_out_contours.append(contour)
            
            print(f"Depth filtering: {len(valid_contours)} kept, {len(self.filtered_out_contours)} filtered (no valid depth)")
            contours = valid_contours
        else:
            contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        # Get depth values for coloring
        if self.params['color_by_depth'] and self.current_depth_data is not None:
            for contour in contours:
                depth = self.get_contour_depth(contour, (x1, y1))
                self.contour_depths.append(depth)
            
            # Get depth range for colormap
            valid_depths = [d for d in self.contour_depths if d is not None]
            if valid_depths:
                min_depth = min(valid_depths)
                max_depth = max(valid_depths)
                print(f"Contour depth range: {min_depth:.2f}m - {max_depth:.2f}m")
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
                color = self.depth_to_color(self.contour_depths[i], min_depth, max_depth)
            else:
                color = (0, 255, 255)  # Default cyan
            
            cv2.drawContours(roi_result, [contour], -1, color, 2)
        
        # Draw filtered contours in red if enabled
        if self.params['show_filtered_pixels'] and len(self.filtered_out_contours) > 0:
            cv2.drawContours(roi_result, self.filtered_out_contours, -1, (0, 0, 255), 2)
            print(f"Showing {len(self.filtered_out_contours)} filtered contours in red")
        
        result[y1:y2, x1:x2] = roi_result
        
        return result


def main():
    app = QApplication(sys.argv)
    window = SVOEdgeDetectionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
