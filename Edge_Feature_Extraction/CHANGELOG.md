# Changelog - Edge Feature Extraction GUI

## [v0.3.0] - 2024-01-XX - Parameter Management Overhaul

### ✨ New Features

#### New Algorithms
- **Structured Edges**: Multi-scale gradient fusion for robust edge detection using adaptive thresholding
- **Active Contours (Snake)**: Energy-minimizing spline that evolves toward edges, initialized at ROI center
  - Uses scikit-image's `active_contour` with configurable elasticity, stiffness, and iterations
  - Fallback to initialization visualization if contour evolution fails

#### Advanced Parameter Management
- **Dynamic Slider Visibility**: Sliders automatically enable/disable based on selected algorithm's requirements
- **Reset Parameters Button**: One-click restore of all parameters to default values
- **Auto-Update Sliders**: Canny (Auto) now updates slider values with computed thresholds in real-time
  - Displays computed low/high thresholds after automatic median-based calculation
  - Blocks signals during update to prevent recursive calls

#### Enhanced UI/UX
- **Thinner Visual Elements**: BBox drawn with 1px thickness (down from 2px), text scale 0.4 (down from 0.6)
- **Better Initial Display**: Images now scale to 0.95x after fitInView for immediate usability
- **Organized Parameter Panel**: 9 parameters in clean two-row layout with descriptive labels

### 🔧 Parameter Additions
- `morph_kernel`: Morphology kernel size for contour silhouette (3-15, default 5)
- `snake_alpha`: Active Contours elasticity (0.001-1.0, default 0.01, 3 decimals)
- `snake_beta`: Active Contours stiffness (0.001-1.0, default 0.1, 3 decimals)
- `snake_iterations`: Active Contours evolution steps (10-500, default 100)

### 🐛 Bug Fixes
- Fixed algorithm selection causing zoom reset (preserved zoom level across algorithm changes)
- Fixed BBox edges being detected as contours (5px inward padding on ROI extraction)
- Fixed sliders not respecting algorithm-specific parameters
- Removed unused import `QFormLayout`

### 🏗️ Architecture Changes
- **Algorithm Dictionary Restructure**: Changed from `{"name": func}` to `{"name": {'func': func, 'params': [list]}}`
  - Enables metadata-driven parameter management
  - Simplifies adding new algorithms with custom parameters
- **Parameter Widget Tracking**: Centralized `self.param_widgets` dict for slider lifecycle management
- **Signal Blocking**: Proper signal blocking during programmatic slider updates prevents feedback loops

### 📊 Algorithm Performance
Confirmed strong performance (user feedback):
- ✅ **Canny (Auto)**: "sehr stark" (very strong)
- ✅ **CLAHE + Canny**: "sehr stark" (very strong)
- ⚠️ Other algorithms: Less robust, require further tuning

---

## [v0.2.0] - Previous Version

### Features
- Interactive GUI with zoomable image viewer
- 8 edge detection algorithms (Canny Auto/Manual, CLAHE+Canny, Sobel, LoG, Multi-scale, Contours, Hough)
- Parameter sliders for real-time tuning
- Session-wise output saving with JSON metadata
- Runtime/FPS measurement
- Zoom preservation on algorithm change
- BBox padding to avoid edge artifacts

### Algorithms
1. **Canny (Auto)**: Auto-threshold Canny with median-based threshold selection
2. **Canny (Manual)**: Manual-threshold Canny with user-defined low/high thresholds
3. **CLAHE + Canny**: Contrast Limited Adaptive Histogram Equalization + Canny
4. **Sobel**: Sobel gradient magnitude with fixed threshold
5. **LoG**: Laplacian of Gaussian edge detection
6. **Multi-scale Canny**: Multi-sigma Canny with OR-fusion
7. **Canny + Contours**: Contour-based silhouette extraction
8. **Canny + Hough**: Hough probabilistic line detection

---

## [v0.1.0] - Initial Release

### Features
- Project structure creation
- Helper script for debug image extraction (depth ≤ 15m filter)
- Git repository initialization on GitHub (feature branch: `feature/edge-extraction`)
- Basic GUI framework with PySide6
- YOLO label parsing for bounding box extraction
- Grayscale preprocessing pipeline

---

## Technical Notes

### Dependencies
- PySide6 >= 6.4 (Qt for Python)
- OpenCV >= 4.5.5
- NumPy >= 1.21
- Pandas >= 1.3
- scikit-image >= 0.19 (for Active Contours)

### Platform Compatibility
- Developed on Linux (Jetson ARM64)
- PySide6 used instead of PyQt5/6 due to ARM64 build issues

### Architecture
- **Pattern**: MVC-style separation
- **GUI**: PySide6 (Qt) - `src/gui/edge_detection_gui.py`
- **Algorithms**: Pure OpenCV/NumPy - `src/algorithms/*.py`
- **Data**: File I/O, YOLO label parsing

### Known Limitations
- Structured Edges uses gradient fusion proxy (not true structured forests, which requires pre-trained model)
- Active Contours requires circular initialization (not bbox-fitted)
- Some algorithms (Sobel, LoG, Hough) less robust than Canny-based methods
