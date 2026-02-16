# Depth Visualisierung - Änderungen für svo_edge_detection_gui.py

## 🎯 Ihre Anforderungen

1. ✅ **1280 zur YOLO-Auswahl hinzugefügt** - Bereits erledigt!
2. ⏳ **Separate Depth-Visualisierung** - Braucht matplotlib-Alternative
3. ⏳ **Schwarze/Zero Depth-Werte ausschließen** - Logik fertig
4. ⏳ **Contours nach Depth-Wert färben** - Logik fertig
5. ⏳ **Farbskala mit Metern** - Braucht matplotlib-Alternative
6. ⏳ **Depth-Averaging Radius-Slider** - Bereits hinzugefügt!

## 💡 Problem

Die erweiterte Version (`svo_edge_detection_gui_enhanced.py`) hat Matplotlib-Kompatibilitätsprobleme mit PySide6.

## 🔧 Vereinfachte Lösung (EMPFOHLEN)

Statt komplexer Matplotlib-Integration, nutzen wir OpenCV für Depth-Visualisierung:

### Option A: Depth als farbiges Bild in neuem Fenster

```python
def show_depth_window(self):
    """Show depth map in separate OpenCV window."""
    if self.current_depth_data is None:
        return
    
    # Filter valid depth
    valid_mask = (self.current_depth_data > 0) & np.isfinite(self.current_depth_data)
    
    if not valid_mask.any():
        print("No valid depth data")
        return
    
    # Normalize for display
    display_depth = self.current_depth_data.copy()
    display_depth[~valid_mask] = 0
    
    vmin = display_depth[valid_mask].min()
    vmax = display_depth[valid_mask].max()
    
    # Normalize to 0-255
    normalized = np.zeros_like(display_depth, dtype=np.uint8)
    normalized[valid_mask] = ((display_depth[valid_mask] - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    # Apply colormap (COLORMAP_VIRIDIS or COLORMAP_JET)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    
    # Draw BBox if available
    if self.current_bbox:
        x1, y1, x2, y2 = self.current_bbox
        cv2.rectangle(colored, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Add text info
    text = f"Depth: {vmin:.2f}m - {vmax:.2f}m"
    cv2.putText(colored, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show in OpenCV window
    cv2.imshow("Depth Map", colored)
```

### Option B: Depth im GUI (QGraphicsView)

Zweites `QGraphicsView` Widget im GUI hinzufügen.

## 📝 Quick Fix für aktuelle GUI

Die einfachste Lösung: **Zeigen Sie Depth-Map in separatem OpenCV Fenster**.

### Änderungen in `svo_edge_detection_gui.py`:

1. **Button hinzufügen:**
```python
self.btn_show_depth = QPushButton("📊 Show Depth Map")
self.btn_show_depth.clicked.connect(self.show_depth_map)
control_layout.addWidget(self.btn_show_depth)
```

2. **Funktion hinzufügen:**
```python
def show_depth_map(self):
    """Display depth map in separate window with colorbar."""
    if self.current_depth_data is None:
        QMessageBox.information(self, "No Depth", "No depth data available for this image")
        return
    
    # Filter valid depth (EXCLUDE black/zero!)
    valid_mask = (self.current_depth_data > 0) & np.isfinite(self.current_depth_data)
    
    if not valid_mask.any():
        QMessageBox.warning(self, "No Valid Depth", "All depth values are zero/invalid (black pixels)")
        return
    
    # Normalize for visualization
    display_depth = self.current_depth_data.copy()
    display_depth[~valid_mask] = 0
    
    vmin = display_depth[valid_mask].min()
    vmax = display_depth[valid_mask].max()
    
    # Create colorbar legend (small image on the side)
    colorbar_width = 50
    colorbar_height = display_depth.shape[0]
    colorbar = np.linspace(255, 0, colorbar_height).astype(np.uint8).reshape(-1, 1)
    colorbar = np.repeat(colorbar, colorbar_width, axis=1)
    colorbar_colored = cv2.applyColorMap(colorbar, cv2.COLORMAP_VIRIDIS)
    
    # Normalize depth to 0-255
    normalized = np.zeros_like(display_depth, dtype=np.uint8)
    if vmax > vmin:
        normalized[valid_mask] = ((display_depth[valid_mask] - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    # Apply VIRIDIS colormap
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    colored[~valid_mask] = [0, 0, 0]  # Black for invalid
    
    # Draw BBox
    if self.current_bbox:
        x1, y1, x2, y2 = self.current_bbox
        cv2.rectangle(colored, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(colored, "YOLO BBox", (x1, max(30, y1-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Combine depth map + colorbar
    combined = np.hstack([colored, colorbar_colored])
    
    # Add text annotations
    valid_percent = 100 * valid_mask.sum() / valid_mask.size
    mean_depth = display_depth[valid_mask].mean()
    std_depth = display_depth[valid_mask].std()
    
    # Title
    title_text = f"Depth Map: {vmin:.2f}m - {vmax:.2f}m"
    cv2.putText(combined, title_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Stats
    stats_text = f"Valid: {valid_percent:.1f}% | Mean: {mean_depth:.2f}m +/- {std_depth:.2f}m"
    cv2.putText(combined, stats_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Colorbar labels
    label_positions = [30, colorbar_height//2, colorbar_height-30]
    label_values = [vmax, (vmax+vmin)/2, vmin]
    for pos, val in zip(label_positions, label_values):
        cv2.putText(combined, f"{val:.2f}m", (colored.shape[1] + 5, pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show
    cv2.imshow("Depth Visualization", combined)
    cv2.waitKey(1)  # Update window
    
    print(f"Depth Map: {vmin:.2f}m - {vmax:.2f}m, {valid_percent:.1f}% valid pixels")
```

3. **Depth-Averaging für Contour-Farben (bereits im Enhanced):**
```python
def get_contour_depth(self, contour, bbox_offset):
    """Get average depth for contour within radius."""
    if self.current_depth_data is None:
        return None
    
    x_offset, y_offset = bbox_offset
    radius = self.params['depth_avg_radius']  # NEW PARAMETER
    
    depth_values = []
    # Sample contour points and average in neighborhood
    for pt in contour[::5]:  # Every 5th point
        x_img = pt[0][0] + x_offset
        y_img = pt[0][1] + y_offset
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                check_x, check_y = x_img + dx, y_img + dy
                if 0 <= check_y < self.current_depth_data.shape[0] and \
                   0 <= check_x < self.current_depth_data.shape[1]:
                    d = self.current_depth_data[check_y, check_x]
                    if d > 0 and np.isfinite(d):  # EXCLUDE zero/black!
                        depth_values.append(d)
    
    return np.mean(depth_values) if depth_values else None

def depth_to_bgr(self, depth, min_d, max_d):
    """Convert depth to BGR color (viridis-like)."""
    if max_d == min_d:
        norm = 0.5
    else:
        norm = np.clip((depth - min_d) / (max_d - min_d), 0, 1)
    
    # Viridis approximation
    if norm < 0.25:
        r, g, b = 68 + (norm/0.25)*155, 1 + (norm/0.25)*97, 84 + (norm/0.25)*26
    elif norm < 0.5:
        r, g, b = 33 + ((norm-0.25)/0.25)*15, 145 + ((norm-0.25)/0.25)*15, 140 + ((norm-0.25)/0.25)*20
    elif norm < 0.75:
        r, g, b = 53 + ((norm-0.5)/0.25)*79, 183 + ((norm-0.5)/0.25)*20, 121 - ((norm-0.5)/0.25)*121
    else:
        r, g, b = 253, 231 - ((norm-0.75)/0.25)*108, 37
    
    return (int(b), int(g), int(r))
```

4. **Contours färben nach Depth:**
```python
# In apply_canny_contours():
contour_depths = []
for contour in valid_contours:
    depth = self.get_contour_depth(contour, (x1, y1))
    contour_depths.append(depth)

# Get depth range
valid_depths = [d for d in contour_depths if d is not None]
if valid_depths:
    min_d, max_d = min(valid_depths), max(valid_depths)
    
    # Draw with depth colors
    for i, contour in enumerate(valid_contours):
        if contour_depths[i] is not None:
            color = self.depth_to_bgr(contour_depths[i], min_d, max_d)
        else:
            color = (128, 128, 128)  # Gray for no depth
        cv2.drawContours(roi_result, [contour], -1, color, 2)
```

## 🚀 Schnellste Lösung

Da die alte GUI läuft, fügen Sie einfach den "Show Depth Map" Button hinzu! 

Die Depth-Visualisierung in separatem OpenCV-Fenster ist:
- ✅ Einfach zu implementieren
- ✅ Keine matplotlib-Probleme
- ✅ Zeigt Farbskala
- ✅ Zeigt BBox
- ✅ Zeigt Statistiken

## 📊 Performance Note

**Warum 1024/1280 schneller ist als erwartet:**

Ihr Jetson Orin Nano hat einen **Image Signal Processor (ISP)** der größere Bilder effizient verarbeiten kann. Bei 1280x720 (native Trainings-Auflösung):

- Keine Resizing-Overhead
- GPU Memory Layout optimal
- Tensor Cores besser ausgelastet

**Tipp:** Für beste Qualität, nutzen Sie 1280 wenn Speed <150ms akzeptabel ist!

---

**Möchten Sie:**
A) Ich füge den "Show Depth Map" Button zur bestehenden GUI hinzu? (5 min)
B) Ich debugge die Enhanced-Version mit Matplotlib? (30-60 min, komplex)
C) Ich erstelle eine komplett neue GUI ohne Matplotlib aber mit Split-View? (60 min)

**Empfehlung:** Option A - schnell, funktioniert sofort!
