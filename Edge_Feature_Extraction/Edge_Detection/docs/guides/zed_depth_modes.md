# ZED SDK Depth Modes — Reference for Edge Detection GUI

**Source**: [Stereolabs Docs – Depth Modes](https://www.stereolabs.com/docs/depth-sensing/depth-modes)  
**SDK version referenced**: ZED SDK v5.x (current as of 2026)

---

## Overview

Starting with ZED SDK v5, Stereolabs replaced all traditional (non-AI) depth modes
(`PERFORMANCE`, `QUALITY`, `ULTRA`) with exclusively **AI-powered neural network modes**.
The old modes are legacy and should not be used in new code.

The three current modes all use the same core AI disparity estimation pipeline, but are
tuned differently for speed vs. accuracy:

| Mode | Ideal Depth Range | Speed | Accuracy | Recommended Use |
|------|------------------|-------|----------|-----------------|
| `NEURAL_LIGHT` | 0.3 – 5 m | ⚡ Fastest | ★★☆ | Multi-camera setups, obstacle avoidance, FPS-critical pipelines |
| `NEURAL` | 0.3 – 9 m | ⚡⚡ Balanced | ★★★ | General-purpose single-camera use (**default**) |
| `NEURAL_PLUS` | 0.3 – 12 m | ⚡⚡⚡ Slow | ★★★★ | High-precision inspection, 3D reconstruction, challenging lighting |
| `NONE` | — | — | — | When depth is disabled (image-only capture) |

---

## Mode Details

### NEURAL_LIGHT
- Fastest AI depth mode
- Optimised for **multi-camera** setups (lowest per-camera compute cost)
- Slightly smaller ideal range and less object detail than `NEURAL`
- Slightly less robust to environmental light changes than `NEURAL` / `NEURAL_PLUS`
- Good for real-time drone obstacle avoidance at < 5 m

**ZED X accuracy:**

| Depth Range | Error | Noise |
|-------------|-------|-------|
| 0.3 – 3 m | < 1 % | Low |
| 3 – 5 m | < 3 % | Medium |
| 5 – 12 m | < 8 % | High |

---

### NEURAL *(default in this project)*
- Balanced depth accuracy and processing speed
- Better object detail than `NEURAL_LIGHT`
- Suitable for most single-camera drone applications
- Same robustness to environmental changes as `NEURAL_PLUS`
- Covers 0.3 – 9 m reliably — fits ZED 2i + drone at 2–8 m range

**ZED X accuracy:**

| Depth Range | Error | Noise |
|-------------|-------|-------|
| 0.3 – 4 m | < 1 % | Low |
| 4 – 6 m | < 2.5 % | Low |
| 6 – 9 m | < 4 % | Medium |
| 10 – 12 m | < 6 % | High |

---

### NEURAL_PLUS
- Highest depth accuracy of all modes
- Best for detecting near, far, and small objects simultaneously
- Most robust to rain, sunlight, and light reflections
- Not recommended for multi-camera setups (too compute-heavy)
- Use when maximum 3D precision is required (e.g. final pose-optimizer runs)

**ZED X accuracy:**

| Depth Range | Error | Noise |
|-------------|-------|-------|
| 0.3 – 9 m | < 1 % | Low |
| 9 – 12 m | < 2 % | Medium |

---

## Legacy Modes (Removed in SDK v5)

The following modes existed in SDK v3/v4 and are **no longer available** in the current SDK.
Do not use them in new code:

- `PERFORMANCE` — traditional SGM-based, fastest legacy
- `QUALITY` — traditional SGM-based, balanced legacy
- `ULTRA` — traditional SGM-based, best legacy quality

---

## Usage in `edge_detection_gui.py`

The SVO2 mode exposes a combo box with the four current valid options:

```python
["NEURAL_PLUS", "NEURAL", "NEURAL_LIGHT", "NONE"]
```

The `depth_mode_map` inside `svo2_open_camera()` maps these to SDK enums:

```python
depth_mode_map = {
    "NEURAL_PLUS":  sl.DEPTH_MODE.NEURAL_PLUS,
    "NEURAL":       sl.DEPTH_MODE.NEURAL,
    "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
    "NONE":         sl.DEPTH_MODE.NONE,
}
```

**Recommended default**: `NEURAL` — covers the full drone operational distance range
(2–8 m) with < 4 % error and balanced frame rate on Jetson Orin NX.

---

## Jetson Orin NX 16 GB Performance Reference (ZED SDK v5.0.1 RC)

| Mode | 1 Camera FPS | 2 Cameras FPS | 4 Cameras FPS |
|------|-------------|--------------|--------------|
| `NEURAL_LIGHT` | 30 | 30 | 30 |
| `NEURAL` | 30 | 30 | 30 |
| `NEURAL_PLUS` | 29 | 17 | 8 |

*(Tests with ZED X + MAXN without Super mode)*

---

## Notes for This Project

- **Depth map alignment**: ZED depth map is aligned to the **left camera** coordinate frame.
  When displayed as an overlay on the right camera image (side-by-side view), there is a
  small sub-pixel offset due to stereo baseline parallax. At drone operational distances
  (> 2 m) and with 120 mm baseline, this offset is typically < 10 px and visually acceptable.
- **cuDNN workaround**: On Jetson with CUDA/cuDNN version mismatches, `torch.backends.cudnn.enabled = False`
  is set before loading YOLO models. This does not affect ZED SDK depth computation.
- Depth values are in **metres** (float32, `coordinate_units = sl.UNIT.METER`).
  Zero and NaN/Inf values indicate invalid / out-of-range measurements and are masked out
  before colormap rendering.
