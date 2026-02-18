# CAD Edge Feature Extraction for 6D Pose Estimation

Optimized silhouette extraction pipeline for RANSAC-based aircraft pose estimation on Jetson Orin Nano Super.

---

## Overview

This pipeline extracts outer boundary silhouette points from CAD models for 6D pose estimation. It uses adaptive hemisphere-aware precomputation to optimize for aircraft tracking scenarios where the camera is primarily positioned behind the target.

**Key Features:**
- **Real-time performance:** 29ms extraction time (30 FPS)
- **Numba JIT optimization:** ~50% faster than Python baseline
- **Adaptive sampling:** Fine resolution (2°) for rear views, coarse (10°) for front views
- **Memory efficient:** 76MB for 5,785 precomputed poses
- **Correct coordinate system:** Azimuth 90° = rear view (primary tracking angle)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  1. CAD Model (.stl)                                        │
│     ↓                                                        │
│  2. Preprocess: Extract edges & simplify mesh               │
│     ↓                                                        │
│  3. Precompute visibility map for pose grid                 │
│     ↓                                                        │
│  4. Runtime: Query nearest pose → Extract silhouette        │
│     ↓                                                        │
│  5. RANSAC: Match silhouette points to detected edges       │
└─────────────────────────────────────────────────────────────┘
```

---

## Coordinate System

**Aircraft Model:**
- **X-axis:** Forward (nose direction)
- **Y-axis:** Left-to-right (wingspan)
- **Z-axis:** Down (toward ground, positive = down)

**Camera Azimuth Convention:**
- **0°:** Right side view (camera at +Y)
- **90°:** Rear view (camera at -X, looking at tail) ← **PRIMARY TRACKING**
- **180°:** Left side view (camera at -Y)
- **270°:** Front view (camera at +X, looking at nose)

**Camera Elevation:**
- **0°:** Level (same height as target)
- **+90°:** Above (looking down)
- **-90°:** Below (looking up)

---

## Directory Structure

```
CAD-Edge-Generation/
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── data/                        # Model data and precomputed maps
│   ├── CAD_Ranger_simplified.stl              # Simplified mesh (44k faces)
│   ├── CAD_Ranger_simplified.edges.npz        # Preprocessed edges
│   ├── visibility_adaptive_5m_FIXED.npz       # Precomputed visibility map
│   └── visibility_adaptive_5m_FIXED.lookup.json  # Pose metadata
│
├── src/                         # Core pipeline source code
│   ├── extract_silhouette.py   # Optimized silhouette extraction
│   └── precompute_visibility.py  # Adaptive precomputation
│
└── output/                      # Generated outputs (gitignored)
    ├── debug/                   # Debug visualizations
    └── renders/                 # Rendered silhouettes
```

---

## Installation

### Requirements
- Python 3.10+
- NumPy 1.26+
- Numba 0.63+
- OpenCV (optional, for visualization only)

### Setup
```bash
# Install dependencies
pip install numpy numba opencv-python

# Verify setup
cd src
python extract_silhouette.py --help
```

---

## Usage

### 1. Extract Silhouette for Single Pose

```bash
python src/extract_silhouette.py \
  --cache data/CAD_Ranger_simplified.edges.npz \
  --azimuth 90 \           # Rear view (primary tracking)
  --elevation 0 \          # Level
  --distance 5.0 \         # 5 meters from target
  --sample-fraction 0.5 \  # Adaptive sampling density
  --benchmark 10 \         # Run 10 iterations for timing
  --output output/silhouette.npz
```

**Output:**
- Extraction time: ~29ms
- Points: ~2,300 3D points in camera space
- File size: ~15 KB (compressed NPZ)

### 2. Precompute Full Visibility Map

```bash
python src/precompute_visibility.py \
  --cache data/CAD_Ranger_simplified.edges.npz \
  --output data/visibility_map.npz \
  --distance 5.0 \
  --az-rear 2.0 --el-rear 2.0 \      # Fine: rear hemisphere (90°-270°)
  --az-front 10.0 --el-front 10.0 \  # Coarse: front hemisphere (270°-90°)
  --sample-fraction 0.5
```

**Output:**
- Total poses: 5,785 (5,551 rear + 234 front)
- Processing time: ~3 minutes
- File size: 76 MB (compressed)
- Lookup table: JSON with pose metadata

### 3. Query Precomputed Visibility Map (Integration Example)

```python
import numpy as np
import json

# Load visibility map
visibility_data = np.load('data/visibility_adaptive_5m_FIXED.npz')
with open('data/visibility_adaptive_5m_FIXED.lookup.json', 'r') as f:
    lookup = json.load(f)

# Find nearest pose
def query_nearest_pose(target_az, target_el, target_dist, lookup):
    """Find nearest precomputed pose using simple distance metric."""
    best_key = None
    best_dist = float('inf')
    
    for key, pose in lookup.items():
        az_diff = min(abs(pose['azimuth'] - target_az), 
                      360 - abs(pose['azimuth'] - target_az))
        el_diff = abs(pose['elevation'] - target_el)
        dist_diff = abs(pose['distance'] - target_dist)
        
        metric = az_diff**2 + el_diff**2 + (dist_diff * 10)**2
        
        if metric < best_dist:
            best_dist = metric
            best_key = key
    
    return best_key, lookup[best_key]

# Query example
pose_key, pose_info = query_nearest_pose(azimuth=92, elevation=5, distance=5.0, lookup=lookup)
points_3d = visibility_data[pose_key]

print(f"Nearest pose: {pose_info['azimuth']:.1f}° az, {pose_info['elevation']:.1f}° el")
print(f"Points available: {len(points_3d):,}")
```

---

## Performance Benchmarks

### Extraction Performance (Jetson Orin Nano Super)

| Metric | Value |
|--------|-------|
| Average extraction time | **29.6 ms** |
| Throughput | **33.8 poses/second** |
| Silhouette edges | 992-1,251 edges |
| Sampled points | 2,068-2,637 points |
| Memory footprint | ~15 KB/pose (compressed) |

**Performance Targets:**
- ✅ **30 FPS** (< 33ms) - **ACHIEVED**
- ❌ 60 FPS (< 17ms) - Possible with further optimization

### Precomputation Performance

| Configuration | Rear Hemisphere | Front Hemisphere | Total Poses | Time |
|--------------|-----------------|------------------|-------------|------|
| Adaptive (2° / 10°) | 5,551 poses | 234 poses | 5,785 | **2.9 min** |
| Uniform (2°) | 13,140 poses | 13,140 poses | 26,280 | ~11 min |

**Storage:**
- Adaptive map: 76 MB
- Uniform map: ~327 MB (estimated)
- **Space savings: 77%**

---

## Optimization Details

### Numba JIT Compilation
Two critical functions are JIT-compiled for maximum performance:

1. **`find_silhouette_edges_numba()`**
   - Determines front-facing triangles via cross product
   - Identifies boundary edges between front/back faces
   - ~60% of total execution time

2. **`sample_edge_adaptive_numba()`**
   - Samples points along edges based on length
   - Adaptive density: 5mm-50mm step size
   - Prevents over/under-sampling

### Adaptive Hemisphere Sampling
Aircraft tracking scenarios typically view the target from behind:
- **Rear hemisphere (90°-270° azimuth):** 2° × 2° resolution → 5,551 poses
- **Front hemisphere (270°-90° azimuth):** 10° × 10° resolution → 234 poses

This reduces precomputation time by **~75%** while maintaining accuracy where needed.

---

## Algorithm Details

### Silhouette Edge Detection

A silhouette edge satisfies one of two conditions:
1. **Boundary edge:** Only one adjacent face exists
2. **Visibility edge:** Adjacent faces have opposite visibility (one front-facing, one back-facing)

**Front-facing test:**
```python
normal = cross(v1 - v0, v2 - v0)
view_dir = -centroid  # Looking toward camera origin
is_front_facing = dot(normal, view_dir) > 0
```

### Adaptive Point Sampling

Each silhouette edge is sampled with adaptive density:
```python
edge_length = ||p2 - p1||
step_size = clamp(edge_length * sample_fraction, 5mm, 50mm)
n_samples = ceil(edge_length / step_size)
```

This ensures:
- Long edges get more samples (prevent gaps)
- Short edges don't get over-sampled (reduce redundancy)
- Consistent ~5mm point spacing in 3D

---

## Data Format

### Extracted Silhouette NPZ
```python
{
    'points_3d': np.ndarray,        # (N, 3) float32, camera space coordinates
    'edge_indices': np.ndarray,     # (M,) int32, indices of silhouette edges
    'cam_pos': np.ndarray,          # (3,) float32, camera position (world)
    'target': np.ndarray,           # (3,) float32, target position (world)
    'azimuth': float,               # Camera azimuth angle
    'elevation': float,             # Camera elevation angle
    'distance': float,              # Camera distance from target
    'sample_fraction': float,       # Sampling density used
    'avg_time_ms': float,           # Extraction time
    'n_edges': int                  # Number of silhouette edges
}
```

### Visibility Map NPZ
```python
{
    'pose_00000': np.ndarray,       # (N, 3) float32, points for pose 0
    'pose_00001': np.ndarray,       # (N, 3) float32, points for pose 1
    ...
    'pose_05784': np.ndarray        # (N, 3) float32, points for pose 5784
}
```

### Lookup Table JSON
```json
{
  "pose_00000": {
    "azimuth": 0.0,
    "elevation": -30.0,
    "distance": 5.0,
    "cam_pos": [-0.155, 5.001, -2.5],
    "n_points": 2637,
    "n_edges": 1251,
    "time_ms": 30.52
  },
  ...
}
```

---

## Troubleshooting

### Extraction is slow (>50ms)
- **Check:** JIT compilation warmup completed?
- **Solution:** First call takes longer (~200ms), subsequent calls are fast
- **Verify:** Run with `--benchmark 10` to see average after warmup

### Precomputation taking too long
- **Check:** Are you using adaptive sampling?
- **Solution:** Use `--az-rear 2.0` and `--az-front 10.0` for 75% speedup
- **Alternative:** Increase front hemisphere steps to 15° or 20°

### Wrong orientation in renders
- **Check:** Is azimuth 90° showing the rear view?
- **Solution:** Use `spherical_to_cartesian_aircraft()` function (already implemented)
- **Verify:** Test with `--azimuth 90` should show tail, `--azimuth 270` should show nose

### Points appear inside silhouette
- **Check:** Are you using `extract_silhouette.py` (optimized version)?
- **Solution:** Ensure `find_silhouette_edges_numba()` is being called
- **Cause:** Old extraction code didn't properly filter internal edges

---

## Future Enhancements

### Short-term
1. **Query optimization:** Implement KD-tree for faster nearest-pose lookup
2. **Interpolation:** Blend between nearby poses for smoother transitions
3. **Distance variants:** Precompute at 3m, 5m, 10m distances

### Long-term
1. **GPU acceleration:** CUDA implementation for <5ms extraction
2. **Streaming:** Load visibility map chunks on-demand to reduce memory
3. **Multi-model support:** Extend to different aircraft types

---

## References

### Key Optimizations Applied
1. **Numba JIT compilation** - 50% speedup on hot paths
2. **Adaptive sampling** - 75% reduction in precomputation time
3. **Mesh simplification** - 82% face reduction (246k → 44k faces)
4. **Compressed storage** - NPZ format with ~7 bytes/point

### Performance Evolution
| Version | Extraction Time | Speedup |
|---------|----------------|---------|
| Python baseline | 57.5 ms | 1.0× |
| NumPy vectorized | 107 ms | 0.5× |
| **Numba optimized** | **29.6 ms** | **1.9×** |

---

## License

This pipeline is part of the 6D Pose Estimation project for aircraft tracking on embedded systems.

**Author:** Angelo  
**Date:** February 2026  
**Platform:** Jetson Orin Nano Super (6 cores, ARM64)

---

## Quick Reference

### Essential Commands
```bash
# Extract single pose (rear view)
python src/extract_silhouette.py --cache data/CAD_Ranger_simplified.edges.npz --azimuth 90 --elevation 0 --distance 5.0 --output output/test.npz

# Precompute visibility map (3 minutes)
python src/precompute_visibility.py --cache data/CAD_Ranger_simplified.edges.npz --output data/visibility_new.npz --distance 5.0 --az-rear 2.0 --el-rear 2.0 --az-front 10.0 --el-front 10.0

# List all precomputed poses
python -c "import json; print(len(json.load(open('data/visibility_adaptive_5m_FIXED.lookup.json'))))"
```

### Directory Shortcuts
```bash
DATA="data/"
SRC="src/"
OUT="output/"
```

---

**Status:** ✅ Production Ready | 🚀 Real-time Performance | 📊 Fully Optimized
