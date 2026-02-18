#!/usr/bin/env python3
"""
Optimized silhouette point extraction with correct coordinate system.

Key improvements:
1. Fixed azimuth mapping for aircraft tracking (az=90° = rear view)
2. Numba JIT optimization for <17ms target
3. Proper outer boundary extraction (no internal points)
"""

import argparse
import time
from pathlib import Path
import numpy as np
import cv2
from numba import jit, prange


def load_cache(path: Path):
    """Load precomputed mesh data from NPZ cache."""
    data = np.load(path)
    return (
        data['vertices'].astype(np.float32),
        data['faces'],
        data['edges'],
        data['edges_faces']
    )


def compute_camera_transform(cam_pos: np.ndarray, target: np.ndarray):
    """Compute camera rotation matrix from position and target.

    STL axes: X=fuselage (nose=+X), Y=wingspan, Z=vertical (+Z=down, -Z=up).

    The image right-vector (r) is kept horizontal (in the X-Y plane) so that
    elevation changes never roll the aircraft in the image.
    """
    f = target - cam_pos
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    f = f / f_norm

    # Derive r from the HORIZONTAL component of f only.
    # This guarantees r stays in the X-Y plane and never rotates when elevation changes.
    f_horiz = np.array([f[0], f[1], 0.0], dtype=np.float32)
    f_horiz_norm = np.linalg.norm(f_horiz)

    if f_horiz_norm > 0.01:   # normal case: camera not directly above/below
        f_horiz = f_horiz / f_horiz_norm
        # Right-handed NED body frame: X=nose, Y=right(starboard), Z=down
        # world_up points in -Z (physically upward).
        # r = cross(f_horiz, world_up) → +Y (starboard) on RIGHT in image
        # u = cross(f, r)              → +Z (down) maps to image-down, so
        #                                 winglets (-Z = world-up) appear at TOP
        world_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        r = np.cross(f_horiz, world_up)   # cross([1,0,0],[0,0,-1]) = [0,+1,0]
    else:
        # Camera nearly straight above/below – fall back to fixed r
        r = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(f, r)        # cross(f,r) keeps u pointing +Z (image-down) → right-side-up
    u = u / (np.linalg.norm(u) + 1e-12)
    return np.vstack([r, u, f])


@jit(nopython=True, parallel=True, cache=True)
def rasterize_depth_and_mask(V_cam, F, width, height, fx, fy):
    """Rasterise front-facing triangles into a depth buffer and a filled mask.

    Both outputs are produced in a single scanline pass over all faces.

    Args:
        V_cam         : (N, 3) float32 — vertices in camera space
        F             : (M, 3) int32   — triangle face vertex indices
        width, height : image size in pixels
        fx, fy        : focal lengths in pixels

    Returns:
        mask : (height, width) uint8   — 255 where object visible, 0 background
        zbuf : (height, width) float32 — closest surface depth per pixel (inf=empty)
    """
    half_w = width  * 0.5
    half_h = height * 0.5
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)

    for fi in prange(F.shape[0]):
        v0 = V_cam[F[fi, 0]]
        v1 = V_cam[F[fi, 1]]
        v2 = V_cam[F[fi, 2]]

        # Skip triangles with any vertex behind or at the camera plane
        if v0[2] <= 0.0 or v1[2] <= 0.0 or v2[2] <= 0.0:
            continue

        # Face normal via cross product of two edges
        e1x = v1[0]-v0[0];  e1y = v1[1]-v0[1];  e1z = v1[2]-v0[2]
        e2x = v2[0]-v0[0];  e2y = v2[1]-v0[1];  e2z = v2[2]-v0[2]
        nx = e1y*e2z - e1z*e2y
        ny = e1z*e2x - e1x*e2z
        nz = e1x*e2y - e1y*e2x

        # Perspective front-face test: view dir = from centroid toward camera origin
        fcx = -(v0[0]+v1[0]+v2[0]) / 3.0
        fcy = -(v0[1]+v1[1]+v2[1]) / 3.0
        fcz = -(v0[2]+v1[2]+v2[2]) / 3.0
        if nx*fcx + ny*fcy + nz*fcz <= 0.0:
            continue  # back-facing — skip

        # Project vertices to pixel coordinates
        x0 = fx*v0[0]/v0[2] + half_w;  y0 = fy*v0[1]/v0[2] + half_h
        x1 = fx*v1[0]/v1[2] + half_w;  y1 = fy*v1[1]/v1[2] + half_h
        x2 = fx*v2[0]/v2[2] + half_w;  y2 = fy*v2[1]/v2[2] + half_h

        # Bounding box clamped to image
        xmin = max(0,        int(min(x0, x1, x2)))
        xmax = min(width-1,  int(max(x0, x1, x2)) + 1)
        ymin = max(0,        int(min(y0, y1, y2)))
        ymax = min(height-1, int(max(y0, y1, y2)) + 1)

        # Barycentric denominator
        denom = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if abs(denom) < 1e-8:
            continue

        for py in range(ymin, ymax+1):
            for px in range(xmin, xmax+1):
                w0 = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) / denom
                w1 = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                depth = w0*v0[2] + w1*v1[2] + w2*v2[2]
                if depth < zbuf[py, px]:
                    zbuf[py, px] = depth
                    mask[py, px] = 255

    return mask, zbuf


def extract_silhouette_contour(V_world, F, cam_pos, target,
                               img_width=1280, img_height=720,
                               fx=800.0, fy=800.0,
                               contour_stride=1,
                               supersample=1):
    """Extract the outer visible silhouette as 3-D points (Approach C).

    Pipeline:
      1. Transform mesh vertices to camera space.
      2. Rasterise all front-facing triangles into a filled uint8 mask and
         depth buffer at (supersample * img_width) x (supersample * img_height)
         resolution.  The focal lengths are scaled by the same factor so that
         the projection is identical — only the pixel grid is finer.
      3. cv2.findContours on the mask -> ordered pixel chain of the outer boundary.
         This is ALWAYS the outer contour; interior geometry is physically impossible
         in the output of a filled-mask contour trace.
      4. Back-project each contour pixel (u, v) + zbuf depth -> 3-D camera-space
         point using the scaled focal lengths.  The 3-D coordinates are identical
         to the non-supersampled case; the difference is purely that the contour
         is traced at finer pixel steps, removing the staircase artefact.

    Supersampling effect at 5 m (f=800)::

        supersample=1 (default) -> 6.2 mm / contour step  (real-time)
        supersample=2            -> 3.1 mm / contour step
        supersample=4            -> 1.6 mm / contour step  (precompute, smooth)
        supersample=8            -> 0.8 mm / contour step  (very fine)

    Args:
        V_world          : (N, 3) float32 — mesh vertices in world coordinates
        F                : (M, 3) int32   — triangle face indices
        cam_pos          : (3,) camera position in world coordinates
        target           : (3,) look-at point in world coordinates
        img_width/height : base rasterisation resolution (pixels)
        fx, fy           : base focal lengths (pixels)
        contour_stride   : take every Nth contour pixel after supersampling
                           (use supersample itself to normalise point density)
        supersample      : integer supersampling factor (1 = no supersampling).
                           Raster size and focal lengths are both scaled by this
                           value; computation cost scales as supersample².

    Returns:
        points_3d : (K, 3) float32 — contour points in camera space
    """
    S = max(1, int(supersample))

    # Scale raster and focal lengths together so projection is unchanged
    ss_width  = img_width  * S
    ss_height = img_height * S
    ss_fx     = fx * S
    ss_fy     = fy * S

    # 1. Camera transform
    R_cam = compute_camera_transform(cam_pos, target)
    V_cam = (R_cam @ (V_world - cam_pos).T).T.astype(np.float32)

    # 2. Rasterise -> mask + depth buffer at supersampled resolution
    mask, zbuf = rasterize_depth_and_mask(V_cam, F.astype(np.int32),
                                          ss_width, ss_height, ss_fx, ss_fy)

    # 3. Outer contour at full supersampled resolution (every pixel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 3), dtype=np.float32)

    half_w = ss_width  * 0.5
    half_h = ss_height * 0.5
    pts_out = []

    for contour in contours:
        if len(contour) < 3:
            continue
        # Shape: (N, 1, 2) -> stride -> (K, 2); columns: [x=col, y=row]
        pixels = contour[::contour_stride, 0, :]
        for k in range(pixels.shape[0]):
            px = int(pixels[k, 0])
            py = int(pixels[k, 1])
            depth = float(zbuf[py, px])

            # Contour pixels are foreground so zbuf should always be valid.
            # 3x3 fallback handles the rare sub-pixel rounding edge case.
            if depth == np.inf:
                best = np.inf
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny2 = py + dy; nx2 = px + dx
                        if 0 <= ny2 < ss_height and 0 <= nx2 < ss_width:
                            d = float(zbuf[ny2, nx2])
                            if d < best:
                                best = d
                depth = best

            if depth == np.inf or depth <= 0.0:
                continue

            # 4. Pinhole back-projection using scaled focal lengths
            # (ss_fx = fx*S cancels the ss_width*S scaling -> same 3D coords)
            X = (px - half_w) / ss_fx * depth
            Y = (py - half_h) / ss_fy * depth
            pts_out.append((X, Y, depth))

    if not pts_out:
        return np.zeros((0, 3), dtype=np.float32)

    return np.array(pts_out, dtype=np.float32)


def spherical_to_cartesian_aircraft(azimuth_deg, elevation_deg, distance):
    """Convert spherical coordinates to Cartesian for aircraft tracking.

    STL coordinate system (confirmed):
    - X-axis: fuselage  (+X = nose,  -X = tail)
    - Y-axis: wingspan  (+Y = right wingtip, -Y = left wingtip)
    - Z-axis: vertical  (+Z = down toward ground, -Z = up)

    Azimuth convention (horizontal camera placement around the aircraft):
    - 0°:   Rear  — camera offset in -X direction (behind tail)  ← PRIMARY
    - 90°:  Right — camera offset in +Y direction (starboard)
    - 180°: Front — camera offset in +X direction (in front of nose)
    - 270°: Left  — camera offset in -Y direction (port)

    Elevation:
    - 0°:   Level  (camera at same height as target)
    - +90°: Above  (camera directly above, Z decreases = world-up)
    - -90°: Below

    Args:
        azimuth_deg: Horizontal angle (0-360°)
        elevation_deg: Vertical angle (-90 to +90°)
        distance: Distance from target (metres)

    Returns:
        Camera position offset [x, y, z] to add to target centroid
    """
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)

    horiz = distance * np.cos(el_rad)

    # az=0   → -X (rear)   az=90  → +Y (right)
    # az=180 → +X (front)  az=270 → -Y (left)
    x = -horiz * np.cos(az_rad)   # -X at az=0 (rear), +X at az=180 (front)
    y =  horiz * np.sin(az_rad)   # +Y at az=90 (right), -Y at az=270 (left)

    # +Z is down, so "above" means negative Z offset
    z = -distance * np.sin(el_rad)

    return np.array([x, y, z], dtype=np.float32)


def save_silhouette_points(points_3d, output_path, metadata=None):
    """Save silhouette points to NPZ file."""
    save_dict = {'points_3d': points_3d}
    
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                save_dict[key] = np.array(value)
            else:
                save_dict[key] = value
    
    np.savez_compressed(output_path, **save_dict)


def main():
    parser = argparse.ArgumentParser(description='Extract silhouette points (optimized)')
    parser.add_argument('--cache', type=str, required=True,
                        help='Path to mesh cache NPZ')
    parser.add_argument('--output', type=str, default=None,
                        help='Output NPZ file')
    parser.add_argument('--azimuth', type=float, default=0.0,
                        help='Azimuth angle (0=rear, 90=right, 180=front, 270=left)')
    parser.add_argument('--elevation', type=float, default=0.0,
                        help='Elevation angle (-90=below, 0=level, 90=above)')
    parser.add_argument('--distance', type=float, default=5.0,
                        help='Camera distance from target (meters)')
    parser.add_argument('--sample-fraction', type=float, default=0.5,
                        help='Contour stride denominator (0-1, ignored in precompute mode)')
    parser.add_argument('--supersample', type=int, default=1,
                        help='Supersampling factor for smoother contours (1=realtime, 4=precompute). '
                             'Raster and focal lengths are both scaled; 3-D coords are unchanged. '
                             'Cost scales as supersample^2.')
    parser.add_argument('--benchmark', type=int, default=10,
                        help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("OPTIMIZED SILHOUETTE EXTRACTION")
    print("="*70)
    
    # Load mesh
    print("\n=== Loading Mesh ===")
    V, F, E, Efaces = load_cache(Path(args.cache))
    print(f"Vertices: {len(V):,}")
    print(f"Faces: {len(F):,}")
    print(f"Edges: {len(E):,}")
    
    # Auto-scale
    extents = V.max(axis=0) - V.min(axis=0)
    if np.max(extents) > 100:
        V = V * 0.001
        print("Auto-scaled from mm to meters")
    
    # Setup camera
    centroid = V.mean(axis=0)
    target = centroid.copy()
    cam_pos = centroid + spherical_to_cartesian_aircraft(
        args.azimuth, args.elevation, args.distance
    )
    
    print(f"\n=== Camera Setup ===")
    print(f"Azimuth: {args.azimuth}° (0=rear, 90=right, 180=front, 270=left)")
    print(f"Elevation: {args.elevation}° (0=level, ±90=above/below)")
    print(f"Distance: {args.distance}m")
    print(f"Target: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
    print(f"Camera: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")
    print(f"Supersample: {args.supersample}x  "
          f"(raster {1280*args.supersample}x{720*args.supersample}, "
          f"step ~{6.25/args.supersample:.1f} mm @ 5m)")

    # Warmup JIT (always at supersample=1 so it's fast)
    print("\n⏱️  Warming up Numba JIT...")
    _ = extract_silhouette_contour(V, F, cam_pos, target, supersample=1)
    print("✓ JIT warmup complete")

    # Benchmark
    print(f"\n=== Benchmarking ({args.benchmark} iterations) ===")
    times = []
    n_points = 0

    for i in range(args.benchmark):
        t_start = time.perf_counter()
        points_3d = extract_silhouette_contour(
            V, F, cam_pos, target,
            supersample=args.supersample,
            contour_stride=max(1, int(1.0 / args.sample_fraction))
        )
        t_elapsed = (time.perf_counter() - t_start) * 1000
        times.append(t_elapsed)
        n_points = len(points_3d)

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\n=== Results ===")
    print(f"Contour points: {n_points}")
    print(f"\nTiming:")
    print(f"  Mean:   {avg_time:6.2f} ms")
    print(f"  Std:    {std_time:6.2f} ms")
    print(f"  Min:    {min_time:6.2f} ms")
    print(f"  Max:    {max_time:6.2f} ms")
    print(f"\nThroughput: {1000/avg_time:.1f} extractions/second")

    # Performance targets
    target_30fps = 33.3
    target_60fps = 16.7
    print(f"\nPerformance targets:")
    print(f"  Real-time (30 FPS, <33ms):  {'✅ YES' if avg_time < target_30fps else '❌ NO'}")
    print(f"  High-speed (60 FPS, <17ms): {'✅ YES' if avg_time < target_60fps else '❌ NO'}")

    # Save output
    if args.output:
        metadata = {
            'cam_pos': cam_pos,
            'target': target,
            'azimuth': args.azimuth,
            'elevation': args.elevation,
            'distance': args.distance,
            'supersample': args.supersample,
            'avg_time_ms': avg_time,
            'n_points': n_points
        }

        save_silhouette_points(points_3d, args.output, metadata)

        file_size = Path(args.output).stat().st_size
        print(f"\n✓ Saved: {args.output}")
        print(f"  File size: {file_size/1024:.1f} KB")
        if n_points > 0:
            print(f"  Bytes/point: {file_size/n_points:.1f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
