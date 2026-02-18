#!/usr/bin/env python3
"""
Debug visualization helper for silhouette extraction.

Outputs both:
  • SVG  — vector scatter plot, every point individually visible, no overlap hiding
  • PNG  — quick raster preview (same layout)

Coordinate convention (right-handed NED):
  +X = nose, +Y = right/starboard, +Z = down
  Azimuth: 0=rear, 90=right, 180=front, 270=left
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def project_to_2d(points_3d, fx=800.0, fy=800.0, cx=640.0, cy=360.0):
    """Perspective-project camera-space points to 2D pixel coordinates."""
    depth = points_3d[:, 2]
    valid = depth > 0
    pts = np.full((len(points_3d), 2), np.nan)
    pts[valid, 0] = fx * points_3d[valid, 0] / depth[valid] + cx
    pts[valid, 1] = fy * points_3d[valid, 1] / depth[valid] + cy
    return pts, valid


def draw_axes_mpl(ax, points_3d_cam, R_cam,
                  fx=800.0, fy=800.0, cx=640.0, cy=360.0, scale=0.3):
    """Draw +X / +Y / -Z world-axis arrows into the matplotlib axes.
      Red   → +X  (nose)
      Green → +Y  (right wing / starboard)
      Cyan  → -Z  (world up)
    """
    if len(points_3d_cam) == 0:
        return
    c_cam = points_3d_cam.mean(axis=0)
    if c_cam[2] <= 0:
        return

    def proj(pt):
        if pt[2] <= 0:
            return None
        return (fx * pt[0] / pt[2] + cx, fy * pt[1] / pt[2] + cy)

    c_px = proj(c_cam)
    if c_px is None:
        return

    for world_offset, color, label in [
        (np.array([scale, 0,      0    ], np.float32), 'red',   '+X NOSE'),
        (np.array([0,     scale,  0    ], np.float32), 'lime',  '+Y WING-R'),
        (np.array([0,     0,     -scale], np.float32), 'cyan',  '-Z UP'),
    ]:
        tip_cam = c_cam + (R_cam @ world_offset)
        tip_px  = proj(tip_cam)
        if tip_px is None:
            continue
        ax.annotate('', xy=tip_px, xytext=c_px,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
                    zorder=5)
        ax.text(tip_px[0] + 5, tip_px[1], label,
                color=color, fontsize=7, va='center', zorder=5)


def visualize_npz(npz_path, output_dir, width=1280, height=720,
                  fx=800.0, fy=800.0):
    """Load NPZ and save SVG + PNG scatter plots."""
    npz_path   = Path(npz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    if 'points_3d' in data:
        points_3d = data['points_3d']
    elif 'arr_0' in data:
        points_3d = data['arr_0']
    else:
        print(f"❌ No points_3d in {npz_path}")
        return

    cx, cy = width / 2.0, height / 2.0
    pts2d, valid = project_to_2d(points_3d, fx, fy, cx, cy)
    pts2d_valid  = pts2d[valid]

    # metadata
    azimuth   = float(data['azimuth'])     if 'azimuth'     in data else None
    elevation = float(data['elevation'])   if 'elevation'   in data else None
    avg_ms    = float(data['avg_time_ms']) if 'avg_time_ms' in data else None

    if azimuth is not None:
        a = azimuth % 360
        if   a <= 15 or a >= 345: vl = "REAR (tail)"
        elif 85  <= a <= 95:      vl = "RIGHT (starboard)"
        elif 175 <= a <= 185:     vl = "FRONT (nose)"
        elif 265 <= a <= 275:     vl = "LEFT (port)"
        else:                     vl = f"Az {a:.0f}°"
        if elevation is not None:
            vl += f"  El {elevation:+.0f}°"
    else:
        vl = npz_path.stem

    n_vis   = int(valid.sum())
    n_total = len(points_3d)
    title   = f"{vl}    {n_vis}/{n_total} pts"
    if avg_ms is not None:
        title += f"    {avg_ms:.1f} ms"

    dpi   = 120
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)    # y-axis flipped: top of image = top of plot
    ax.set_aspect('equal')
    ax.axis('off')

    if len(pts2d_valid) > 0:
        in_frame = (
            (pts2d_valid[:, 0] >= 0) & (pts2d_valid[:, 0] < width) &
            (pts2d_valid[:, 1] >= 0) & (pts2d_valid[:, 1] < height)
        )
        pts_plot   = pts2d_valid[in_frame]
        depths_plot = points_3d[valid][in_frame, 2]
        sc = ax.scatter(
            pts_plot[:, 0], pts_plot[:, 1],
            c=depths_plot, cmap='plasma',
            s=8, linewidths=0, zorder=3
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.018, pad=0.01)
        cbar.set_label('depth (m)', color='white', fontsize=7)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor('white')

    if 'cam_pos' in data and 'target' in data:
        from extract_silhouette import compute_camera_transform
        R_cam = compute_camera_transform(
            data['cam_pos'].astype(np.float32),
            data['target'].astype(np.float32)
        )
        draw_axes_mpl(ax, points_3d, R_cam, fx, fy, cx, cy)

    ax.set_title(title, color='white', fontsize=8, pad=3)
    plt.tight_layout(pad=0.2)

    stem  = npz_path.stem
    svg_p = output_dir / f"{stem}_render.svg"
    png_p = output_dir / f"{stem}_render.png"
    fig.savefig(str(svg_p), format='svg', bbox_inches='tight')
    fig.savefig(str(png_p), format='png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"✓ {stem}: {n_vis} pts  →  SVG + PNG")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize silhouette NPZ (SVG + PNG scatter)')
    parser.add_argument('input', help='NPZ file or directory of NPZ files')
    parser.add_argument('--output-dir', default='../output/renders')
    parser.add_argument('--width',  type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_file() and inp.suffix == '.npz':
        visualize_npz(inp, args.output_dir, args.width, args.height)
    elif inp.is_dir():
        files = sorted(inp.glob('*.npz'))
        if not files:
            print(f"❌ No .npz files in {inp}")
            return
        for f in files:
            visualize_npz(f, args.output_dir, args.width, args.height)
        print(f"\n✅ {len(files)} files → {args.output_dir}")
    else:
        print(f"❌ Invalid input: {inp}")


if __name__ == '__main__':
    main()
