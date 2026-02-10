#!/usr/bin/env python3
"""
Helper script to extract debug images from a YOLO training folder.

Behavior:
- Scan a source directory for image files whose filename contains a depth tag like `depth-11.73m`.
- Only select images with depth <= `--max-depth` (default 15.0).
- Copy the original image and the matching .txt YOLO label file into the project's `dest` folder (subfolders `originals/` and `labels/`).
- Create a debug image with bounding boxes drawn (in `debug/`).
- Produce a CSV `metadata.csv` with original path, copied paths, depth and bbox count.

The script never modifies the original files; it only copies and writes into `dest`.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd


DEPTH_RE = re.compile(r"depth-([0-9]+(?:\.[0-9]+)?)m")


def parse_depth_from_filename(filename: str) -> float | None:
    m = DEPTH_RE.search(filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int, int]]:
    """Return list of bboxes as (class, x1, y1, x2, y2) in pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line:
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        try:
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        # convert normalized to pixel coords
        x1 = int((x_c - w / 2) * img_w)
        y1 = int((y_c - h / 2) * img_h)
        x2 = int((x_c + w / 2) * img_w)
        y2 = int((y_c + h / 2) * img_h)
        # clamp
        x1 = max(0, min(img_w - 1, x1))
        x2 = max(0, min(img_w - 1, x2))
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(0, min(img_h - 1, y2))
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_boxes(img: np.ndarray, boxes: List[Tuple[int, int, int, int, int]]) -> np.ndarray:
    out = img.copy()
    for cls, x1, y1, x2, y2 in boxes:
        color = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, str(cls), (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_and_copy(source: Path, dest: Path, max_depth: float) -> None:
    source = source.expanduser().resolve()
    dest = dest.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    dest_originals = dest / "originals"
    dest_labels = dest / "labels"
    dest_debug = dest / "debug"
    dest.mkdir(parents=True, exist_ok=True)
    dest_originals.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    dest_debug.mkdir(parents=True, exist_ok=True)

    rows = []
    # Walk shallowly (non-recursive) to avoid scanning huge project root; user folder may be shallow
    for p in sorted(source.iterdir()):
        if not is_image_file(p):
            continue
        depth = parse_depth_from_filename(p.name)
        if depth is None:
            continue
        if depth > max_depth:
            continue
        label_file = p.with_suffix('.txt')

        # copy original image and label (if present)
        dest_img = dest_originals / p.name
        if not dest_img.exists():
            shutil.copy2(p, dest_img)

        if label_file.exists():
            dest_label = dest_labels / label_file.name
            if not dest_label.exists():
                shutil.copy2(label_file, dest_label)
        else:
            dest_label = None

        # create debug image with boxes
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: cannot read image {p}")
            bbox_count = 0
            debug_path = None
        else:
            h, w = img.shape[:2]
            boxes = []
            if dest_label is not None:
                boxes = load_yolo_labels(dest_label, w, h)
            debug_img = draw_boxes(img, boxes)
            debug_path = dest_debug / p.name
            cv2.imwrite(str(debug_path), debug_img)
            bbox_count = len(boxes)

        rows.append({
            'original_path': str(p),
            'copied_image': str(dest_img),
            'copied_label': str(dest_label) if dest_label is not None else '',
            'debug_image': str(debug_path) if debug_path is not None else '',
            'depth_m': depth,
            'bbox_count': bbox_count,
        })

    df = pd.DataFrame(rows)
    df.to_csv(dest / 'metadata.csv', index=False)
    print(f"Done. Copied {len(df)} images to {dest}. Metadata at {dest / 'metadata.csv'}")


def main():
    p = argparse.ArgumentParser(description="Extract debug images from YOLO training folder")
    p.add_argument('--source', '-s', required=True, help='Source folder with images and .txt labels')
    p.add_argument('--dest', '-d', default='data/debug_images', help='Destination folder in project')
    p.add_argument('--max-depth', type=float, default=15.0, help='Maximum depth in meters (inclusive)')
    args = p.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    collect_and_copy(src, dst, args.max_depth)


if __name__ == '__main__':
    main()
