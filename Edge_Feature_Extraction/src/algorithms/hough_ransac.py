"""Hough Transform and RANSAC line extraction."""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple


def hough_lines_probabilistic(
    edges: np.ndarray,
    rho: float = 1.0,
    theta: float = np.pi / 180,
    threshold: int = 50,
    min_line_length: float = 100,
    max_line_gap: float = 20,
) -> List[Tuple[int, int, int, int]]:
    """
    Extract lines using Probabilistic Hough Transform.
    
    Args:
        edges: Binary edge map.
        rho: Distance resolution in pixels.
        theta: Angle resolution in radians.
        threshold: Minimum votes to consider a line.
        min_line_length: Minimum line length in pixels.
        max_line_gap: Maximum gap between line segments.
    
    Returns:
        List of lines as (x1, y1, x2, y2) tuples.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    
    if lines is None:
        return []
    
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines[:, 0]]


def draw_lines(
    img: np.ndarray,
    lines: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw lines on image.
    
    Args:
        img: Input image.
        lines: List of lines as (x1, y1, x2, y2).
        color: Line color (BGR).
        thickness: Line thickness.
    
    Returns:
        Image with lines drawn.
    """
    result = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    return result
