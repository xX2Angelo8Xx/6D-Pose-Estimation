"""Contour-based silhouette extraction."""

from __future__ import annotations

import cv2
import numpy as np


def extract_silhouette_from_contours(
    edges: np.ndarray,
    original_shape: tuple[int, int],
    morph_kernel_size: int = 5,
    min_area_ratio: float = 0.01,
) -> np.ndarray:
    """
    Extract silhouette mask from edge map using contours.
    
    Pipeline:
    1. Apply morphological closing to connect edge gaps.
    2. Find contours.
    3. Select largest contour (by area) that meets minimum threshold.
    4. Fill contour to create solid mask.
    
    Args:
        edges: Binary edge map (uint8, 0 or 255).
        original_shape: Shape (H, W) of the original image.
        morph_kernel_size: Size of morphological kernel for closing.
        min_area_ratio: Minimum contour area as fraction of image area.
    
    Returns:
        Binary silhouette mask (uint8, 0 or 255).
    """
    h, w = original_shape[:2]
    total_area = h * w
    
    # Morphological closing to connect gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and select largest contour
    mask = np.zeros((h, w), dtype=np.uint8)
    if not contours:
        return mask
    
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_ratio * total_area]
    if not valid_contours:
        return mask
    
    # Take largest contour
    largest = max(valid_contours, key=cv2.contourArea)
    
    # Fill contour
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    
    return mask
