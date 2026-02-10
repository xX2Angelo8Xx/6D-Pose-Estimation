"""Canny edge detection with automatic threshold selection."""

from __future__ import annotations

import cv2
import numpy as np

from .preproc import gaussian_blur


def auto_canny(
    img: np.ndarray,
    sigma: float = 0.33,
) -> np.ndarray:
    """
    Apply Canny edge detection with automatic threshold selection.
    
    Uses the median of the image gradient magnitude to set thresholds:
    - lower = max(0, (1.0 - sigma) * median)
    - upper = min(255, (1.0 + sigma) * median)
    
    Args:
        img: Input grayscale image.
        sigma: Factor for threshold adjustment (default 0.33).
    
    Returns:
        Binary edge map.
    """
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


def canny_edge_detection(
    img: np.ndarray,
    low_threshold: int | None = None,
    high_threshold: int | None = None,
    blur_sigma: float = 1.0,
    auto_threshold: bool = True,
) -> np.ndarray:
    """
    Apply Canny edge detection with optional preprocessing.
    
    Args:
        img: Input image (BGR or grayscale).
        low_threshold: Lower hysteresis threshold (None for auto).
        high_threshold: Upper hysteresis threshold (None for auto).
        blur_sigma: Gaussian blur sigma before edge detection.
        auto_threshold: If True, use automatic threshold selection.
    
    Returns:
        Binary edge map (uint8, 0 or 255).
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply Gaussian blur
    if blur_sigma > 0:
        gray = gaussian_blur(gray, blur_sigma)
    
    # Apply Canny
    if auto_threshold or low_threshold is None or high_threshold is None:
        edges = auto_canny(gray, sigma=0.33)
    else:
        edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    return edges
