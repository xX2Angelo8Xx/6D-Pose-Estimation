"""Preprocessing utilities for edge detection."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        img: Input image (grayscale or BGR).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
    
    Returns:
        CLAHE-enhanced image (same format as input).
    """
    if len(img.shape) == 3:
        # Convert BGR to LAB, apply CLAHE to L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)


def gaussian_blur(
    img: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian blur.
    
    Args:
        img: Input image.
        sigma: Standard deviation for Gaussian kernel.
    
    Returns:
        Blurred image.
    """
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def bilateral_filter(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Apply bilateral filter (edge-preserving smoothing).
    
    Args:
        img: Input image.
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.
    
    Returns:
        Filtered image.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
