"""Edge detection algorithms module."""

from .preproc import apply_clahe, gaussian_blur
from .canny_pipeline import canny_edge_detection, auto_canny
from .contour_silhouette import extract_silhouette_from_contours

__all__ = [
    "apply_clahe",
    "gaussian_blur",
    "canny_edge_detection",
    "auto_canny",
    "extract_silhouette_from_contours",
]
