#!/usr/bin/env python3
"""
Adaptive Edge Detection Pipeline with Temporal Filtering.

Features:
- Adaptive sigma computation (ROI size, contrast, depth variance)
- Auto-mode selection (CLAHE vs standard)
- Edge-preserving smoothing (bilateral filter)
- Multi-threshold Canny
- Temporal filtering with exponential decay
- Comprehensive contour validation
- Debug visualization of all intermediate steps
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import time
from collections import deque

import cv2
import numpy as np


@dataclass
class PipelineStage:
    """Container for a pipeline stage result."""
    name: str
    image: np.ndarray
    time_ms: float
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ContourInfo:
    """Information about a detected contour."""
    contour: np.ndarray
    area: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    solidity: float
    aspect_ratio: float
    depth_mean: Optional[float] = None
    depth_std: Optional[float] = None
    depth_consistency: Optional[float] = None  # std/mean ratio
    score: float = 0.0  # temporal weight
    age: int = 0  # frames since first detection


class AdaptiveEdgePipeline:
    """
    Deterministic, adaptive edge detection pipeline.
    
    Automatically adapts to:
    - ROI size (small objects → less smoothing)
    - Image contrast (low → apply CLAHE)
    - Depth variance (high → more smoothing)
    
    Temporal filtering:
    - Tracks contours across frames
    - Exponential decay (recent frames weighted higher)
    - Rejects outliers (short-lived contours)
    """
    
    def __init__(
        self,
        temporal_history: int = 5,
        temporal_decay: float = 0.7,
        min_temporal_score: float = 0.3,
        debug_mode: bool = False,
        minimal_preprocessing: bool = False
    ):
        """
        Args:
            temporal_history: Number of frames to keep in history
            temporal_decay: Weight decay per frame (0-1, higher = recent frames matter more)
            min_temporal_score: Minimum score to keep contour (0-1)
            debug_mode: Enable detailed debug output and stage saving
            minimal_preprocessing: If True, skip CLAHE + bilateral (nur Grayscale + Gaussian)
        """
        self.temporal_history = temporal_history
        self.temporal_decay = temporal_decay
        self.min_temporal_score = min_temporal_score
        self.debug_mode = debug_mode
        self.minimal_preprocessing = minimal_preprocessing
        
        # Temporal storage
        self.frame_history: deque = deque(maxlen=temporal_history)
        self.contour_history: List[List[ContourInfo]] = []
        
        # Debug stages
        self.stages: List[PipelineStage] = []
        
        # Adaptive thresholds
        self.sigma_min = 0.3
        self.sigma_max = 2.5
        self.sigma_base_scale = 0.003  # Scale factor for ROI diagonal
        
        # Contour validation thresholds
        self.min_area_ratio = 0.0005  # Min area as ratio of ROI area
        self.min_absolute_area = 10  # Min absolute pixel area
        self.max_aspect_ratio = 50.0
        self.min_aspect_ratio = 0.02
        self.min_solidity = 0.25
        self.max_depth_consistency = 0.4  # Max std/mean ratio for depth
        
        print("🔧 Adaptive Pipeline initialized:")
        print(f"   Temporal: {temporal_history} frames, decay={temporal_decay:.2f}")
        print(f"   Min score: {min_temporal_score:.2f}")
        print(f"   Preprocessing: {'MINIMAL (Grayscale+Gaussian only)' if minimal_preprocessing else 'FULL (CLAHE+Bilateral+Gaussian)'}")
        print(f"   Debug: {debug_mode}")
    
    def reset_temporal(self):
        """Reset temporal tracking (e.g., when switching images)."""
        self.frame_history.clear()
        self.contour_history.clear()
    
    def compute_adaptive_sigma(
        self,
        roi_gray: np.ndarray,
        depth_roi: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict]:
        """
        Compute adaptive Gaussian sigma based on image characteristics.
        
        Returns:
            sigma: Computed sigma value
            metadata: Dict with computation details
        """
        h, w = roi_gray.shape
        diag = np.sqrt(h*h + w*w)
        
        # Base sigma from ROI size
        sigma = diag * self.sigma_base_scale
        factors = {"base": sigma}
        
        # Contrast factor (low contrast → more smoothing to reduce noise)
        contrast = float(np.percentile(roi_gray, 95) - np.percentile(roi_gray, 5))
        factors["contrast"] = contrast
        if contrast < 30:
            sigma *= 1.4
            factors["contrast_boost"] = 1.4
        elif contrast < 50:
            sigma *= 1.2
            factors["contrast_boost"] = 1.2
        else:
            factors["contrast_boost"] = 1.0
        
        # Depth variance factor (high variance → more smoothing for stability)
        if depth_roi is not None:
            valid_mask = (depth_roi > 0) & np.isfinite(depth_roi)
            if valid_mask.any():
                depth_std = depth_roi[valid_mask].std()
                factors["depth_std"] = depth_std
                if depth_std > 0.5:  # >0.5m variation
                    sigma *= 1.3
                    factors["depth_boost"] = 1.3
                else:
                    factors["depth_boost"] = 1.0
        
        # Clamp to min/max
        sigma = max(self.sigma_min, min(self.sigma_max, sigma))
        factors["final_sigma"] = sigma
        
        return sigma, factors
    
    def should_use_clahe(self, roi_gray: np.ndarray) -> Tuple[bool, Dict]:
        """
        Intelligente CLAHE-Entscheidung.
        
        CLAHE ist hilfreich wenn:
        - Niedriger Kontrast UND strukturierte Inhalte (z.B. Heck im Schatten)
        
        CLAHE ist schädlich wenn:
        - Rauschen im Hintergrund (flache Regionen mit niedrigem Gradient)
        - Bereits guter Kontrast
        
        Strategie:
        1. Messe globalen Kontrast
        2. Messe Kanten-Stärke (wie viel strukturierte Information?)
        3. Messe Rauschen (flache Regionen mit Textur?)
        4. CLAHE nur wenn: niedriger Kontrast + starke Kanten + wenig Rauschen
        """
        h, w = roi_gray.shape
        
        # 1. Global contrast
        global_min, global_max = roi_gray.min(), roi_gray.max()
        global_contrast = float(global_max - global_min)
        
        # 2. Edge strength (Sobel gradients)
        grad_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Strong edges: gradient > threshold
        edge_threshold = 30
        strong_edge_ratio = (grad_mag > edge_threshold).sum() / grad_mag.size
        mean_edge_strength = grad_mag.mean()
        
        # 3. Noise estimation (flat regions with high variance)
        # Use Laplacian variance: high value = edges, low value = flat/noisy
        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Noise indicator: low Laplacian variance but some texture
        # If Laplacian variance is low (<100), image is smooth/noisy (not structured)
        is_noisy = laplacian_var < 100
        
        # 4. Dynamic range check (bright + dark regions?)
        # Histogram-based: check if pixels span multiple brightness levels
        hist, _ = np.histogram(roi_gray, bins=8, range=(0, 256))
        hist_norm = hist / hist.sum()
        # Entropy: high = many brightness levels, low = concentrated
        hist_entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
        needs_dynamic_range = hist_entropy > 2.0  # >2 bits = multiple levels
        
        metadata = {
            "global_contrast": global_contrast,
            "strong_edge_ratio": strong_edge_ratio,
            "mean_edge_strength": mean_edge_strength,
            "laplacian_var": laplacian_var,
            "hist_entropy": hist_entropy,
            "is_noisy": is_noisy
        }
        
        # === DECISION LOGIC ===
        # CLAHE hilft bei: niedriger Kontrast + strukturiertem Inhalt (z.B. Heck)
        # CLAHE schadet bei: Rauschen (niedriger Laplacian + niedriger Kontrast)
        
        # Case 1: Hoher Kontrast → CLAHE unnötig
        if global_contrast > 120:
            use_clahe = False
            metadata["reason"] = "high_contrast"
        
        # Case 2: Niedriger Kontrast + Rauschen → CLAHE verstärkt Rauschen
        elif global_contrast < 80 and is_noisy and strong_edge_ratio < 0.05:
            use_clahe = False
            metadata["reason"] = "noisy_background"
        
        # Case 3: Niedriger Kontrast + starke Kanten → CLAHE hilft (Heck sichtbar machen)
        elif global_contrast < 100 and (strong_edge_ratio > 0.08 or mean_edge_strength > 15):
            use_clahe = True
            metadata["reason"] = "low_contrast_with_structure"
        
        # Case 4: Hohe dynamische Range (Hell+Dunkel) → CLAHE hilft
        elif needs_dynamic_range and global_contrast < 150:
            use_clahe = True
            metadata["reason"] = "high_dynamic_range"
        
        # Case 5: Default → kein CLAHE (bei Zweifel besser weglassen)
        else:
            use_clahe = False
            metadata["reason"] = "default_skip"
        metadata["decision"] = use_clahe
        
        return use_clahe, metadata
    
    def should_use_bilateral(self, roi_gray: np.ndarray, used_clahe: bool) -> Tuple[bool, Dict]:
        """
        Intelligente Bilateral-Filter-Entscheidung.
        
        Bilateral Filter ist hilfreich wenn:
        - Rauschen vorhanden (z.B. sensor noise)
        - Kanten sollen erhalten bleiben
        
        Bilateral Filter ist schädlich wenn:
        - Bild bereits glatt (übermäßiges Smoothing)
        - CLAHE wurde angewendet (CLAHE+Bilateral kann zu stark sein)
        - Feine Details wichtig sind
        """
        # Noise estimation: compute local variance in small patches
        h, w = roi_gray.shape
        patch_size = 5
        noise_estimates = []
        
        # Sample 20 random patches
        np.random.seed(42)  # Deterministic
        for _ in range(20):
            y = np.random.randint(0, max(1, h - patch_size))
            x = np.random.randint(0, max(1, w - patch_size))
            patch = roi_gray[y:y+patch_size, x:x+patch_size]
            if patch.size > 0:
                noise_estimates.append(patch.std())
        
        noise_level = np.mean(noise_estimates) if noise_estimates else 0
        
        # Edge density
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = edges.sum() / (edges.size * 255)
        
        metadata = {
            "noise_level": noise_level,
            "edge_density": edge_density,
            "used_clahe": used_clahe
        }
        
        # === DECISION LOGIC ===
        # Case 1: CLAHE wurde verwendet → skip bilateral (zu viel preprocessing)
        if used_clahe:
            use_bilateral = False
            metadata["reason"] = "clahe_already_applied"
        
        # Case 2: Hoher Noise-Level → bilateral hilft
        elif noise_level > 18:  # Erhöht von 15 → bilateral seltener (nur bei starkem Rauschen)
            use_bilateral = True
            metadata["reason"] = "high_noise"
        
        # Case 3: Sehr hohe Edge-Dichte → bilateral kann Details verschmieren
        elif edge_density > 0.12:  # Gesenkt von 0.15 → skip bilateral öfter bei Kanten
            use_bilateral = False
            metadata["reason"] = "high_edge_density"
        
        # Case 4: Default → skip bilateral (weniger preprocessing = besser)
        else:
            use_bilateral = False
            metadata["reason"] = "default_skip"
        
        metadata["decision"] = use_bilateral
        
        return use_bilateral, metadata
    
    def preprocess(
        self,
        roi_bgr: np.ndarray,
        depth_roi: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[PipelineStage]]:
        """
        Adaptive preprocessing pipeline.
        
        MINIMAL mode (wenn minimal_preprocessing=True):
        1. Convert to grayscale
        2. Adaptive Gaussian blur
        
        FULL mode (wenn minimal_preprocessing=False):
        1. Convert to grayscale
        2. Auto-decide: apply CLAHE if needed
        3. Edge-preserving smoothing (bilateral filter)
        4. Adaptive Gaussian blur
        
        Returns:
            preprocessed: Final preprocessed image
            stages: List of intermediate stages (if debug_mode)
        """
        stages = []
        t0 = time.time()
        
        # Step 1: Grayscale (IMMER)
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if self.debug_mode:
            stages.append(PipelineStage(
                "1_Grayscale",
                cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR),
                (time.time() - t0) * 1000
            ))
        
        if self.minimal_preprocessing:
            # === MINIMAL MODE: Nur Gaussian ===
            # Step 4: Adaptive Gaussian blur (skip Step 2+3)
            t3 = time.time()
            sigma, sigma_meta = self.compute_adaptive_sigma(roi_gray, depth_roi)
            ksize = max(3, int(2 * round(3 * sigma) + 1))
            smoothed = cv2.GaussianBlur(roi_gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            
            if self.debug_mode:
                # Add dummy stages for consistent numbering
                stages.append(PipelineStage(
                    "2_CLAHE_Skipped",
                    cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR),
                    0.0,
                    {"skipped": True, "reason": "minimal_preprocessing"}
                ))
                stages.append(PipelineStage(
                    "3_Bilateral_Skipped",
                    cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR),
                    0.0,
                    {"skipped": True, "reason": "minimal_preprocessing"}
                ))
                stages.append(PipelineStage(
                    f"4_Gaussian_s{sigma:.2f}",
                    cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR),
                    (time.time() - t3) * 1000,
                    sigma_meta
                ))
        else:
            # === FULL MODE: CLAHE + Bilateral + Gaussian ===
            # Step 2: CLAHE (conditional)
            t1 = time.time()
            use_clahe, clahe_meta = self.should_use_clahe(roi_gray)
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                roi_gray_processed = clahe.apply(roi_gray)
                label = "2_CLAHE_Applied"
            else:
                roi_gray_processed = roi_gray
                label = "2_CLAHE_Skipped"
            
            if self.debug_mode:
                stages.append(PipelineStage(
                    label,
                    cv2.cvtColor(roi_gray_processed, cv2.COLOR_GRAY2BGR),
                    (time.time() - t1) * 1000,
                    clahe_meta
                ))
            
            # Step 3: Bilateral filter (conditional, adaptive)
            t2 = time.time()
            use_bilateral, bilateral_meta = self.should_use_bilateral(roi_gray_processed, use_clahe)
            if use_bilateral:
                # Sanftere Parameter: d=5 (kleiner Radius), sigmaColor=30, sigmaSpace=30
                # Verhindert zu starkes Verschmieren von Kanten (Heck!)
                bilateral = cv2.bilateralFilter(roi_gray_processed, d=5, sigmaColor=30, sigmaSpace=30)
                label = "3_Bilateral_Applied"
            else:
                bilateral = roi_gray_processed
                label = "3_Bilateral_Skipped"
            
            if self.debug_mode:
                stages.append(PipelineStage(
                    label,
                    cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR),
                    (time.time() - t2) * 1000,
                    bilateral_meta
                ))
            
            # Step 4: Adaptive Gaussian blur
            t3 = time.time()
            sigma, sigma_meta = self.compute_adaptive_sigma(bilateral, depth_roi)
            ksize = max(3, int(2 * round(3 * sigma) + 1))
            smoothed = cv2.GaussianBlur(bilateral, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            
            if self.debug_mode:
                stages.append(PipelineStage(
                    f"4_Gaussian_s{sigma:.2f}",
                    cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR),
                    (time.time() - t3) * 1000,
                    sigma_meta
                ))
        
        return smoothed, stages
    
    def detect_edges(
        self,
        smoothed: np.ndarray
    ) -> Tuple[np.ndarray, List[PipelineStage]]:
        """
        Multi-threshold Canny edge detection.
        
        Uses median-based auto-thresholding with 3 variants:
        - Conservative (high thresholds)
        - Medium
        - Sensitive (low thresholds)
        
        Final edges = union of all three, then morphological cleanup.
        """
        stages = []
        t0 = time.time()
        
        # Compute median-based thresholds
        v = np.median(smoothed[smoothed > 0]) if (smoothed > 0).any() else 128
        
        # Three threshold variants
        thresholds = [
            (int(max(0, 0.8 * v)), int(min(255, 1.5 * v)), "Conservative"),
            (int(max(0, 0.66 * v)), int(min(255, 1.33 * v)), "Medium"),
            (int(max(0, 0.5 * v)), int(min(255, 1.2 * v)), "Sensitive")
        ]
        
        edge_maps = []
        for lower, upper, name in thresholds:
            edges = cv2.Canny(smoothed, lower, upper)
            edge_maps.append(edges)
            
            if self.debug_mode:
                stages.append(PipelineStage(
                    f"5_Canny_{name}",
                    cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
                    (time.time() - t0) * 1000,
                    {"lower": lower, "upper": upper}
                ))
        
        # Combine: OR all three
        combined = np.zeros_like(edge_maps[0])
        for em in edge_maps:
            combined = cv2.bitwise_or(combined, em)
        
        # Morphological cleanup: close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        if self.debug_mode:
            # Invert edges for better visibility (white background, black edges)
            cleaned_inv = 255 - cleaned
            stages.append(PipelineStage(
                "6_Combined_Cleaned",
                cv2.cvtColor(cleaned_inv, cv2.COLOR_GRAY2BGR),
                (time.time() - t0) * 1000,
                {"edge_pixels": int(np.sum(cleaned > 0))}
            ))
        
        return cleaned, stages
    
    def validate_contour(
        self,
        contour: np.ndarray,
        roi_area: int,
        depth_roi: Optional[np.ndarray] = None,
        bbox_offset: Tuple[int, int] = (0, 0)
    ) -> Tuple[bool, ContourInfo]:
        """
        Validate a contour based on geometric and depth criteria.
        
        Returns:
            valid: Whether contour passes validation
            info: ContourInfo object with details
        """
        area = cv2.contourArea(contour)
        
        # Min area check
        min_area = max(self.min_absolute_area, roi_area * self.min_area_ratio)
        if area < min_area:
            return False, None
        
        # Bounding box & aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / (h + 1e-6)
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False, None
        
        # Solidity (compactness)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        if solidity < self.min_solidity:
            return False, None
        
        # Depth consistency (if available)
        depth_mean = None
        depth_std = None
        depth_consistency = None
        
        if depth_roi is not None:
            # Create mask for contour
            mask = np.zeros(depth_roi.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Extract depth values
            depth_vals = depth_roi[mask == 255]
            valid_depths = depth_vals[(depth_vals > 0) & np.isfinite(depth_vals)]
            
            if valid_depths.size > 5:  # Need enough samples
                depth_mean = float(valid_depths.mean())
                depth_std = float(valid_depths.std())
                depth_consistency = depth_std / (depth_mean + 1e-6)
                
                # Reject if too inconsistent (contour spans multiple depth planes)
                if depth_consistency > self.max_depth_consistency:
                    return False, None
        
        # Create info object
        info = ContourInfo(
            contour=contour,
            area=area,
            bbox=(x, y, w, h),
            solidity=solidity,
            aspect_ratio=aspect_ratio,
            depth_mean=depth_mean,
            depth_std=depth_std,
            depth_consistency=depth_consistency,
            score=1.0,  # Initial score
            age=0
        )
        
        return True, info
    
    def temporal_filter(
        self,
        current_contours: List[ContourInfo]
    ) -> List[ContourInfo]:
        """
        Apply temporal filtering with exponential decay.
        
        Tracks contours across frames:
        - Match current contours to previous ones (by spatial proximity)
        - Update scores with exponential decay
        - Keep contours with score > threshold
        """
        # Add current frame to history
        self.frame_history.append(current_contours)
        
        if len(self.frame_history) < 2:
            # Not enough history yet, return all current
            return current_contours
        
        # Build tracking: match current contours to previous frames
        tracked = []
        
        for curr_info in current_contours:
            # Try to match with contours in previous frames
            best_score = 1.0  # Current frame score
            
            # Look back through history with decaying weights
            for frame_idx in range(len(self.frame_history) - 2, -1, -1):
                decay_factor = self.temporal_decay ** (len(self.frame_history) - 1 - frame_idx)
                prev_frame = self.frame_history[frame_idx]
                
                # Find closest contour by centroid distance
                curr_m = cv2.moments(curr_info.contour)
                if curr_m['m00'] == 0:
                    continue
                curr_cx = curr_m['m10'] / curr_m['m00']
                curr_cy = curr_m['m01'] / curr_m['m00']
                
                min_dist = float('inf')
                matched = False
                
                for prev_info in prev_frame:
                    prev_m = cv2.moments(prev_info.contour)
                    if prev_m['m00'] == 0:
                        continue
                    prev_cx = prev_m['m10'] / prev_m['m00']
                    prev_cy = prev_m['m01'] / prev_m['m00']
                    
                    dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                    
                    # Match if within reasonable distance (20 pixels)
                    if dist < 20 and dist < min_dist:
                        min_dist = dist
                        matched = True
                
                if matched:
                    best_score += decay_factor
            
            # Update score and age
            curr_info.score = best_score
            curr_info.age = min(len(self.frame_history), curr_info.age + 1)
            tracked.append(curr_info)
        
        # Filter by minimum score
        filtered = [info for info in tracked if info.score >= self.min_temporal_score]
        
        return filtered
    
    def process(
        self,
        roi_bgr: np.ndarray,
        depth_roi: Optional[np.ndarray] = None,
        bbox_offset: Tuple[int, int] = (0, 0)
    ) -> Tuple[List[ContourInfo], List[PipelineStage]]:
        """
        Full pipeline: preprocess → edges → contours → validate → temporal filter.
        
        Returns:
            contours: List of validated ContourInfo objects
            stages: List of all pipeline stages (for debug visualization)
        """
        self.stages = []
        
        # 1. Preprocessing
        preprocessed, preproc_stages = self.preprocess(roi_bgr, depth_roi)
        self.stages.extend(preproc_stages)
        
        # 2. Edge detection
        edges, edge_stages = self.detect_edges(preprocessed)
        self.stages.extend(edge_stages)
        
        # 3. Find contours
        t0 = time.time()
        contours_raw, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.debug_mode:
            # Visualize raw contours with thicker lines
            vis = roi_bgr.copy()
            cv2.drawContours(vis, contours_raw, -1, (0, 255, 255), 2)  # Thicker: 2px
            self.stages.append(PipelineStage(
                f"7_Raw_Contours_n{len(contours_raw)}",
                vis,
                (time.time() - t0) * 1000,
                {"count": len(contours_raw)}
            ))
        
        # 4. Validate contours
        t1 = time.time()
        roi_area = roi_bgr.shape[0] * roi_bgr.shape[1]
        validated = []
        rejected = []
        
        for contour in contours_raw:
            valid, info = self.validate_contour(contour, roi_area, depth_roi, bbox_offset)
            if valid:
                validated.append(info)
            else:
                rejected.append(contour)
        
        if self.debug_mode:
            # Visualize validated vs rejected with thicker lines
            vis = roi_bgr.copy()
            cv2.drawContours(vis, [info.contour for info in validated], -1, (0, 255, 0), 3)  # Green, thick
            cv2.drawContours(vis, rejected, -1, (0, 0, 255), 1)  # Red, thin
            self.stages.append(PipelineStage(
                f"8_Validated_n{len(validated)}",
                vis,
                (time.time() - t1) * 1000,
                {"kept": len(validated), "rejected": len(rejected)}
            ))
        
        # 5. Temporal filtering
        t2 = time.time()
        filtered = self.temporal_filter(validated)
        
        if self.debug_mode:
            # Visualize with temporal scores (thicker lines + better text)
            vis = roi_bgr.copy()
            for info in filtered:
                # Color by score: green (high) → yellow (low)
                score_norm = min(1.0, info.score / 2.0)  # Normalize to 0-1
                score_color = int(255 * (1 - score_norm))
                color = (0, 255 - score_color, score_color)
                
                # Thickness by score (more persistent = thicker)
                thickness = max(2, int(3 * score_norm))
                cv2.drawContours(vis, [info.contour], -1, color, thickness)
                
                # Add score text with background for readability
                m = cv2.moments(info.contour)
                if m['m00'] > 0:
                    cx = int(m['m10'] / m['m00'])
                    cy = int(m['m01'] / m['m00'])
                    text = f"{info.score:.1f}"
                    # Black background
                    cv2.putText(vis, text, (cx-16, cy+1),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    # White text
                    cv2.putText(vis, text, (cx-15, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            self.stages.append(PipelineStage(
                f"9_Temporal_n{len(filtered)}",
                vis,
                (time.time() - t2) * 1000,
                {
                    "kept": len(filtered),
                    "history_size": len(self.frame_history),
                    "avg_score": np.mean([i.score for i in filtered]) if filtered else 0
                }
            ))
        
        return filtered, self.stages
