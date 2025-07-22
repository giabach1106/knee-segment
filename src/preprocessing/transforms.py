"""Preprocessing transforms for ultrasound image segmentation.

This module implements various preprocessing techniques to enhance
ultrasound images before feeding them to the segmentation model.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import logging

import cv2
import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.draw import polygon2mask
from skimage import io
import torch
from torch import nn

logger = logging.getLogger(__name__)


class Transform(ABC):
    """Abstract base class for preprocessing transforms."""
    
    @abstractmethod
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply the transform to an image and optionally its mask.
        
        Args:
            image: Input image as numpy array.
            mask: Optional segmentation mask.
            
        Returns:
            Tuple of (transformed_image, transformed_mask).
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get transform parameters for logging/reproducibility.
        
        Returns:
            Dictionary of transform parameters.
        """
        return {}


class CLAHETransform(Transform):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE improves local contrast while preventing over-amplification
    of noise in homogeneous regions.
    
    Args:
        clip_limit: Contrast limiting threshold.
        tile_grid_size: Size of the neighborhood area.
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply CLAHE to the image.
        
        Args:
            image: Input image.
            mask: Optional segmentation mask (unchanged).
            
        Returns:
            Tuple of (CLAHE-enhanced image, unchanged mask).
        """
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced_image = clahe.apply(image)
        return enhanced_image, mask
        
    def get_params(self) -> Dict[str, Any]:
        """Get CLAHE parameters."""
        return {
            "clip_limit": self.clip_limit,
            "tile_grid_size": self.tile_grid_size
        }


class FixedCrop(Transform):
    """Fixed rectangular crop to focus on ROI.
    
    Removes irrelevant borders and focuses on the central knee joint region.
    
    Args:
        crop_top: Pixels to crop from top.
        crop_bottom: Pixels to crop from bottom.
        crop_left: Pixels to crop from left.
        crop_right: Pixels to crop from right.
    """
    
    def __init__(
        self, 
        crop_top: int = 62, 
        crop_bottom: int = 94, 
        crop_left: int = 84, 
        crop_right: int = 44
    ) -> None:
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply fixed crop to image and mask.
        
        Args:
            image: Input image.
            mask: Optional segmentation mask.
            
        Returns:
            Tuple of (cropped_image, cropped_mask).
        """
        h, w = image.shape[:2]
        
        # Calculate crop boundaries
        y1 = self.crop_top
        y2 = h - self.crop_bottom
        x1 = self.crop_left
        x2 = w - self.crop_right
        
        # Ensure valid crop
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)
        
        # Crop image
        cropped_image = image[y1:y2, x1:x2]
        
        # Crop mask if provided
        cropped_mask = mask[y1:y2, x1:x2] if mask is not None else None
        
        return cropped_image, cropped_mask
    
    def get_params(self) -> Dict[str, Any]:
        """Get crop parameters."""
        return {
            "crop_top": self.crop_top,
            "crop_bottom": self.crop_bottom,
            "crop_left": self.crop_left,
            "crop_right": self.crop_right
        }


class SnakeROI(Transform):
    """Active contour (Snake) based ROI extraction for ultrasound images.
    
    Uses active contour algorithm to automatically find and crop the anatomical
    Region of Interest by detecting the boundary between ultrasound content
    and black background/border areas. This method is more robust than simple
    thresholding as it can handle dark regions within the anatomy.
    
    The approach:
    1. Detect black border regions via thresholding and connectivity analysis
    2. Initialize snake ellipse around detected content region  
    3. Evolve snake to fit ROI boundary using edge and smoothness forces
    4. Mask and crop image to extracted ROI
    
    Args:
        alpha: Snake tension parameter (elasticity/contraction). Lower values
            allow more expansion. Default: 0.01
        beta: Snake rigidity parameter (smoothness). Higher values create 
            smoother contours. Default: 5.0
        w_edge: Weight for edge attraction forces. Default: 1.0
        w_line: Weight for line/intensity forces. Default: 0.0
        gamma: Time stepping parameter. Default: 0.1
        max_iterations: Maximum snake evolution iterations. Default: 1000
        convergence: Convergence threshold for early stopping. Default: 0.1
        intensity_threshold: Threshold for detecting black pixels. Default: 10
        min_content_size: Minimum size for content regions (removes text). Default: 1000
        hole_fill_threshold: Size threshold for filling holes. Default: 2000
        blur_sigma: Gaussian blur sigma for preprocessing. Default: 2.0
        snake_points: Number of points in snake contour. Default: 200
        content_padding: Padding factor for snake initialization (0.45 = 90% of region). Default: 0.45
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 5.0, 
        w_edge: float = 1.0,
        w_line: float = 0.0,
        gamma: float = 0.1,
        max_iterations: int = 1000,
        convergence: float = 0.1,
        intensity_threshold: int = 10,
        min_content_size: int = 1000,
        hole_fill_threshold: int = 2000,
        blur_sigma: float = 2.0,
        snake_points: int = 200,
        content_padding: float = 0.45
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.w_edge = w_edge
        self.w_line = w_line
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.convergence = convergence
        self.intensity_threshold = intensity_threshold
        self.min_content_size = min_content_size
        self.hole_fill_threshold = hole_fill_threshold
        self.blur_sigma = blur_sigma
        self.snake_points = snake_points
        self.content_padding = content_padding
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply snake-based ROI extraction.
        
        Args:
            image: Input grayscale ultrasound image.
            mask: Optional segmentation mask.
            
        Returns:
            Tuple of (ROI extracted image, cropped mask).
        """
        # Validate input
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
            
        # Store original shape for consistency checks
        original_shape = image.shape
        
        try:
            # Step 1: Detect black border and content region
            content_mask = self._detect_content_region(image)
            
            # Step 2: Find content bounding box for initial cropping
            coords = np.column_stack(np.nonzero(content_mask))
            if len(coords) == 0:
                logger.warning("No content detected, returning original image")
                return image, mask
                
            top_left = coords.min(axis=0)
            bottom_right = coords.max(axis=0) 
            r0, c0 = top_left
            r1, c1 = bottom_right
            
            # Add margin for snake initialization
            margin = 5
            r0m = max(r0 - margin, 0)
            r1m = min(r1 + margin, image.shape[0])
            c0m = max(c0 - margin, 0) 
            c1m = min(c1 + margin, image.shape[1])
            
            # Crop to content region for efficiency
            image_cropped = image[r0m:r1m, c0m:c1m]
            
            # Validate cropped image
            if image_cropped.size == 0:
                logger.warning("Cropped region is empty, returning original image")
                return image, mask
            
            # Step 3: Initialize and evolve snake
            snake = self._run_active_contour(image_cropped)
            
            # Step 4: Create ROI mask and crop final image
            final_image, final_mask = self._apply_snake_crop(
                image, mask, snake, (r0m, c0m), image_cropped.shape
            )
            
            # Validate output shape consistency
            if final_image.shape != original_shape:
                logger.error(f"Shape mismatch: input {original_shape}, output {final_image.shape}")
                return image, mask
                
            return final_image, final_mask
            
        except Exception as e:
            logger.warning(f"Snake ROI extraction failed: {e}. Returning original image.")
            # Ensure we always return the same shape
            return image, mask
    
    def _detect_content_region(self, image: np.ndarray) -> np.ndarray:
        """Detect content region by removing black border and small objects.
        
        Args:
            image: Input grayscale image.
            
        Returns:
            Binary mask where True indicates content region.
        """
        # Step 1: Threshold to separate content from black background
        content_mask = image > self.intensity_threshold
        
        # Step 2: Remove small objects (text, artifacts)  
        content_mask_clean = remove_small_objects(
            content_mask, min_size=self.min_content_size
        )
        
        # Step 3: Fill small holes within content region
        content_mask_filled = remove_small_holes(
            content_mask_clean, area_threshold=self.hole_fill_threshold
        )
        
        return content_mask_filled
    
    def _initialize_snake(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Initialize snake as ellipse within image bounds.
        
        Args:
            image_shape: Shape of the cropped image (height, width).
            
        Returns:
            Initial snake coordinates as (N, 2) array [[y, x], ...].
        """
        rows, cols = image_shape
        center_y, center_x = rows / 2, cols / 2
        
        # Create ellipse with padding from edges
        radius_y = rows * self.content_padding
        radius_x = cols * self.content_padding
        
        theta = np.linspace(0, 2 * np.pi, self.snake_points)
        init_y = center_y + radius_y * np.sin(theta)
        init_x = center_x + radius_x * np.cos(theta)
        
        init_snake = np.vstack([init_y, init_x]).T
        
        return init_snake
    
    def _run_active_contour(self, image_cropped: np.ndarray) -> np.ndarray:
        """Run active contour algorithm on cropped image.
        
        Args:
            image_cropped: Cropped image focused on content region.
            
        Returns:
            Final snake coordinates.
        """
        # Normalize and smooth image for snake
        img_float = image_cropped.astype(float) / 255.0
        blurred = gaussian(img_float, sigma=self.blur_sigma)
        
        # Initialize snake
        init_snake = self._initialize_snake(blurred.shape)
        
        # Run active contour evolution
        snake = active_contour(
            blurred,
            init_snake,
            alpha=self.alpha,
            beta=self.beta, 
            w_edge=self.w_edge,
            w_line=self.w_line,
            gamma=self.gamma,
            max_num_iter=self.max_iterations,
            convergence=self.convergence
        )
        
        return snake
    
    def _apply_snake_crop(
        self, 
        original_image: np.ndarray,
        original_mask: Optional[np.ndarray],
        snake: np.ndarray,
        crop_offset: Tuple[int, int],
        cropped_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply snake-based cropping to original image and mask.
        
        Args:
            original_image: Original full-size image.
            original_mask: Original full-size mask.
            snake: Snake coordinates relative to cropped image.
            crop_offset: (row_offset, col_offset) of cropped region.
            cropped_shape: Shape of cropped region.
            
        Returns:
            Tuple of (final ROI image, final ROI mask).
        """
        # Store original dimensions for consistent output size
        original_height, original_width = original_image.shape[:2]
        
        # Create snake mask on cropped region
        snake_mask = polygon2mask(cropped_shape, snake)
        
        # Get bounding box of snake mask
        ys, xs = np.nonzero(snake_mask)
        if len(ys) == 0 or len(xs) == 0:
            logger.warning("Snake mask is empty, returning original image")
            return original_image, original_mask
        
        miny, maxy = ys.min(), ys.max()
        minx, maxx = xs.min(), xs.max()
        
        # Adjust coordinates to original image space
        r0, c0 = crop_offset
        final_r0 = r0 + miny
        final_r1 = r0 + maxy + 1
        final_c0 = c0 + minx  
        final_c1 = c0 + maxx + 1
        
        # Ensure bounds are valid
        final_r0 = max(0, final_r0)
        final_r1 = min(original_image.shape[0], final_r1)
        final_c0 = max(0, final_c0)
        final_c1 = min(original_image.shape[1], final_c1)
        
        # Extract ROI from original image
        roi_image = original_image[final_r0:final_r1, final_c0:final_c1].copy()
        
        # Check if ROI is too small for meaningful processing
        if roi_image.size == 0 or roi_image.shape[0] < 5 or roi_image.shape[1] < 5:
            logger.warning("Snake ROI is too small, returning original image")
            return original_image, original_mask
        
        # Apply snake mask to remove background in cropped region
        if roi_image.size > 0:
            # Create mask for final cropped region
            crop_snake_mask = snake_mask[miny:maxy+1, minx:maxx+1]
            if crop_snake_mask.shape == roi_image.shape:
                roi_image[~crop_snake_mask] = 0
        
        # CRITICAL: Resize back to original dimensions to maintain consistent size
        # This ensures all images have the same dimensions for DataLoader batching
        final_image = cv2.resize(roi_image, (original_width, original_height))
        
        # Debug logging for tensor shape consistency
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Snake ROI: input shape {original_image.shape}, output shape {final_image.shape}")
        
        # Crop and resize mask if provided
        final_mask = None
        if original_mask is not None:
            roi_mask = original_mask[final_r0:final_r1, final_c0:final_c1]
            if roi_mask.size > 0:
                final_mask = cv2.resize(roi_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            else:
                final_mask = original_mask
        
        return final_image, final_mask
    
    def get_params(self) -> Dict[str, Any]:
        """Get snake parameters."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "w_edge": self.w_edge, 
            "w_line": self.w_line,
            "gamma": self.gamma,
            "max_iterations": self.max_iterations,
            "convergence": self.convergence,
            "intensity_threshold": self.intensity_threshold,
            "min_content_size": self.min_content_size,
            "hole_fill_threshold": self.hole_fill_threshold,
            "blur_sigma": self.blur_sigma,
            "snake_points": self.snake_points,
            "content_padding": self.content_padding
        }


class HistogramEqualization(Transform):
    """Global histogram equalization.
    
    Applies standard histogram equalization to improve overall contrast.
    """
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply histogram equalization.
        
        Args:
            image: Input grayscale image.
            mask: Optional segmentation mask (passed through unchanged).
            
        Returns:
            Tuple of (equalized image, mask).
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        else:
            image_uint8 = image
            
        # Apply histogram equalization
        equalized = cv2.equalizeHist(image_uint8)
        
        # Convert back to original scale
        if image.dtype == np.float32 or image.dtype == np.float64:
            equalized = equalized.astype(image.dtype) / 255.0
            
        return equalized, mask


class GaussianBlur(Transform):
    """Gaussian blur for noise reduction.
    
    Reduces speckle noise in ultrasound images while preserving larger structures.
    
    Args:
        kernel_size: Size of the Gaussian kernel (default: 3).
        sigma: Standard deviation for Gaussian kernel (default: 1.0).
    """
    
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply Gaussian blur.
        
        Args:
            image: Input image.
            mask: Optional segmentation mask (passed through unchanged).
            
        Returns:
            Tuple of (blurred image, mask).
        """
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        return blurred, mask
    
    def get_params(self) -> Dict[str, Any]:
        """Get blur parameters."""
        return {
            "kernel_size": self.kernel_size,
            "sigma": self.sigma
        }


class Sharpening(Transform):
    """Image sharpening to enhance edges.
    
    Uses unsharp masking to make boundaries more pronounced.
    
    Args:
        alpha: Sharpening strength (default: 1.5).
        sigma: Gaussian blur sigma for unsharp mask (default: 1.0).
    """
    
    def __init__(self, alpha: float = 1.5, sigma: float = 1.0) -> None:
        self.alpha = alpha
        self.sigma = sigma
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply sharpening.
        
        Args:
            image: Input image.
            mask: Optional segmentation mask (passed through unchanged).
            
        Returns:
            Tuple of (sharpened image, mask).
        """
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (0, 0), self.sigma)
        
        # Unsharp mask: sharpened = original + alpha * (original - blurred)
        sharpened = cv2.addWeighted(image, 1 + self.alpha, blurred, -self.alpha, 0)
        
        # Clip values to valid range
        if image.dtype == np.uint8:
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            sharpened = np.clip(sharpened, 0, 1)
            
        return sharpened, mask
    
    def get_params(self) -> Dict[str, Any]:
        """Get sharpening parameters."""
        return {
            "alpha": self.alpha,
            "sigma": self.sigma
        }


class IntensityNormalization(Transform):
    """Intensity normalization.
    
    Normalizes pixel intensities to a standard range or distribution.
    
    Args:
        method: Normalization method ('minmax' or 'zscore').
        target_range: Target range for minmax normalization (default: (0, 1)).
    """
    
    def __init__(
        self, 
        method: str = "minmax", 
        target_range: Tuple[float, float] = (0, 1)
    ) -> None:
        assert method in ["minmax", "zscore"], f"Unknown normalization method: {method}"
        self.method = method
        self.target_range = target_range
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply intensity normalization.
        
        Args:
            image: Input image.
            mask: Optional segmentation mask (passed through unchanged).
            
        Returns:
            Tuple of (normalized image, mask).
        """
        image_float = image.astype(np.float32)
        
        if self.method == "minmax":
            # Min-max normalization
            min_val = image_float.min()
            max_val = image_float.max()
            
            if max_val > min_val:
                normalized = (image_float - min_val) / (max_val - min_val)
                # Scale to target range
                range_min, range_max = self.target_range
                normalized = normalized * (range_max - range_min) + range_min
            else:
                normalized = np.zeros_like(image_float) + self.target_range[0]
                
        else:  # zscore
            # Z-score normalization
            mean = image_float.mean()
            std = image_float.std()
            
            if std > 0:
                normalized = (image_float - mean) / std
            else:
                normalized = image_float - mean
                
        return normalized.astype(np.float32), mask
    
    def get_params(self) -> Dict[str, Any]:
        """Get normalization parameters."""
        return {
            "method": self.method,
            "target_range": self.target_range
        }


class MultiChannelTransform(Transform):
    """Create multi-channel input by stacking processed versions.
    
    Combines original image with one or more processed versions to create
    multi-channel input for the model.
    
    Args:
        transforms: List of transforms to apply for additional channels.
        include_original: Whether to include original image as first channel.
    """
    
    def __init__(
        self, 
        transforms: List[Transform],
        include_original: bool = True
    ) -> None:
        self.transforms = transforms
        self.include_original = include_original
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create multi-channel image.
        
        Args:
            image: Input image (H, W).
            mask: Optional segmentation mask.
            
        Returns:
            Tuple of (multi-channel image (C, H, W), mask).
        """
        channels = []
        
        # Add original image if requested
        if self.include_original:
            channels.append(image)
            
        # Apply transforms for additional channels
        for transform in self.transforms:
            transformed_img, _ = transform(image, None)
            channels.append(transformed_img)
            
        # Stack channels
        multi_channel = np.stack(channels, axis=0)
        
        return multi_channel, mask
    
    def get_params(self) -> Dict[str, Any]:
        """Get multi-channel parameters."""
        return {
            "num_channels": len(self.transforms) + (1 if self.include_original else 0),
            "include_original": self.include_original,
            "transforms": [t.__class__.__name__ for t in self.transforms]
        }


class PreprocessingPipeline:
    """Compose multiple preprocessing transforms into a pipeline.
    
    Args:
        transforms: List of transforms to apply in sequence.
    """
    
    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply all transforms in sequence.
        
        Args:
            image: Input image.
            mask: Optional segmentation mask.
            
        Returns:
            Tuple of (transformed image, transformed mask).
        """
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask
    
    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        params = {
            "num_transforms": len(self.transforms),
            "transforms": []
        }
        
        for i, transform in enumerate(self.transforms):
            transform_info = {
                "name": transform.__class__.__name__,
                "params": transform.get_params()
            }
            params["transforms"].append(transform_info)
            
        return params 