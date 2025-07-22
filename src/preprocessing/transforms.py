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
    """Contrast Limited Adaptive Histogram Equalization.
    
    CLAHE improves local contrast in ultrasound images, making fluid-filled
    lesion areas more distinguishable from surrounding tissue.
    
    Args:
        clip_limit: Threshold for contrast limiting (default: 2.0).
        tile_grid_size: Size of grid for histogram equalization (default: (8, 8)).
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply CLAHE to the image.
        
        Args:
            image: Input grayscale image.
            mask: Optional segmentation mask (passed through unchanged).
            
        Returns:
            Tuple of (CLAHE-enhanced image, mask).
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        else:
            image_uint8 = image
            
        # Apply CLAHE
        enhanced = self.clahe.apply(image_uint8)
        
        # Convert back to original scale
        if image.dtype == np.float32 or image.dtype == np.float64:
            enhanced = enhanced.astype(image.dtype) / 255.0
            
        return enhanced, mask
    
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
    """Snake-based (active contour) ROI extraction.
    
    Uses active contour algorithm to automatically find and crop the
    Region of Interest based on image content.
    
    Args:
        alpha: Snake length shape parameter (default: 0.015).
        beta: Snake smoothness shape parameter (default: 10).
        gamma: Explicit time stepping parameter (default: 0.001).
        max_iterations: Maximum iterations for snake evolution (default: 2500).
        convergence: Convergence criteria (default: 0.1).
        edge_threshold: Threshold for edge detection preprocessing (default: 100).
    """
    
    def __init__(
        self, 
        alpha: float = 0.015,
        beta: float = 10,
        gamma: float = 0.001,
        max_iterations: int = 2500,
        convergence: float = 0.1,
        edge_threshold: float = 100
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.convergence = convergence
        self.edge_threshold = edge_threshold
        
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply snake-based ROI extraction.
        
        Args:
            image: Input grayscale image.
            mask: Optional segmentation mask.
            
        Returns:
            Tuple of (ROI image, ROI mask).
        """
        # Ensure image is float
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)
            
        # Apply Gaussian smoothing for snake
        img_smooth = gaussian(img_float, 3)
        
        # Initialize snake at image border
        h, w = img_smooth.shape
        margin = 10
        init_snake = np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ], dtype=np.float32)
        
        # Close the contour
        init_snake = np.vstack([init_snake, init_snake[0]])
        
        # Run active contour
        try:
            snake = active_contour(
                img_smooth,
                init_snake,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                max_iterations=self.max_iterations,
                convergence=self.convergence
            )
            
            # Create mask from snake contour
            roi_mask = np.zeros(img_smooth.shape, dtype=np.uint8)
            snake_int = np.array(snake, dtype=np.int32)
            cv2.fillPoly(roi_mask, [snake_int], 255)
            
            # Apply mask to image
            roi_image = img_float.copy()
            roi_image[roi_mask == 0] = 0
            
            # Find bounding box of ROI
            y_indices, x_indices = np.where(roi_mask > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                
                # Crop to bounding box
                roi_image = roi_image[y_min:y_max, x_min:x_max]
                
                # Crop mask if provided
                if mask is not None:
                    mask = mask[y_min:y_max, x_min:x_max]
                    
            # Convert back to original dtype
            if image.dtype == np.uint8:
                roi_image = (roi_image * 255).astype(np.uint8)
                
        except Exception as e:
            logger.warning(f"Snake ROI extraction failed: {e}. Returning original image.")
            roi_image = image
            
        return roi_image, mask
    
    def get_params(self) -> Dict[str, Any]:
        """Get snake parameters."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "max_iterations": self.max_iterations,
            "convergence": self.convergence,
            "edge_threshold": self.edge_threshold
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