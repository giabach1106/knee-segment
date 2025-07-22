from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging
import cv2
import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.draw import polygon2mask

logger = logging.getLogger(__name__)


class Transform(ABC):
    """Base class cho transforms"""
    
    @abstractmethod
    def __call__(self, image, mask=None):
        pass
    
    def get_params(self):
        return {}


class CLAHETransform(Transform):
    """CLAHE
    Cải thiện local contrast ultrasound"""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, image, mask=None):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced_image = clahe.apply(image)
        return enhanced_image, mask


class FixedCrop(Transform):
    """Crop cố định để focus vào knee joint ROI"""
    
    def __init__(self, crop_top=62, crop_bottom=94, crop_left=84, crop_right=44):
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        
    def __call__(self, image, mask=None):
        h, w = image.shape[:2]
        
        y1 = max(0, self.crop_top)
        y2 = min(h, h - self.crop_bottom)
        x1 = max(0, self.crop_left)
        x2 = min(w, w - self.crop_right)
        
        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2] if mask is not None else None
        
        return cropped_image, cropped_mask


class SnakeROI(Transform):
    """Active contour (Snake) ROI extraction cho ultrasound
    
    SnakeROI dùng active contour algorithm để tự động tìm và crop vùng anatomical
    ROI bằng cách detect boundary giữa ultrasound content và black background.
    Phương pháp này robust hơn simple thresholding vì có thể handle dark regions
    trong anatomy.
    
    Quy trình:
    1. Detect black border regions qua thresholding + connectivity analysis
    2. Initialize snake ellipse quanh detected content region  
    3. Evolve snake để fit ROI boundary dùng edge + smoothness forces
    4. Mask và crop image về extracted ROI
    
    Args:
        alpha: Snake tension (elasticity/contraction). Thấp hơn = expand nhiều hơn
        beta: Snake rigidity (smoothness). Cao hơn = contour mượt hơn  
        w_edge: Weight cho edge attraction forces
        gamma: Time stepping parameter
        max_iterations: Max snake evolution iterations
        intensity_threshold: Threshold detect black pixels
        blur_sigma: Gaussian blur sigma cho preprocessing
        snake_points: Số points trong snake contour
        content_padding: Padding factor cho snake init (0.45 = 90% của region)
    """
    
    def __init__(self, alpha=0.01, beta=5.0, w_edge=1.0, gamma=0.1, 
                 max_iterations=1000, intensity_threshold=10, blur_sigma=2.0,
                 snake_points=200, content_padding=0.45):
        self.alpha = alpha
        self.beta = beta  
        self.w_edge = w_edge
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.intensity_threshold = intensity_threshold
        self.blur_sigma = blur_sigma
        self.snake_points = snake_points
        self.content_padding = content_padding
        
    def __call__(self, image, mask=None):
        try:
            # Detect content region (không phải black border)
            content_mask = self._detect_content_region(image)
            
            # Tìm bounding box content
            coords = np.column_stack(np.nonzero(content_mask))
            if len(coords) == 0:
                return image, mask
                
            top_left = coords.min(axis=0)
            bottom_right = coords.max(axis=0) 
            r0, c0 = top_left
            r1, c1 = bottom_right
            
            # Crop để efficiency
            margin = 5
            r0m = max(r0 - margin, 0)
            r1m = min(r1 + margin, image.shape[0])
            c0m = max(c0 - margin, 0) 
            c1m = min(c1 + margin, image.shape[1])
            
            image_cropped = image[r0m:r1m, c0m:c1m]
            if image_cropped.size == 0:
                return image, mask
            
            # Run active contour
            snake = self._run_active_contour(image_cropped)
            
            # Apply snake crop và resize về original size
            final_image, final_mask = self._apply_snake_crop(
                image, mask, snake, (r0m, c0m), image_cropped.shape
            )
            
            return final_image, final_mask
            
        except:
            logger.warning("SnakeROI failed, returning original")
            return image, mask
    
    def _detect_content_region(self, image):
        """Detect content region bằng cách remove black border + small objects"""
        content_mask = image > self.intensity_threshold
        content_mask = remove_small_objects(content_mask, min_size=1000)
        content_mask = remove_small_holes(content_mask, area_threshold=2000)
        return content_mask
    
    def _initialize_snake(self, image_shape):
        """Initialize snake như ellipse trong image bounds"""
        rows, cols = image_shape
        center_y, center_x = rows / 2, cols / 2
        
        radius_y = rows * self.content_padding
        radius_x = cols * self.content_padding
        
        theta = np.linspace(0, 2 * np.pi, self.snake_points)
        init_y = center_y + radius_y * np.sin(theta)
        init_x = center_x + radius_x * np.cos(theta)
        
        return np.vstack([init_y, init_x]).T
    
    def _run_active_contour(self, image_cropped):
        """Run active contour evolution"""
        img_float = image_cropped.astype(float) / 255.0
        blurred = gaussian(img_float, sigma=self.blur_sigma)
        
        init_snake = self._initialize_snake(blurred.shape)
        
        snake = active_contour(
            blurred, init_snake,
            alpha=self.alpha, beta=self.beta, 
            w_edge=self.w_edge, gamma=self.gamma,
            max_num_iter=self.max_iterations
        )
        return snake
    
    def _apply_snake_crop(self, original_image, original_mask, snake, 
                         crop_offset, cropped_shape):
        """Apply snake crop và resize về original dimensions"""
        original_height, original_width = original_image.shape[:2]
        
        # Create snake mask
        snake_mask = polygon2mask(cropped_shape, snake)
        
        # Get bounding box
        ys, xs = np.nonzero(snake_mask)
        if len(ys) == 0:
            return original_image, original_mask
        
        miny, maxy = ys.min(), ys.max()
        minx, maxx = xs.min(), xs.max()
        
        # Adjust coordinates về original image space
        r0, c0 = crop_offset
        final_r0 = max(0, r0 + miny)
        final_r1 = min(original_image.shape[0], r0 + maxy + 1)
        final_c0 = max(0, c0 + minx)  
        final_c1 = min(original_image.shape[1], c0 + maxx + 1)
        
        # Extract ROI
        roi_image = original_image[final_r0:final_r1, final_c0:final_c1].copy()
        
        if roi_image.size == 0:
            return original_image, original_mask
        
        # Apply snake mask
        crop_snake_mask = snake_mask[miny:maxy+1, minx:maxx+1]
        if crop_snake_mask.shape == roi_image.shape:
            roi_image[~crop_snake_mask] = 0
        
        # QUAN TRỌNG: Resize về original size để consistency
        final_image = cv2.resize(roi_image, (original_width, original_height))
        
        final_mask = None
        if original_mask is not None:
            roi_mask = original_mask[final_r0:final_r1, final_c0:final_c1]
            if roi_mask.size > 0:
                final_mask = cv2.resize(roi_mask, (original_width, original_height), 
                                      interpolation=cv2.INTER_NEAREST)
            else:
                final_mask = original_mask
        
        return final_image, final_mask


class HistogramEqualization(Transform):
    """Global histogram equalization"""
    
    def __call__(self, image, mask=None):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        equalized = cv2.equalizeHist(image)
        return equalized, mask


class GaussianBlur(Transform):
    """Gaussian blur để reduce speckle noise"""
    
    def __init__(self, kernel_size=3, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __call__(self, image, mask=None):
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        return blurred, mask


class Sharpening(Transform):
    """Unsharp masking để enhance edges"""
    
    def __init__(self, alpha=1.5, sigma=1.0):
        self.alpha = alpha
        self.sigma = sigma
        
    def __call__(self, image, mask=None):
        blurred = cv2.GaussianBlur(image, (0, 0), self.sigma)
        sharpened = cv2.addWeighted(image, 1 + self.alpha, blurred, -self.alpha, 0)
        
        if image.dtype == np.uint8:
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            sharpened = np.clip(sharpened, 0, 1)
            
        return sharpened, mask


class IntensityNormalization(Transform):
    """Intensity normalization: minmax hoặc zscore"""
    
    def __init__(self, method="minmax", target_range=(0, 1)):
        self.method = method
        self.target_range = target_range
        
    def __call__(self, image, mask=None):
        image_float = image.astype(np.float32)
        
        if self.method == "minmax":
            min_val, max_val = image_float.min(), image_float.max()
            if max_val > min_val:
                normalized = (image_float - min_val) / (max_val - min_val)
                range_min, range_max = self.target_range
                normalized = normalized * (range_max - range_min) + range_min
            else:
                normalized = np.zeros_like(image_float) + self.target_range[0]
        else:  # zscore
            mean, std = image_float.mean(), image_float.std()
            normalized = (image_float - mean) / std if std > 0 else image_float - mean
                
        return normalized.astype(np.float32), mask


class MultiChannelTransform(Transform):
    """Tạo multi-channel input bằng cách stack processed versions"""
    
    def __init__(self, transforms, include_original=True):
        self.transforms = transforms
        self.include_original = include_original
        
    def __call__(self, image, mask=None):
        """Tạo multi-channel: (H, W) -> (C, H, W)"""
        channels = []
        
        if self.include_original:
            channels.append(image)
            
        for transform in self.transforms:
            transformed_img, _ = transform(image, None)
            channels.append(transformed_img)
            
        # Stack thành multi-channel
        multi_channel = np.stack(channels, axis=0)
        return multi_channel, mask


class PreprocessingPipeline:
    """Pipeline gộm nhiều transforms"""
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, mask=None):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask 