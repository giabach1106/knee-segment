"""PyTorch dataset for knee ultrasound segmentation.

This module provides a custom Dataset class for loading ultrasound images
and their corresponding segmentation masks.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A

logger = logging.getLogger(__name__)


class KneeUltrasoundDataset(Dataset):
    """Dataset for knee ultrasound images and segmentation masks.
    
    This dataset handles loading of ultrasound images and their corresponding
    binary masks. It supports preprocessing transforms and data augmentation.
    
    Args:
        image_dir: Path to directory containing ultrasound images.
        mask_dir: Path to directory containing segmentation masks.
        image_ids: Optional list of image IDs to include. If None, uses all images.
        preprocessing: Optional preprocessing transform to apply.
        augmentation: Optional augmentation transform to apply.
        resize: Optional target size for resizing images (height, width).
        
    Example:
        >>> dataset = KneeUltrasoundDataset(
        ...     image_dir="image/raw",
        ...     mask_dir="image/mask",
        ...     preprocessing=CLAHETransform(),
        ...     augmentation=get_training_augmentation()
        ... )
        >>> image, mask = dataset[0]
    """
    
    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        image_ids: Optional[List[str]] = None,
        preprocessing: Optional[Callable] = None,
        augmentation: Optional[A.Compose] = None,
        resize: Optional[Tuple[int, int]] = None
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.resize = resize
        
        # Get list of images
        if image_ids is not None:
            self.image_ids = image_ids
        else:
            # Find all images in the directory
            self.image_ids = []
            for img_path in self.image_dir.glob("*.png"):
                # Extract ID without extension
                img_id = img_path.stem
                # Check if corresponding mask exists
                mask_path = self.mask_dir / f"{img_id}_mask.png"
                if mask_path.exists():
                    self.image_ids.append(img_id)
                else:
                    logger.warning(f"No mask found for image {img_id}")
                    
        if len(self.image_ids) == 0:
            raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
            
        logger.info(f"Found {len(self.image_ids)} valid image-mask pairs")
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Tuple of (image_tensor, mask_tensor).
        """
        # Get image ID
        image_id = self.image_ids[idx]
        
        # Load image and mask
        image = self._load_image(image_id)
        mask = self._load_mask(image_id)
        
        # Apply resize if specified
        if self.resize is not None:
            image = cv2.resize(image, (self.resize[1], self.resize[0]))
            mask = cv2.resize(mask, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_NEAREST)
            
        # Apply preprocessing
        if self.preprocessing is not None:
            image, mask = self.preprocessing(image, mask)
            
        # Apply augmentation
        if self.augmentation is not None:
            transformed = self.augmentation(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        # Convert to tensors
        image_tensor = self._to_tensor(image)
        mask_tensor = self._mask_to_tensor(mask)
        
        return image_tensor, mask_tensor
    
    def _load_image(self, image_id: str) -> np.ndarray:
        """Load an ultrasound image.
        
        Args:
            image_id: ID of the image to load.
            
        Returns:
            Image as numpy array.
        """
        image_path = self.image_dir / f"{image_id}.png"
        
        # Load as grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        return image
    
    def _load_mask(self, image_id: str) -> np.ndarray:
        """Load a segmentation mask.
        
        Args:
            image_id: ID of the image whose mask to load.
            
        Returns:
            Mask as numpy array.
        """
        mask_path = self.mask_dir / f"{image_id}_mask.png"
        
        # Load as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
            
        # Ensure binary mask (0 or 255)
        # Use adaptive threshold - any non-zero value is positive
        mask = (mask > 0).astype(np.uint8) * 255
        
        return mask
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor.
        
        Args:
            image: Image as numpy array.
            
        Returns:
            Image as tensor with shape (C, H, W).
        """
        # Handle different input shapes
        if len(image.shape) == 2:
            # Grayscale image (H, W) -> (1, H, W)
            image = image[np.newaxis, ...]
        elif len(image.shape) == 3 and image.shape[0] not in [1, 3]:
            # Multi-channel from preprocessing (H, W, C) -> (C, H, W)
            image = image.transpose(2, 0, 1)
            
        # Convert to float tensor
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            
        return torch.from_numpy(image)
    
    def _mask_to_tensor(self, mask: np.ndarray) -> torch.Tensor:
        """Convert mask to tensor.
        
        Args:
            mask: Mask as numpy array.
            
        Returns:
            Mask as tensor with shape (1, H, W).
        """
        # Convert to binary (0 or 1) 
        # Use adaptive threshold - any non-zero value is positive
        mask = (mask > 0).astype(np.float32)
        
        # Add channel dimension
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]
            
        return torch.from_numpy(mask)
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a sample without loading it.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Dictionary with sample information.
        """
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}.png"
        mask_path = self.mask_dir / f"{image_id}_mask.png"
        
        # Get image shape without fully loading
        image_info = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image_info is not None:
            shape = image_info.shape
        else:
            shape = None
            
        return {
            "index": idx,
            "image_id": image_id,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "shape": shape
        }


def get_training_augmentation(
    rotation_limit: int = 15,
    scale_limit: float = 0.1,
    shift_limit: float = 0.1,
    elastic_alpha: float = 120,
    elastic_sigma: float = 9,
    p_flip: float = 0.5,
    p_rotate: float = 0.5,
    p_scale: float = 0.5,
    p_elastic: float = 0.3
) -> A.Compose:
    """Get augmentation pipeline for training.
    
    Args:
        rotation_limit: Maximum rotation angle in degrees.
        scale_limit: Maximum scaling factor.
        shift_limit: Maximum shift as fraction of image size.
        elastic_alpha: Alpha parameter for elastic transform.
        elastic_sigma: Sigma parameter for elastic transform.
        p_flip: Probability of applying flips.
        p_rotate: Probability of applying rotation.
        p_scale: Probability of applying scaling.
        p_elastic: Probability of applying elastic deformation.
        
    Returns:
        Albumentations composition of transforms.
    """
    return A.Compose([
        # Flips
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=p_flip),
        
        # Rotation
        A.Rotate(limit=rotation_limit, p=p_rotate, border_mode=cv2.BORDER_CONSTANT),
        
        # Scaling and shifting
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=0,  # Already handled by Rotate
            p=p_scale,
            border_mode=cv2.BORDER_CONSTANT
        ),
        
        # Elastic deformation (as in original U-Net paper)
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            alpha_affine=0,
            p=p_elastic
        ),
        
        # Small random brightness/contrast changes
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),
        
        # Small amount of Gaussian noise
        A.GaussNoise(var_limit=(0, 0.05), p=0.2),
    ])


def get_validation_augmentation() -> Optional[A.Compose]:
    """Get augmentation pipeline for validation/test (usually none).
    
    Returns:
        None or minimal augmentation composition.
    """
    # No augmentation for validation/test
    return None 