from pathlib import Path
import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

logger = logging.getLogger(__name__)


class KneeUltrasoundDataset(Dataset):
    """Dataset loading ultrasound images và masks
    
    Args:
        image_dir: thư mục ảnh ultrasound
        mask_dir: nhãn
        image_ids: list IDs để load (None = load all)
        preprocessing: transform preprocessing
        augmentation
        resize: resize về size (H, W)
    """
    
    def __init__(self, image_dir, mask_dir, image_ids=None, preprocessing=None, 
                 augmentation=None, resize=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.resize = resize
        
        # Tìm image IDs
        if image_ids:
            self.image_ids = image_ids
        else:
            self.image_ids = []
            for img_path in self.image_dir.glob("*.png"):
                img_id = img_path.stem
                mask_path = self.mask_dir / f"{img_id}_mask.png"
                if mask_path.exists():
                    self.image_ids.append(img_id)
        
        logger.info(f"Found {len(self.image_ids)} image-mask pairs")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Load và process 1 sample: return (image_tensor, mask_tensor)"""
        image_id = self.image_ids[idx]
        
        image = self._load_image(image_id)
        mask = self._load_mask(image_id)
        
        if self.resize:
            image = cv2.resize(image, (self.resize[1], self.resize[0]))
            mask = cv2.resize(mask, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_NEAREST)
        
        # Preprocessing
        if self.preprocessing:
            image, mask = self.preprocessing(image, mask)
        
        # Augmentation
        if self.augmentation:
            transformed = self.augmentation(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        
        # Convert to tensors
        image_tensor = self._to_tensor(image)
        mask_tensor = self._mask_to_tensor(mask)
        
        return image_tensor, mask_tensor
    
    def _load_image(self, image_id):
        """Load ultrasound image grayscale"""
        image_path = self.image_dir / f"{image_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        return image
    
    def _load_mask(self, image_id):
        """Load segmentation mask"""
        mask_path = self.mask_dir / f"{image_id}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # Binary mask: >0 = positive
        mask = (mask > 0).astype(np.uint8) * 255
        return mask
    
    def _to_tensor(self, image):
        """Convert image to tensor: (H, W) -> (1, H, W)"""
        # Handle different shapes
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        elif len(image.shape) == 3 and image.shape[0] not in [1, 3]:
            image = image.transpose(2, 0, 1)
        
        # Normalize to [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        return torch.from_numpy(image.copy())
    
    def _mask_to_tensor(self, mask):
        """Convert mask to tensor: (H, W) -> (1, H, W)"""
        mask = (mask > 0).astype(np.float32)
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]
        return torch.from_numpy(mask.copy())


def get_training_augmentation(
    rotation_limit=0,     # Không xoay cho medical images
    scale_limit=0.02,     # Scale nhẹ ±2%
    shift_limit=0.03,     # Shift nhẹ ±3%
    p_flip=0.5,
    p_scale=0.3
):
    """Augmentation cho training"""
    return A.Compose([
        # Flips
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=0.3),
        
        # Shift + scale nhẹ
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=0,  # Không xoay
            p=p_scale,
            border_mode=cv2.BORDER_CONSTANT
        ),
        
        # Brightness/contrast chỉ ảnh, không mask
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.4
        ),
    ])


def get_validation_augmentation():
    """Không augmen cho val"""
    return None 