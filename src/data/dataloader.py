
from pathlib import Path
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from .dataset import KneeUltrasoundDataset, get_training_augmentation, get_validation_augmentation

logger = logging.getLogger(__name__)


def stratified_split(image_ids, mask_dir, train_ratio=0.6, val_ratio=0.2, 
                    test_ratio=0.2, random_state=42):
    """Train/val/test split với stratification đơn giản"""
    mask_dir = Path(mask_dir)
    
    # Kiểm tra mask có positive pixels không
    stratify_labels = []
    for img_id in image_ids:
        mask_path = mask_dir / f"{img_id}_mask.png"
        if mask_path.exists():
            import cv2
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            has_positive = np.any(mask > 0)
            stratify_labels.append(int(has_positive))
        else:
            stratify_labels.append(0)
    
    positive_count = sum(stratify_labels)
    negative_count = len(stratify_labels) - positive_count
    
    logger.info(f"Dataset: {positive_count} positive, {negative_count} negative masks")
    
    # Stratify nếu có đủ samples
    stratify = stratify_labels if min(positive_count, negative_count) >= 2 else None
    
    # Split train+val vs test
    train_val_size = train_ratio + val_ratio
    train_val_ids, test_ids = train_test_split(
        image_ids, test_size=test_ratio, random_state=random_state, stratify=stratify
    )
    
    # Split train vs val
    val_size_adjusted = val_ratio / train_val_size
    if stratify:
        train_val_indices = [image_ids.index(id) for id in train_val_ids]
        train_val_labels = [stratify_labels[i] for i in train_val_indices]
        stratify_val = train_val_labels if len(set(train_val_labels)) > 1 else None
    else:
        stratify_val = None
        
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size_adjusted, random_state=random_state, 
        stratify=stratify_val
    )
    
    logger.info(f"Split sizes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return train_ids, val_ids, test_ids


def create_data_loaders(image_dir, mask_dir, batch_size=4, num_workers=4,
                       image_size=(256, 256), train_ratio=0.6, val_ratio=0.2, 
                       test_ratio=0.2, random_state=42, preprocessing_transform=None,
                       enable_augmentation=True, augmentation_params=None):
    """Tạo train/val/test data loaders"""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    # Lấy tất cả image IDs
    dataset = KneeUltrasoundDataset(image_dir, mask_dir)
    all_image_ids = dataset.image_ids
    
    # Stratified split
    train_ids, val_ids, test_ids = stratified_split(
        all_image_ids, mask_dir, train_ratio=train_ratio, 
        val_ratio=val_ratio, test_ratio=test_ratio, random_state=random_state
    )
    
    # Augmentation
    if enable_augmentation:
        train_augmentation = get_training_augmentation(**(augmentation_params or {}))
    else:
        train_augmentation = None
    val_augmentation = get_validation_augmentation()
    
    # Datasets
    train_dataset = KneeUltrasoundDataset(
        image_dir=image_dir, mask_dir=mask_dir, image_ids=train_ids,
        preprocessing=preprocessing_transform, augmentation=train_augmentation, resize=image_size
    )
    
    val_dataset = KneeUltrasoundDataset(
        image_dir=image_dir, mask_dir=mask_dir, image_ids=val_ids,
        preprocessing=preprocessing_transform, augmentation=val_augmentation, resize=image_size
    )
    
    test_dataset = KneeUltrasoundDataset(
        image_dir=image_dir, mask_dir=mask_dir, image_ids=test_ids,
        preprocessing=preprocessing_transform, augmentation=val_augmentation, resize=image_size
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    # Split info
    split_info = {
        "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
        "total_samples": len(all_image_ids), "train_samples": len(train_ids),
        "val_samples": len(val_ids), "test_samples": len(test_ids)
    }
    
    return train_loader, val_loader, test_loader, split_info 