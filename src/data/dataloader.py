"""DataLoader utilities for knee ultrasound segmentation.

This module provides functions for creating data loaders with
stratified train/validation/test splits.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from .dataset import KneeUltrasoundDataset, get_training_augmentation, get_validation_augmentation

logger = logging.getLogger(__name__)


def stratified_split(
    image_ids: List[str],
    mask_dir: Union[str, Path],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    stratify_by_mask_presence: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """Perform stratified train/validation/test split.
    
    Stratification ensures balanced distribution of samples with/without
    inflamed regions across all splits.
    
    Args:
        image_ids: List of all image IDs.
        mask_dir: Directory containing mask files.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        random_state: Random seed for reproducibility.
        stratify_by_mask_presence: Whether to stratify by presence of positive pixels in mask.
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    mask_dir = Path(mask_dir)
    
    # Calculate stratification labels if requested
    if stratify_by_mask_presence:
        stratify_labels = []
        for img_id in image_ids:
            mask_path = mask_dir / f"{img_id}_mask.png"
            if mask_path.exists():
                # Check if mask has any positive pixels
                import cv2
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                has_positive = np.any(mask > 0)
                stratify_labels.append(int(has_positive))
            else:
                # If mask doesn't exist, assume no positive pixels
                stratify_labels.append(0)
                logger.warning(f"Mask not found for stratification: {mask_path}")
    else:
        stratify_labels = None
        
    # Check if we have enough samples for proper stratification
    if stratify_by_mask_presence:
        positive_count = sum(stratify_labels)
        negative_count = len(stratify_labels) - positive_count
        
        logger.info(f"Dataset composition: {positive_count} positive, {negative_count} negative masks")
        
        # Ensure each split gets at least one positive and one negative sample
        min_samples_needed = 3  # At least 1 for each split
        if positive_count < min_samples_needed or negative_count < min_samples_needed:
            logger.warning(f"Insufficient samples for stratification. Switching to random split.")
            stratify_labels = None
            stratify_by_mask_presence = False
    
    # First split: train+val vs test
    train_val_size = train_ratio + val_ratio
    
    try:
        train_val_ids, test_ids = train_test_split(
            image_ids,
            test_size=test_ratio,
            random_state=random_state,
            stratify=stratify_labels if stratify_by_mask_presence else None
        )
    except ValueError as e:
        logger.warning(f"Stratification failed: {e}. Using random split.")
        train_val_ids, test_ids = train_test_split(
            image_ids,
            test_size=test_ratio,
            random_state=random_state,
            stratify=None
        )
        stratify_by_mask_presence = False
    
    # Second split: train vs val
    val_size_adjusted = val_ratio / train_val_size
    if stratify_by_mask_presence:
        # Get stratify labels for train_val subset
        train_val_indices = [image_ids.index(id) for id in train_val_ids]
        train_val_labels = [stratify_labels[i] for i in train_val_indices]
        
        # Check if we still have both classes in train_val
        if len(set(train_val_labels)) < 2:
            logger.warning("Train+val subset has only one class. Using random split.")
            train_val_labels = None
    else:
        train_val_labels = None
        
    try:
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_labels
        )
    except ValueError as e:
        logger.warning(f"Train/val stratification failed: {e}. Using random split.")
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=None
        )
    
    logger.info(f"Split sizes - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Log split composition for debugging
    if stratify_labels is not None:
        def count_positives(ids):
            indices = [image_ids.index(id) for id in ids]
            return sum(stratify_labels[i] for i in indices)
        
        train_pos = count_positives(train_ids)
        val_pos = count_positives(val_ids) 
        test_pos = count_positives(test_ids)
        
        logger.info(f"Positive masks - Train: {train_pos}/{len(train_ids)}, "
                   f"Val: {val_pos}/{len(val_ids)}, Test: {test_pos}/{len(test_ids)}")
        
        # Warn if any split has no positive examples
        if val_pos == 0:
            logger.warning("⚠️ Validation set has NO positive masks! This may cause issues.")
        if test_pos == 0:
            logger.warning("⚠️ Test set has NO positive masks! This may cause issues.")
    
    return train_ids, val_ids, test_ids


def create_data_loaders(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: Optional[Tuple[int, int]] = (256, 256),
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    preprocessing_transform: Optional[Any] = None,
    enable_augmentation: bool = True,
    augmentation_params: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[str]]]:
    """Create train, validation, and test data loaders.
    
    Args:
        image_dir: Directory containing ultrasound images.
        mask_dir: Directory containing segmentation masks.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.
        image_size: Target size for resizing images (height, width).
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        random_state: Random seed for reproducibility.
        preprocessing_transform: Optional preprocessing to apply to all splits.
        enable_augmentation: Whether to apply augmentation to training data.
        augmentation_params: Optional parameters for augmentation.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, split_info).
        split_info contains the image IDs for each split.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    # Get all available image IDs
    dataset = KneeUltrasoundDataset(image_dir, mask_dir)
    all_image_ids = dataset.image_ids
    
    # Perform stratified split
    train_ids, val_ids, test_ids = stratified_split(
        all_image_ids,
        mask_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    
    # Get augmentation
    if enable_augmentation:
        if augmentation_params is None:
            train_augmentation = get_training_augmentation()
        else:
            train_augmentation = get_training_augmentation(**augmentation_params)
    else:
        train_augmentation = None
        
    val_augmentation = get_validation_augmentation()
    
    # Create datasets
    train_dataset = KneeUltrasoundDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_ids=train_ids,
        preprocessing=preprocessing_transform,
        augmentation=train_augmentation,
        resize=image_size
    )
    
    val_dataset = KneeUltrasoundDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_ids=val_ids,
        preprocessing=preprocessing_transform,
        augmentation=val_augmentation,
        resize=image_size
    )
    
    test_dataset = KneeUltrasoundDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_ids=test_ids,
        preprocessing=preprocessing_transform,
        augmentation=val_augmentation,
        resize=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Split info for reproducibility
    split_info = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "total_samples": len(all_image_ids),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids)
    }
    
    return train_loader, val_loader, test_loader, split_info 