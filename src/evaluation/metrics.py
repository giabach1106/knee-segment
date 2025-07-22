"""Evaluation metrics for segmentation model performance.

This module implements common segmentation metrics including Dice coefficient,
Intersection over Union (IoU), and Pixel Accuracy.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    threshold: float = 0.5
) -> torch.Tensor:
    """Calculate Dice coefficient (F1 score) for binary segmentation.
    
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    
    Special handling for empty masks:
    - If both pred and target are empty: Dice = 1.0 (true negative)
    - If only one is empty: Dice = 0.0 (false positive/negative)
    
    Args:
        pred: Predicted segmentation logits or probabilities (B, 1, H, W).
        target: Ground truth binary mask (B, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
        threshold: Threshold for converting probabilities to binary.
        
    Returns:
        Dice coefficient averaged over batch.
    """
    # Apply sigmoid if logits
    if threshold is not None:
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > threshold).float()
    
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Calculate intersection and sums
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    
    # Handle empty masks properly
    dice_scores = []
    for i in range(pred_flat.size(0)):
        p_sum = pred_sum[i]
        t_sum = target_sum[i]
        intersect = intersection[i]
        
        if t_sum == 0 and p_sum == 0:
            # Both empty - perfect match (true negative)
            dice_scores.append(torch.tensor(1.0, device=pred.device))
        elif t_sum == 0 or p_sum == 0:
            # Only one empty - no overlap possible
            dice_scores.append(torch.tensor(0.0, device=pred.device))
        else:
            # Normal case - both have positive pixels
            dice = (2.0 * intersect + smooth) / (p_sum + t_sum + smooth)
            dice_scores.append(dice)
    
    return torch.stack(dice_scores).mean()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    threshold: float = 0.5
) -> torch.Tensor:
    """Calculate Intersection over Union (IoU) for binary segmentation.
    
    IoU = |P ∩ G| / |P ∪ G|
    
    Special handling for empty masks:
    - If both pred and target are empty: IoU = 1.0 (true negative)
    - If only one is empty: IoU = 0.0 (false positive/negative)
    
    Args:
        pred: Predicted segmentation logits or probabilities (B, 1, H, W).
        target: Ground truth binary mask (B, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
        threshold: Threshold for converting probabilities to binary.
        
    Returns:
        IoU score averaged over batch.
    """
    # Apply sigmoid if logits
    if threshold is not None:
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > threshold).float()
    
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    union = pred_sum + target_sum - intersection
    
    # Handle empty masks properly
    iou_scores = []
    for i in range(pred_flat.size(0)):
        p_sum = pred_sum[i]
        t_sum = target_sum[i]
        u = union[i]
        intersect = intersection[i]
        
        if t_sum == 0 and p_sum == 0:
            # Both empty - perfect match (true negative)
            iou_scores.append(torch.tensor(1.0, device=pred.device))
        elif u == 0:
            # Union is 0 but we know one isn't empty - impossible case
            iou_scores.append(torch.tensor(0.0, device=pred.device))
        else:
            # Normal case
            iou = (intersect + smooth) / (u + smooth)
            iou_scores.append(iou)
    
    return torch.stack(iou_scores).mean()


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """Calculate pixel-wise accuracy for binary segmentation.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        pred: Predicted segmentation logits or probabilities (B, 1, H, W).
        target: Ground truth binary mask (B, 1, H, W).
        threshold: Threshold for converting probabilities to binary.
        
    Returns:
        Pixel accuracy averaged over batch.
    """
    # Apply sigmoid if logits
    if threshold is not None:
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > threshold).float()
    
    # Calculate correct predictions
    correct = (pred == target).float()
    
    # Average over all pixels
    accuracy = correct.mean()
    
    return accuracy


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute all segmentation metrics.
    
    Args:
        pred: Predicted segmentation logits or probabilities (B, 1, H, W).
        target: Ground truth binary mask (B, 1, H, W).
        threshold: Threshold for converting probabilities to binary.
        
    Returns:
        Dictionary containing dice, iou, and pixel_accuracy scores.
    """
    with torch.no_grad():
        dice = dice_coefficient(pred, target, threshold=threshold)
        iou = iou_score(pred, target, threshold=threshold)
        pa = pixel_accuracy(pred, target, threshold=threshold)
        
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "pixel_accuracy": pa.item()
    }


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    save_predictions: bool = False
) -> Tuple[Dict[str, float], Optional[List[Dict]]]:
    """Evaluate model on a dataset.
    
    Args:
        model: Segmentation model to evaluate.
        data_loader: DataLoader for the evaluation dataset.
        device: Device to run evaluation on.
        threshold: Threshold for converting probabilities to binary.
        save_predictions: Whether to save predictions for visualization.
        
    Returns:
        Tuple of (metrics_dict, predictions_list).
        metrics_dict contains average metrics over the dataset.
        predictions_list contains per-sample predictions if save_predictions=True.
    """
    model.eval()
    
    total_dice = 0.0
    total_iou = 0.0
    total_pa = 0.0
    num_samples = 0
    
    predictions_list = [] if save_predictions else None
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute metrics
            metrics = compute_metrics(outputs, masks, threshold=threshold)
            
            # Accumulate metrics
            batch_size = images.size(0)
            total_dice += metrics["dice"] * batch_size
            total_iou += metrics["iou"] * batch_size
            total_pa += metrics["pixel_accuracy"] * batch_size
            num_samples += batch_size
            
            # Save predictions if requested
            if save_predictions:
                # Convert to binary predictions
                pred_binary = (torch.sigmoid(outputs) > threshold).float()
                
                for i in range(batch_size):
                    prediction_info = {
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "global_idx": batch_idx * data_loader.batch_size + i,
                        "prediction": pred_binary[i].cpu().numpy(),
                        "ground_truth": masks[i].cpu().numpy(),
                        "dice": dice_coefficient(
                            outputs[i:i+1], masks[i:i+1], threshold=threshold
                        ).item(),
                        "iou": iou_score(
                            outputs[i:i+1], masks[i:i+1], threshold=threshold
                        ).item()
                    }
                    predictions_list.append(prediction_info)
    
    # Calculate average metrics
    avg_metrics = {
        "dice": total_dice / num_samples,
        "iou": total_iou / num_samples,
        "pixel_accuracy": total_pa / num_samples,
        "num_samples": num_samples
    }
    
    # Convert to percentages for readability
    avg_metrics["dice_percent"] = avg_metrics["dice"] * 100
    avg_metrics["iou_percent"] = avg_metrics["iou"] * 100
    avg_metrics["pixel_accuracy_percent"] = avg_metrics["pixel_accuracy"] * 100
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Dice: {avg_metrics['dice_percent']:.2f}%")
    logger.info(f"  IoU: {avg_metrics['iou_percent']:.2f}%")
    logger.info(f"  Pixel Accuracy: {avg_metrics['pixel_accuracy_percent']:.2f}%")
    
    return avg_metrics, predictions_list


class DiceLoss(torch.nn.Module):
    """Dice loss for training segmentation models.
    
    Loss = 1 - Dice Coefficient
    
    Args:
        smooth: Smoothing factor to avoid division by zero.
        sigmoid: Whether to apply sigmoid to predictions.
    """
    
    def __init__(self, smooth: float = 1e-6, sigmoid: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss.
        
        Args:
            pred: Predicted segmentation logits (B, 1, H, W).
            target: Ground truth binary mask (B, 1, H, W).
            
        Returns:
            Dice loss value.
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)
            
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate intersection and sums
        intersection = (pred_flat * target_flat).sum(dim=1)
        pred_sum = pred_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Dice loss
        loss = 1.0 - dice.mean()
        
        return loss


class CombinedLoss(torch.nn.Module):
    """Combined loss using both BCE and Dice loss.
    
    This often works better than either loss alone for segmentation.
    
    Args:
        bce_weight: Weight for BCE loss component.
        dice_weight: Weight for Dice loss component.
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss.
        
        Args:
            pred: Predicted segmentation logits (B, 1, H, W).
            target: Ground truth binary mask (B, 1, H, W).
            
        Returns:
            Combined loss value.
        """
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        combined = self.bce_weight * bce + self.dice_weight * dice
        
        return combined 