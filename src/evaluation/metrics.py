import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def dice_coefficient(pred, target, smooth=1e-6, threshold=0.5):
    """Dice coefficient: Dice = 2 * |P ∩ G| / (|P| + |G|)"""
    if threshold is not None:
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > threshold).float()
    
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.mean()


def iou_score(pred, target, smooth=1e-6, threshold=0.5):
    """IoU = |P ∩ G| / |P ∪ G|"""
    if threshold is not None:
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > threshold).float()
    
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def pixel_accuracy(pred, target, threshold=0.5):
    """Pixel accuracy = correct pixels / total pixels"""
    if threshold is not None:
        pred = torch.sigmoid(pred) if pred.requires_grad else pred
        pred = (pred > threshold).float()
    
    correct = (pred == target).float()
    return correct.mean()


def compute_metrics(pred, target, threshold=0.5):
    """Tính tất cả metrics: (B, 1, H, W) -> dict scores"""
    with torch.no_grad():
        dice = dice_coefficient(pred, target, threshold=threshold)
        iou = iou_score(pred, target, threshold=threshold)
        pa = pixel_accuracy(pred, target, threshold=threshold)
    
    return {
        "dice": dice.item(),
        "iou": iou.item(), 
        "pixel_accuracy": pa.item()
    }


def evaluate_model(model, data_loader, device, threshold=0.5):
    """Evaluate model on dataset."""
    model.eval()
    total_dice = total_iou = total_pa = num_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            metrics = compute_metrics(outputs, masks, threshold)
            batch_size = images.size(0)
            
            total_dice += metrics["dice"] * batch_size
            total_iou += metrics["iou"] * batch_size  
            total_pa += metrics["pixel_accuracy"] * batch_size
            num_samples += batch_size
    
    avg_metrics = {
        "dice": total_dice / num_samples,
        "iou": total_iou / num_samples,
        "pixel_accuracy": total_pa / num_samples,
        "dice_percent": (total_dice / num_samples) * 100,
        "iou_percent": (total_iou / num_samples) * 100,
        "pixel_accuracy_percent": (total_pa / num_samples) * 100
    }
    
    return avg_metrics, None


class DiceLoss(torch.nn.Module):
    """Dice Loss = 1 - Dice Coefficient"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class CombinedLoss(torch.nn.Module):
    """BCE + Dice loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice 