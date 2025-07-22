"""Visualization utilities for segmentation results.

This module provides functions for visualizing segmentation predictions,
creating comparison plots, and generating metric charts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd

logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_segmentation_results(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    title: Optional[str] = None,
    dice_score: Optional[float] = None,
    iou_score: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """Plot segmentation results with original image, ground truth, and prediction.
    
    Args:
        image: Original ultrasound image (H, W) or (1, H, W).
        ground_truth: Ground truth mask (H, W) or (1, H, W).
        prediction: Predicted mask (H, W) or (1, H, W).
        title: Title for the plot.
        dice_score: Dice coefficient score to display.
        iou_score: IoU score to display.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure object.
    """
    # Handle different input shapes
    if len(image.shape) == 3:
        image = image.squeeze()
    if len(ground_truth.shape) == 3:
        ground_truth = ground_truth.squeeze()
    if len(prediction.shape) == 3:
        prediction = prediction.squeeze()
        
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(ground_truth, cmap='binary')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction, cmap='binary')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    overlay = create_overlay(image, ground_truth, prediction)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (GT: Green, Pred: Red)')
    axes[3].axis('off')
    
    # Add metrics to title if provided
    if title:
        metrics_str = title
        if dice_score is not None:
            metrics_str += f" | Dice: {dice_score:.3f}"
        if iou_score is not None:
            metrics_str += f" | IoU: {iou_score:.3f}"
        fig.suptitle(metrics_str, fontsize=14)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
        
    return fig


def create_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Create an overlay image showing ground truth and prediction.
    
    Args:
        image: Original image (H, W).
        ground_truth: Ground truth mask (H, W).
        prediction: Predicted mask (H, W).
        alpha: Transparency for overlay.
        
    Returns:
        RGB overlay image.
    """
    # Normalize image to 0-255 range
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
        
    # Convert to RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
        
    # Create colored overlays
    overlay = image_rgb.copy()
    
    # Green for ground truth
    gt_mask = ground_truth > 0.5
    overlay[gt_mask] = (0, 255, 0)
    
    # Red for prediction
    pred_mask = prediction > 0.5
    overlay[pred_mask] = (255, 0, 0)
    
    # Yellow for overlap (both GT and pred)
    overlap_mask = gt_mask & pred_mask
    overlay[overlap_mask] = (255, 255, 0)
    
    # Blend with original image
    result = cv2.addWeighted(image_rgb, 1-alpha, overlay, alpha, 0)
    
    return result


def save_prediction_overlay(
    image_path: Union[str, Path],
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    output_path: Union[str, Path] = None,
    dice_score: Optional[float] = None,
    iou_score: Optional[float] = None
) -> None:
    """Save prediction overlay on original image.
    
    Args:
        image_path: Path to original image.
        prediction: Predicted mask.
        ground_truth: Optional ground truth mask.
        output_path: Path to save overlay. If None, uses image_path with _overlay suffix.
        dice_score: Optional Dice score to add to filename.
        iou_score: Optional IoU score to add to filename.
    """
    # Load original image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if output_path is None:
        image_path = Path(image_path)
        suffix = "_overlay"
        if dice_score is not None:
            suffix += f"_dice{dice_score:.3f}"
        if iou_score is not None:
            suffix += f"_iou{iou_score:.3f}"
        output_path = image_path.parent / f"{image_path.stem}{suffix}.png"
        
    # Create overlay
    if ground_truth is not None:
        overlay = create_overlay(image, ground_truth, prediction)
    else:
        # Just prediction overlay
        overlay = create_overlay(image, prediction, prediction)
        
    # Save
    cv2.imwrite(str(output_path), overlay)
    logger.info(f"Saved overlay to {output_path}")


def plot_metrics_comparison(
    experiments_data: Dict[str, Dict[str, float]],
    metric_names: List[str] = ["dice", "iou", "pixel_accuracy"],
    title: str = "Experiment Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Create bar plot comparing metrics across experiments.
    
    Args:
        experiments_data: Dictionary mapping experiment names to metrics dictionaries.
        metric_names: List of metric names to plot.
        title: Title for the plot.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure object.
    """
    # Convert to DataFrame for easier plotting
    df_data = []
    for exp_name, metrics in experiments_data.items():
        for metric in metric_names:
            value = metrics.get(f"{metric}_percent", metrics.get(metric, 0) * 100)
            df_data.append({
                "Experiment": exp_name,
                "Metric": metric.replace("_", " ").title(),
                "Value": value
            })
    
    df = pd.DataFrame(df_data)
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create grouped bar plot
    x_pos = np.arange(len(experiments_data))
    width = 0.25
    
    for i, metric in enumerate(metric_names):
        metric_label = metric.replace("_", " ").title()
        metric_data = df[df["Metric"] == metric_label]["Value"].values
        offset = (i - len(metric_names)/2 + 0.5) * width
        bars = ax.bar(x_pos + offset, metric_data, width, label=metric_label)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Customize plot
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(experiments_data.keys()), rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved metrics comparison to {save_path}")
        
    return fig


def create_experiment_comparison_plot(
    baseline_metrics: Dict[str, float],
    experiment_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Create comprehensive comparison plot showing improvement over baseline.
    
    Args:
        baseline_metrics: Metrics for baseline experiment.
        experiment_metrics: Dictionary of experiment names to metrics.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure object.
    """
    # Add baseline to experiments
    all_experiments = {"Baseline": baseline_metrics}
    all_experiments.update(experiment_metrics)
    
    # Create main comparison plot
    fig = plt.figure(figsize=figsize)
    
    # Main metrics comparison
    ax1 = plt.subplot(2, 2, (1, 2))
    plot_metrics_comparison(all_experiments, ax=ax1)
    
    # Improvement over baseline
    ax2 = plt.subplot(2, 2, 3)
    improvements = {}
    for exp_name, metrics in experiment_metrics.items():
        imp_data = {}
        for metric in ["dice", "iou", "pixel_accuracy"]:
            baseline_val = baseline_metrics.get(f"{metric}_percent", 
                                              baseline_metrics.get(metric, 0) * 100)
            exp_val = metrics.get(f"{metric}_percent", 
                                 metrics.get(metric, 0) * 100)
            imp_data[metric] = exp_val - baseline_val
        improvements[exp_name] = imp_data
    
    # Plot improvements
    improvement_df = pd.DataFrame(improvements).T
    improvement_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Improvement over Baseline (%)', fontsize=12)
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Improvement (%)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.legend(title='Metric')
    ax2.grid(True, alpha=0.3)
    
    # Best performing experiments
    ax3 = plt.subplot(2, 2, 4)
    best_by_metric = {}
    for metric in ["dice", "iou", "pixel_accuracy"]:
        best_exp = max(all_experiments.items(), 
                      key=lambda x: x[1].get(f"{metric}_percent", 
                                           x[1].get(metric, 0) * 100))
        best_by_metric[metric.replace("_", " ").title()] = {
            "Experiment": best_exp[0],
            "Score": best_exp[1].get(f"{metric}_percent", 
                                   best_exp[1].get(metric, 0) * 100)
        }
    
    # Create text summary
    ax3.axis('off')
    summary_text = "Best Performing Experiments:\n\n"
    for metric, data in best_by_metric.items():
        summary_text += f"{metric}:\n"
        summary_text += f"  {data['Experiment']}: {data['Score']:.2f}%\n\n"
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle("Comprehensive Experiment Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comprehensive comparison to {save_path}")
        
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot training history including loss and metrics.
    
    Args:
        history: Dictionary containing training history.
        save_path: Path to save the figure.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Dice
    if 'train_dice' in history:
        axes[1].plot(history['train_dice'], label='Train')
    if 'val_dice' in history:
        axes[1].plot(history['val_dice'], label='Validation')
    axes[1].set_title('Dice Coefficient')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot IoU
    if 'train_iou' in history:
        axes[2].plot(history['train_iou'], label='Train')
    if 'val_iou' in history:
        axes[2].plot(history['val_iou'], label='Validation')
    axes[2].set_title('IoU Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if 'learning_rate' in history:
        axes[3].plot(history['learning_rate'])
        axes[3].set_title('Learning Rate')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('LR')
        axes[3].set_yscale('log')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].axis('off')
    
    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training history to {save_path}")
        
    return fig 