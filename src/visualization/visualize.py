
from pathlib import Path
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_segmentation_results(image, ground_truth, prediction, title=None,
                            dice_score=None, iou_score=None, save_path=None, figsize=(15, 5)):
    """Plot segmentation: original, GT, prediction, overlay"""
    # Handle shapes
    if len(image.shape) == 3:
        image = image.squeeze()
    if len(ground_truth.shape) == 3:
        ground_truth = ground_truth.squeeze()
    if len(prediction.shape) == 3:
        prediction = prediction.squeeze()
        
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
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
    
    # Title với metrics
    if title:
        metrics_str = title
        if dice_score is not None:
            metrics_str += f" | Dice: {dice_score:.3f}"
        if iou_score is not None:
            metrics_str += f" | IoU: {iou_score:.3f}"
        fig.suptitle(metrics_str, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def create_overlay(image, ground_truth, prediction, alpha=0.5):
    """Tạo overlay image: GT=green, pred=red, overlap=yellow"""
    # Normalize image
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
        
    # Convert to RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
        
    # Create overlay
    overlay = image_rgb.copy()
    
    # GT = green, pred = red, overlap = yellow
    gt_mask = ground_truth > 0.5
    pred_mask = prediction > 0.5
    
    overlay[gt_mask] = (0, 255, 0)
    overlay[pred_mask] = (255, 0, 0)
    overlay[gt_mask & pred_mask] = (255, 255, 0)
    
    # Blend
    result = cv2.addWeighted(image_rgb, 1-alpha, overlay, alpha, 0)
    return result


def plot_metrics_comparison(experiments_data, metric_names=["dice", "iou", "pixel_accuracy"],
                          title="Experiment Comparison", save_path=None, figsize=(12, 8)):
    """Bar plot comparing metrics across experiments"""
    # Convert to DataFrame
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
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Grouped bar plot
    x_pos = np.arange(len(experiments_data))
    width = 0.25
    
    for i, metric in enumerate(metric_names):
        metric_label = metric.replace("_", " ").title()
        metric_data = df[df["Metric"] == metric_label]["Value"].values
        offset = (i - len(metric_names)/2 + 0.5) * width
        bars = ax.bar(x_pos + offset, metric_data, width, label=metric_label)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score (%)')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(experiments_data.keys()), rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def create_experiment_comparison_plot(baseline_metrics, experiment_metrics, 
                                    save_path=None, figsize=(14, 8)):
    """Comprehensive comparison với baseline"""
    all_experiments = {"Baseline": baseline_metrics}
    all_experiments.update(experiment_metrics)
    
    fig = plt.figure(figsize=figsize)
    
    # Main comparison
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
    
    improvement_df = pd.DataFrame(improvements).T
    improvement_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Improvement over Baseline (%)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Best performers
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
    
    ax3.axis('off')
    summary_text = "Best Performers:\n\n"
    for metric, data in best_by_metric.items():
        summary_text += f"{metric}: {data['Experiment']} ({data['Score']:.2f}%)\n"
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.suptitle("Comprehensive Experiment Comparison", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_training_history(history, save_path=None, figsize=(12, 8)):
    """Plot training history: loss, dice, IoU, LR"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    # Loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice
    if 'train_dice' in history:
        axes[1].plot(history['train_dice'], label='Train')
    if 'val_dice' in history:
        axes[1].plot(history['val_dice'], label='Validation')
    axes[1].set_title('Dice Coefficient')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # IoU
    if 'train_iou' in history:
        axes[2].plot(history['train_iou'], label='Train')
    if 'val_iou' in history:
        axes[2].plot(history['val_iou'], label='Validation')
    axes[2].set_title('IoU Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        axes[3].plot(history['learning_rate'])
        axes[3].set_title('Learning Rate')
        axes[3].set_yscale('log')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].axis('off')
    
    plt.suptitle('Training History')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig 