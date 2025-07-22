#!/usr/bin/env python3
"""Preprocessing visualization tool for presentation."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.preprocessing import (
    CLAHETransform, FixedCrop, HistogramEqualization, 
    GaussianBlur, Sharpening, IntensityNormalization
)
from src.data import get_training_augmentation


def visualize_preprocessing_pipeline(image_id: str, save_dir: str = "results/preprocessing"):
    """Visualize all preprocessing steps on a sample image."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original image
    img_path = Path("image/raw") / f"{image_id}.png"
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    # Define preprocessing steps
    steps = [
        ("Original", lambda x, _: (x, None)),
        ("CLAHE", CLAHETransform(clip_limit=2.0, tile_grid_size=[8, 8])),
        ("Histogram EQ", HistogramEqualization()),
        ("Gaussian Blur", GaussianBlur(kernel_size=3, sigma=1.0)),
        ("Sharpening", Sharpening(alpha=1.5, sigma=1.0)),
        ("Normalization", IntensityNormalization(method="minmax")),
        ("Fixed Crop", FixedCrop(crop_top=62, crop_bottom=94, crop_left=84, crop_right=44))
    ]
    
    # Apply augmentation
    aug = get_training_augmentation()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    current_img = original.copy()
    
    for i, (name, transform) in enumerate(steps):
        if name == "Original":
            processed_img = current_img
        else:
            processed_img, _ = transform(current_img, None)
            current_img = processed_img
        
        axes[i].imshow(processed_img, cmap='gray')
        axes[i].set_title(f'{name}\nShape: {processed_img.shape}')
        axes[i].axis('off')
    
    # Show augmentation example
    if len(axes) > len(steps):
        if aug:
            aug_result = aug(image=current_img, mask=np.zeros_like(current_img))
            aug_img = aug_result['image']
            axes[len(steps)].imshow(aug_img, cmap='gray')
            axes[len(steps)].set_title(f'Augmented\nShape: {aug_img.shape}')
        axes[len(steps)].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f"preprocessing_pipeline_{image_id}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_dir / f"preprocessing_pipeline_{image_id}.png"


def create_preprocessing_comparison(num_samples: int = 3):
    """Create preprocessing comparison for multiple sample images."""
    image_dir = Path("image/raw")
    image_files = list(image_dir.glob("*.png"))[:num_samples]
    
    results = []
    for img_file in image_files:
        img_id = img_file.stem
        result_path = visualize_preprocessing_pipeline(img_id)
        results.append(result_path)
        print(f"✅ Created preprocessing visualization: {result_path}")
    
    return results


def visualize_experiment_results(experiments_dir: str = "experiments"):
    """Create comprehensive visualization of all experiment results."""
    exp_dir = Path(experiments_dir)
    results_dir = Path("results/comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results from all experiments
    experiment_data = []
    
    for exp_path in exp_dir.glob("*/"):
        if not exp_path.is_dir():
            continue
            
        test_metrics_file = exp_path / "test_metrics.json"
        if test_metrics_file.exists():
            import json
            with open(test_metrics_file) as f:
                metrics = json.load(f)
            
            experiment_data.append({
                'Experiment': exp_path.name.replace('_', ' ').title(),
                'Dice (%)': metrics.get('dice_percent', 0),
                'IoU (%)': metrics.get('iou_percent', 0),
                'Pixel Accuracy (%)': metrics.get('pixel_accuracy_percent', 0),
                'Samples': metrics.get('num_samples', 0)
            })
    
    if not experiment_data:
        print("❌ No experiment results found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(experiment_data)
    df = df.sort_values('Dice (%)', ascending=False)
    
    # Save to CSV for Google Sheets
    csv_path = results_dir / "experiment_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['Dice (%)', 'IoU (%)', 'Pixel Accuracy (%)']
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = axes[i].bar(range(len(df)), df[metric], color=color, alpha=0.7)
        axes[i].set_xlabel('Experiments')
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'Comparison: {metric}')
        axes[i].set_xticks(range(len(df)))
        axes[i].set_xticklabels(df['Experiment'], rotation=45, ha='right')
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(results_dir / "experiment_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = df.round(2).values
    table = ax.table(cellText=table_data, colLabels=df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the best results
    for i in range(len(df)):
        for j in range(1, 4):  # Metrics columns
            if i == 0:  # Best result (sorted by Dice)
                table[(i+1, j)].set_facecolor('#90EE90')  # Light green
            elif df.iloc[i, j] > df.iloc[:, j].mean():
                table[(i+1, j)].set_facecolor('#FFFFE0')  # Light yellow
    
    plt.title("Experiment Results Summary", fontsize=16, fontweight='bold', pad=20)
    plt.savefig(results_dir / "results_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Results saved to: {results_dir}")
    print(f"📊 CSV file: {csv_path}")
    print(f"📈 Charts: experiment_comparison.png, results_table.png")
    
    return df, results_dir


def create_presentation_report():
    """Create comprehensive presentation materials."""
    print("🎯 Creating Presentation Materials")
    print("=" * 50)
    
    # 1. Preprocessing visualizations
    print("\n📸 Creating preprocessing visualizations...")
    preprocessing_results = create_preprocessing_comparison(num_samples=3)
    
    # 2. Experiment results comparison
    print("\n📊 Creating experiment comparison...")
    df, results_dir = visualize_experiment_results()
    
    # 3. Create summary statistics
    if df is not None:
        best_method = df.iloc[0]['Experiment']
        best_dice = df.iloc[0]['Dice (%)']
        best_iou = df.iloc[0]['IoU (%)']
        
        summary = f"""
# Knee Ultrasound Segmentation Results

## Best Performing Method: {best_method}
- **Dice Score**: {best_dice:.1f}%
- **IoU Score**: {best_iou:.1f}%
- **Pixel Accuracy**: {df.iloc[0]['Pixel Accuracy (%)']:.1f}%

## Top 3 Methods:
"""
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            summary += f"{i+1}. {row['Experiment']}: Dice {row['Dice (%)']:.1f}%, IoU {row['IoU (%)']:.1f}%\n"
        
        summary_path = results_dir / "summary_report.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"✅ Summary report: {summary_path}")
    
    print("\n🎉 Presentation materials ready!")
    print(f"📁 All files in: results/")
    print(f"📈 Charts: results/comparison/")
    print(f"📸 Preprocessing: results/preprocessing/")


if __name__ == "__main__":
    create_presentation_report() 