#!/usr/bin/env python3
"""Preprocessing visualization tool for presentation."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import albumentations as A

from src.preprocessing import (
    CLAHETransform, FixedCrop, HistogramEqualization, 
    GaussianBlur, Sharpening, IntensityNormalization
)


def get_gentle_augmentation() -> A.Compose:
    """Get gentle augmentation for visualization - more suitable for presentation."""
    return A.Compose([
        # Simple flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        
        # NO rotation - not suitable for medical imaging
        
        # Very light shift and scale only
        A.ShiftScaleRotate(
            shift_limit=0.03,    # Giảm từ 0.05 xuống 0.03 (3% shift)
            scale_limit=0.02,    # Giảm từ 0.05 xuống 0.02 (2% scale) 
            rotate_limit=0,      # NO rotation
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT
        ),
        
        # Gentle brightness/contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.1,   # Giảm từ 0.15 xuống 0.1
            contrast_limit=0.1,     # Giảm từ 0.15 xuống 0.1
            p=0.4
        ),
    ])


def visualize_preprocessing_pipeline(image_id: str, save_dir: str = "results/preprocessing"):
    """Visualize all preprocessing steps independently on a sample image."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original image
    img_path = Path("image/raw") / f"{image_id}.png"
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if original is None:
        print(f"❌ Could not load image: {img_path}")
        return None
    
    print(f"📸 Processing image: {image_id} (shape: {original.shape})")
    
    # Define preprocessing steps - each applied independently to original image
    steps = [
        ("Original", None),
        ("CLAHE", CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))),
        ("Histogram EQ", HistogramEqualization()),
        ("Gaussian Blur", GaussianBlur(kernel_size=3, sigma=1.0)),
        ("Sharpening", Sharpening(alpha=1.5, sigma=1.0)),
        ("Normalization", IntensityNormalization(method="minmax")),
        ("Fixed Crop", FixedCrop(crop_top=62, crop_bottom=94, crop_left=84, crop_right=44))
    ]
    
    # Create figure - larger to accommodate multiple augmentations
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    processed_images = []
    
    # Apply each preprocessing step independently to the original image
    for i, (name, transform) in enumerate(steps):
        if transform is None:  # Original image
            processed_img = original.copy()
        else:
            # Apply transform to ORIGINAL image, not the previous result
            try:
                processed_img, _ = transform(original.copy(), None)
                print(f"  ✅ Applied {name}: shape {processed_img.shape}")
            except Exception as e:
                print(f"  ❌ Failed to apply {name}: {e}")
                processed_img = original.copy()
        
        processed_images.append(processed_img)
        
        # Display the result
        if i < len(axes):
            # Fix normalization display by using proper range
            if name == "Normalization":
                # For normalized images, show with proper range
                axes[i].imshow(processed_img, cmap='gray', vmin=processed_img.min(), vmax=processed_img.max())
            else:
                # For other images, use standard range
                if processed_img.dtype in [np.float32, np.float64]:
                    axes[i].imshow(processed_img, cmap='gray')
                else:
                    axes[i].imshow(processed_img, cmap='gray', vmin=0, vmax=255)
            
            axes[i].set_title(f'{name}\nShape: {processed_img.shape}', fontsize=12, fontweight='bold')
            axes[i].axis('off')
            
            # Add some statistics to the title
            if processed_img.dtype in [np.float32, np.float64]:
                min_val, max_val = processed_img.min(), processed_img.max()
                axes[i].set_title(f'{name}\nShape: {processed_img.shape}\nRange: [{min_val:.2f}, {max_val:.2f}]', 
                                fontsize=10, fontweight='bold')
            else:
                min_val, max_val = processed_img.min(), processed_img.max()
                axes[i].set_title(f'{name}\nShape: {processed_img.shape}\nRange: [{min_val}, {max_val}]', 
                                fontsize=10, fontweight='bold')
    
    # Create multiple gentle augmentation examples
    aug = get_gentle_augmentation()
    if aug and len(processed_images) > 0:
        print(f"  📸 Creating gentle augmentation examples...")
        
        # Create 4 different augmented versions
        for aug_idx in range(4):
            slot_idx = len(steps) + aug_idx
            if slot_idx < len(axes):
                try:
                    # Apply gentle augmentation to the original image
                    aug_result = aug(image=original.copy(), mask=np.zeros_like(original))
                    aug_img = aug_result['image']
                    
                    axes[slot_idx].imshow(aug_img, cmap='gray', vmin=0, vmax=255)
                    axes[slot_idx].set_title(f'Augmentation {aug_idx+1}\nShape: {aug_img.shape}', 
                                           fontsize=12, fontweight='bold')
                    axes[slot_idx].axis('off')
                    
                except Exception as e:
                    print(f"  ❌ Failed to create augmentation {aug_idx+1}: {e}")
                    axes[slot_idx].axis('off')
        
        print(f"  ✅ Created 4 gentle augmentation examples")
    
    # Hide any remaining empty subplots
    for i in range(len(steps) + 4, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Independent Preprocessing Effects + Gentle Augmentations - Image: {image_id}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the visualization
    output_path = save_dir / f"preprocessing_gentle_{image_id}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  💾 Saved visualization: {output_path}")
    return output_path


def create_preprocessing_comparison(num_samples: int = 3):
    """Create independent preprocessing comparison for multiple sample images."""
    print("🔄 Creating Independent Preprocessing Comparisons with Gentle Augmentation")
    print("=" * 70)
    
    image_dir = Path("image/raw")
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return []
    
    image_files = list(image_dir.glob("*.png"))
    if not image_files:
        print(f"❌ No PNG images found in: {image_dir}")
        return []
    
    # Use the first few images
    image_files = image_files[:num_samples]
    
    results = []
    for i, img_file in enumerate(image_files):
        img_id = img_file.stem
        print(f"\n[{i+1}/{len(image_files)}] Processing {img_id}...")
        
        result_path = visualize_preprocessing_pipeline(img_id)
        if result_path:
            results.append(result_path)
    
    print(f"\n✅ Created {len(results)} preprocessing visualizations with gentle augmentation")
    return results


def create_combined_preprocessing_overview():
    """Create a combined overview showing all preprocessing methods side by side."""
    print("\n📊 Creating Combined Preprocessing Overview...")
    
    # Get a representative image
    image_dir = Path("image/raw")
    image_files = list(image_dir.glob("*.png"))
    if not image_files:
        print("❌ No images found for overview")
        return None
    
    # Use the first image
    img_path = image_files[0]
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    image_id = img_path.stem
    
    # Define all preprocessing methods
    methods = {
        "Original": None,
        "CLAHE": CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        "Histogram\nEqualization": HistogramEqualization(),
        "Gaussian\nBlur": GaussianBlur(kernel_size=3, sigma=1.0),
        "Sharpening": Sharpening(alpha=1.5, sigma=1.0),
        "Normalization": IntensityNormalization(method="minmax"),
        "Fixed Crop": FixedCrop(crop_top=62, crop_bottom=94, crop_left=84, crop_right=44)
    }
    
    # Create a large comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    for i, (name, transform) in enumerate(methods.items()):
        if i >= len(axes):
            break
            
        if transform is None:
            processed = original.copy()
        else:
            try:
                processed, _ = transform(original.copy(), None)
            except Exception as e:
                print(f"Warning: Failed to apply {name}: {e}")
                processed = original.copy()
        
        # Display with proper scaling for each type
        if name == "Normalization":
            # For normalized images, use their actual range
            axes[i].imshow(processed, cmap='gray', vmin=processed.min(), vmax=processed.max())
            axes[i].set_title(f'{name}\nRange: [{processed.min():.2f}, {processed.max():.2f}]', 
                            fontsize=12, fontweight='bold', pad=15)
        else:
            # For other images
            if processed.dtype in [np.float32, np.float64]:
                axes[i].imshow(processed, cmap='gray')
            else:
                axes[i].imshow(processed, cmap='gray', vmin=0, vmax=255)
            axes[i].set_title(name, fontsize=14, fontweight='bold', pad=15)
        
        axes[i].axis('off')
        
        # Add border for better distinction
        for spine in axes[i].spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(2)
    
    # Hide unused axes
    for i in range(len(methods), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Preprocessing Methods Comparison - Sample Image: {image_id}', 
                 fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save overview
    results_dir = Path("results/preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    overview_path = results_dir / "preprocessing_methods_overview_fixed.png"
    
    plt.savefig(overview_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved fixed preprocessing overview: {overview_path}")
    return overview_path


def create_augmentation_showcase():
    """Create a dedicated showcase of gentle augmentation effects."""
    print("\n🎨 Creating Gentle Augmentation Showcase...")
    
    # Get a representative image  
    image_dir = Path("image/raw")
    image_files = list(image_dir.glob("*.png"))
    if not image_files:
        print("❌ No images found for augmentation showcase")
        return None
    
    # Use the first image
    img_path = image_files[0]
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    image_id = img_path.stem
    
    # Create gentle augmentation
    aug = get_gentle_augmentation()
    
    # Create figure for showcase
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Show original first
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Create 7 different gentle augmentation examples
    for i in range(1, 8):
        if i < len(axes):
            try:
                aug_result = aug(image=original.copy(), mask=np.zeros_like(original))
                aug_img = aug_result['image']
                
                axes[i].imshow(aug_img, cmap='gray', vmin=0, vmax=255)
                axes[i].set_title(f'Gentle Aug {i}', fontsize=14, fontweight='bold')
                axes[i].axis('off')
                
            except Exception as e:
                print(f"Warning: Failed to create augmentation {i}: {e}")
                axes[i].axis('off')
    
    plt.suptitle(f'Gentle Data Augmentation Examples - Image: {image_id}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save showcase
    results_dir = Path("results/preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)
    showcase_path = results_dir / "gentle_augmentation_showcase.png"
    
    plt.savefig(showcase_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved gentle augmentation showcase: {showcase_path}")
    return showcase_path


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
        return None, results_dir
    
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
    print("🎯 Creating Presentation Materials with Gentle Augmentation")
    print("=" * 60)
    
    # 1. Independent preprocessing visualizations with gentle augmentation
    print("\n📸 Creating independent preprocessing visualizations...")
    preprocessing_results = create_preprocessing_comparison(num_samples=3)
    
    # 2. Combined preprocessing overview (fixed normalization)
    overview_result = create_combined_preprocessing_overview()
    
    # 3. Dedicated augmentation showcase
    augmentation_result = create_augmentation_showcase()
    
    # 4. Experiment results comparison
    print("\n📊 Creating experiment comparison...")
    df, results_dir = visualize_experiment_results()
    
    # 5. Create summary statistics
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
    
    # Print summary of created files
    print(f"\nFiles created:")
    print(f"  📊 Methods overview (fixed): results/preprocessing/preprocessing_methods_overview_fixed.png")
    print(f"  🎨 Gentle augmentation: results/preprocessing/gentle_augmentation_showcase.png")
    for i, result in enumerate(preprocessing_results):
        print(f"  📸 Sample {i+1} (with 4 augmentations): {result}")


if __name__ == "__main__":
    create_presentation_report() 