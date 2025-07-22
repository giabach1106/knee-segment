from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime
import torch
import cv2

from ..models import UNet
from ..data import create_data_loaders
from ..preprocessing import (
    CLAHETransform, FixedCrop, SnakeROI, HistogramEqualization,
    GaussianBlur, Sharpening, IntensityNormalization,
    MultiChannelTransform, PreprocessingPipeline
)
from ..evaluation import evaluate_model
from ..visualization import plot_segmentation_results, plot_training_history
from .trainer import train_model

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Config mẫu"""
    name: str
    preprocessing: list = field(default_factory=list)
    model_config: dict = field(default_factory=lambda: {
        "n_channels": 1, "n_classes": 1, "bilinear": True
    })
    training_config: dict = field(default_factory=lambda: {
        "num_epochs": 60, "batch_size": 16, "learning_rate": 1e-3,
        "optimizer_type": "adam", "loss_function": "combined"
    })
    data_config: dict = field(default_factory=lambda: {
        "image_dir": "image/raw", "mask_dir": "image/mask",
        "image_size": [256, 256], "train_ratio": 0.6,
        "val_ratio": 0.2, "test_ratio": 0.2, "random_state": 42
    })
    enable_augmentation: bool = False
    multi_channel: dict = None
    wandb_config: dict = field(default_factory=lambda: {
        "use_wandb": True, "project": "knee-ultrasound-segmentation"
    })
    
    def save(self, path):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
            
    @classmethod
    def load(cls, path):
        """Load config from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def create_preprocessing_pipeline(preprocessing_config):
    """Tạo preprocessing pipeline từ config"""
    if not preprocessing_config:
        return None
        
    transforms = []
    
    for step in preprocessing_config:
        transform_type = step["type"]
        params = step.get("params", {})
        
        if transform_type == "clahe":
            transform = CLAHETransform(**params)
        elif transform_type == "fixed_crop":
            transform = FixedCrop(**params)
        elif transform_type == "snake_roi":
            transform = SnakeROI(**params)
        elif transform_type == "histogram_equalization":
            transform = HistogramEqualization(**params)
        elif transform_type == "gaussian_blur":
            transform = GaussianBlur(**params)
        elif transform_type == "sharpening":
            transform = Sharpening(**params)
        elif transform_type == "intensity_normalization":
            transform = IntensityNormalization(**params)
        else:
            logger.warning(f"Unknown transform: {transform_type}")
            continue
            
        transforms.append(transform)
        
    return PreprocessingPipeline(transforms)


def run_experiment(config, base_dir="experiments", device=None):
    """Run 1 experiment hoàn chỉnh"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    experiment_dir = Path(base_dir) / config.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(experiment_dir / "experiment_config.json")
    
    logger.info(f"Running experiment: {config.name}")
    
    # Tạo preprocessing pipeline
    preprocessing = create_preprocessing_pipeline(config.preprocessing)
    
    # Handle multi-channel
    if config.multi_channel:
        extra_transforms = []
        for transform_config in config.multi_channel["transforms"]:
            extra_pipeline = create_preprocessing_pipeline([transform_config])
            extra_transforms.append(extra_pipeline)
            
        multi_channel = MultiChannelTransform(
            transforms=extra_transforms,
            include_original=config.multi_channel.get("include_original", True)
        )
        
        if preprocessing:
            preprocessing = PreprocessingPipeline([preprocessing, multi_channel])
        else:
            preprocessing = multi_channel
            
        # Update model channels
        num_channels = len(extra_transforms)
        if config.multi_channel.get("include_original", True):
            num_channels += 1
        config.model_config["n_channels"] = num_channels
    
    # Adjust workers cho snake ROI
    data_config = config.data_config.copy()
    if any("snake_roi" in str(step) for step in config.preprocessing):
        num_workers = 0  # Snake ROI có thể gây tensor sharing issues
    else:
        num_workers = 4
    
    # Tạo data loaders
    train_loader, val_loader, test_loader, split_info = create_data_loaders(
        preprocessing_transform=preprocessing,
        enable_augmentation=config.enable_augmentation,
        batch_size=config.training_config["batch_size"],
        num_workers=num_workers,
        **data_config
    )
    
    # Save split info
    with open(experiment_dir / "data_split.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Tạo model
    model = UNet(**config.model_config)
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training_config["num_epochs"],
        device=device,
        experiment_dir=experiment_dir,
        learning_rate=config.training_config["learning_rate"],
        optimizer_type=config.training_config["optimizer_type"],
        loss_function=config.training_config["loss_function"],
        use_wandb=config.wandb_config["use_wandb"],
        wandb_project=config.wandb_config["project"]
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=experiment_dir / "training_history.png"
    )
    
    # Evaluate trên test set
    logger.info("Evaluating on test set...")
    test_metrics, _ = evaluate_model(
        model=trained_model,
        data_loader=test_loader,
        device=device
    )
    
    # Save test metrics
    with open(experiment_dir / "test_metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Results summary
    results = {
        "experiment_name": config.name,
        "timestamp": datetime.now().isoformat(),
        "preprocessing_steps": config.preprocessing,
        "training_epochs": len(history["train_loss"]),
        "best_val_dice": max(history["val_dice"]),
        "test_metrics": test_metrics,
        "data_split": split_info
    }
    
    # Save results
    with open(experiment_dir / "results_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment completed!")
    logger.info(f"Test Dice: {test_metrics['dice_percent']:.2f}%")
    
    return results


def create_standard_experiments():
    """Tạo standard experiments cho preprocessing evaluation"""
    experiments = []
    
    # Baseline - không preprocessing
    experiments.append(ExperimentConfig(
        name="baseline",
        preprocessing=[]
    ))
    
    # CLAHE only
    experiments.append(ExperimentConfig(
        name="clahe",
        preprocessing=[{"type": "clahe", "params": {"clip_limit": 2.0}}]
    ))
    
    # Fixed crop only
    experiments.append(ExperimentConfig(
        name="crop_fixed",
        preprocessing=[{"type": "fixed_crop", "params": {
            "crop_top": 62, "crop_bottom": 94, "crop_left": 84, "crop_right": 44
        }}]
    ))
    
    # Snake ROI only
    experiments.append(ExperimentConfig(
        name="snake_crop",
        preprocessing=[{"type": "snake_roi", "params": {}}]
    ))
    
    # Histogram equalization
    experiments.append(ExperimentConfig(
        name="hist_eq",
        preprocessing=[{"type": "histogram_equalization", "params": {}}]
    ))
    
    # Gaussian blur
    experiments.append(ExperimentConfig(
        name="blur",
        preprocessing=[{"type": "gaussian_blur", "params": {"kernel_size": 3}}]
    ))
    
    # Sharpening
    experiments.append(ExperimentConfig(
        name="sharpen",
        preprocessing=[{"type": "sharpening", "params": {"alpha": 1.5}}]
    ))
    
    # Intensity normalization
    experiments.append(ExperimentConfig(
        name="normalize",
        preprocessing=[{"type": "intensity_normalization", "params": {"method": "minmax"}}]
    ))
    
    # Augmentation only
    experiments.append(ExperimentConfig(
        name="augmentation",
        preprocessing=[],
        enable_augmentation=True
    ))
    
    # SGD optimizer
    sgd_config = ExperimentConfig(name="optimizer_sgd", preprocessing=[])
    sgd_config.training_config["optimizer_type"] = "sgd"
    experiments.append(sgd_config)
    
    # Multi-channel (original + CLAHE)
    experiments.append(ExperimentConfig(
        name="multichannel",
        preprocessing=[],
        multi_channel={
            "include_original": True,
            "transforms": [{"type": "clahe", "params": {"clip_limit": 2.0}}]
        }
    ))
    
    # Combined: CLAHE + Fixed Crop + Augmentation
    experiments.append(ExperimentConfig(
        name="combined_clahe_crop_aug",
        preprocessing=[
            {"type": "clahe", "params": {"clip_limit": 2.0}},
            {"type": "fixed_crop", "params": {
                "crop_top": 62, "crop_bottom": 94, "crop_left": 84, "crop_right": 44
            }}
        ],
        enable_augmentation=True
    ))
    
    # Snake ROI + Augmentation
    experiments.append(ExperimentConfig(
        name="combined_snake_aug",
        preprocessing=[{"type": "snake_roi", "params": {}}],
        enable_augmentation=True
    ))
    
    return experiments 