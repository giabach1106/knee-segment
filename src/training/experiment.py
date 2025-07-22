"""Experiment configuration and runner for systematic preprocessing evaluation.

This module provides utilities for defining and running experiments with
different preprocessing configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging
from datetime import datetime

import torch
from omegaconf import DictConfig, OmegaConf
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
    """Configuration for a segmentation experiment.
    
    Attributes:
        name: Experiment name.
        preprocessing: List of preprocessing steps to apply.
        model_config: Model configuration.
        training_config: Training configuration.
        data_config: Data loading configuration.
        enable_augmentation: Whether to use data augmentation.
        multi_channel: Configuration for multi-channel input.
    """
    name: str
    preprocessing: List[Dict[str, Any]] = field(default_factory=list)
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "n_channels": 1,
        "n_classes": 1,
        "bilinear": True
    })
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        "num_epochs": 60,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "optimizer_type": "adam",
        "loss_function": "combined"
    })
    data_config: Dict[str, Any] = field(default_factory=lambda: {
        "image_dir": "image/raw",
        "mask_dir": "image/mask",
        "image_size": [256, 256],
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 42
    })
    enable_augmentation: bool = False
    multi_channel: Optional[Dict[str, Any]] = None
    wandb_config: Dict[str, Any] = field(default_factory=lambda: {
        "use_wandb": True,
        "project": "knee-ultrasound-segmentation",
        "entity": None
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "preprocessing": self.preprocessing,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "data_config": self.data_config,
            "enable_augmentation": self.enable_augmentation,
            "multi_channel": self.multi_channel,
            "wandb_config": self.wandb_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def create_preprocessing_pipeline(
    preprocessing_config: List[Dict[str, Any]]
) -> Optional[PreprocessingPipeline]:
    """Create preprocessing pipeline from configuration.
    
    Args:
        preprocessing_config: List of preprocessing step configurations.
        
    Returns:
        PreprocessingPipeline or None if no preprocessing.
    """
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
            raise ValueError(f"Unknown preprocessing type: {transform_type}")
            
        transforms.append(transform)
        
    return PreprocessingPipeline(transforms)


def run_experiment(
    config: ExperimentConfig,
    base_dir: Union[str, Path] = "experiments",
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Run a complete experiment with the given configuration.
    
    Args:
        config: Experiment configuration.
        base_dir: Base directory for saving results.
        device: Device to use for training.
        
    Returns:
        Dictionary containing experiment results.
    """
    # Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    experiment_dir = Path(base_dir) / config.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save(experiment_dir / "experiment_config.json")
    
    # Setup logging for this experiment
    log_file = experiment_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Device: {device}")
    
    try:
        # Create preprocessing pipeline
        preprocessing = create_preprocessing_pipeline(config.preprocessing)
        
        # Handle multi-channel configuration
        if config.multi_channel:
            # Create additional transforms for multi-channel
            extra_transforms = []
            for transform_config in config.multi_channel["transforms"]:
                extra_pipeline = create_preprocessing_pipeline([transform_config])
                extra_transforms.append(extra_pipeline)
                
            # Wrap in multi-channel transform
            multi_channel = MultiChannelTransform(
                transforms=extra_transforms,
                include_original=config.multi_channel.get("include_original", True)
            )
            
            # Update preprocessing
            if preprocessing:
                preprocessing = PreprocessingPipeline([preprocessing, multi_channel])
            else:
                preprocessing = multi_channel
                
            # Update model channels
            num_channels = len(extra_transforms)
            if config.multi_channel.get("include_original", True):
                num_channels += 1
            config.model_config["n_channels"] = num_channels
            
        # Adjust num_workers for complex preprocessing
        data_config = config.data_config.copy()
        if any("snake_roi" in str(step) for step in config.preprocessing):
            # Snake ROI processing is complex and can cause multiprocessing issues
            logger.info("Snake ROI detected - reducing DataLoader workers to avoid tensor sharing issues")
            num_workers = 0
        else:
            num_workers = 4
        
        # Create data loaders
        train_loader, val_loader, test_loader, split_info = create_data_loaders(
            preprocessing_transform=preprocessing,
            enable_augmentation=config.enable_augmentation,
            batch_size=config.training_config["batch_size"],
            num_workers=num_workers,
            **data_config
        )
        
        # Save split information
        with open(experiment_dir / "data_split.json", 'w') as f:
            json.dump(split_info, f, indent=2)
            
        # Create model
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
            wandb_project=config.wandb_config["project"],
            wandb_entity=config.wandb_config["entity"]
        )
        
        # Plot training history
        history_plot = plot_training_history(
            history,
            save_path=experiment_dir / "training_history.png"
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics, test_predictions = evaluate_model(
            model=trained_model,
            data_loader=test_loader,
            device=device,
            save_predictions=True
        )
        
        # Save test metrics
        with open(experiment_dir / "test_metrics.json", 'w') as f:
            json.dump(test_metrics, f, indent=2)
            
        # Save some prediction visualizations
        predictions_dir = experiment_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        # Visualize first 5 test predictions
        for i in range(min(5, len(test_predictions))):
            pred_info = test_predictions[i]
            
            # Get processed image that matches the model input size
            test_dataset = test_loader.dataset
            image_id = test_dataset.image_ids[pred_info["global_idx"]]
            
            # Load and process image the same way as during training
            raw_image = test_dataset._load_image(image_id)
            raw_mask = test_dataset._load_mask(image_id)
            
            # Apply the same preprocessing pipeline as used during training
            if test_dataset.resize is not None:
                processed_image = cv2.resize(raw_image, (test_dataset.resize[1], test_dataset.resize[0]))
            else:
                processed_image = raw_image
                
            if test_dataset.preprocessing is not None:
                processed_image, _ = test_dataset.preprocessing(processed_image, None)
            
            # Create visualization using processed image that matches prediction size
            fig = plot_segmentation_results(
                image=processed_image,
                ground_truth=pred_info["ground_truth"].squeeze(),
                prediction=pred_info["prediction"].squeeze(),
                title=f"Test Image {i+1} - {image_id}",
                dice_score=pred_info["dice"],
                iou_score=pred_info["iou"],
                save_path=predictions_dir / f"test_{i+1}_{image_id}.png"
            )
            
        # Prepare results summary
        results = {
            "experiment_name": config.name,
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "preprocessing_steps": config.preprocessing,
            "model_params": sum(p.numel() for p in trained_model.parameters()),
            "training_epochs": len(history["train_loss"]),
            "best_val_dice": max(history["val_dice"]),
            "best_val_iou": max(history["val_iou"]),
            "test_metrics": test_metrics,
            "data_split": split_info
        }
        
        # Save results summary
        with open(experiment_dir / "results_summary.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Test Dice: {test_metrics['dice_percent']:.2f}%")
        logger.info(f"Test IoU: {test_metrics['iou_percent']:.2f}%")
        logger.info(f"Test Pixel Accuracy: {test_metrics['pixel_accuracy_percent']:.2f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise
        
    finally:
        # Remove file handler
        logger.removeHandler(file_handler)
        file_handler.close()


def create_standard_experiments() -> List[ExperimentConfig]:
    """Create standard set of experiments for preprocessing evaluation.
    
    Returns:
        List of experiment configurations.
    """
    experiments = []
    
    # Baseline - no preprocessing
    experiments.append(ExperimentConfig(
        name="baseline",
        preprocessing=[],
        enable_augmentation=False
    ))
    
    # CLAHE only
    experiments.append(ExperimentConfig(
        name="clahe",
        preprocessing=[{
            "type": "clahe",
            "params": {"clip_limit": 2.0, "tile_grid_size": [8, 8]}
        }],
        enable_augmentation=False
    ))
    
    # Fixed crop only
    experiments.append(ExperimentConfig(
        name="crop_fixed",
        preprocessing=[{
            "type": "fixed_crop",
            "params": {
                "crop_top": 62,
                "crop_bottom": 94,
                "crop_left": 84,
                "crop_right": 44
            }
        }],
        enable_augmentation=False
    ))
    
    # Snake ROI only
    experiments.append(ExperimentConfig(
        name="snake_crop",
        preprocessing=[{
            "type": "snake_roi",
            "params": {}
        }],
        enable_augmentation=False
    ))
    
    # Histogram equalization only
    experiments.append(ExperimentConfig(
        name="hist_eq",
        preprocessing=[{
            "type": "histogram_equalization",
            "params": {}
        }],
        enable_augmentation=False
    ))
    
    # Gaussian blur only
    experiments.append(ExperimentConfig(
        name="blur",
        preprocessing=[{
            "type": "gaussian_blur",
            "params": {"kernel_size": 3, "sigma": 1.0}
        }],
        enable_augmentation=False
    ))
    
    # Sharpening only
    experiments.append(ExperimentConfig(
        name="sharpen",
        preprocessing=[{
            "type": "sharpening",
            "params": {"alpha": 1.5, "sigma": 1.0}
        }],
        enable_augmentation=False
    ))
    
    # Intensity normalization only
    experiments.append(ExperimentConfig(
        name="normalize",
        preprocessing=[{
            "type": "intensity_normalization",
            "params": {"method": "minmax"}
        }],
        enable_augmentation=False
    ))
    
    # Augmentation only (with baseline preprocessing)
    experiments.append(ExperimentConfig(
        name="augmentation",
        preprocessing=[],
        enable_augmentation=True
    ))
    
    # SGD optimizer
    sgd_config = ExperimentConfig(
        name="optimizer_sgd",
        preprocessing=[],
        enable_augmentation=False
    )
    sgd_config.training_config["optimizer_type"] = "sgd"
    experiments.append(sgd_config)
    
    # Multi-channel (original + CLAHE)
    experiments.append(ExperimentConfig(
        name="multichannel",
        preprocessing=[],
        enable_augmentation=False,
        multi_channel={
            "include_original": True,
            "transforms": [{
                "type": "clahe",
                "params": {"clip_limit": 2.0}
            }]
        }
    ))
    
    # Combined: CLAHE + Fixed Crop + Augmentation
    experiments.append(ExperimentConfig(
        name="combined_clahe_crop_aug",
        preprocessing=[
            {
                "type": "clahe",
                "params": {"clip_limit": 2.0}
            },
            {
                "type": "fixed_crop",
                "params": {
                    "crop_top": 62,
                    "crop_bottom": 94,
                    "crop_left": 84,
                    "crop_right": 44
                }
            }
        ],
        enable_augmentation=True
    ))
    
    # Combined: Snake ROI + Augmentation
    experiments.append(ExperimentConfig(
        name="combined_snake_aug",
        preprocessing=[{
            "type": "snake_roi",
            "params": {}
        }],
        enable_augmentation=True
    ))
    
    return experiments 