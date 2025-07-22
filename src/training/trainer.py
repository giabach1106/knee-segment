"""Training utilities for U-Net segmentation model.

This module provides a Trainer class and training functions for
systematic model training with logging and checkpointing.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import json
import time

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb

from ..evaluation.metrics import (
    compute_metrics, evaluate_model, DiceLoss, CombinedLoss
)
from ..models import UNet

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for U-Net segmentation model.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Args:
        model: The segmentation model to train.
        device: Device to train on (cuda or cpu).
        experiment_dir: Directory to save experiment results.
        loss_function: Loss function to use. Options: 'bce', 'dice', 'combined'.
        optimizer_type: Optimizer type. Options: 'adam', 'sgd'.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight decay.
        use_tensorboard: Whether to log to TensorBoard.
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
        wandb_entity: W&B entity/team name (optional).
        checkpoint_interval: Save checkpoint every N epochs.
        early_stopping_patience: Stop if no improvement for N epochs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        experiment_dir: Union[str, Path],
        loss_function: str = "combined",
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_tensorboard: bool = True,
        use_wandb: bool = True,
        wandb_project: str = "knee-ultrasound-segmentation",
        wandb_entity: Optional[str] = None,
        checkpoint_interval: int = 5,
        early_stopping_patience: int = 15
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loss function
        if loss_function == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_function == "dice":
            self.criterion = DiceLoss()
        elif loss_function == "combined":
            self.criterion = CombinedLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
            
        # Initialize optimizer
        if optimizer_type == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
            
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_interval = checkpoint_interval
        
        # History tracking
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
            "train_iou": [],
            "val_iou": [],
            "learning_rate": []
        }
        
        # TensorBoard
        if use_tensorboard:
            tb_dir = self.experiment_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
        else:
            self.writer = None
            
        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            experiment_name = self.experiment_dir.name
            # Create config dictionary with training parameters
            config_dict = {
                "model": "UNet",
                "loss_function": loss_function,
                "optimizer": optimizer_type,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "early_stopping_patience": early_stopping_patience,
                "checkpoint_interval": checkpoint_interval,
                "experiment_dir": str(self.experiment_dir)
            }
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=experiment_name,
                config=config_dict,
                dir=str(self.experiment_dir),
                resume="allow"
            )
            # Watch model for gradients and parameters
            wandb.watch(self.model, log="all", log_freq=100)
            
        # Save configuration
        self.config = {
            "model_type": model.__class__.__name__,
            "loss_function": loss_function,
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(device),
            "num_parameters": sum(p.numel() for p in model.parameters())
        }
        self._save_config()
        
    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(outputs, masks)
                
            # Update totals
            total_loss += loss.item()
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{metrics['dice']:.4f}",
                "iou": f"{metrics['iou']:.4f}"
            })
            
            # Log to TensorBoard and W&B
            if batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar("Train/Loss_step", loss.item(), global_step)
                    self.writer.add_scalar("Train/Dice_step", metrics["dice"], global_step)
                    self.writer.add_scalar("Train/IoU_step", metrics["iou"], global_step)
                
                # W&B logging
                if self.use_wandb:
                    wandb.log({
                        "train/loss_step": loss.item(),
                        "train/dice_step": metrics["dice"],
                        "train/iou_step": metrics["iou"],
                        "global_step": global_step
                    })
                
        # Average metrics
        avg_metrics = {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches
        }
        
        return avg_metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_pa = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Compute metrics
                metrics = compute_metrics(outputs, masks)
                
                # Update totals
                total_loss += loss.item()
                total_dice += metrics["dice"]
                total_iou += metrics["iou"]
                total_pa += metrics["pixel_accuracy"]
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{metrics['dice']:.4f}",
                    "iou": f"{metrics['iou']:.4f}"
                })
                
        # Average metrics
        avg_metrics = {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
            "pixel_accuracy": total_pa / num_batches
        }
        
        return avg_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        start_epoch: int = 0
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            start_epoch: Starting epoch (for resuming).
            
        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model has {self.config['num_parameters']:,} parameters")
        
        epochs_without_improvement = 0
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader, epoch)
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_dice"].append(train_metrics["dice"])
            self.history["train_iou"].append(train_metrics["iou"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics["dice"])
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch}/{num_epochs-1} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Dice: {train_metrics['dice']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Dice: {val_metrics['dice']:.4f}, "
                f"Val IoU: {val_metrics['iou']:.4f}"
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
                self.writer.add_scalar("Train/Dice", train_metrics["dice"], epoch)
                self.writer.add_scalar("Train/IoU", train_metrics["iou"], epoch)
                self.writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("Val/Dice", val_metrics["dice"], epoch)
                self.writer.add_scalar("Val/IoU", val_metrics["iou"], epoch)
                self.writer.add_scalar("Val/PixelAccuracy", val_metrics["pixel_accuracy"], epoch)
                self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]['lr'], epoch)
                
            # W&B logging
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/dice": train_metrics["dice"],
                    "train/iou": train_metrics["iou"],
                    "val/loss": val_metrics["loss"],
                    "val/dice": val_metrics["dice"],
                    "val/iou": val_metrics["iou"],
                    "val/pixel_accuracy": val_metrics["pixel_accuracy"],
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch_time": epoch_time
                })
                
            # Check for improvement
            if val_metrics["dice"] > self.best_val_metric:
                self.best_val_metric = val_metrics["dice"]
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint("best_model.pth", epoch, val_metrics)
                logger.info(f"New best model! Dice: {self.best_val_metric:.4f}")
            else:
                epochs_without_improvement += 1
                
            # Regular checkpointing
            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth", epoch, val_metrics)
                
            # Save history
            self.save_history()
            
            # Early stopping
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered. No improvement for {epochs_without_improvement} epochs."
                )
                break
                
        # Final checkpoint
        self.save_checkpoint("final_model.pth", self.current_epoch, val_metrics)
        
        # Close TensorBoard and W&B
        if self.writer:
            self.writer.close()
            
        if self.use_wandb:
            # Log final metrics
            wandb.log({
                "final/best_val_dice": self.best_val_metric,
                "final/best_epoch": self.best_epoch,
                "final/total_epochs": self.current_epoch + 1
            })
            wandb.finish()
            
        return self.history
    
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
            epoch: Current epoch.
            metrics: Current metrics.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch,
            "config": self.config
        }
        
        checkpoint_path = self.experiment_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Epoch number from checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_metric = checkpoint["best_val_metric"]
        self.best_epoch = checkpoint["best_epoch"]
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]
    
    def save_history(self) -> None:
        """Save training history to JSON file."""
        history_path = self.experiment_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    experiment_dir: Union[str, Path],
    use_wandb: bool = True,
    wandb_project: str = "knee-ultrasound-segmentation",
    **trainer_kwargs
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Convenience function to train a model.
    
    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        num_epochs: Number of epochs.
        device: Device to train on.
        experiment_dir: Directory for experiment results.
        **trainer_kwargs: Additional arguments for Trainer.
        
    Returns:
        Tuple of (trained_model, history).
    """
    trainer = Trainer(
        model=model,
        device=device,
        experiment_dir=experiment_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        **trainer_kwargs
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )
    
    return model, history 