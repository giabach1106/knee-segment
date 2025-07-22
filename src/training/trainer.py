from pathlib import Path
import logging
import json
import time
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from ..evaluation.metrics import compute_metrics, DiceLoss, CombinedLoss
from ..models import UNet

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, device, experiment_dir, loss_function="combined",
                 optimizer_type="adam", learning_rate=1e-3, weight_decay=1e-4,
                 use_tensorboard=True, use_wandb=True, wandb_project="knee-ultrasound-segmentation"):
        self.model = model.to(device)
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        if loss_function == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_function == "dice":
            self.criterion = DiceLoss()
        else:  # combined
            self.criterion = CombinedLoss()
            
        # Optimizer
        if optimizer_type == "adam":
            self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:  # sgd
            self.optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.best_epoch = 0
        
        # History tracking
        self.history = {
            "train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [],
            "train_iou": [], "val_iou": [], "learning_rate": []
        }
        
        # TensorBoard
        if use_tensorboard:
            tb_dir = self.experiment_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
        else:
            self.writer = None
            
        # wandb 
        self.use_wandb = use_wandb
        if use_wandb:
            experiment_name = self.experiment_dir.name
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                dir=str(self.experiment_dir)
            )
            wandb.watch(self.model, log="all", log_freq=100)
    
    def train_epoch(self, train_loader, epoch):
        """Train 1 epoch"""
        self.model.train()
        total_loss = total_dice = total_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                metrics = compute_metrics(outputs, masks)
                
            total_loss += loss.item()
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{metrics['dice']:.4f}"
            })
        
        return {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches
        }
    
    def validate(self, val_loader, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = total_dice = total_iou = total_pa = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                metrics = compute_metrics(outputs, masks)
                
                total_loss += loss.item()
                total_dice += metrics["dice"]
                total_iou += metrics["iou"]
                total_pa += metrics["pixel_accuracy"]
                num_batches += 1
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{metrics['dice']:.4f}"
                })
        
        return {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
            "pixel_accuracy": total_pa / num_batches
        }
    
    def train(self, train_loader, val_loader, num_epochs, start_epoch=0):
        """Main training loop"""
        logger.info(f"Training for {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train & validate
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_dice"].append(train_metrics["dice"])
            self.history["train_iou"].append(train_metrics["iou"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["learning_rate"].append(self.optimizer.param_groups[0]['lr'])
            
            # Scheduler step
            self.scheduler.step(val_metrics["dice"])
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch}/{num_epochs-1} ({epoch_time:.1f}s) - "
                f"Train Dice: {train_metrics['dice']:.4f}, "
                f"Val Dice: {val_metrics['dice']:.4f}"
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
                self.writer.add_scalar("Train/Dice", train_metrics["dice"], epoch)
                self.writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("Val/Dice", val_metrics["dice"], epoch)
                
            # W&B logging
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/dice": train_metrics["dice"],
                    "val/loss": val_metrics["loss"],
                    "val/dice": val_metrics["dice"],
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Best model checkpoint
            if val_metrics["dice"] > self.best_val_metric:
                self.best_val_metric = val_metrics["dice"]
                self.best_epoch = epoch
                self.save_checkpoint("best_model.pth", epoch, val_metrics)
                logger.info(f"New best model! Dice: {self.best_val_metric:.4f}")
            
            # Save history
            self.save_history()
        
        # Final checkpoint
        self.save_checkpoint("final_model.pth", self.current_epoch, val_metrics)
        
        # Close loggers
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()
            
        return self.history
    
    def save_checkpoint(self, filename, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch
        }
        torch.save(checkpoint, self.experiment_dir / filename)
    
    def save_history(self):
        """Save training history"""
        with open(self.experiment_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)


def train_model(model, train_loader, val_loader, num_epochs, device, experiment_dir,
                use_wandb=True, wandb_project="knee-ultrasound-segmentation", **trainer_kwargs):
    """Training function"""
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