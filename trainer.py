"""
Trainer class for Deep Metric Learning
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler  # Add this for schedulers
import numpy as np
import time
import logging  # Add this missing import
import matplotlib.pyplot as plt
from tqdm import tqdm

from loss_functions import TripletLoss, ContrastiveLoss, ArcFaceLoss
from model import EmbeddingNet, ArcFaceModel
from utils import plot_loss_curve
# Import our custom scheduler if you've created lr_schedulers.py
from lr_schedulers import WarmupCosineScheduler

class Trainer:
    """Trainer class for Deep Metric Learning"""
    def __init__(self, config, data_module):
        self.config = config
        self.data_module = data_module
        self.device = config.DEVICE
        self.logger = logging.getLogger(__name__)
        
        # Create model
        if config.LOSS_TYPE == 'arcface':
            self.model = ArcFaceModel(config, data_module.num_classes).to(self.device)
        else:
            self.model = EmbeddingNet(config).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Create loss function
        if config.LOSS_TYPE == 'triplet':
            self.criterion = TripletLoss(
                margin=config.MARGIN,
                mining_type=config.MINING_TYPE
            )
        elif config.LOSS_TYPE == 'contrastive':
            self.criterion = ContrastiveLoss(
                margin=config.MARGIN
            )
        elif config.LOSS_TYPE == 'arcface':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {config.LOSS_TYPE}")
        
        # Initialize learning rate scheduler
        self._init_scheduler()
        
        # Initialize early stopping
        self.early_stopping_counter = 0
        self.early_stopping_best_loss = float('inf')
        
        # Initialize training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.learning_rates = []  # Track learning rates
    
    def _init_scheduler(self):
        """Initialize the learning rate scheduler based on config"""
        if self.config.SCHEDULER_TYPE == "step":
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.SCHEDULER_STEP_SIZE,
                gamma=self.config.SCHEDULER_GAMMA
            )
        elif self.config.SCHEDULER_TYPE == "plateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.SCHEDULER_GAMMA,
                patience=self.config.SCHEDULER_PATIENCE,
                min_lr=self.config.SCHEDULER_MIN_LR,
                verbose=True
            )
        elif self.config.SCHEDULER_TYPE == "cosine":
            t_max = self.config.SCHEDULER_T_MAX if self.config.SCHEDULER_T_MAX is not None else self.config.NUM_EPOCHS
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=self.config.SCHEDULER_MIN_LR
            )
        elif self.config.SCHEDULER_TYPE == "warmup_cosine":
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config.WARMUP_EPOCHS,
                total_epochs=self.config.NUM_EPOCHS,
                min_lr=self.config.SCHEDULER_MIN_LR
            )
        else:
            self.scheduler = None
            
    def _step_scheduler(self, val_loss):
        """Step the learning rate scheduler"""
        if self.scheduler is not None:
            # For plateau scheduler, we need to pass the validation loss
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
                
            # Log the learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            self.logger.info(f"Current learning rate: {current_lr:.8f}")
    
    def _check_early_stopping(self, val_loss):
        """Check if training should stop early"""
        if not self.config.EARLY_STOPPING:
            return False
            
        # Check if current validation loss is better than the best so far
        if val_loss < self.early_stopping_best_loss - self.config.EARLY_STOPPING_DELTA:
            self.early_stopping_best_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            self.logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.config.EARLY_STOPPING_PATIENCE}")
            
            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.info("Early stopping triggered!")
                return True
                
        return False

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.LOSS_TYPE == 'arcface':
                outputs, embeddings = self.model(images, labels)
                loss = self.criterion(outputs, labels)
            else:
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        return epoch_loss
        
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.config.LOSS_TYPE == 'arcface':
                    outputs, embeddings = self.model(images, labels)
                    loss = self.criterion(outputs, labels)
                else:
                    embeddings = self.model(images)
                    loss = self.criterion(embeddings, labels)
                
                running_loss += loss.item()
        
        # Calculate average loss for the validation set
        val_loss = running_loss / len(dataloader)
        return val_loss

    def train(self, num_epochs=None):
        """Train the model for the specified number of epochs"""
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        # Get dataloaders
        dataloaders = self.data_module.get_dataloaders()
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Log progress
            self.logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Step the learning rate scheduler
            self._step_scheduler(val_loss)
            
            # Check early stopping
            if self._check_early_stopping(val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Plot and save loss curves
            self._plot_training_curves()
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time/60:.2f} minutes")
        
        return self.train_losses, self.val_losses
        
    def _plot_training_curves(self):
        """Plot and save loss and learning rate curves"""
        # Loss curve
        plot_loss_curve(
            self.train_losses, 
            self.val_losses, 
            os.path.join(self.config.RESULTS_DIR, "loss_curve.png")
        )
        
        # Learning rate curve (if we're using a scheduler)
        if self.scheduler is not None and self.learning_rates:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(self.learning_rates) + 1), self.learning_rates)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')  # Log scale for better visualization
            plt.savefig(os.path.join(self.config.RESULTS_DIR, "lr_schedule.png"))
            plt.close()


    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.MODEL_DIR, f"checkpoint_epoch_{epoch}.pth")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model separately
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(self.config.MODEL_DIR, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {epoch} with validation loss: {val_loss:.4f}")
        
        return epoch