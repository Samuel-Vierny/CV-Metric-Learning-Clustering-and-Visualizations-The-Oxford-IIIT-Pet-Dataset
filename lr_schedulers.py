"""
Custom learning rate schedulers for metric learning
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup and then cosine decay scheduler.
    Linearly increases learning rate from 0 to base_lr over `warmup_epochs`.
    Then follows cosine decay to min_lr over remaining epochs.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, progress)  # Ensure we don't go beyond 1.0
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay 
                   for base_lr in self.base_lrs]