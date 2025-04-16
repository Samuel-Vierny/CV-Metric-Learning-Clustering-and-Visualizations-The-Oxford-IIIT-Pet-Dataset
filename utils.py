"""
Utility functions for the project
"""
import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_directories(config):
    """Create necessary directories"""
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

def setup_logging(config):
    """Set up logging"""
    log_file = os.path.join(config.SAVE_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def plot_loss_curve(train_losses, val_losses, save_path):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, auc, save_path):
    """Plot ROC curve for verification task"""
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, 'b', label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()

def compute_eer(fpr, tpr, thresholds):
    """Compute Equal Error Rate (EER)"""
    fnr = 1 - tpr
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]
    return eer, eer_threshold
