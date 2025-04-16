"""
Main script for Deep Metric Learning on the Oxford-IIIT Pet Dataset
"""
# Add these lines at the very beginning
import os
# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

#Silence a conflict issue regarding Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import json
import logging
import numpy as np
import inspect
from datetime import datetime

from config import Config
from utils import set_seed, create_directories, setup_logging
from data_module import OxfordPetsDataModule
from model import EmbeddingNet, ArcFaceModel
from trainer import Trainer
from evaluator import Evaluator
from visualization import Visualizer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Deep Metric Learning on Oxford-IIIT Pet Dataset')
    
    # General
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory with dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save outputs')
    
    # Dataset
    parser.add_argument('--img_size', type=int, default=None, help='Image size for resizing')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers for data loading')
    
    # Model
    parser.add_argument('--backbone', type=str, default=None, help='Backbone CNN (resnet18, resnet34, resnet50)')
    parser.add_argument('--embedding_size', type=int, default=None, help='Embedding dimension')
    # Changed to store_true/store_false with no default for pretrained
    parser.add_argument('--pretrained', action='store_true', dest='pretrained', help='Use pretrained backbone')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', help='Do not use pretrained backbone')
    parser.set_defaults(pretrained=None)  # Set default to None to detect if it was specified
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    
    # Loss Function
    parser.add_argument('--loss_type', type=str, default=None, help='Loss function (triplet, contrastive, arcface)')
    parser.add_argument('--margin', type=float, default=None, help='Margin for triplet/contrastive loss')
    parser.add_argument('--mining_type', type=str, default=None, help='Mining type for triplet loss')
    
    # Mode
    parser.add_argument('--mode', type=str, default=None, 
                        choices=['train', 'evaluate', 'visualize', 'train_eval', 'all'],
                        help='Mode to run (train, evaluate, visualize, train_eval, all)')
    
    # Load model
    parser.add_argument('--load_model', type=str, default=None, help='Path to model checkpoint to load')
    
    return parser.parse_args()

def train(config, data_module):
    """Train the model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")
    
    # Create trainer
    trainer = Trainer(config, data_module)
    
    # Load checkpoint if specified
    if hasattr(config, 'LOAD_MODEL') and config.LOAD_MODEL:
        trainer.load_checkpoint(config.LOAD_MODEL)
    
    # Train model
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, "final_model.pth")
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config
    }, final_model_path)
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    
    return trainer.model

def evaluate(config, data_module, model=None):
    """Evaluate the model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation...")
    
    # Load model if not provided
    if model is None:
        # Try to load best model first
        best_model_path = os.path.join(config.MODEL_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            if hasattr(config, 'LOSS_TYPE') and config.LOSS_TYPE == 'arcface':
                model = ArcFaceModel(config, data_module.num_classes)
            else:
                model = EmbeddingNet(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path}")
        elif hasattr(config, 'LOAD_MODEL') and config.LOAD_MODEL:
            checkpoint = torch.load(config.LOAD_MODEL)
            if hasattr(config, 'LOSS_TYPE') and config.LOSS_TYPE == 'arcface':
                model = ArcFaceModel(config, data_module.num_classes)
            else:
                model = EmbeddingNet(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {config.LOAD_MODEL}")
        else:
            logger.error("No model found for evaluation")
            return None
    
    # Create evaluator
    evaluator = Evaluator(config, model, data_module)
    
    # Run all evaluations
    results = evaluator.run_all_evaluations()
    
    # Save results
    results_path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    
    # Convert numpy values to Python scalars for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    logger.info(f"Evaluation completed. Results saved to {results_path}")
    
    return results

def visualize(config, data_module, model=None):
    """Visualize the embeddings"""
    logger = logging.getLogger(__name__)
    logger.info("Starting visualization...")
    
    # Load model if not provided
    if model is None:
        # Try to load best model first
        best_model_path = os.path.join(config.MODEL_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            if hasattr(config, 'LOSS_TYPE') and config.LOSS_TYPE == 'arcface':
                model = ArcFaceModel(config, data_module.num_classes)
            else:
                model = EmbeddingNet(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path}")
        elif hasattr(config, 'LOAD_MODEL') and config.LOAD_MODEL:
            checkpoint = torch.load(config.LOAD_MODEL)
            if hasattr(config, 'LOSS_TYPE') and config.LOSS_TYPE == 'arcface':
                model = ArcFaceModel(config, data_module.num_classes)
            else:
                model = EmbeddingNet(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {config.LOAD_MODEL}")
        else:
            logger.error("No model found for visualization")
            return None
    
    # Create visualizer
    visualizer = Visualizer(config, model, data_module)
    
    # Get dataloaders
    dataloaders = data_module.get_dataloaders()
    test_loader = dataloaders['test']
    
    # Create t-SNE visualization
    visualizer.visualize_tsne(test_loader)
    
    # Create UMAP visualization
    visualizer.visualize_umap(test_loader)
    
    # Create retrieval examples
    visualizer.visualize_retrieval_examples(test_loader)
    
    logger.info("Visualization completed")

def get_all_config_attributes(config):
    """Get all attributes from a config object, including class attributes"""
    # Get class attributes
    class_attrs = {key: value for key, value in vars(type(config)).items() 
                   if not key.startswith('__') and not inspect.isfunction(value) and not inspect.ismethod(value)}
    
    # Get instance attributes
    instance_attrs = vars(config)
    
    # Combine them, with instance attributes taking precedence
    all_attrs = {**class_attrs, **instance_attrs}
    
    # Filter out methods, functions, and private attributes
    filtered_attrs = {k: v for k, v in all_attrs.items() 
                    if not k.startswith('__') and not inspect.isfunction(v) and not inspect.ismethod(v)}
    
    return filtered_attrs

def main():
    """Main function"""
    # Create config with default values
    config = Config()
    
    # Parse command line arguments
    args = parse_args()
    
    # Update config ONLY for arguments explicitly specified in command line
    # This preserves config file values when not overridden by command line
    for key, value in vars(args).items():
        if value is not None:  # Only update if argument was explicitly provided
            key_upper = key.upper()
            setattr(config, key_upper, value)
    
    # Create output directories
    create_directories(config)
    
    # Set up logging
    logger = setup_logging(config)
    
    # Set random seed
    set_seed(config.SEED)
    
    # Print config - FIXED to show all config properties
    logger.info("Configuration:")
    config_attrs = get_all_config_attributes(config)
    for key, value in sorted(config_attrs.items()):
        if not key.startswith('__') and not callable(value):
            logger.info(f"  {key}: {value}")
    
    # Set up data module
    data_module = OxfordPetsDataModule(config)
    data_module.setup()
    
    model = None
    
    # Run based on mode
    mode = config.MODE if hasattr(config, 'MODE') else 'train_eval'
    
    if mode == 'train' or mode == 'train_eval' or mode == 'all':
        model = train(config, data_module)
    
    if mode == 'evaluate' or mode == 'train_eval' or mode == 'all':
        evaluate(config, data_module, model)
    
    if mode == 'visualize' or mode == 'all':
        visualize(config, data_module, model)
    
    logger.info("All tasks completed successfully")

if __name__ == "__main__":
    main()