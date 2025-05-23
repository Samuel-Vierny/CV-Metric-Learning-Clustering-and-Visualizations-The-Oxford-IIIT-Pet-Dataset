"""
Visualization utilities for Deep Metric Learning
"""
# Silence TensorFlow warnings - add at the top of the file
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow logging (0=all, 1=info, 2=warning, 3=error)

# Silence warnings from other libraries
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore sklearn and other deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")  # Ignore UMAP parallelism warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
import logging
import seaborn as sns
import pandas as pd
import argparse
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    """Class for visualizing embeddings and results"""
    def __init__(self, config, model, data_module):
        self.config = config
        self.model = model
        self.data_module = data_module
        self.device = config.DEVICE
        self.logger = logging.getLogger(__name__)
        
        # Set model to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def compute_embeddings(self, dataloader):
        """Compute embeddings for all samples in the dataloader"""
        all_embeddings = []
        all_labels = []
        all_paths = []  # Keep track of image paths for later visualization
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Computing embeddings")):
                images = images.to(self.device)
                
                # Get embeddings
                if hasattr(self.model, 'embedding_net'):
                    embeddings = self.model.embedding_net(images)
                else:
                    embeddings = self.model(images)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
                
                # Get file paths for these samples (if needed for image display)
                if hasattr(dataloader.dataset, 'file_list'):
                    batch_paths = [dataloader.dataset.file_list[dataloader.dataset.indices[idx + batch_idx * dataloader.batch_size]][0] 
                                   if hasattr(dataloader.dataset, 'indices') else 
                                   dataloader.dataset.file_list[idx + batch_idx * dataloader.batch_size][0]
                                   for idx in range(len(labels))]
                    all_paths.extend(batch_paths)
        
        # Concatenate embeddings and labels
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        
        return all_embeddings, all_labels, all_paths if all_paths else None
    
    def visualize_tsne(self, dataloader, perplexity=None, save_path=None):
        """Visualize embeddings using t-SNE"""
        if perplexity is None:
            perplexity = self.config.TSNE_PERPLEXITY
        if save_path is None:
            save_path = os.path.join(self.config.RESULTS_DIR, "tsne_visualization.png")
        
        self.logger.info(f"Creating t-SNE visualization with perplexity {perplexity}...")
        
        # Compute embeddings
        embeddings, labels, _ = self.compute_embeddings(dataloader)
        
        # Apply t-SNE for dimensionality reduction
        # Use max_iter instead of n_iter (following sklearn 1.5+ convention)
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=self.config.SEED)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create unique labels and a colormap
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # Build a label map for cleaner plotting
        label_map = {i: i for i in unique_labels}
        
        if hasattr(self.data_module, 'breed_names'):
            # Use actual breed names if available
            label_map = {i: name for i, name in enumerate(self.data_module.breed_names)}
        
        # Create a DataFrame for easier plotting with seaborn
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': [label_map[l] for l in labels]
        })
        
        # Plot with seaborn for better aesthetics
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='x', y='y', hue='label', data=df, palette='tab20', s=50, alpha=0.7)
        
        plt.title(f't-SNE Visualization of Embeddings (perplexity={perplexity})')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        
        # Improve legend if there are many classes
        if num_classes > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"t-SNE visualization saved to {save_path}")
        
        return embeddings_2d, labels
    
    def visualize_umap(self, dataloader, n_neighbors=None, save_path=None):
        """Visualize embeddings using UMAP"""
        if n_neighbors is None:
            n_neighbors = self.config.UMAP_N_NEIGHBORS
        if save_path is None:
            save_path = os.path.join(self.config.RESULTS_DIR, "umap_visualization.png")
        
        self.logger.info(f"Creating UMAP visualization with n_neighbors {n_neighbors}...")
        
        # Compute embeddings
        embeddings, labels, _ = self.compute_embeddings(dataloader)
        
        # Apply UMAP for dimensionality reduction
        # Set random_state to None if you want parallelism (n_jobs > 1)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=2, 
            random_state=self.config.SEED,
            n_jobs=1  # Explicitly set to 1 since we're using random_state
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create unique labels and a colormap
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # Build a label map for cleaner plotting
        label_map = {i: i for i in unique_labels}
        
        if hasattr(self.data_module, 'breed_names'):
            # Use actual breed names if available
            label_map = {i: name for i, name in enumerate(self.data_module.breed_names)}
        
        # Create a DataFrame for easier plotting with seaborn
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': [label_map[l] for l in labels]
        })
        
        # Plot with seaborn for better aesthetics
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='x', y='y', hue='label', data=df, palette='tab20', s=50, alpha=0.7)
        
        plt.title(f'UMAP Visualization of Embeddings (n_neighbors={n_neighbors})')
        plt.xlabel('UMAP dimension 1')
        plt.ylabel('UMAP dimension 2')
        
        # Improve legend if there are many classes
        if num_classes > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"UMAP visualization saved to {save_path}")
        
        return embeddings_2d, labels
    
    def visualize_retrieval_examples(self, dataloader, num_queries=5, k=5, save_dir=None):
        """Visualize some retrieval examples"""
        if save_dir is None:
            save_dir = os.path.join(self.config.RESULTS_DIR, "retrieval_examples")
            os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Creating retrieval example visualizations for {num_queries} queries...")
        
        # Compute embeddings
        embeddings, labels, image_paths = self.compute_embeddings(dataloader)
        
        if image_paths is None:
            self.logger.warning("Image paths not available, skipping retrieval visualization")
            return
        
        # Function to load and prepare an image for display
        def load_image(path):
            from PIL import Image
            img_path = os.path.join(self.data_module.data_dir, path)
            img = Image.open(img_path).convert('RGB')
            return img
        
        # Select random query images
        num_samples = len(labels)
        query_indices = np.random.choice(num_samples, size=min(num_queries, num_samples), replace=False)
        
        for i, query_idx in enumerate(query_indices):
            query_embedding = embeddings[query_idx]
            query_label = labels[query_idx]
            query_path = image_paths[query_idx]
            
            # Compute distances to all other embeddings
            distances = np.linalg.norm(embeddings - query_embedding, axis=1)
            
            # Get indices of k+1 closest embeddings (including the query itself)
            closest_indices = np.argsort(distances)[:k+1]
            
            # Remove the query itself if it's among the closest
            if closest_indices[0] == query_idx:
                closest_indices = closest_indices[1:k+2]
            else:
                closest_indices = closest_indices[:k+1]
            
            # Get paths and labels of retrieved images
            retrieved_paths = [image_paths[idx] for idx in closest_indices]
            retrieved_labels = [labels[idx] for idx in closest_indices]
            
            # Plot query and retrieved images
            fig, axes = plt.subplots(1, k+1, figsize=(15, 3))
            
            # Display query image
            query_img = load_image(query_path)
            axes[0].imshow(query_img)
            axes[0].set_title(f"Query\nClass: {query_label}")
            axes[0].axis('off')
            
            # Display retrieved images
            for j, (path, label) in enumerate(zip(retrieved_paths, retrieved_labels)):
                retrieved_img = load_image(path)
                axes[j+1].imshow(retrieved_img)
                
                # Mark correct/incorrect retrievals
                is_correct = label == query_label
                color = 'green' if is_correct else 'red'
                axes[j+1].set_title(f"Rank {j+1}\nClass: {label}", color=color)
                axes[j+1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"retrieval_query_{i+1}.png"), dpi=200, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Retrieval examples saved to {save_dir}")

def main():
    """Main function to run visualization as a standalone script"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualization for Deep Metric Learning')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to model checkpoint (default: outputs/models/best_model.pth)')
    parser.add_argument('--perplexity', type=int, default=None, 
                       help='Perplexity for t-SNE (default: from config)')
    parser.add_argument('--n_neighbors', type=int, default=None,
                       help='Number of neighbors for UMAP (default: from config)')
    parser.add_argument('--num_queries', type=int, default=5,
                       help='Number of query examples for retrieval visualization')
    args = parser.parse_args()
    
    try:
        # Import necessary modules here to avoid circular imports
        from config import Config
        from data_module import OxfordPetsDataModule
        from model import EmbeddingNet, ArcFaceModel
        
        # Load config
        config = Config()
        
        # Override config values if provided
        if args.perplexity is not None:
            config.TSNE_PERPLEXITY = args.perplexity
        if args.n_neighbors is not None:
            config.UMAP_N_NEIGHBORS = args.n_neighbors
        
        # Create output directories
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # Set up data module
        logger.info("Setting up data module...")
        data_module = OxfordPetsDataModule(config)
        data_module.setup()
        dataloaders = data_module.get_dataloaders()
        
        # Set default model path if not provided
        model_path = args.model_path if args.model_path else os.path.join(config.MODEL_DIR, "best_model.pth")
        
        logger.info(f"Starting visualization process...")
        logger.info(f"Loading model from: {model_path}")
        
        # Load the checkpoint with weights_only=True to avoid PyTorch warning
        checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
        logger.info(f"Checkpoint loaded successfully")
        
        # Initialize correct model type based on config or checkpoint
        if 'config' in checkpoint and hasattr(checkpoint['config'], 'LOSS_TYPE'):
            config_in_checkpoint = checkpoint['config']
            loss_type = config_in_checkpoint.LOSS_TYPE
        else:
            loss_type = config.LOSS_TYPE if hasattr(config, 'LOSS_TYPE') else 'triplet'
        
        logger.info(f"Using loss type: {loss_type}")
        if loss_type == 'arcface':
            model = ArcFaceModel(config, data_module.num_classes)
        else:
            model = EmbeddingNet(config)
        
        # Load the state dictionary
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Try loading as direct state dict
            model.load_state_dict(checkpoint)
            
        logger.info(f"Model initialized and weights loaded successfully")
        
        # Create visualizer
        visualizer = Visualizer(config, model, data_module)
        
        # Run visualizations
        test_loader = dataloaders['test']
        
        logger.info("Creating t-SNE visualization...")
        visualizer.visualize_tsne(test_loader)
        
        logger.info("Creating UMAP visualization...")
        visualizer.visualize_umap(test_loader)
        
        logger.info(f"Creating retrieval examples (n={args.num_queries})...")
        visualizer.visualize_retrieval_examples(test_loader, num_queries=args.num_queries)
        
        logger.info(f"All visualizations completed successfully!")
        logger.info(f"Visualizations saved to {config.RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Error in visualization process: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()