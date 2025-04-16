"""
Hard negative mining implementation for Deep Metric Learning
"""
import torch
import numpy as np
import logging
from tqdm import tqdm

class HardNegativeMiner:
    """
    Class implementing hard negative mining for improved training of deep metric learning models

     # Initialize with model, data, and configuration

    """
    def __init__(self, model, data_module, config):
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = config.DEVICE
        self.logger = logging.getLogger(__name__)
        
        # Set model to evaluation mode for mining
        self.model.eval()
    
    def mine_hard_negatives(self, num_hard_negatives=100):
        """
        Mine hard negative examples from the training set
        
        Returns a list of (anchor_idx, positive_idx, negative_idx) triplets
        with hard negatives that are close to the anchor but from a different class

        # 1. Compute embeddings for all training samples
        # 2. Calculate pairwise distances between embeddings
        # 3. For each anchor:
        #    - Find samples of the same class (positives)
        #    - Find samples of different classes (negatives)
        #    - Sort negatives by distance and select the closest ones
        #    - Create triplets: (anchor, positive, hard negative)
        # 4. Return list of hard triplets

        """
        self.logger.info("Mining hard negative examples...")
        
        # Get training dataloader
        dataloaders = self.data_module.get_dataloaders()
        train_loader = dataloaders['train']
        
        # Compute embeddings for all training samples
        all_embeddings = []
        all_labels = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Computing embeddings")):
                images = images.to(self.device)
                
                # Get embeddings
                if hasattr(self.model, 'embedding_net'):
                    embeddings = self.model.embedding_net(images)
                else:
                    embeddings = self.model(images)
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
                
                # Keep track of indices in the original dataset
                if hasattr(train_loader.dataset, 'indices'):
                    indices = torch.tensor([train_loader.dataset.indices[i + batch_idx * train_loader.batch_size] 
                                          for i in range(len(labels))])
                else:
                    indices = torch.tensor([i + batch_idx * train_loader.batch_size for i in range(len(labels))])
                
                all_indices.append(indices)
        
        # Concatenate all embeddings and labels
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        
        # Compute pairwise distances between all embeddings
        num_samples = all_embeddings.size(0)
        
        # Create a distance matrix in a memory-efficient way
        distance_matrix = torch.zeros(num_samples, num_samples)
        
        batch_size = 128  # Process in batches to avoid OOM
        for i in range(0, num_samples, batch_size):
            end_i = min(i + batch_size, num_samples)
            for j in range(0, num_samples, batch_size):
                end_j = min(j + batch_size, num_samples)
                
                # Compute squared Euclidean distances
                dist_batch = torch.cdist(all_embeddings[i:end_i], all_embeddings[j:end_j], p=2)
                distance_matrix[i:end_i, j:end_j] = dist_batch
        
        # Mine hard negative triplets
        hard_triplets = []
        
        for anchor_idx in tqdm(range(num_samples), desc="Mining hard triplets"):
            anchor_label = all_labels[anchor_idx].item()
            
            # Find all positives (same class)
            positive_indices = torch.where(all_labels == anchor_label)[0]
            positive_indices = positive_indices[positive_indices != anchor_idx]  # Exclude anchor itself
            
            if len(positive_indices) == 0:
                continue  # Skip if no positives
            
            # Find all negatives (different class)
            negative_indices = torch.where(all_labels != anchor_label)[0]
            
            if len(negative_indices) == 0:
                continue  # Skip if no negatives
            
            # Get distances to all negatives
            neg_distances = distance_matrix[anchor_idx, negative_indices]
            
            # Sort negatives by distance (ascending) and take the closest ones
            _, sorted_neg_indices = torch.sort(neg_distances)
            hard_negative_indices = negative_indices[sorted_neg_indices[:min(len(sorted_neg_indices), 10)]]
            
            # For each hard negative, create a triplet with a random positive
            for neg_idx in hard_negative_indices:
                pos_idx = positive_indices[torch.randint(0, len(positive_indices), (1,))]
                
                # Store original dataset indices
                anchor_orig_idx = all_indices[anchor_idx].item()
                pos_orig_idx = all_indices[pos_idx].item()
                neg_orig_idx = all_indices[neg_idx].item()
                
                hard_triplets.append((anchor_orig_idx, pos_orig_idx, neg_orig_idx))
                
                if len(hard_triplets) >= num_hard_negatives:
                    break
            
            if len(hard_triplets) >= num_hard_negatives:
                break
        
        self.logger.info(f"Mined {len(hard_triplets)} hard negative triplets")
        return hard_triplets
    
    def create_hard_triplet_batch(self, triplets, batch_size=32):
        """
        Create a batch of images from hard triplets
        
        Args:
            triplets: List of (anchor_idx, positive_idx, negative_idx) triplets
            batch_size: Number of triplets to include in the batch
        
        Returns:
            images: Tensor of shape (batch_size * 3, C, H, W)
            labels: Tensor of shape (batch_size * 3) with labels indicating triplet membership

        # Create training batches from the mined triplets

        """
        # Sample a subset of triplets
        if len(triplets) > batch_size:
            selected_triplets = np.random.choice(len(triplets), batch_size, replace=False)
            triplets = [triplets[i] for i in selected_triplets]
        
        # Create a batch
        images = []
        labels = []
        
        # Get file list from the dataset
        file_list = self.data_module.train_dataset.file_list
        transform = self.data_module.train_transform
        
        for i, (anchor_idx, pos_idx, neg_idx) in enumerate(triplets):
            # Load anchor image
            anchor_path, anchor_label = file_list[anchor_idx]
            anchor_img = Image.open(os.path.join(self.data_module.data_dir, anchor_path)).convert('RGB')
            anchor_img = transform(anchor_img)
            
            # Load positive image
            pos_path, pos_label = file_list[pos_idx]
            pos_img = Image.open(os.path.join(self.data_module.data_dir, pos_path)).convert('RGB')
            pos_img = transform(pos_img)
            
            # Load negative image
            neg_path, neg_label = file_list[neg_idx]
            neg_img = Image.open(os.path.join(self.data_module.data_dir, neg_path)).convert('RGB')
            neg_img = transform(neg_img)
            
            # Add to batch
            images.extend([anchor_img, pos_img, neg_img])
            labels.extend([anchor_label, pos_label, neg_label])
        
        # Convert to tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return images, labels
    
    def train_with_hard_mining(self, trainer, num_epochs=5, mining_frequency=2, num_hard_triplets=100):
        """
        Train the model with periodic hard negative mining
        
        Args:
            trainer: Trainer object
            num_epochs: Number of epochs to train
            mining_frequency: Mine hard negatives every this many epochs
            num_hard_triplets: Number of hard triplets to mine each time


        # Train model with periodic hard negative mining
        # 1. Train normally for some epochs
        # 2. Mine hard negatives periodically
        # 3. Train specifically on hard triplets
        # 4. Continue regular training
        
        """
        self.logger.info(f"Training with hard negative mining every {mining_frequency} epochs")
        
        # Get dataloaders
        dataloaders = self.data_module.get_dataloaders()
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss = trainer.train_epoch(train_loader)
            trainer.train_losses.append(train_loss)
            
            # Mine hard negatives periodically
            if epoch % mining_frequency == 0:
                self.logger.info(f"Epoch {epoch}: Mining hard negatives...")
                hard_triplets = self.mine_hard_negatives(num_hard_triplets)
                
                # Train on hard triplets
                if hard_triplets:
                    self.logger.info(f"Training on {len(hard_triplets)} hard triplets")
                    trainer.model.train()
                    
                    # Process in batches
                    batch_size = min(32, len(hard_triplets))
                    num_batches = (len(hard_triplets) + batch_size - 1) // batch_size
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(hard_triplets))
                        batch_triplets = hard_triplets[start_idx:end_idx]
                        
                        images, labels = self.create_hard_triplet_batch(batch_triplets)
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Forward pass
                        trainer.optimizer.zero_grad()
                        
                        if hasattr(trainer.model, 'embedding_net'):
                            embeddings = trainer.model.embedding_net(images)
                        else:
                            embeddings = trainer.model(images)
                        
                        loss = trainer.criterion(embeddings, labels)
                        
                        # Backward pass and optimize
                        loss.backward()
                        trainer.optimizer.step()
            
            # Validate
            val_loss = trainer.validate(val_loader)
            trainer.val_losses.append(val_loss)
            
            # Log progress
            self.logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            trainer.save_checkpoint(epoch, val_loss)
        
        self.logger.info("Training with hard negative mining completed")
