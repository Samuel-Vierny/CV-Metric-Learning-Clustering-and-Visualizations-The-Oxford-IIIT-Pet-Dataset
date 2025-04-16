"""
Loss functions for Deep Metric Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining
    """
    def __init__(self, margin=0.3, mining_type='batch_hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining_type = mining_type
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: tensor of shape (batch_size, embedding_size)
            labels: tensor of shape (batch_size)
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            # Need at least 2 samples to form a pair
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Compute pairwise distances
        dist_matrix = self._get_pairwise_distances(embeddings)
        
        # Get positive and negative mask based on labels
        pos_mask = self._get_positive_mask(labels)
        neg_mask = self._get_negative_mask(labels)
        
        # Check if we have any valid triplets
        if not pos_mask.any() or not neg_mask.any():
            # Create a dummy loss that's connected to the computation graph
            return torch.sum(embeddings) * 0.0
        
        if self.mining_type == 'batch_all':
            loss = self._batch_all_triplet_loss(dist_matrix, pos_mask, neg_mask)
        elif self.mining_type == 'batch_hard':
            loss = self._batch_hard_triplet_loss(dist_matrix, pos_mask, neg_mask)
        elif self.mining_type == 'batch_semi_hard':
            loss = self._batch_semi_hard_triplet_loss(dist_matrix, pos_mask, neg_mask)
        else:
            raise ValueError(f"Unsupported mining type: {self.mining_type}")
        
        return loss
    
    def _get_pairwise_distances(self, embeddings):
        """Compute the 2D matrix of distances between all embeddings."""
        # For numerical stability, we work with squared distances first
        squared_norm = torch.sum(embeddings ** 2, dim=1, keepdim=True)
        dot_product = torch.mm(embeddings, embeddings.t())
        distances_squared = squared_norm + squared_norm.t() - 2.0 * dot_product
        
        # Deal with numerical inaccuracies (can't have negative distances)
        distances_squared = torch.clamp(distances_squared, min=0.0)
        
        # Compute Euclidean distances
        distances = torch.sqrt(distances_squared + 1e-8)  # Small epsilon for numerical stability
        
        return distances
    
    def _get_positive_mask(self, labels):
        """Return a mask where mask[i, j] is True if embeddings i and j have the same label."""
        # Create a mask for indices that are equal
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        # We want embeddings from the same class but different indices
        indices_not_equal = ~indices_equal
        
        # Check if the labels are the same
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # We need both conditions: different indices and same labels
        mask = labels_equal & indices_not_equal
        
        return mask
    
    def _get_negative_mask(self, labels):
        """Return a mask where mask[i, j] is True if embeddings i and j have different labels."""
        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))
    
    def _batch_all_triplet_loss(self, dist_matrix, pos_mask, neg_mask):
        """
        Compute the triplet loss for all valid triplets and average over those that are active
        """
        # For each anchor-positive pair, compute loss for all negatives
        # Shape: [batch_size, batch_size, batch_size]
        anchor_positive_dist = dist_matrix.unsqueeze(2)
        anchor_negative_dist = dist_matrix.unsqueeze(1)
        
        # Compute triplet loss for all possible triplets
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Only consider valid triplets (anchor, positive, negative)
        # A triplet is valid if:
        # - anchor and positive are the same class (pos_mask)
        # - anchor and negative are different classes (neg_mask)
        mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        
        # Apply mask to triplet loss
        triplet_loss = torch.where(mask, triplet_loss, torch.zeros_like(triplet_loss))
        
        # Remove negative losses (easy triplets)
        triplet_loss = torch.clamp(triplet_loss, min=0.0)
        
        # Count number of positive (non-zero) triplets
        valid_triplets = (triplet_loss > 1e-16) & mask
        num_positive_triplets = valid_triplets.sum().float()
        
        # Average over positive triplets
        if num_positive_triplets < 1:
            # No triplets found, return a dummy loss
            return torch.sum(dist_matrix) * 0.0
        
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        
        return triplet_loss
    
    def _batch_hard_triplet_loss(self, dist_matrix, pos_mask, neg_mask):
        """
        For each anchor, consider the hardest positive and hardest negative
        """
        # Check if we have any valid positives or negatives
        if not pos_mask.any() or not neg_mask.any():
            # Return a dummy loss connected to input
            return torch.sum(dist_matrix) * 0.0
        
        # For each anchor, get the hardest positive
        # Set invalid positives to large negative values to be ignored in max operation
        max_dist = torch.max(dist_matrix) + 1.0
        masked_dist = torch.where(pos_mask, dist_matrix, -max_dist * torch.ones_like(dist_matrix))
        
        # Shape: [batch_size]
        hardest_positive_dist = masked_dist.max(dim=1)[0]
        
        # For each anchor, get the hardest negative
        # Set invalid negatives to large positive values to be ignored in min operation
        masked_dist = torch.where(neg_mask, dist_matrix, max_dist * torch.ones_like(dist_matrix))
        
        # Shape: [batch_size]
        hardest_negative_dist = masked_dist.min(dim=1)[0]
        
        # Combine the hardest positive and hardest negative distances
        triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0)
        
        # Compute mean over active triplets
        hard_triplets = triplet_loss > 1e-16
        num_hard_triplets = hard_triplets.sum().float()
        
        if num_hard_triplets == 0:
            # Return a dummy loss connected to input
            return torch.sum(dist_matrix) * 0.0
        
        triplet_loss = triplet_loss[hard_triplets].mean()
        
        return triplet_loss
    
    def _batch_semi_hard_triplet_loss(self, dist_matrix, pos_mask, neg_mask):
        """
        For each anchor, select semi-hard negatives: negatives that are 
        farther than the positive but still within the margin
        """
        # Check if we have any valid positives or negatives
        if not pos_mask.any() or not neg_mask.any():
            # Return a dummy loss connected to input
            return torch.sum(dist_matrix) * 0.0
        
        # Get all anchor-positive pairs
        anchor_positive_dist = dist_matrix.unsqueeze(2)
        
        # Get all anchor-negative pairs
        anchor_negative_dist = dist_matrix.unsqueeze(1)
        
        # Semi-hard negative: negative that is farther than positive but within margin
        semi_hard = (anchor_negative_dist > anchor_positive_dist) & \
                   (anchor_negative_dist < anchor_positive_dist + self.margin)
        
        # Valid triplets: anchor-positive pairs with semi-hard negatives
        valid_mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1) & semi_hard
        
        if not valid_mask.any():
            # If no semi-hard triplets, fall back to batch_hard
            return self._batch_hard_triplet_loss(dist_matrix, pos_mask, neg_mask)
        
        # Compute triplet loss for semi-hard triplets
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Apply mask (set non-valid to zero)
        triplet_loss = torch.where(valid_mask, triplet_loss, torch.zeros_like(triplet_loss))
        
        # Remove negative losses (easy triplets)
        triplet_loss = torch.clamp(triplet_loss, min=0.0)
        
        # Count number of positive (non-zero) triplets
        valid_triplets = (triplet_loss > 1e-16) & valid_mask
        num_positive_triplets = valid_triplets.sum().float()
        
        if num_positive_triplets < 1:
            # No triplets found, fall back to batch_hard
            return self._batch_hard_triplet_loss(dist_matrix, pos_mask, neg_mask)
        
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        
        return triplet_loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=0.3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: tensor of shape (batch_size, embedding_size)
            labels: tensor of shape (batch_size)
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            # Need at least 2 samples to form a pair
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Compute pairwise distances
        dist_matrix = self._get_pairwise_distances(embeddings)
        
        # Get positive and negative mask based on labels
        pos_mask = self._get_positive_mask(labels)
        neg_mask = self._get_negative_mask(labels)
        
        # Check if we have any valid pairs
        if not pos_mask.any() and not neg_mask.any():
            # Create a dummy loss that's connected to the computation graph
            return torch.sum(embeddings) * 0.0
        
        # Calculate the contrastive loss
        loss = 0.0
        
        # For positive pairs: distance should be small
        if pos_mask.any():
            pos_loss = dist_matrix[pos_mask].mean()
            loss += 0.5 * pos_loss
        
        # For negative pairs: distance should be at least margin
        if neg_mask.any():
            # Calculate how much each negative pair violates the margin
            neg_loss = torch.clamp(self.margin - dist_matrix, min=0.0)
            neg_loss = neg_loss[neg_mask].mean()
            loss += 0.5 * neg_loss
        
        return loss
    
    def _get_pairwise_distances(self, embeddings):
        """Compute the 2D matrix of distances between all embeddings."""
        # For numerical stability, we work with squared distances first
        squared_norm = torch.sum(embeddings ** 2, dim=1, keepdim=True)
        dot_product = torch.mm(embeddings, embeddings.t())
        distances_squared = squared_norm + squared_norm.t() - 2.0 * dot_product
        
        # Deal with numerical inaccuracies (can't have negative distances)
        distances_squared = torch.clamp(distances_squared, min=0.0)
        
        # Compute Euclidean distances
        distances = torch.sqrt(distances_squared + 1e-8)  # Small epsilon for numerical stability
        
        return distances
    
    def _get_positive_mask(self, labels):
        """Return a mask where mask[i, j] is True if embeddings i and j have the same label."""
        # Create a mask for indices that are equal
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        # We want embeddings from the same class but different indices
        indices_not_equal = ~indices_equal
        
        # Check if the labels are the same
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # We need both conditions: different indices and same labels
        mask = labels_equal & indices_not_equal
        
        return mask
    
    def _get_negative_mask(self, labels):
        """Return a mask where mask[i, j] is True if embeddings i and j have different labels."""
        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

class ArcFaceLoss(nn.Module):
    """
    ArcFace loss function for deep metric learning
    """
    def __init__(self, device, in_features, out_features, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.s = s  # scale factor
        self.m = m  # margin
        
        # Initialize weights
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-calculate constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        
        # Calculate cosine similarity
        cosine = F.linear(embeddings, weights)
        
        # Calculate arcface formula
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Ensure numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert labels to one-hot encoding
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply arcface margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        # Apply cross entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss