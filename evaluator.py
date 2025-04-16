"""
Evaluator for Deep Metric Learning tasks
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
import logging
import faiss
import random

from utils import plot_roc_curve, compute_eer

class Evaluator:
    """Evaluator class for verification, retrieval, and few-shot tasks"""
    def __init__(self, config, model, data_module):
        self.config = config
        self.model = model
        self.data_module = data_module
        self.device = config.DEVICE
        self.logger = logging.getLogger(__name__)
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def compute_embeddings(self, dataloader):
        """Compute embeddings for all samples in the dataloader"""
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Computing embeddings"):
                images = images.to(self.device)
                
                # Get embeddings
                if hasattr(self.model, 'embedding_net'):
                    embeddings = self.model.embedding_net(images)
                else:
                    embeddings = self.model(images)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
        
        # Concatenate embeddings and labels
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        
        return all_embeddings, all_labels
    
    def evaluate_verification(self, test_loader):
        """Evaluate the model on the verification task"""
        self.logger.info("Evaluating on verification task...")
        
        # Compute embeddings for test set
        embeddings, labels = self.compute_embeddings(test_loader)
        
        # Generate verification pairs
        num_samples = embeddings.shape[0]
        num_pairs = min(10000, num_samples * (num_samples - 1) // 2)  # Limit number of pairs for memory efficiency
        
        pairs = []
        pair_labels = []
        
        # Sample positive pairs (same class)
        class_indices = {}
        for i, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        for label, indices in class_indices.items():
            if len(indices) >= 2:
                num_positive_pairs = min(len(indices) * (len(indices) - 1) // 2, num_pairs // 2)
                sampled_pairs = random.sample([(i, j) for i in indices for j in indices if i < j], 
                                              k=min(num_positive_pairs, len(indices) * (len(indices) - 1) // 2))
                pairs.extend(sampled_pairs)
                pair_labels.extend([1] * len(sampled_pairs))
        
        # Sample negative pairs (different classes)
        num_negative_pairs = num_pairs - len(pairs)
        neg_pair_candidates = []
        
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                if labels[i] != labels[j]:
                    neg_pair_candidates.append((i, j))
        
        if neg_pair_candidates:
            sampled_neg_pairs = random.sample(neg_pair_candidates, k=min(num_negative_pairs, len(neg_pair_candidates)))
            pairs.extend(sampled_neg_pairs)
            pair_labels.extend([0] * len(sampled_neg_pairs))
        
        # Compute distances for all pairs
        distances = []
        for i, j in pairs:
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(dist)
        
        # For verification, we want to predict that pairs with small distances are the same class
        # So we need to convert distances to similarities
        max_dist = max(distances)
        similarities = [1 - (dist / max_dist) for dist in distances]
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(pair_labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Compute Equal Error Rate (EER)
        eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
        
        # Plot ROC curve
        plot_path = os.path.join(self.config.RESULTS_DIR, "verification_roc.png")
        plot_roc_curve(fpr, tpr, roc_auc, plot_path)
        
        # Log results
        self.logger.info(f"Verification Results:")
        self.logger.info(f"  ROC AUC: {roc_auc:.4f}")
        self.logger.info(f"  Equal Error Rate (EER): {eer:.4f}")
        self.logger.info(f"  EER Threshold: {eer_threshold:.4f}")
        
        return {
            'roc_auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def evaluate_retrieval(self, test_loader, k_values=None):
        """Evaluate the model on the retrieval task"""
        if k_values is None:
            k_values = self.config.RETRIEVAL_K_VALUES
            
        self.logger.info(f"Evaluating on retrieval task for k values: {k_values}...")
        
        # Compute embeddings for test set
        embeddings, labels = self.compute_embeddings(test_loader)
        
        # Initialize FAISS index for fast similarity search
        d = embeddings.shape[1]  # Embedding dimension
        
        # Using L2 distance for similarity search
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.astype(np.float32))
        
        # Initialize metrics for each k
        precision_at_k = {k: [] for k in k_values}
        recall_at_k = {k: [] for k in k_values}
        
        # For each embedding, retrieve k-nearest neighbors
        for i, query_embedding in enumerate(embeddings):
            query_label = labels[i]
            
            # Find k-nearest neighbors (add 1 since the first result will be the query itself)
            max_k = max(k_values)
            distances, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), max_k + 1)
            
            # Skip the first result (the query itself)
            retrieved_indices = indices[0][1:]
            retrieved_labels = labels[retrieved_indices]
            
            # Compute precision and recall for each k
            for k in k_values:
                top_k_labels = retrieved_labels[:k]
                
                # Precision@k: proportion of retrieved items that are relevant
                precision = np.sum(top_k_labels == query_label) / k
                precision_at_k[k].append(precision)
                
                # Recall@k: proportion of relevant items that are retrieved
                # Here, we only have one relevant class, so recall is either 0 or 1
                # We consider recall as 1 if at least one of the retrieved items has the same label
                recall = 1.0 if np.any(top_k_labels == query_label) else 0.0
                recall_at_k[k].append(recall)
        
        # Compute average precision and recall for each k
        results = {}
        for k in k_values:
            avg_precision = np.mean(precision_at_k[k])
            avg_recall = np.mean(recall_at_k[k])
            
            results[f'precision@{k}'] = avg_precision
            results[f'recall@{k}'] = avg_recall
            
            self.logger.info(f"  Precision@{k}: {avg_precision:.4f}")
            self.logger.info(f"  Recall@{k}: {avg_recall:.4f}")
        
        return results
    
    def evaluate_few_shot(self, few_shot_loader=None, n_way=None, k_shot=None, num_tasks=None):
        """Evaluate the model on few-shot learning tasks"""
        if n_way is None:
            n_way = self.config.N_WAY
        if k_shot is None:
            k_shot = self.config.K_SHOT
        if num_tasks is None:
            num_tasks = self.config.NUM_FEW_SHOT_TASKS
            
        self.logger.info(f"Evaluating on {n_way}-way {k_shot}-shot learning with {num_tasks} tasks...")
        
        # If no specific few-shot loader is provided, use the test loader
        if few_shot_loader is None:
            few_shot_loader = self.data_module.get_dataloaders().get('few_shot', self.data_module.get_dataloaders()['test'])
        
        # Compute embeddings for few-shot dataset
        embeddings, labels = self.compute_embeddings(few_shot_loader)
        
        # Get unique classes
        unique_classes = np.unique(labels)
        
        if len(unique_classes) < n_way:
            self.logger.warning(f"Not enough classes for {n_way}-way few-shot learning. Found only {len(unique_classes)} classes.")
            n_way = len(unique_classes)
        
        # Initialize accuracy list
        accuracies = []
        
        # Run multiple random tasks
        for task_idx in tqdm(range(num_tasks), desc=f"{n_way}-way {k_shot}-shot evaluation"):
            # Randomly select n_way classes
            selected_classes = np.random.choice(unique_classes, size=n_way, replace=False)
            
            # Initialize support and query sets
            support_embeddings = []
            support_labels = []
            query_embeddings = []
            query_labels = []
            
            # For each selected class
            for class_idx, class_label in enumerate(selected_classes):
                # Get indices of samples with this class
                class_indices = np.where(labels == class_label)[0]
                
                if len(class_indices) < k_shot + 1:
                    self.logger.warning(f"Not enough samples for class {class_label}. Skipping task.")
                    continue
                
                # Randomly select k_shot samples for support set
                support_indices = np.random.choice(class_indices, size=k_shot, replace=False)
                
                # Use remaining samples for query set
                query_indices = np.array([idx for idx in class_indices if idx not in support_indices])
                
                # If there are too many query samples, limit them
                if len(query_indices) > 10:  # Arbitrarily limit to 10 query samples per class
                    query_indices = np.random.choice(query_indices, size=10, replace=False)
                
                # Add to support and query sets with new labels (0 to n_way-1)
                support_embeddings.append(embeddings[support_indices])
                support_labels.extend([class_idx] * len(support_indices))
                
                query_embeddings.append(embeddings[query_indices])
                query_labels.extend([class_idx] * len(query_indices))
            
            # Convert to numpy arrays
            support_embeddings = np.vstack(support_embeddings)
            support_labels = np.array(support_labels)
            query_embeddings = np.vstack(query_embeddings)
            query_labels = np.array(query_labels)
            
            # Train a k-nearest neighbors classifier on the support set
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(support_embeddings, support_labels)
            
            # Evaluate on query set
            predictions = knn.predict(query_embeddings)
            accuracy = np.mean(predictions == query_labels)
            accuracies.append(accuracy)
        
        # Compute average accuracy
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        self.logger.info(f"  {n_way}-way {k_shot}-shot Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return {
            'accuracy': mean_accuracy,
            'std': std_accuracy,
            'all_accuracies': accuracies
        }
    
    def run_all_evaluations(self):
        """Run all evaluation tasks"""
        dataloaders = self.data_module.get_dataloaders()
        test_loader = dataloaders['test']
        few_shot_loader = dataloaders.get('few_shot', None)
        
        # Verification
        verification_results = self.evaluate_verification(test_loader)
        
        # Retrieval
        retrieval_results = self.evaluate_retrieval(test_loader)
        
        # Few-shot learning
        if few_shot_loader is not None:
            few_shot_results = self.evaluate_few_shot(few_shot_loader)
        else:
            few_shot_results = self.evaluate_few_shot(test_loader)
        
        return {
            'verification': verification_results,
            'retrieval': retrieval_results,
            'few_shot': few_shot_results
        }
