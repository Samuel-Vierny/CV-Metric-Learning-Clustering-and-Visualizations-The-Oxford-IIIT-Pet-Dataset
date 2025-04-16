"""
Grad-CAM implementation for Deep Metric Learning Model visualization
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from torchvision import transforms
import logging

class GradCAM:
    """
    Grad-CAM implementation for visualizing which parts of an image the model focuses on
    """
    def __init__(self, model, layer_name=None):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.logger = logging.getLogger(__name__)
        
        # Get a hook for the selected layer
        if layer_name is None:
            # Default to the last convolutional layer in ResNet
            if hasattr(model, 'embedding_net'):
                target_layer = model.embedding_net.backbone[-2][-1].conv2
            else:
                target_layer = model.backbone[-2][-1].conv2
        else:
            # Find the layer by name
            target_layer = self._get_layer_by_name(self.model, layer_name)
        
        if target_layer is None:
            raise ValueError(f"Could not find layer: {layer_name}")
        
        self.target_layer = target_layer
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        # Register forward hook
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        
        # Register backward hook
        self.backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _get_layer_by_name(self, model, layer_name):
        """Get a layer module by name"""
        # Split the layer name by dots for nested access
        parts = layer_name.split('.')
        curr_module = model
        
        for part in parts:
            if part.isdigit():
                curr_module = curr_module[int(part)]
            else:
                curr_module = getattr(curr_module, part, None)
                if curr_module is None:
                    return None
        
        return curr_module
    
    def _forward_hook(self, module, input, output):
        """Hook for saving the activations"""
        self.activations = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook for saving the gradients"""
        self.gradients = grad_output[0]
    
    def __del__(self):
        """Clean up hooks on deletion"""
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def _preprocess_image(self, image_path, transform=None):
        """Load and preprocess an image"""
        image = Image.open(image_path).convert('RGB')
        
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def _compute_gradcam(self, image_tensor, target_idx=None):
        """Compute Grad-CAM for the given image and target index"""
        # Forward pass
        if hasattr(self.model, 'embedding_net'):
            embeddings = self.model.embedding_net(image_tensor)
        else:
            embeddings = self.model(image_tensor)
        
        # If no target index is specified, use the embedding norm
        if target_idx is None:
            outputs = torch.sum(embeddings)
        else:
            outputs = embeddings[0, target_idx]
        
        # Backward pass
        self.model.zero_grad()
        outputs.backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu().numpy()[0]  # [C, H, W]
        activations = self.activations.detach().cpu().numpy()[0]  # [C, H, W]
        
        # Calculate weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Compute weighted activation map
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU to focus on features that have a positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize between 0-1
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-10)
        
        return cam
    
    def overlay_heatmap(self, image_path, save_path, target_idx=None, alpha=0.5, color_map=cv2.COLORMAP_JET, transform=None):
        """Generate and overlay Grad-CAM heatmap on the original image"""
        # Preprocess image
        image_tensor = self._preprocess_image(image_path, transform)
        
        # Compute Grad-CAM
        cam = self._compute_gradcam(image_tensor, target_idx)
        
        # Load the original image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Resize CAM to match original image size
        cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Apply color map to create heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), color_map)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlaid = original_image * (1 - alpha) + heatmap * alpha
        overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
        
        # Save the result
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlaid)
        plt.title('Overlaid Result')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Grad-CAM visualization saved to {save_path}")
        
        return overlaid
    
    def generate_multiple(self, data_module, num_samples=5, save_dir=None, transform=None):
        """Generate Grad-CAM visualizations for multiple samples"""
        if save_dir is None:
            save_dir = os.path.join(data_module.config.RESULTS_DIR, "grad_cam")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get dataloader
        dataloaders = data_module.get_dataloaders()
        test_loader = dataloaders['test']
        
        # Get a few samples from the test set
        samples = []
        labels = []
        image_paths = []
        
        for batch_idx, (images, batch_labels) in enumerate(test_loader):
            for i in range(len(batch_labels)):
                # Get image path
                if hasattr(test_loader.dataset, 'file_list'):
                    img_idx = i + batch_idx * test_loader.batch_size
                    if hasattr(test_loader.dataset, 'indices'):
                        img_idx = test_loader.dataset.indices[img_idx]
                    img_path = os.path.join(data_module.data_dir, test_loader.dataset.file_list[img_idx][0])
                    
                    samples.append(images[i])
                    labels.append(batch_labels[i].item())
                    image_paths.append(img_path)
                    
                    if len(samples) >= num_samples:
                        break
            
            if len(samples) >= num_samples:
                break
        
        # Generate Grad-CAM for each sample
        for i, (image, label, image_path) in enumerate(zip(samples, labels, image_paths)):
            save_path = os.path.join(save_dir, f"gradcam_sample_{i+1}_class_{label}.png")
            
            # Create tensor for the image
            image_tensor = image.unsqueeze(0).to(self.device)
            
            # Compute Grad-CAM
            cam = self._compute_gradcam(image_tensor)
            
            # Load the original image
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Resize CAM to match original image size
            cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
            
            # Apply color map to create heatmap
            heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap on original image
            alpha = 0.5
            overlaid = original_image * (1 - alpha) + heatmap * alpha
            overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
            
            # Save the result
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title(f'Original Image\nClass: {label}')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap)
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlaid)
            plt.title('Overlaid Result')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Grad-CAM visualization saved to {save_path}")
        
        self.logger.info(f"Generated {len(samples)} Grad-CAM visualizations in {save_dir}")
