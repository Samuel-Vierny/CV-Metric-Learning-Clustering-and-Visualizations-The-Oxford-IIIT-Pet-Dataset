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
import argparse
import sys

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
            # Default to the last convolutional layer in ResNet based on architecture
            if hasattr(model, 'embedding_net'):
                # For ResNet50, the last conv layer is conv3 in the last bottleneck block
                if 'resnet50' in str(model.embedding_net.backbone):
                    target_layer = self._find_last_conv_layer(model.embedding_net.backbone)
                # For ResNet18/34, the last conv layer is conv2 in the last basic block
                else:
                    target_layer = self._find_last_conv_layer(model.embedding_net.backbone)
            else:
                target_layer = self._find_last_conv_layer(model.backbone)
        else:
            # Find the layer by name
            target_layer = self._get_layer_by_name(self.model, layer_name)
        
        if target_layer is None:
            raise ValueError(f"Could not find appropriate layer for GradCAM. Please specify a valid layer_name.")
        
        self.logger.info(f"Using layer {type(target_layer).__name__} for GradCAM")
        self.target_layer = target_layer
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        # Register forward hook
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        
        # Register backward hook
        self.backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _find_last_conv_layer(self, backbone):
        """Find the last convolutional layer in the backbone"""
        # Navigate through the backbone to find the last convolutional layer
        last_conv = None
        
        # Approach 1: Try to find the last layer group and its last block
        try:
            # For ResNet architectures, the last layer is usually in layer4/layer group 7
            if hasattr(backbone, 'layer4'):
                last_block = backbone.layer4[-1]
                if hasattr(last_block, 'conv3'):  # Bottleneck block (ResNet50)
                    last_conv = last_block.conv3
                elif hasattr(last_block, 'conv2'):  # Basic block (ResNet18/34)
                    last_conv = last_block.conv2
            # Alternative approach if the above doesn't work
            elif len(list(backbone.children())) >= 8:
                # Try to access the last layer group and its last block
                last_layer_group = list(backbone.children())[-3]  # Usually the last layer group is 3rd from end
                last_block = last_layer_group[-1]
                
                # Check for conv2 (ResNet18/34) or conv3 (ResNet50)
                if hasattr(last_block, 'conv3'):
                    last_conv = last_block.conv3
                elif hasattr(last_block, 'conv2'):
                    last_conv = last_block.conv2
        except (AttributeError, IndexError) as e:
            self.logger.warning(f"Could not find last conv layer using standard approach: {str(e)}")
        
        # Approach 2: Fallback to searching through all modules
        if last_conv is None:
            for name, module in reversed(list(backbone.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    self.logger.info(f"Using fallback approach, found conv layer: {name}")
                    last_conv = module
                    break
        
        return last_conv
    
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
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
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


# Add standalone script functionality
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for a trained model')
    parser.add_argument('--model_path', type=str, default='outputs/models/best_model.pth',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/archive/images',
                        help='Directory containing the dataset images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save Grad-CAM visualizations (default: outputs/results/grad_cam)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize (default: 5)')
    parser.add_argument('--layer_name', type=str, default=None,
                        help='Name of the layer to use for Grad-CAM (default: last conv layer)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for data loading (default: 64)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for processing (default: 224)')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Backbone CNN architecture (resnet18, resnet34, resnet50), defaults to checkpoint value')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Embedding dimension size (default: 256)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Import necessary modules
        logger.info("Importing required modules...")
        from config import Config
        from data_module import OxfordPetsDataModule
        from model import EmbeddingNet, ArcFaceModel
        
        # Load checkpoint to determine model architecture
        logger.info(f"Loading checkpoint from {args.model_path} to determine configuration...")
        checkpoint = torch.load(args.model_path, map_location='cpu')
        
        # Extract backbone from checkpoint if available
        backbone = args.backbone  # Default to arg-provided backbone
        if backbone is None:
            if 'config' in checkpoint and hasattr(checkpoint['config'], 'BACKBONE'):
                backbone = checkpoint['config'].BACKBONE
                logger.info(f"Using backbone from checkpoint: {backbone}")
            else:
                logger.warning("Could not determine backbone from checkpoint. Defaulting to resnet18.")
                backbone = "resnet18"
        
        # Create config with the correct backbone
        logger.info(f"Setting up configuration with backbone: {backbone}...")
        config = Config()
        config.DATA_DIR = args.data_dir
        config.BATCH_SIZE = args.batch_size
        config.IMG_SIZE = args.img_size
        config.BACKBONE = backbone
        config.EMBEDDING_SIZE = args.embedding_size
        
        # Determine loss type from checkpoint if available
        loss_type = "triplet"  # Default
        if 'config' in checkpoint and hasattr(checkpoint['config'], 'LOSS_TYPE'):
            loss_type = checkpoint['config'].LOSS_TYPE
            config.LOSS_TYPE = loss_type
            logger.info(f"Using loss type from checkpoint: {loss_type}")
        
        # Create output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join('outputs', 'results', 'grad_cam')
        os.makedirs(output_dir, exist_ok=True)
        config.RESULTS_DIR = os.path.dirname(output_dir)
        
        # Set up data module
        logger.info("Setting up data module...")
        data_module = OxfordPetsDataModule(config)
        data_module.setup()
        
        # Create model with the correct architecture
        if loss_type == 'arcface':
            logger.info(f"Creating ArcFace model with {backbone} backbone...")
            model = ArcFaceModel(config, data_module.num_classes)
        else:
            logger.info(f"Creating Embedding model with {backbone} backbone...")
            model = EmbeddingNet(config)
        
        # Load model weights
        try:
            logger.info("Loading model weights...")
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume it's just the state dict
                model.load_state_dict(checkpoint)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            sys.exit(1)
        
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"Model loaded successfully to {device}")
        
        # Create Grad-CAM instance
        logger.info("Creating Grad-CAM instance...")
        gradcam = GradCAM(model, layer_name=args.layer_name)
        
        # Generate visualizations
        logger.info(f"Generating Grad-CAM visualizations for {args.num_samples} samples...")
        gradcam.generate_multiple(data_module, num_samples=args.num_samples, save_dir=output_dir)
        
        logger.info(f"Grad-CAM visualizations completed. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()