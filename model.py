"""
Model definitions for Deep Metric Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class EmbeddingNet(nn.Module):
    """
    CNN backbone with a projection head for metric learning
    """
    def __init__(self, config):
        super(EmbeddingNet, self).__init__()
        self.config = config
        
        # Initialize backbone with correct weights parameter
        if config.BACKBONE == "resnet18":
            if config.PRETRAINED:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                self.backbone = models.resnet18(weights=None)
            feature_dim = 512
        elif config.BACKBONE == "resnet34":
            if config.PRETRAINED:
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                self.backbone = models.resnet34(weights=None)
            feature_dim = 512
        elif config.BACKBONE == "resnet50":
            if config.PRETRAINED:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                self.backbone = models.resnet50(weights=None)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {config.BACKBONE}")
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection head - maps features to embedding space with dropout for regularization
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),  # Add dropout after activation
            nn.Linear(512, config.EMBEDDING_SIZE)
        )
        
    def forward(self, x):
        """Forward pass to get embeddings"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection_head(features)
        
        # Normalize embeddings to be on the unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class ArcFaceModel(nn.Module):
    """
    Model with ArcFace head for classification during training
    """
    def __init__(self, config, num_classes):
        super(ArcFaceModel, self).__init__()
        self.embedding_net = EmbeddingNet(config)
        self.arc_margin_product = ArcMarginProduct(
            config.EMBEDDING_SIZE, 
            num_classes, 
            s=30.0, 
            m=0.5
        )
        
    def forward(self, x, labels=None):
        embeddings = self.embedding_net(x)
        
        if labels is not None:
            outputs = self.arc_margin_product(embeddings, labels)
            return outputs, embeddings
        return embeddings

class ArcMarginProduct(nn.Module):
    """
    ArcFace loss implementation
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(math.pi - m))
        self.mm = torch.sin(torch.tensor(math.pi - m)) * m
        
    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output