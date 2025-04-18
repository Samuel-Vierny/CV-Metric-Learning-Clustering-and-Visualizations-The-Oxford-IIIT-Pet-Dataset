"""
Configuration for Deep Metric Learning on Oxford-IIIT Pet Dataset
"""
import os
import torch

class Config:
    # General
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = os.path.join("data", "archive", "images")
    SAVE_DIR = "outputs"
    RESULTS_DIR = os.path.join(SAVE_DIR, "results")
    MODEL_DIR = os.path.join(SAVE_DIR, "models")
    
    # Dataset
    IMG_SIZE = 224
    TRAIN_VAL_SPLIT = 0.8  # 80% of non-test data for training, 20% for validation
    TRAIN_TEST_SPLIT = 0.9  # 90% for train+val, 10% for test
    NUM_WORKERS = 13
    BATCH_SIZE = 64
    
    # Model
    BACKBONE = "resnet18"  # Options: "resnet18", "resnet34", "resnet50"
    PRETRAINED = True
    EMBEDDING_SIZE = 256
    
    # Training
    NUM_EPOCHS = 2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss Function
    LOSS_TYPE = "arcface"  # Options: "triplet", "contrastive", "arcface"
    MARGIN = 0.3  # For triplet and contrastive loss
    
    # Triplet Mining
    MINING_TYPE = "batch_hard"  # Options: "batch_all", "batch_hard", "batch_semi_hard"
    
    # Verification Task
    VERIFICATION_THRESHOLD = 0.5
    
    # Retrieval Task
    RETRIEVAL_K_VALUES = [1, 5, 10]
    
    # Few-shot Learning
    N_WAY = 5
    K_SHOT = 5
    NUM_FEW_SHOT_TASKS = 100
    
    # Visualization
    TSNE_PERPLEXITY = 30
    UMAP_N_NEIGHBORS = 15
    
    # Holdout Breeds for Few-shot Learning
    FEW_SHOT_HOLDOUT_BREEDS = [
        "american_bulldog",
        "Egyptian_Mau",
        "samoyed",
        "Siamese",
        "wheaten_terrier"
    ]
    
    # Learning Rate Scheduler
    SCHEDULER_TYPE = "warmup_cosine"  # Options: "none", "step", "plateau", "cosine", "warmup_cosine"
    SCHEDULER_STEP_SIZE = 10  # For StepLR
    SCHEDULER_GAMMA = 0.1     # Learning rate decay factor
    SCHEDULER_PATIENCE = 5    # For ReduceLROnPlateau
    SCHEDULER_MIN_LR = 1e-6   # Minimum learning rate
    SCHEDULER_T_MAX = None    # For cosine annealing (uses NUM_EPOCHS if None)
    WARMUP_EPOCHS = 1         # Warmup epochs for warmup scheduler

    # Regularization
    DROPOUT_RATE = 0.2        # Dropout rate in network
    EARLY_STOPPING = True    # Whether to use early stopping
    EARLY_STOPPING_PATIENCE = 3  # Patience for early stopping
    EARLY_STOPPING_DELTA = 0.0001 # Minimum change to qualify as improvement


    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def update(self, **kwargs):
        """Update config parameters"""
        for key, value in kwargs.items():
            setattr(self, key, value)
