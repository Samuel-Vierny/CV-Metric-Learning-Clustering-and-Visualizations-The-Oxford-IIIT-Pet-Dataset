Deep Metric Learning for Oxford-IIIT Pet Dataset
This project implements deep metric learning techniques for the Oxford-IIIT Pet Dataset, focusing on training a model that learns to recognize how similar or different pet breeds are based on their images.
Project Structure

class_names.py: Utility to extract breed names from dataset
config.py: Configuration parameters
data_module.py: Dataset and dataloader implementation
evaluator.py: Evaluation for verification, retrieval, and few-shot tasks
grad_cam.py: Grad-CAM visualization implementation (can be run as standalone)
hard_mining.py: Hard negative mining implementation
loss_functions.py: Loss functions (Triplet, Contrastive, ArcFace)
lr_schedulers.py: Custom learning rate schedulers
main.py: Main script to run the code
model.py: Model architecture definitions
suppress_tf_messages.py: Utility to silence TensorFlow warnings
trainer.py: Training loop implementation with learning rate scheduling and early stopping
utils.py: Utility functions
visualization.py: Embedding space visualization tools (can be run as standalone)

Requirements

Python 3.7+
PyTorch 1.7+
torchvision
scikit-learn
matplotlib
numpy
pandas
tqdm
faiss
umap-learn
opencv-python
seaborn

Dataset
The Oxford-IIIT Pet Dataset contains images of 37 pet breeds (25 dogs, 12 cats) with roughly 200 images for each class. The dataset includes breed labels and pet segmentation masks. Cat breeds begin with an uppercase letter, while dog breeds begin with a lowercase letter.
Features
Metric Learning Implementation

CNN backbone (ResNet18/34/50) with projection head
Loss functions: Triplet Loss, Contrastive Loss, ArcFace
Hard negative mining strategies: batch_all, batch_hard, batch_semi_hard
Learning rate scheduling: step, plateau, cosine, warmup_cosine
Early stopping with configurable patience

Evaluation Tasks

Verification: Predict whether two images belong to the same breed

ROC-AUC
Equal Error Rate (EER)


Retrieval: Given a query image, retrieve top-K most similar images

Recall@K and Precision@K (K = 1, 5, 10)


Few-shot Classification: Perform N-way K-shot classification

Classification accuracy on holdout breeds



Visualization

t-SNE and UMAP embedding space visualization
Grad-CAM attention maps
Retrieval examples visualization

Usage
Training
python main.py --mode train --backbone resnet18 --embedding_size 256 --loss_type triplet --batch_size 64 --num_epochs 30 --learning_rate 1e-4 --scheduler_type warmup_cosine
Evaluation
python main.py --mode evaluate --load_model outputs/models/best_model.pth
Visualization
python main.py --mode visualize --load_model outputs/models/best_model.pth
All-in-one
python main.py --mode all
Standalone Grad-CAM
python grad_cam.py --model_path outputs/models/best_model.pth --num_samples 10
Standalone Visualization
python visualization.py --model_path outputs/models/best_model.pth --num_queries 8
Configuration
You can modify the configuration parameters in config.py or pass them as command-line arguments. Some key parameters include:
General

SEED: Random seed for reproducibility (default: 42)
DEVICE: Device to use (cuda or cpu)
DATA_DIR: Directory with the dataset
SAVE_DIR: Directory to save outputs

Dataset

IMG_SIZE: Image size for resizing (default: 224)
TRAIN_VAL_SPLIT: Portion of non-test data for training (default: 0.8)
TRAIN_TEST_SPLIT: Portion of data for training+validation (default: 0.9)
BATCH_SIZE: Batch size (default: 64)
NUM_WORKERS: Number of workers for data loading (default: 13)

Model

BACKBONE: CNN backbone architecture (resnet18, resnet34, resnet50)
PRETRAINED: Whether to use pretrained backbone (default: True)
EMBEDDING_SIZE: Dimensionality of the embedding space (default: 256)

Training

NUM_EPOCHS: Number of training epochs (default: 4)
LEARNING_RATE: Learning rate (default: 1e-4)
WEIGHT_DECAY: Weight decay (default: 1e-5)

Loss Function

LOSS_TYPE: Loss function to use (triplet, contrastive, arcface)
MARGIN: Margin for triplet and contrastive loss (default: 0.3)
MINING_TYPE: Mining strategy for triplet loss (batch_all, batch_hard, batch_semi_hard)

Learning Rate Scheduler

SCHEDULER_TYPE: Type of scheduler (none, step, plateau, cosine, warmup_cosine)
SCHEDULER_STEP_SIZE: Steps between learning rate decay for StepLR (default: 10)
SCHEDULER_GAMMA: Learning rate decay factor (default: 0.1)
SCHEDULER_PATIENCE: Patience for ReduceLROnPlateau (default: 5)
SCHEDULER_MIN_LR: Minimum learning rate (default: 1e-6)
WARMUP_EPOCHS: Number of epochs for linear warmup (default: 1)

Regularization

DROPOUT_RATE: Dropout rate in network (default: 0.2)
EARLY_STOPPING: Whether to use early stopping (default: True)
EARLY_STOPPING_PATIENCE: Patience for early stopping (default: 3)
EARLY_STOPPING_DELTA: Minimum change to count as improvement (default: 0.0001)

Few-shot Learning

N_WAY: Number of classes for few-shot tasks (default: 5)
K_SHOT: Number of samples per class for few-shot tasks (default: 5)
NUM_FEW_SHOT_TASKS: Number of few-shot tasks to evaluate (default: 100)
FEW_SHOT_HOLDOUT_BREEDS: List of specific breeds to hold out for few-shot evaluation

Visualization

TSNE_PERPLEXITY: Perplexity parameter for t-SNE (default: 30)
UMAP_N_NEIGHBORS: Number of neighbors for UMAP (default: 15)

Results
The results of the evaluation (verification, retrieval, few-shot classification) are saved in JSON format in the outputs/results directory. Visualization outputs are saved as PNG images in the same directory.
Key visualizations include:

Loss curves
Learning rate schedule curves
t-SNE and UMAP embedding visualizations
ROC curves for verification
Retrieval examples
Grad-CAM attention maps

Customization
The code is designed to be modular and easily customizable:

Add new loss functions in loss_functions.py
Implement different backbones in model.py
Add new learning rate schedulers in lr_schedulers.py
Add new evaluation metrics in evaluator.py
Customize visualizations in visualization.py