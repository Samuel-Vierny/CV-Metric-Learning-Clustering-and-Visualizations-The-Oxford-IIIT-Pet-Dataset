# Deep Metric Learning for Oxford-IIIT Pet Dataset

This project implements deep metric learning techniques for the Oxford-IIIT Pet Dataset, focusing on training a model that learns to recognize how similar or different pet breeds are based on their images.

## Project Structure

- `config.py`: Configuration parameters
- `data_module.py`: Dataset and dataloader implementation
- `evaluator.py`: Evaluation for verification, retrieval, and few-shot tasks
- `grad_cam.py`: Grad-CAM visualization implementation
- `hard_mining.py`: Hard negative mining implementation
- `loss_functions.py`: Loss functions (Triplet, Contrastive, ArcFace)
- `main.py`: Main script to run the code
- `model.py`: Model architecture definitions
- `trainer.py`: Training loop implementation
- `utils.py`: Utility functions
- `visualization.py`: Embedding space visualization tools

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- scikit-learn
- matplotlib
- numpy
- pandas
- tqdm
- faiss
- umap-learn
- opencv-python

## Dataset

The Oxford-IIIT Pet Dataset contains images of 37 pet breeds (25 dogs, 12 cats) with roughly 200 images for each class. The dataset includes breed labels and bounding boxes.

## Features

### Metric Learning Implementation

- CNN backbone (ResNet18/34/50) with projection head
- Loss functions: Triplet Loss, Contrastive Loss, ArcFace
- Hard negative mining

### Evaluation Tasks

1. **Verification**: Predict whether two images belong to the same breed
   - ROC-AUC
   - Equal Error Rate (EER)

2. **Retrieval**: Given a query image, retrieve top-K most similar images
   - Recall@K and Precision@K (K = 1, 5, 10)

3. **Few-shot Classification**: Perform N-way K-shot classification
   - Classification accuracy

### Visualization

- t-SNE and UMAP embedding space visualization
- Grad-CAM attention maps
- Retrieval examples visualization

## Usage

### Training

```bash
python main.py --mode train --backbone resnet18 --embedding_size 128 --loss_type triplet --batch_size 64 --num_epochs 30
```

### Evaluation

```bash
python main.py --mode evaluate --load_model outputs/models/best_model.pth
```

### Visualization

```bash
python main.py --mode visualize --load_model outputs/models/best_model.pth
```

### All-in-one

```bash
python main.py --mode all
```

## Configuration

You can modify the configuration parameters in `config.py` or pass them as command-line arguments. Some key parameters include:

- `SEED`: Random seed for reproducibility
- `BACKBONE`: CNN backbone architecture (resnet18, resnet34, resnet50)
- `EMBEDDING_SIZE`: Dimensionality of the embedding space
- `LOSS_TYPE`: Loss function to use (triplet, contrastive, arcface)
- `MARGIN`: Margin for triplet and contrastive loss
- `MINING_TYPE`: Mining strategy for triplet loss (batch_all, batch_hard, batch_semi_hard)

## Customization

The code is designed to be modular and easily customizable:

- Add new loss functions in `loss_functions.py`
- Implement different backbones in `model.py`
- Add new evaluation metrics in `evaluator.py`
- Customize visualizations in `visualization.py`

## Results

The results of the evaluation (verification, retrieval, few-shot classification) are saved in JSON format in the `outputs/results` directory. Visualization outputs are saved as PNG images in the same directory.
