"""
Data module for the Oxford-IIIT Pet Dataset
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class OxfordPetsDataset(Dataset):
    """Dataset class for Oxford-IIIT Pet Dataset"""
    def __init__(self, data_dir, file_list, transform=None):
        """
        Args:
            data_dir: Directory with all the images
            file_list: List of (image_path, class_id) tuples
            transform: Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path, class_id = self.file_list[idx]
        try:
            image = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(class_id, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the class_id
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, torch.tensor(class_id, dtype=torch.long)

class OxfordPetsDataModule:
    """Data module for loading and processing the Oxford-IIIT Pet Dataset"""
    def __init__(self, config):
        self.config = config
        self.data_dir = config.DATA_DIR
        
        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
    def _parse_annotations(self):
        """Parse the dataset annotations"""
        # Get all image files
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        
        # FIXED: Extract breed names correctly by preserving compound names
        # and maintaining original capitalization
        breed_names = []
        breed_to_idx = {}
        seen_breeds = set()
        
        for img_file in image_files:
            # Extract everything before the last underscore and number
            import re
            match = re.match(r'(.+)_(\d+)\.jpg$', img_file)
            if match:
                breed = match.group(1)  # Preserve original capitalization
                if breed not in seen_breeds:
                    seen_breeds.add(breed)
                    breed_names.append(breed)
        
        # Sort the breed names and create the mapping
        breed_names = sorted(breed_names)
        breed_to_idx = {breed: idx for idx, breed in enumerate(breed_names)}
        
        # Create file list with (image_path, class_id)
        file_list = []
        for img_file in image_files:
            match = re.match(r'(.+)_(\d+)\.jpg$', img_file)
            if match:
                breed = match.group(1)
                class_id = breed_to_idx[breed]
                file_list.append((img_file, class_id))
        
        self.breed_names = breed_names
        self.num_classes = len(breed_names)
        self.breed_to_idx = breed_to_idx
        
        # Debug: Print detected breeds
        if hasattr(self.config, 'FEW_SHOT_HOLDOUT_BREEDS'):
            print("Configured holdout breeds:", self.config.FEW_SHOT_HOLDOUT_BREEDS)
            print("Found these in dataset:", [b for b in self.config.FEW_SHOT_HOLDOUT_BREEDS if b in self.breed_to_idx])
        
        return file_list, breed_names, breed_to_idx
    
    def _split_data(self, file_list):
        """Split data into train, validation, and test sets"""
        # First split into train+val and test
        train_val_list, test_list = train_test_split(
            file_list, 
            test_size=1-self.config.TRAIN_TEST_SPLIT if hasattr(self.config, 'TRAIN_TEST_SPLIT') else 0.1,
            stratify=[item[1] for item in file_list],
            random_state=self.config.SEED
        )
        
        # Then split train+val into train and val
        train_list, val_list = train_test_split(
            train_val_list,
            test_size=1-self.config.TRAIN_VAL_SPLIT if hasattr(self.config, 'TRAIN_VAL_SPLIT') else 0.2,
            stratify=[item[1] for item in train_val_list],
            random_state=self.config.SEED
        )
        
        return train_list, val_list, test_list
    
    def _holdout_breeds_for_few_shot(self, train_list, val_list, test_list):
        """Hold out specific breeds for few-shot learning"""
        if hasattr(self.config, 'FEW_SHOT_HOLDOUT_BREEDS') and self.config.FEW_SHOT_HOLDOUT_BREEDS:
            holdout_indices = [self.breed_to_idx[breed] for breed in self.config.FEW_SHOT_HOLDOUT_BREEDS 
                              if breed in self.breed_to_idx]
        else:
            # If no specific breeds defined, hold out a random selection of breeds
            num_holdout = 5  # Hold out 5 breeds by default
            all_classes = list(range(self.num_classes))
            np.random.seed(self.config.SEED)
            holdout_indices = np.random.choice(all_classes, size=min(num_holdout, len(all_classes)), replace=False)
        
        # Filter out holdout breeds from training and validation sets
        train_list_filtered = [(img, cls) for img, cls in train_list if cls not in holdout_indices]
        val_list_filtered = [(img, cls) for img, cls in val_list if cls not in holdout_indices]
        
        # Create a separate few-shot dataset with only holdout breeds
        few_shot_list = [(img, cls) for img, cls in test_list if cls in holdout_indices]
        
        # Add remaining test examples to test set
        test_list_filtered = [(img, cls) for img, cls in test_list if cls not in holdout_indices]
        
        return train_list_filtered, val_list_filtered, test_list_filtered, few_shot_list
    
    def setup(self):
        """Set up the datasets and dataloaders"""
        # Parse annotations and get file list
        file_list, breed_names, breed_to_idx = self._parse_annotations()
        
        # Split into train, validation, and test sets
        train_list, val_list, test_list = self._split_data(file_list)
        
        # Optionally holdout breeds for few-shot learning
        if hasattr(self.config, 'FEW_SHOT_HOLDOUT_BREEDS') or True:  # Always create few-shot dataset
            train_list, val_list, test_list, few_shot_list = self._holdout_breeds_for_few_shot(
                train_list, val_list, test_list
            )
            self.few_shot_dataset = OxfordPetsDataset(self.data_dir, few_shot_list, self.val_transform)
        
        # Create datasets
        self.train_dataset = OxfordPetsDataset(self.data_dir, train_list, self.train_transform)
        self.val_dataset = OxfordPetsDataset(self.data_dir, val_list, self.val_transform)
        self.test_dataset = OxfordPetsDataset(self.data_dir, test_list, self.val_transform)
        
        print(f"Dataset split: Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        if hasattr(self, 'few_shot_dataset'):
            print(f"Few-shot dataset: {len(self.few_shot_dataset)}")
    
    def get_dataloaders(self):
        """Get dataloaders for train, validation, and test sets"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False
        )
        
        loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        
        if hasattr(self, 'few_shot_dataset'):
            few_shot_loader = DataLoader(
                self.few_shot_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=False
            )
            loaders['few_shot'] = few_shot_loader
        
        return loaders