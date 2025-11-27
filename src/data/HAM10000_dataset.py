import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class HAM10000Dataset(Dataset):
    
    def __init__(self, 
                 dataset_root: str,
                 csv_file: str = "GroundTruth.csv",
                 image_folder: str = "images",
                 image_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 normalize: bool = True,
                 augmentation_strength: float = 0.5,
                 train_ratio: float = 0.8,
                 split: str = "train"):
        """
        HAM10000 Dataset for skin lesion classification.
        
        Args:
            dataset_root: Root directory containing images/, masks/, and GroundTruth.csv
            csv_file: Name of the CSV file with ground truth labels
            image_folder: Name of the folder containing images
            image_size: Target size for images (height, width)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images
            augmentation_strength: Strength of augmentation (0.0 to 1.0)
            train_ratio: Ratio of training data (rest goes to validation/test)
            split: 'train' or 'val' or 'test'
        """
        self.dataset_root = dataset_root
        self.image_folder = os.path.join(dataset_root, image_folder)
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.augmentation_strength = augmentation_strength
        self.split = split
        
        # Load ground truth CSV
        csv_path = os.path.join(dataset_root, csv_file)
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file {csv_path} does not exist.")
        
        self.df = pd.read_csv(csv_path)
        
        # Create label mapping
        self._create_label_mapping()
        
        # Split dataset
        self._split_dataset(train_ratio)
        
        # Define augmentation and normalization pipelines
        self.augmentation_transforms = self._get_augmentation_transforms()
        self.normalization_transforms = self._get_normalization_transforms()
        
        print(f"Loaded {len(self.images)} images from {self.split} split.")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _create_label_mapping(self):
        """Create mapping from class names to indices."""
        # Assuming the CSV has columns like: image, nv, mel, bcc, akiec, bkl, df, vasc
        # Or a single column 'dx' with class names
        
        # Check if 'dx' column exists (single label column)
        if 'dx' in self.df.columns:
            self.classes = sorted(self.df['dx'].unique())
            self.label_map = {cls: idx for idx, cls in enumerate(self.classes)}
            self.df['label_idx'] = self.df['dx'].map(self.label_map)
        else:
            # Assume one-hot encoded columns
            class_columns = ['nv', 'mel', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
            available_cols = [col for col in class_columns if col in self.df.columns]
            
            if len(available_cols) == 0:
                raise ValueError("Could not find class labels in CSV. Expected 'dx' column or one-hot encoded columns.")
            
            self.classes = available_cols
            self.label_map = {cls: idx for idx, cls in enumerate(self.classes)}
            
            # Convert one-hot to single label
            self.df['label_idx'] = self.df[available_cols].idxmax(axis=1).map(self.label_map)
        
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Label mapping: {self.label_map}")
    
    def _split_dataset(self, train_ratio: float):
        """Split dataset into train/val/test."""
        n_total = len(self.df)
        n_train = int(n_total * train_ratio)
        n_val = (n_total - n_train) // 2
        
        # Shuffle with a fixed seed for reproducibility
        df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if self.split == "train":
            self.df_split = df_shuffled[:n_train]
        elif self.split == "val":
            self.df_split = df_shuffled[n_train:n_train + n_val]
        elif self.split == "test":
            self.df_split = df_shuffled[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'.")
        
        # Get image paths and labels
        self.images = []
        self.labels = []
        
        # Determine the image column name
        image_col = 'image' if 'image' in self.df_split.columns else 'image_id'
        if image_col not in self.df_split.columns:
            raise ValueError("Could not find image column. Expected 'image' or 'image_id'.")
        
        for _, row in self.df_split.iterrows():
            img_name = row[image_col]
            
            # Handle different image naming conventions
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_name = f"{img_name}.jpg"
            
            img_path = os.path.join(self.image_folder, img_name)
            
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(int(row['label_idx']))
            else:
                print(f"Warning: Image not found: {img_path}")
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in current split."""
        from collections import Counter
        label_counts = Counter(self.labels)
        return {self.classes[idx]: count for idx, count in sorted(label_counts.items())}
    
    def _get_augmentation_transforms(self) -> transforms.Compose:
        """Create augmentation transformations."""
        aug_list = [
            transforms.RandomRotation(degrees=20 * self.augmentation_strength),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1 * self.augmentation_strength, 0.1 * self.augmentation_strength)
            ),
            transforms.RandomHorizontalFlip(p=0.5 * self.augmentation_strength),
            transforms.RandomVerticalFlip(p=0.3 * self.augmentation_strength),
            transforms.ColorJitter(
                brightness=0.2 * self.augmentation_strength,
                contrast=0.2 * self.augmentation_strength,
                saturation=0.2 * self.augmentation_strength,
                hue=0.1 * self.augmentation_strength
            ),
        ]
        
        return transforms.Compose(aug_list)
    
    def _get_normalization_transforms(self) -> transforms.Compose:
        """Create normalization transformations using ImageNet stats."""
        return transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single item from the dataset."""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image (already in RGB format for dermoscopic images)
        img = cv2.imread(img_path)
        
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img_resized = cv2.resize(img_rgb, (self.image_size[1], self.image_size[0]))
        
        # Convert to PIL Image for torchvision transforms
        from PIL import Image
        img_pil = Image.fromarray(img_resized)
        
        # Convert to tensor and normalize to [0, 1]
        img_tensor = transforms.ToTensor()(img_pil)
        
        # Apply augmentation if training
        if self.augment and self.split == "train":
            img_tensor = self.augmentation_transforms(img_tensor)
        
        # Clip values to [0, 1] after augmentation
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Apply normalization
        if self.normalize:
            img_tensor = self.normalization_transforms(img_tensor)
        
        return img_tensor, label
    
    def visualize_samples(self, n_samples: int = 8, figsize: Tuple[int, int] = (15, 8)):
        """Visualize random samples from the dataset."""
        indices = np.random.choice(len(self), min(n_samples, len(self)), replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            img_tensor, label = self[idx]
            
            # Denormalize for visualization
            if self.normalize:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
            
            # Convert to numpy and transpose
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            axes[i].imshow(img_np)
            axes[i].set_title(f"Class: {self.classes[label]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
