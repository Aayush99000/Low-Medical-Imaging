"""PyTorch Dataset class for chest X-ray image preprocessing."""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class ChestXrayDataset(Dataset):
    
    def __init__(self, 
                 dataset_root: str,
                 split: str = "train",
                 image_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 normalize: bool = True,
                 augmentation_strength: float = 0.5):
        self.dataset_root = dataset_root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.augmentation_strength = augmentation_strength
        
        self.label_map = {"Normal": 0, "Pneumonia": 1}
        self.images = []
        self.labels = []
        
        self._load_image_paths()
        
        # Define augmentation and normalization pipelines
        self.augmentation_transforms = self._get_augmentation_transforms()
        self.normalization_transforms = self._get_normalization_transforms()
    
    def _load_image_paths(self):
        """Load all image paths from the dataset directory."""
        split_dir = os.path.join(self.dataset_root, self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist.")
        
        for label, label_idx in self.label_map.items():
            label_dir = os.path.join(split_dir, label)
            
            if not os.path.exists(label_dir):
                print(f"Warning: Label directory {label_dir} does not exist. Skipping.")
                continue
            
            for img_name in os.listdir(label_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(label_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label_idx)
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {split_dir}")
        
        print(f"Loaded {len(self.images)} images from {self.split} split.")
    
    def _get_augmentation_transforms(self) -> transforms.Compose:
        """Create augmentation transformations."""
        aug_list = [
            transforms.RandomRotation(degrees=20 * self.augmentation_strength),
            transforms.RandomAffine(degrees=0, translate=(0.1 * self.augmentation_strength, 0.1 * self.augmentation_strength)),
            transforms.RandomHorizontalFlip(p=0.5 * self.augmentation_strength),
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
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Resize to target size
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        
        # Convert to RGB (stack grayscale into 3 channels)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image for torchvision transforms
        from PIL import Image
        img_pil = Image.fromarray(img_rgb)
        
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

data = ChestXrayDataset(
    dataset_root="/Users/aayushkatoch/Desktop/DS_project/chest_xray_pheumonia/chest_xray",
    split="train")
print(f"Dataset size: {len(data)}")
# Visualize some samples
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for i in range(5):
    img, label = data[i]
    
    # Denormalize
    if data.normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
    
    # Convert to numpy and transpose from (C, H, W) to (H, W, C)
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # Display
    axes[i].imshow(img_np)
    axes[i].set_title(f"{'PNEUMONIA' if label == 1 else 'NORMAL'}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()


"""
Load the fixed subset for training.
Use this script every time you want to train with the same subset.
"""

import torch
from torch.utils.data import DataLoader
from dataset import ChestXrayDataset  # Import your dataset class
from subset_utils import load_subset_from_indices, verify_subset

# Configuration
DATA_ROOT = '/kaggle/input/chest-xray-pneumonia/chest_xray'
SUBSET_PATH = 'train_subset_300_seed42.json'
BATCH_SIZE = 32

def main():
    print("Loading datasets...")
    
    # Load full training dataset
    full_train_dataset = ChestXrayDataset(
        dataset_root=DATA_ROOT,
        split='train',
        image_size=(224, 224),
        augment=True,
        normalize=True
    )
    
    # Load validation dataset
    val_dataset = ChestXrayDataset(
        dataset_root=DATA_ROOT,
        split='val',
        image_size=(224, 224),
        augment=False,
        normalize=True
    )
    
    # Load test dataset
    test_dataset = ChestXrayDataset(
        dataset_root=DATA_ROOT,
        split='test',
        image_size=(224, 224),
        augment=False,
        normalize=True
    )
    
    # Load the fixed subset
    train_subset = load_subset_from_indices(
        dataset=full_train_dataset,
        indices_path=SUBSET_PATH
    )
    
    # Verify subset
    verify_subset(full_train_dataset, train_subset, SUBSET_PATH)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"\n{'='*60}")
    print("DATALOADERS READY")
    print(f"{'='*60}")
    print(f"Train: {len(train_subset)} images ({len(train_loader)} batches)")
    print(f"Val: {len(val_dataset)} images ({len(val_loader)} batches)")
    print(f"Test: {len(test_dataset)} images ({len(test_loader)} batches)")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = main()
    
    # Test by loading one batch
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"\n✓ Ready for training!")

