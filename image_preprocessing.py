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
    """Custom PyTorch Dataset for chest X-ray images with preprocessing and augmentation.
    
    Supports loading images from directory structure:
    dataset_root/
        train/
            Normal/
            Pneumonia/
        val/
            Normal/
            Pneumonia/
        test/
            Normal/
            Pneumonia/
    """
    
    def __init__(self, 
                 dataset_root: str,
                 split: str = "train",
                 image_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 normalize: bool = True,
                 augmentation_strength: float = 0.5):
        """Initialize ChestXrayDataset.
        
        Args:
            dataset_root: Root directory containing train/val/test splits.
            split: Data split to load ("train", "val", or "test").
            image_size: Target size for images (height, width).
            augment: Whether to apply data augmentation.
            normalize: Whether to apply ImageNet normalization.
            augmentation_strength: Strength of augmentation (0.0 to 1.0).
        """
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
        """Load and return an image and its label.
        
        Args:
            idx: Index of the image to load.
        
        Returns:
            Tuple of (image_tensor, label).
        """
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


def get_data_loaders(dataset_root: str,
                     batch_size: int = 32,
                     image_size: Tuple[int, int] = (224, 224),
                     num_workers: int = 4,
                     augmentation_strength: float = 0.5) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create PyTorch DataLoaders for train, val, and test splits.
    
    Args:
        dataset_root: Root directory containing train/val/test splits.
        batch_size: Batch size for DataLoaders.
        image_size: Target image size.
        num_workers: Number of worker processes for data loading.
        augmentation_strength: Strength of augmentation for training data.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Training set with augmentation
    train_dataset = ChestXrayDataset(
        dataset_root=dataset_root,
        split="train",
        image_size=image_size,
        augment=True,
        normalize=True,
        augmentation_strength=augmentation_strength
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation set without augmentation
    val_loader = None
    if os.path.exists(os.path.join(dataset_root, "val")):
        val_dataset = ChestXrayDataset(
            dataset_root=dataset_root,
            split="val",
            image_size=image_size,
            augment=False,
            normalize=True,
            augmentation_strength=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Test set without augmentation
    test_loader = None
    if os.path.exists(os.path.join(dataset_root, "test")):
        test_dataset = ChestXrayDataset(
            dataset_root=dataset_root,
            split="test",
            image_size=image_size,
            augment=False,
            normalize=True,
            augmentation_strength=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


def get_dataset_statistics(dataset_root: str) -> Dict[str, int]:
    """Get statistics about the dataset (count of images per class/split).
    
    Args:
        dataset_root: Root directory containing train/val/test splits.
    
    Returns:
        Dictionary with statistics.
    """
    stats = {}
    label_map = {"Normal": 0, "Pneumonia": 1}
    
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_root, split)
        if not os.path.exists(split_path):
            continue
        
        for label in label_map.keys():
            label_path = os.path.join(split_path, label)
            if os.path.exists(label_path):
                count = len([f for f in os.listdir(label_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
                stats[f"{split}_{label}"] = count
    
    return stats


def visualize_batch(images: torch.Tensor, labels: torch.Tensor, 
                   num_samples: int = 4, normalize_back: bool = True):
    """Visualize a batch of images with labels.
    
    Args:
        images: Batch of images with shape (B, 3, H, W).
        labels: Batch of labels with shape (B,).
        num_samples: Number of images to display.
        normalize_back: Whether to denormalize images back to [0, 1] range.
    """
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
    label_names = {0: "Normal", 1: "Pneumonia"}
    
    # ImageNet normalization stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(num_samples):
        ax = axes[i] if num_samples > 1 else axes
        img = images[i].cpu().numpy()
        
        # Denormalize if needed
        if normalize_back:
            img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            img = (img * std) + mean
            img = np.clip(img, 0, 1)
        else:
            img = img.transpose(1, 2, 0)
        
        ax.imshow(img)
        ax.set_title(f"Label: {label_names[int(labels[i].item())]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def denormalize_image(img: torch.Tensor) -> np.ndarray:
    """Denormalize an image from normalized form back to [0, 1].
    
    Args:
        img: Normalized image tensor with shape (3, H, W).
    
    Returns:
        Denormalized numpy array with shape (H, W, 3) in range [0, 1].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_np = img.cpu().numpy()
    img_np = img_np.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)
    
    return img_np
