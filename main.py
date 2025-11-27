import os
import random
import json
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from chest_xray_dataset import ChestXrayDataset
from chexpert_dataset import CheXpertDataset, getImagesLabels
from src.utils.subset_utils import create_subset_indices

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


create_subset_indices(
    dataset=data,
    subset_size=300,              # or 200, or per_class=N
    seed=42,                      # FIXED seed = reproducibility
    save_path="subset_indices_300.json",
    patient_col=None              # or "patient_id" if you have it
)


