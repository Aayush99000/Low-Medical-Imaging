import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HAM10000_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def _find_original_images(image_dirs: Sequence[str]) -> Dict[str, str]:
    """Map image_id -> path for original ISIC_*.jpg files (augmented copies such as _0_123.jpg are ignored)."""
    id_to_path = {}
    for image_dir in image_dirs:
        for path in Path(image_dir).rglob("ISIC_*.jpg"):
            id_to_path.setdefault(path.stem, str(path))
    return id_to_path


def make_ham10000_splits(
    metadata_csv: str,
    image_dirs: Sequence[str],
    test_size: float = 0.15,
    subset_size: Optional[int] = 300,
    per_class: Optional[int] = None,
    min_per_class: int = 5,
    seed: int = 42,
    subset_seed: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Lesion-grouped train/test split plus a small training subset drawn only from the training lesions.

    1. Test split: StratifiedGroupKFold grouped by lesion_id (stratified by dx), so every photo of a lesion
       is on one side. test_size is approximate (one fold of round(1 / test_size) folds).
    2. Training subset: one image per training lesion, then a stratified sample of subset_size images
       (or per_class images per class). Each class gets at least min_per_class images so k-fold CV can
       stratify; keep min_per_class >= n_folds.
    Because the subset has at most one image per lesion, a plain StratifiedKFold on it has no lesion leakage.
    seed fixes the test split; subset_seed (default: seed) picks the training subset, so repeated subset
    draws keep the same test set.

    metadata_csv: HAM10000_metadata.csv (Harvard Dataverse, doi:10.7910/DVN/DBW86T)
    image_dirs: folders searched recursively for the original ISIC_*.jpg images
    Returns lists of image paths / labels for "subset", "test" and the full "train_pool", plus class names.
    """
    subset_seed = seed if subset_seed is None else subset_seed
    rng = np.random.default_rng(subset_seed)
    df = pd.read_csv(metadata_csv)
    id_to_path = _find_original_images(image_dirs)
    missing = sorted(set(df["image_id"]) - set(id_to_path))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images from the metadata were not found, e.g. {missing[:3]}")

    label_map = {c: i for i, c in enumerate(HAM10000_CLASSES)}
    df["path"] = df["image_id"].map(id_to_path)
    df["label"] = df["dx"].map(label_map)

    # 1. Lesion-grouped, stratified test split
    n_splits = max(2, int(round(1 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["lesion_id"]))
    train_pool, test = df.iloc[train_idx], df.iloc[test_idx]
    assert not set(train_pool["lesion_id"]) & set(test["lesion_id"]), "lesion leakage between train and test"

    # 2. One image per lesion, then a stratified (or per-class) subset
    one_per_lesion = train_pool.sample(frac=1.0, random_state=subset_seed).drop_duplicates("lesion_id")
    if per_class is not None:
        quotas = {c: per_class for c in range(len(HAM10000_CLASSES))}
    elif subset_size is not None:
        props = one_per_lesion["label"].value_counts(normalize=True)
        quotas = {c: max(min_per_class, int(round(props.get(c, 0) * subset_size))) for c in range(len(HAM10000_CLASSES))}
        # largest class absorbs the rounding / minimum-per-class difference so the total is subset_size
        largest = int(props.idxmax())
        quotas[largest] -= sum(quotas.values()) - subset_size
        if quotas[largest] < min_per_class:
            raise ValueError(f"subset_size={subset_size} is too small for min_per_class={min_per_class} "
                             f"across {len(HAM10000_CLASSES)} classes; use per_class instead.")
    else:
        quotas = None

    if quotas is None:
        subset = one_per_lesion
    else:
        parts = []
        for c, k in quotas.items():
            pool = one_per_lesion[one_per_lesion["label"] == c]
            if len(pool) < k:
                raise ValueError(f"Not enough lesions in class {HAM10000_CLASSES[c]}: requested {k}, available {len(pool)}")
            parts.append(pool.iloc[rng.choice(len(pool), size=k, replace=False)])
        subset = pd.concat(parts).sample(frac=1.0, random_state=subset_seed)

    def _counts(part):
        return {HAM10000_CLASSES[c]: int(n) for c, n in part["label"].value_counts().sort_index().items()}

    splits = {
        "classes": HAM10000_CLASSES,
        "subset_images": subset["path"].tolist(), "subset_labels": subset["label"].tolist(),
        "test_images": test["path"].tolist(), "test_labels": test["label"].tolist(),
        "train_pool_images": train_pool["path"].tolist(), "train_pool_labels": train_pool["label"].tolist(),
        "info": {
            "seed": int(seed), "subset_seed": int(subset_seed),
            "n_test_folds": n_splits,
            "n_train_pool": len(train_pool), "n_train_pool_lesions": int(train_pool["lesion_id"].nunique()),
            "n_test": len(test), "n_test_lesions": int(test["lesion_id"].nunique()),
            "n_subset": len(subset),
            "subset_class_counts": _counts(subset), "test_class_counts": _counts(test),
        },
    }
    print(f"[HAM10000] train pool {len(train_pool)} images / {splits['info']['n_train_pool_lesions']} lesions, "
          f"test {len(test)} images / {splits['info']['n_test_lesions']} lesions, subset {len(subset)} images")
    print(f"[HAM10000] subset class counts: {splits['info']['subset_class_counts']}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"info": splits["info"], "classes": HAM10000_CLASSES,
                       "subset_image_ids": subset["image_id"].tolist(),
                       "test_image_ids": test["image_id"].tolist()}, f, indent=2)
        print(f"[HAM10000] Saved split image ids to {save_path}")
    return splits


class HAM10000Dataset(Dataset):

    def __init__(self,
                 images: List[str],
                 labels: List[int],
                 image_size: Tuple[int, int] = (224, 224),
                 augment: bool = False,
                 normalize: bool = True,
                 augmentation_strength: float = 0.5,
                 mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD,
                 classes: Sequence[str] = HAM10000_CLASSES):
        """
        images / labels: e.g. splits["subset_images"], splits["subset_labels"] from make_ham10000_splits().
        mean / std: ImageNet stats for Models A/B; for Model C pass MedSigLIP's (get_medsiglip_preprocessing()).
        """
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length.")
        self.images = list(images)
        self.labels = list(labels)
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.augmentation_strength = augmentation_strength
        self.mean = list(mean)
        self.std = list(std)
        self.classes = list(classes)

        # Define augmentation and normalization pipelines
        self.augmentation_transforms = self._get_augmentation_transforms()
        self.normalization_transforms = self._get_normalization_transforms()

        print(f"Loaded {len(self.images)} images. Class distribution: {self._get_class_distribution()}")

    def _get_class_distribution(self) -> Dict[str, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {self.classes[v]: int(c) for v, c in zip(values, counts)}

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
        """Create normalization transformations using the configured mean / std."""
        return transforms.Compose([
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single item from the dataset."""
        img_path = self.images[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img_resized = cv2.resize(img_rgb, (self.image_size[1], self.image_size[0]))

        # Convert to PIL Image for torchvision transforms
        img_pil = Image.fromarray(img_resized)

        # Convert to tensor and normalize to [0, 1]
        img_tensor = transforms.ToTensor()(img_pil)

        # Apply augmentation (enable only for training data)
        if self.augment:
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
                mean = torch.tensor(self.mean).view(3, 1, 1)
                std = torch.tensor(self.std).view(3, 1, 1)
                img_tensor = img_tensor * std + mean

            # Convert to numpy and transpose
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)

            axes[i].imshow(img_np)
            axes[i].set_title(f"Class: {self.classes[label]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
