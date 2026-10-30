import os
import json
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.data.chest_xray_dataset import ChestXrayDataset, IMAGENET_MEAN, IMAGENET_STD


def getImagesLabels(filename, policy):
    """
    filename: path to the csv file containing all the imagepaths and associated labels
    """
    df = pd.read_csv(filename)
    relevant_cols = ['Path', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    df = df[relevant_cols]
    df = df.replace(np.nan, 0.0)

    if policy == 'zeros':
        df = df.replace(-1.0, 0.0)
    elif policy == 'ones':
        df = df.replace(-1.0, 1.0)
    elif policy == 'both':
        df['Cardiomegaly'] = df['Cardiomegaly'].replace(-1.0, 0.0)
        df['Consolidation'] = df['Consolidation'].replace(-1.0, 0.0)

        df['Atelectasis'] = df['Atelectasis'].replace(-1.0, 1.0)
        df['Edema'] = df['Edema'].replace(-1.0, 1.0)
        df['Pleural Effusion'] = df['Pleural Effusion'].replace(-1.0, 1.0)

    elif policy == 'multi' or policy == 'ignore':
        df = df.replace(-1.0, 2.0)

    # df = df[df['Path'].str.contains('frontal')] ###

    X = df['Path']
    y = df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']]

    return np.asarray(X), np.asarray(y)

def multihot(labels):
    """
    Convert labels and create uncertainty mask for loss computation.
    """
    multihot_labels = labels.copy().astype(np.float32)
    
    # For training: replace uncertain with 0 (will be masked in loss)
    multihot_labels = np.where(multihot_labels == 2, 0, multihot_labels)
    
    return multihot_labels

def clahe(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # or COLOR
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe_obj.apply(img)
    return Image.fromarray(enhanced)

class CheXpertDataset(Dataset):
    def __init__(self, image_list, labels, transform=None, test=False):
        """
        image_list: list of paths containing images
        labels: corresponding multi-class labels of image_list
        transform: optional transform to be applied on a sample.
        """
        self.image_names = image_list
        self.gt = labels
        if (len(np.unique(labels)) > 2):
            self.labels = multihot(labels)
        else:
            self.labels = labels

        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""

        image_name = self.image_names[index]
        # image = Image.open(image_name).convert('RGB')
        image = clahe(image_name)

        label = self.labels[index]
        gt = self.gt[index]

        if self.transform is not None:
            image = self.transform(image)
        else:
            self.transform = transform = transforms.Compose([
                    transforms.Resize((320, 320)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
            image = self.transform(image)
        if self.test:
            return image, torch.FloatTensor(gt)
        else:
            return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
    

# ---------------------------------------------------------------------------
# Binary single-finding task for the low-data comparison (run_experiment.py --dataset chexpert)
# ---------------------------------------------------------------------------
CHEXPERT_TASK = "Pleural Effusion"


def _load_chexpert_csv(root: str, split: str, task: str, blank_as_negative: bool) -> pd.DataFrame:
    """Frontal AP/PA views with a definite label for `task` (uncertain -1 labels are dropped)."""
    df = pd.read_csv(Path(root) / f"{split}.csv")
    df = df[(df["Frontal/Lateral"] == "Frontal") & df["AP/PA"].isin(["AP", "PA"])].copy()
    y = df[task].fillna(0.0) if blank_as_negative else df[task]
    df = df[y.isin([0.0, 1.0])].copy()
    df["label"] = y[df.index].astype(int)
    # CSV paths start with the archive folder (CheXpert-v1.0-small/train/...); resolve them under root
    df["path"] = [str(Path(root, *p.split("/")[1:])) for p in df["Path"]]
    df["patient"] = df["Path"].str.extract(r"(patient\d+)", expand=False)
    return df


def make_chexpert_splits(
    root: str,
    task: str = CHEXPERT_TASK,
    subset_size: Optional[int] = 300,
    per_class: Optional[int] = None,
    min_per_class: int = 5,
    blank_as_negative: bool = True,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Binary CheXpert task (one finding, frontal views) with a patient-disjoint expert-labelled test set.

    1. Test set: the official validation split (valid.csv), labelled by radiologist consensus with no
       uncertain labels. Its 200 patients do not appear in train.csv. It is small (~200 frontal images),
       so report AUC with a confidence interval.
    2. Training subset: frontal train.csv images with a definite label (uncertain -1 dropped; blank =
       not mentioned, treated as negative unless blank_as_negative=False), one image per patient, then a
       stratified sample of subset_size images (or per_class images per class).
    Because the subset has at most one image per patient, a plain StratifiedKFold on it has no patient leakage.

    root: folder holding train.csv, valid.csv and the train/ and valid/ image folders (CheXpert-v1.0-small)
    Returns lists of image paths / labels for "subset", "test" and the full "train_pool", plus class names.
    """
    rng = np.random.default_rng(seed)
    classes = [f"no {task}", task]
    train_pool = _load_chexpert_csv(root, "train", task, blank_as_negative)
    test = _load_chexpert_csv(root, "valid", task, blank_as_negative)
    assert not set(train_pool["patient"]) & set(test["patient"]), "patient leakage between train and test"

    one_per_patient = train_pool.sample(frac=1.0, random_state=seed).drop_duplicates("patient")
    if per_class is not None:
        quotas = {0: per_class, 1: per_class}
    elif subset_size is not None:
        n_pos = max(min_per_class, int(round(one_per_patient["label"].mean() * subset_size)))
        quotas = {0: subset_size - n_pos, 1: n_pos}
        if quotas[0] < min_per_class:
            raise ValueError(f"subset_size={subset_size} is too small for min_per_class={min_per_class}.")
    else:
        quotas = None

    if quotas is None:
        subset = one_per_patient
    else:
        parts = []
        for c, k in quotas.items():
            pool = one_per_patient[one_per_patient["label"] == c]
            if len(pool) < k:
                raise ValueError(f"Not enough patients in class {classes[c]!r}: requested {k}, available {len(pool)}")
            parts.append(pool.iloc[rng.choice(len(pool), size=k, replace=False)])
        subset = pd.concat(parts).sample(frac=1.0, random_state=seed)

    missing = [p for p in list(subset["path"]) + list(test["path"]) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} subset/test images were not found under {root}, e.g. {missing[:3]}")

    def _counts(part):
        return {classes[c]: int(n) for c, n in part["label"].value_counts().sort_index().items()}

    splits = {
        "classes": classes,
        "subset_images": subset["path"].tolist(), "subset_labels": subset["label"].tolist(),
        "test_images": test["path"].tolist(), "test_labels": test["label"].tolist(),
        "train_pool_images": train_pool["path"].tolist(), "train_pool_labels": train_pool["label"].tolist(),
        "info": {
            "task": task, "seed": int(seed), "blank_as_negative": blank_as_negative,
            "n_train_pool": len(train_pool), "n_train_pool_patients": int(train_pool["patient"].nunique()),
            "n_test": len(test), "n_test_patients": int(test["patient"].nunique()),
            "n_subset": len(subset),
            "subset_class_counts": _counts(subset), "test_class_counts": _counts(test),
        },
    }
    print(f"[CheXpert] {task}: train pool {len(train_pool)} images / {splits['info']['n_train_pool_patients']} patients, "
          f"test {len(test)} images / {splits['info']['n_test_patients']} patients, subset {len(subset)} images")
    print(f"[CheXpert] subset class counts: {splits['info']['subset_class_counts']}, "
          f"test class counts: {splits['info']['test_class_counts']}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"info": splits["info"], "classes": classes,
                       "subset_paths": subset["Path"].tolist(), "test_paths": test["Path"].tolist()}, f, indent=2)
        print(f"[CheXpert] Saved split image paths to {save_path}")
    return splits


class CheXpertBinaryDataset(ChestXrayDataset):
    """Images and labels from make_chexpert_splits(), loaded, augmented and normalized exactly like ChestXrayDataset."""

    def __init__(self,
                 images: List[str],
                 labels: List[int],
                 image_size: Tuple[int, int] = (448, 448),
                 augment: bool = False,
                 normalize: bool = True,
                 augmentation_strength: float = 0.5,
                 mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD):
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length.")
        self._image_list, self._label_list = list(images), list(labels)
        # split="train" only gates augmentation in ChestXrayDataset; augment decides whether it is applied
        super().__init__(dataset_root="", split="train", image_size=image_size, augment=augment, normalize=normalize,
                         augmentation_strength=augmentation_strength, mean=mean, std=std)

    def _load_image_paths(self):
        self.images, self.labels = self._image_list, self._label_list
