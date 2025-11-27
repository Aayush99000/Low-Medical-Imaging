'''Creating subset of 300 images for Pheumonia detection task'''

# src/utils/subset_utils.py
import json
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset

def _rng(seed: int):
    """Return a numpy Generator for reproducible sampling."""
    return np.random.default_rng(seed)

def create_subset_indices(
    dataset,
    subset_size: Optional[int] = None,
    per_class: Optional[int] = None,
    seed: int = 42,
    save_path: str = "data/processed/subset_indices.json",
    patient_col: Optional[str] = None
) -> Dict:
    """
    Create stratified subset indices for `dataset` and save to JSON.

    Args:
        dataset: dataset instance that must expose `labels` (list/np.array) aligned with dataset ordering.
                 Optionally dataset.df with patient column if patient_col provided.
        subset_size: total number of images desired (ignored if per_class provided).
        per_class: number of images per class (overrides subset_size).
        seed: random seed.
        save_path: path to write JSON metadata containing indices and info.
        patient_col: optional column name (in dataset.df) for patient-level grouping.

    """
    rng = _rng(seed)

    # Validate dataset exposes labels
    if not hasattr(dataset, "labels"):
        raise ValueError("Dataset must expose `labels` (list or array) in the same order as indexing.")

    labels_arr = np.array(dataset.labels)
    unique_classes = np.unique(labels_arr).tolist()

    # utility to save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Patient-level sampling (preferred) ---------------------------------------------------
    if patient_col and hasattr(dataset, "df") and patient_col in dataset.df.columns:
        # build patient -> list(indices) mapping and patient -> class mapping (dominant label)
        patient_to_indices = defaultdict(list)
        for idx, pid in enumerate(dataset.df[patient_col].values):
            patient_to_indices[pid].append(idx)
        # map patient -> class (majority class of that patient's images)
        patient_to_class = {}
        for pid, inds in patient_to_indices.items():
            labs = labels_arr[inds]
            # majority label for patient
            vals, counts = np.unique(labs, return_counts=True)
            patient_to_class[pid] = int(vals[np.argmax(counts)])

        # Prepare per-class list of patients
        class_to_patients = defaultdict(list)
        for pid, cls in patient_to_class.items():
            class_to_patients[int(cls)].append(pid)

        indices_selected = []

        if per_class is not None:
            # select patients per class until we have enough images or hit patient count
            for cls in unique_classes:
                pats = class_to_patients[int(cls)]
                rng.shuffle(pats)
                sel_pats = []
                img_count = 0
                for p in pats:
                    sel_pats.append(p)
                    img_count += len(patient_to_indices[p])
                    if img_count >= per_class:
                        break
                # expand to indices
                for p in sel_pats:
                    indices_selected.extend(patient_to_indices[p])
        else:
            # sample patients (balanced across classes proportionally) until >= subset_size images
            # flatten a patient list with class labels to sample from
            all_patients = list(patient_to_indices.keys())
            rng.shuffle(all_patients)
            selected_patients = []
            total_images = 0
            # simple greedy collect until reaching subset_size
            for p in all_patients:
                selected_patients.append(p)
                total_images += len(patient_to_indices[p])
                if subset_size is not None and total_images >= subset_size:
                    break
            for p in selected_patients:
                indices_selected.extend(patient_to_indices[p])

        indices = np.array(sorted(set(indices_selected)), dtype=int)

    else:
        # Image-level sampling ----------------------------------------------------------------
        class_indices = {int(c): np.where(labels_arr == c)[0] for c in unique_classes}

        if per_class is not None:
            # sample exactly per_class from each class (if available)
            indices = []
            for cls, inds in class_indices.items():
                if len(inds) < per_class:
                    raise ValueError(f"Not enough samples in class {cls}: requested {per_class}, available {len(inds)}")
                sel = rng.choice(inds, size=per_class, replace=False)
                indices.append(sel)
            indices = np.concatenate(indices)
        else:
            # stratified sampling proportional to class frequencies
            if subset_size is None:
                raise ValueError("Either subset_size or per_class must be provided.")
            # compute class quotas (rounding: last class gets remainder)
            counts = {cls: len(inds) for cls, inds in class_indices.items()}
            total_available = sum(counts.values())
            proportions = {cls: counts[cls] / total_available for cls in counts}
            indices = []
            remaining = subset_size
            classes = list(class_indices.keys())
            for i, cls in enumerate(classes):
                if i == len(classes) - 1:
                    k = remaining
                else:
                    k = int(round(proportions[cls] * subset_size))
                    remaining -= k
                k = min(k, len(class_indices[cls]))
                sel = rng.choice(class_indices[cls], size=k, replace=False)
                indices.append(sel)
            indices = np.concatenate(indices)

    # final shuffle of indices
    indices = np.array(indices, dtype=int)
    rng.shuffle(indices)

    # info metadata to save
    info = {
        "indices": indices.tolist(),
        "subset_size": len(indices),
        "seed": int(seed),
        "per_class": int(per_class) if per_class is not None else None,
        "patient_level": bool(patient_col is not None and hasattr(dataset, "df") and patient_col in dataset.df.columns),
    }
    # compute class counts
    unique, counts = np.unique(labels_arr[indices], return_counts=True)
    info["class_counts"] = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}

    with open(save_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"[subset_utils] Saved subset indices ({info['subset_size']}) to {save_path}")
    return info


def load_subset(dataset, indices_path: str = "data/processed/subset_indices.json"):
    """
    Load subset JSON and return a torch.utils.data.Subset for the given dataset.
    """
    path = Path(indices_path)
    if not path.exists():
        raise FileNotFoundError(f"Subset indices JSON not found: {path}")
    with open(path, "r") as f:
        info = json.load(f)
    indices = info["indices"]
    print(f"[subset_utils] Loaded {len(indices)} indices from {indices_path}")
    return Subset(dataset, indices)


def create_multiple_subsets(
    dataset,
    sizes: List[int] = [100, 200, 300],
    seed: int = 42,
    out_dir: str = "data/processed/subsets",
    patient_col: Optional[str] = None
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for s in sizes:
        save_path = out_dir / f"subset_indices_{s}.json"
        info = create_subset_indices(
            dataset,
            subset_size=s,
            per_class=None,
            seed=seed,
            save_path=str(save_path),
            patient_col=patient_col
        )
        created.append(str(save_path))
    return created
