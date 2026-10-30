'''Stratified k-fold training that returns and saves the real per-fold test results'''
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.train_utils import ModelTrainer, compute_class_weights

CLASS_WEIGHTING = (None, "balanced")

SUMMARY_METRICS = ["test_acc", "test_balanced_acc", "test_auc", "test_f1", "test_precision", "test_recall"]


def _test_class_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Macro-averaged precision / recall / F1 and balanced accuracy on the test set."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            logits = model(images.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(np.asarray(y))
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return {
        "test_balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "test_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "test_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "test_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


def train_with_kfold(
    model_fn: Callable[[], nn.Module],
    optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
    train_dataset: Dataset,
    eval_dataset: Dataset,
    labels: Sequence[int],
    test_loader: DataLoader,
    run_name: str,
    n_folds: int = 5,
    num_epochs: int = 20,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
    criterion: Optional[nn.Module] = None,
    class_weights: Optional[str] = None,
    num_classes: Optional[int] = None,
    scheduler_fn: Optional[Callable[[torch.optim.Optimizer], object]] = None,
    early_stop_patience: Optional[int] = None,
    output_dir: str = "experiments",
    device: Optional[torch.device] = None,
    **trainer_kwargs,
) -> List[Dict]:
    """
    Run stratified k-fold CV on a small training pool and evaluate every fold on the held-out test set.

    train_dataset / eval_dataset: the same images in the same order, with and without augmentation
        (e.g. ChestXrayDataset(..., augment=True) and ChestXrayDataset(..., augment=False)).
    labels: label of each item in that order (used for stratification).
    model_fn / optimizer_fn / scheduler_fn: build a fresh model, optimizer and scheduler for each fold.
        optimizer_fn receives the model (so it can set per-group learning rates), e.g.
        lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3).
    class_weights: None (plain cross-entropy) or "balanced": cross-entropy weighted by n / (C * n_c),
        computed from each fold's own training labels. num_classes defaults to max(labels) + 1.
    Pass monitor_metric="val_auc" | "val_balanced_acc" | "val_acc" | "val_loss" (trainer kwarg) to choose the best epoch.

    Each fold logs to its own folder (output_dir/<run_name>_fold<k>), which is cleared first so
    reruns never append to old logs. Results are saved to output_dir/<run_name>_kfold_results.json.
    """
    if len(train_dataset) != len(eval_dataset) or len(train_dataset) != len(labels):
        raise ValueError("train_dataset, eval_dataset and labels must have the same length and order.")

    if class_weights not in CLASS_WEIGHTING:
        raise ValueError(f"class_weights must be one of {CLASS_WEIGHTING}, got {class_weights}")
    if class_weights is not None and criterion is not None:
        raise ValueError("Pass either criterion or class_weights, not both.")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # keep the best model plus the final checkpoint instead of one checkpoint per epoch
    trainer_kwargs.setdefault("save_interval", num_epochs)

    labels_arr = np.asarray(labels)
    num_classes = num_classes or int(labels_arr.max()) + 1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_arr)), labels_arr)):
        fold = fold_idx + 1
        print(f"\n{'='*60}\nFOLD {fold}/{n_folds} - {run_name}\n{'='*60}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        exp_name = f"{run_name}_fold{fold}"
        shutil.rmtree(Path(output_dir) / exp_name, ignore_errors=True)  # no stale rows from earlier runs

        train_loader = DataLoader(Subset(train_dataset, train_idx.tolist()), batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(Subset(eval_dataset, val_idx.tolist()), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True)

        model = model_fn()
        optimizer = optimizer_fn(model)
        scheduler = scheduler_fn(optimizer) if scheduler_fn is not None else None

        fold_weights = None
        if class_weights == "balanced":
            fold_weights = compute_class_weights(labels_arr[train_idx], num_classes)
            fold_criterion = nn.CrossEntropyLoss(weight=fold_weights.to(device))
            print(f"Class weights (from this fold's training labels): {np.round(fold_weights.numpy(), 3).tolist()}")
        else:
            fold_criterion = criterion or nn.CrossEntropyLoss()

        trainer = ModelTrainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=fold_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            experiment_name=exp_name,
            **trainer_kwargs,
        )
        results = trainer.train(num_epochs=num_epochs, early_stop_patience=early_stop_patience)
        # trainer.model now holds the best checkpoint (reloaded by ModelTrainer.train)
        class_metrics = _test_class_metrics(trainer.model, test_loader, trainer.device)

        fold_results.append({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "best_epoch": results["best_epoch"],
            "monitor_metric": trainer.monitor_metric,
            "best_val_metric": float(results["best_metric"]),
            "class_weights": fold_weights.tolist() if fold_weights is not None else None,
            "test_loss": float(results["test_loss"]),
            "test_acc": float(results["test_acc"]),
            "test_auc": float(results["test_auc"]) if results["test_auc"] is not None else None,
            **class_metrics,
            "history": results["history"],
        })

    summarize_kfold_results(fold_results, title=run_name)

    results_path = Path(output_dir) / f"{run_name}_kfold_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(fold_results, f, indent=2, default=float)
    print(f"[kfold_utils] Saved fold results to {results_path}")
    return fold_results


def load_kfold_results(results_path: str) -> List[Dict]:
    """Load the per-fold results saved by train_with_kfold (for summary tables and training curves)."""
    with open(results_path, "r") as f:
        return json.load(f)


def summarize_kfold_results(
    results: Union[str, List[Dict]],
    metrics: Sequence[str] = SUMMARY_METRICS,
    title: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Print and return mean ± sample std (ddof=1) of the real per-fold test metrics.

    results: a list from train_with_kfold, or the path of its saved JSON file.
    Returns {metric: {"mean", "std", "per_fold", "n"}}; metrics missing from every fold are skipped.
    """
    folds = load_kfold_results(results) if isinstance(results, str) else results

    print(f"\n{'='*60}\nK-FOLD TEST SUMMARY{' - ' + title if title else ''}\n{'='*60}")
    summary = {}
    for m in metrics:
        vals = np.array([f[m] for f in folds if f.get(m) is not None], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary[m] = {"mean": float(vals.mean()), "std": std, "per_fold": vals.tolist(), "n": int(len(vals))}
        print(f"{m:18s} {vals.mean():.4f} ± {std:.4f}   per fold: {np.round(vals, 4).tolist()}")
    return summary
