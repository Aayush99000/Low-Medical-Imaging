'''Stratified k-fold training that returns and saves the real per-fold test results'''
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.train_utils import ModelTrainer


def train_with_kfold(
    model_fn: Callable[[], nn.Module],
    optimizer_fn: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
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

    Each fold logs to its own folder (output_dir/<run_name>_fold<k>), which is cleared first so
    reruns never append to old logs. Results are saved to output_dir/<run_name>_kfold_results.json.
    """
    if len(train_dataset) != len(eval_dataset) or len(train_dataset) != len(labels):
        raise ValueError("train_dataset, eval_dataset and labels must have the same length and order.")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion or nn.CrossEntropyLoss()
    # keep the best model plus the final checkpoint instead of one checkpoint per epoch
    trainer_kwargs.setdefault("save_interval", num_epochs)

    labels_arr = np.asarray(labels)
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
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer) if scheduler_fn is not None else None

        trainer = ModelTrainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            experiment_name=exp_name,
            **trainer_kwargs,
        )
        results = trainer.train(num_epochs=num_epochs, early_stop_patience=early_stop_patience)

        fold_results.append({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "best_epoch": results["best_epoch"],
            "monitor_metric": trainer.monitor_metric,
            "best_val_metric": float(results["best_metric"]),
            "test_loss": float(results["test_loss"]),
            "test_acc": float(results["test_acc"]),
            "test_auc": float(results["test_auc"]) if results["test_auc"] is not None else None,
            "history": results["history"],
        })

    # Summary across all folds (sample std, ddof=1)
    print(f"\n{'='*60}\nK-FOLD TEST SUMMARY - {run_name}\n{'='*60}")
    for key in ("test_acc", "test_auc"):
        vals = np.array([r[key] for r in fold_results if r[key] is not None], dtype=float)
        if len(vals) > 0:
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            print(f"{key}: {vals.mean():.4f} ± {std:.4f}  (n={len(vals)})")
    for r in fold_results:
        auc_str = f"{r['test_auc']:.4f}" if r["test_auc"] is not None else "N/A"
        print(f"  Fold {r['fold']}: best {r['monitor_metric']}={r['best_val_metric']:.4f} "
              f"(epoch {r['best_epoch']}), test_acc={r['test_acc']:.4f}, test_auc={auc_str}")

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
