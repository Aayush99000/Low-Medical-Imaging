'''Stratified k-fold training that returns and saves the real per-fold test results'''
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.train_utils import ModelTrainer, compute_class_weights

CLASS_WEIGHTING = (None, "balanced")

# headline metrics first: on imbalanced data (HAM10000) accuracy mostly tracks the majority class
SUMMARY_METRICS = ["test_auc", "test_balanced_acc", "test_f1", "test_acc", "test_precision", "test_recall"]


def _test_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    """float32 softmax probabilities (N, C) and labels (N,) on the test set, in loader order."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            logits = model(images.to(device)).float()
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(probs).astype(np.float32), np.concatenate(labels).astype(int)


def _test_class_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Macro-averaged precision / recall / F1 and balanced accuracy on the test set."""
    preds = probs.argmax(axis=1)
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
    reruns never append to old logs. Results are saved to output_dir/<run_name>_kfold_results.json, and every
    fold model's test probabilities to output_dir/<run_name>_test_predictions.npz (labels (N,), probs
    (n_folds, N, C)) for bootstrap CIs and DeLong tests (src/utils/stats_utils.py).
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
    fold_results, fold_probs, test_labels = [], [], None

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

        # same init / data order / augmentation stream for a given (seed, fold), whatever ran before
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
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
        probs, test_labels = _test_predictions(trainer.model, test_loader, trainer.device)
        fold_probs.append(probs)
        class_metrics = _test_class_metrics(probs, test_labels)

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
    np.savez_compressed(test_predictions_path(output_dir, run_name), labels=test_labels, probs=np.stack(fold_probs))
    return fold_results


def test_predictions_path(output_dir: str, run_name: str) -> Path:
    return Path(output_dir) / f"{run_name}_test_predictions.npz"


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


def _fold_values(folds: Union[str, List[Dict]], metric: str) -> np.ndarray:
    folds = load_kfold_results(folds) if isinstance(folds, str) else folds
    vals = np.array([f[metric] for f in folds if f.get(metric) is not None], dtype=float)
    return vals[~np.isnan(vals)]


def _subset_score(folds: Union[str, List[Dict]], metric: str) -> float:
    """One subset's score: the mean test metric of its fold models."""
    vals = _fold_values(folds, metric)
    return float(vals.mean()) if len(vals) else float("nan")


def _t_interval(vals: np.ndarray, level: float = 0.95) -> List[float]:
    if len(vals) < 2:
        return [float("nan"), float("nan")]
    half = stats.t.ppf(0.5 + level / 2, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
    return [float(vals.mean() - half), float(vals.mean() + half)]


def summarize_repeated_subsets(
    results_by_subset: Dict[int, Union[str, List[Dict]]],
    metrics: Sequence[str] = SUMMARY_METRICS,
    title: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Aggregate k-fold runs repeated on independent draws of the training subset (same test set).

    results_by_subset: {subset_seed: train_with_kfold results or the path of their JSON}.
    The subset draw is the replicate: each subset scores the mean of its fold models, and the spread
    across subsets measures how much a result depends on which images were sampled. The folds of one
    subset reuse the same images, so their spread (within_subset_std) understates that.
    The fixed test set adds its own sampling error on top; it is not included here.
    Returns {metric: {"mean", "between_subset_std", "ci95", "within_subset_std", "per_subset", "n_subsets"}}.
    """
    print(f"\n{'='*60}\nREPEATED-SUBSET SUMMARY{' - ' + title if title else ''}\n{'='*60}")
    seeds = sorted(results_by_subset)
    folds_by_seed = {s: load_kfold_results(r) if isinstance(r, str) else r for s, r in results_by_subset.items()}
    summary = {}
    for m in metrics:
        per_subset = {s: _subset_score(folds_by_seed[s], m) for s in seeds}
        vals = np.array([v for v in per_subset.values() if not np.isnan(v)])
        if len(vals) == 0:
            continue
        within = [v.std(ddof=1) for v in (_fold_values(folds_by_seed[s], m) for s in seeds) if len(v) > 1]
        summary[m] = {
            "mean": float(vals.mean()),
            "between_subset_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "ci95": _t_interval(vals),
            "within_subset_std": float(np.mean(within)) if within else float("nan"),
            "per_subset": per_subset,
            "n_subsets": int(len(vals)),
        }
        r = summary[m]
        print(f"{m:18s} {r['mean']:.4f}  between-subset SD {r['between_subset_std']:.4f}  "
              f"95% CI [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]  within-subset (fold) SD {r['within_subset_std']:.4f}")
    return summary


def compare_paired_subsets(
    results_a: Dict[int, Union[str, List[Dict]]],
    results_b: Dict[int, Union[str, List[Dict]]],
    metric: str = "test_auc",
) -> Dict:
    """
    Paired difference (a - b) of one metric over the subset draws both models were trained on.

    Pairing on the same subsets (and the same test set) removes the variation that both models share,
    so a difference can be clear even when the two models' between-subset intervals overlap.
    """
    seeds = sorted(set(results_a) & set(results_b))
    if len(seeds) < 2:
        raise ValueError("Need at least 2 shared subset seeds to compare models.")
    diffs = np.array([_subset_score(results_a[s], metric) - _subset_score(results_b[s], metric) for s in seeds])
    return {
        "metric": metric, "subset_seeds": seeds, "per_subset_diff": diffs.tolist(),
        "mean_diff": float(diffs.mean()), "std_diff": float(diffs.std(ddof=1)), "ci95": _t_interval(diffs),
        "n_a_better": int((diffs > 0).sum()),
    }
