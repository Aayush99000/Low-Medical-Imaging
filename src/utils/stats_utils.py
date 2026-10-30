'''Confidence intervals and significance tests on the held-out test set (bootstrap, DeLong, Holm)'''
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
from scipy import stats

BOOTSTRAP_METRICS = ("test_auc", "test_balanced_acc", "test_f1", "test_acc")

Predictions = Union[str, Path, Dict[str, np.ndarray]]


def load_test_predictions(preds: Predictions) -> Tuple[np.ndarray, np.ndarray]:
    """labels (N,) and probs (n_folds, N, C) from a <run_name>_test_predictions.npz file (or an equivalent dict)."""
    if isinstance(preds, (str, Path)):
        with np.load(preds) as f:
            return f["labels"].astype(int), f["probs"].astype(np.float32)
    return np.asarray(preds["labels"], dtype=int), np.asarray(preds["probs"], dtype=np.float32)


def _stack_subsets(preds_by_subset: Dict[int, Predictions], seeds: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """labels (N,) and probs (n_subsets, n_folds, N, C); every draw must share the same test set."""
    labels, probs = None, []
    for s in seeds:
        y, p = load_test_predictions(preds_by_subset[s])
        if labels is not None and not np.array_equal(labels, y):
            raise ValueError(f"Subset {s} was evaluated on a different test set.")
        labels = y
        probs.append(p)
    return labels, np.stack(probs)


# ---------------------------------------------------------------------------
# Metrics for many models at once: probs (M, N, C) -> (M,). Same definitions as sklearn / train_utils.
# ---------------------------------------------------------------------------
def _rank_auc(scores: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """ROC AUC of each row of scores (M, N) via the Mann-Whitney U statistic (midranks handle ties)."""
    n1, n0 = positive.sum(), (~positive).sum()
    ranks = stats.rankdata(scores, axis=1)
    return (ranks[:, positive].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def batch_metric(probs: np.ndarray, labels: np.ndarray, metric: str) -> np.ndarray:
    """
    test_auc: binary AUC, or macro one-vs-rest AUC over the classes with positives and negatives in labels.
    test_balanced_acc: mean recall over the classes in labels. test_f1: macro F1 over the classes in labels
    or predictions (zero_division=0). test_acc: accuracy.
    """
    n_classes = probs.shape[2]
    if metric == "test_auc":
        if n_classes == 2:
            return _rank_auc(probs[:, :, 1], labels == 1)
        present = [c for c in range(n_classes) if 0 < (labels == c).sum() < len(labels)]
        return np.mean([_rank_auc(probs[:, :, c], labels == c) for c in present], axis=0)
    preds = probs.argmax(axis=2)
    if metric == "test_acc":
        return (preds == labels).mean(axis=1)
    if metric == "test_balanced_acc":
        return np.mean([(preds[:, labels == c] == c).mean(axis=1) for c in np.unique(labels)], axis=0)
    if metric == "test_f1":
        f1, used = [], []
        for c in range(n_classes):
            tp = ((preds == c) & (labels == c)).sum(axis=1)
            fp = ((preds == c) & (labels != c)).sum(axis=1)
            fn = ((preds != c) & (labels == c)).sum(axis=1)
            denom = 2 * tp + fp + fn
            f1.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
            used.append((labels == c).any() | (preds == c).any(axis=1))
        f1, used = np.array(f1), np.array(used)
        return (f1 * used).sum(axis=0) / used.sum(axis=0)
    raise ValueError(f"metric must be one of {BOOTSTRAP_METRICS}, got {metric}")


def _per_subset_scores(probs: np.ndarray, labels: np.ndarray, metric: str) -> np.ndarray:
    """probs (S, F, N, C) -> (S,): each draw scores the mean of its fold models (as in summarize_repeated_subsets)."""
    S, F, N, C = probs.shape
    return batch_metric(probs.reshape(S * F, N, C), labels, metric).reshape(S, F).mean(axis=1)


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample test cases with replacement within each class, so every class stays present."""
    return np.concatenate([rng.choice(idx, size=len(idx), replace=True)
                           for idx in (np.where(labels == c)[0] for c in np.unique(labels))])


def _percentile_ci(values: np.ndarray, level: float) -> List[float]:
    return [float(np.percentile(values, 50 * (1 - level))), float(np.percentile(values, 50 * (1 + level)))]


def _bootstrap_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for 'difference = 0' (the share of resamples on the far side of 0, doubled)."""
    n = len(diffs)
    return float(min(1.0, 2 * min((np.sum(diffs <= 0) + 1) / (n + 1), (np.sum(diffs >= 0) + 1) / (n + 1))))


def bootstrap_test_metric(
    preds_by_subset: Dict[int, Predictions],
    metric: str = "test_auc",
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Dict:
    """
    Confidence intervals for a model's headline score: the mean over subset draws of the mean over fold models.

    preds_by_subset: {subset_seed: <run_name>_test_predictions.npz}, from repeated draws on the same test set.
    ci_test_set: stratified bootstrap over test cases only (uncertainty from the finite test set).
    ci_total: two-level bootstrap that also resamples the subset draws, so it covers both which test cases
        and which training images were sampled. With one draw the two intervals are the same.
    """
    seeds = sorted(preds_by_subset)
    labels, probs = _stack_subsets(preds_by_subset, seeds)
    point = _per_subset_scores(probs, labels, metric)
    rng = np.random.default_rng(seed)
    boot_test, boot_total = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = _stratified_indices(labels, rng)
        per_subset = _per_subset_scores(probs[:, :, idx], labels[idx], metric)
        boot_test[b] = per_subset.mean()
        boot_total[b] = per_subset[rng.integers(0, len(seeds), len(seeds))].mean()
    return {
        "metric": metric, "mean": float(point.mean()),
        "per_subset": {int(s): float(v) for s, v in zip(seeds, point)},
        "ci_test_set": _percentile_ci(boot_test, level), "ci_total": _percentile_ci(boot_total, level),
        "level": level, "n_boot": n_boot, "n_test": int(len(labels)), "n_subsets": len(seeds),
    }


def delong_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """
    DeLong's test for two correlated ROC AUCs on the same test cases (fast version: Sun & Xu, 2014).
    labels: binary (N,); scores_a / scores_b: positive-class scores (N,). Returns AUCs, difference, SE, z, p.
    """
    positive = np.asarray(labels).astype(bool)
    m, n = positive.sum(), (~positive).sum()
    scores = np.vstack([scores_a, scores_b]).astype(float)
    x, y = scores[:, positive], scores[:, ~positive]
    tx, ty = stats.rankdata(x, axis=1), stats.rankdata(y, axis=1)
    tz = stats.rankdata(np.hstack([x, y]), axis=1)
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n          # per-positive structural components
    v10 = 1.0 - (tz[:, m:] - ty) / m    # per-negative structural components
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(diff),
            "se": float(np.sqrt(max(var, 0.0))), "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def compare_models_on_test(
    preds_a: Dict[int, Predictions],
    preds_b: Dict[int, Predictions],
    metric: str = "test_auc",
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Dict:
    """
    Paired comparison (a - b) of two models trained on the same subset draws and tested on the same cases.

    Main test: two-level paired bootstrap of the headline-score difference (the same resampled test cases
    and draws for both models), giving a CI and a two-sided p-value.
    For binary tasks with metric="test_auc", also DeLong's test on each draw, using each model's
    fold-ensemble scores (mean probability over its fold models) on that draw.
    """
    seeds = sorted(set(preds_a) & set(preds_b))
    if not seeds:
        raise ValueError("The two models share no subset draws.")
    labels, pa = _stack_subsets(preds_a, seeds)
    labels_b, pb = _stack_subsets(preds_b, seeds)
    if not np.array_equal(labels, labels_b):
        raise ValueError("The two models were evaluated on different test sets.")
    point = _per_subset_scores(pa, labels, metric) - _per_subset_scores(pb, labels, metric)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stratified_indices(labels, rng)
        d = _per_subset_scores(pa[:, :, idx], labels[idx], metric) - _per_subset_scores(pb[:, :, idx], labels[idx], metric)
        diffs[b] = d[rng.integers(0, len(seeds), len(seeds))].mean()
    result = {
        "metric": metric, "subset_seeds": [int(s) for s in seeds],
        "mean_diff": float(point.mean()), "per_subset_diff": point.tolist(),
        "ci_total": _percentile_ci(diffs, level), "p_value": _bootstrap_p(diffs),
        "level": level, "n_boot": n_boot, "n_test": int(len(labels)),
    }
    if metric == "test_auc" and pa.shape[3] == 2:
        result["delong_per_subset"] = {
            int(s): delong_test(labels, pa[i, :, :, 1].mean(axis=0), pb[i, :, :, 1].mean(axis=0))
            for i, s in enumerate(seeds)}
    return result


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values for a family of comparisons (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
