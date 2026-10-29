'''Training curves and test-result plots built from the per-fold results saved by kfold_utils'''
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.utils.kfold_utils import load_kfold_results

# Validated categorical palette (colorblind-safe pair); train and val also differ by line style
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"

METRIC_LABELS = {
    "test_acc": "Accuracy", "test_balanced_acc": "Balanced acc.", "test_auc": "AUC",
    "test_f1": "Macro F1", "test_precision": "Macro precision", "test_recall": "Macro recall",
}


def _as_folds(results: Union[str, List[Dict]]) -> List[Dict]:
    return load_kfold_results(results) if isinstance(results, str) else results


def _mean_std(stacked: np.ndarray):
    """Column-wise mean and sample std (ddof=1), ignoring NaN (folds that stopped early / missing AUC)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0, ddof=1)
    return mean, np.nan_to_num(std)


def kfold_history_stats(
    results: Union[str, List[Dict]],
    keys: Sequence[str] = ("train_loss", "val_loss", "train_acc", "val_acc", "val_auc"),
) -> Dict[str, np.ndarray]:
    """
    Stack each fold's training history and return per-epoch mean / std across folds.

    Folds that early-stopped are padded with NaN, so later epochs average over the folds that reached them.
    Returns {"epochs", "n_folds", "<key>_mean", "<key>_std"} for every key present in the history.
    """
    folds = _as_folds(results)
    histories = [f["history"] for f in folds]
    n_epochs = max(len(h["train_loss"]) for h in histories)
    stats = {"epochs": np.arange(1, n_epochs + 1), "n_folds": len(histories)}

    for key in keys:
        if not all(key in h for h in histories):
            continue
        stacked = np.full((len(histories), n_epochs), np.nan)
        for i, h in enumerate(histories):
            vals = np.array([np.nan if v is None else v for v in h[key]], dtype=float)
            stacked[i, :len(vals)] = vals
        stats[f"{key}_mean"], stats[f"{key}_std"] = _mean_std(stacked)
    return stats


def _style_axis(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=12, color=TEXT, loc="left")
    ax.set_xlabel("Epoch", color=TEXT_MUTED)
    ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED)


def _plot_band(ax, epochs, stats, key, color, label, linestyle):
    mean, std = stats[f"{key}_mean"], stats[f"{key}_std"]
    ax.plot(epochs, mean, color=color, linewidth=2, linestyle=linestyle, label=label,
            solid_capstyle="round", dash_capstyle="round")
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)


def plot_kfold_curves(
    results: Union[str, List[Dict]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Loss, accuracy and validation-AUC curves: mean across folds with a ±1 SD band.

    results: list from train_with_kfold, or the path of its saved JSON file.
    """
    stats = kfold_history_stats(results)
    epochs = stats["epochs"]
    has_auc = "val_auc_mean" in stats and not np.all(np.isnan(stats["val_auc_mean"]))

    n_panels = 3 if has_auc else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.2), facecolor=SURFACE)

    ax = axes[0]
    _plot_band(ax, epochs, stats, "train_loss", TRAIN_COLOR, "Train", "-")
    _plot_band(ax, epochs, stats, "val_loss", VAL_COLOR, "Validation", "--")
    _style_axis(ax, "Loss", "Cross-entropy loss")
    ax.legend(frameon=False, labelcolor=TEXT)

    ax = axes[1]
    _plot_band(ax, epochs, stats, "train_acc", TRAIN_COLOR, "Train", "-")
    _plot_band(ax, epochs, stats, "val_acc", VAL_COLOR, "Validation", "--")
    _style_axis(ax, "Accuracy", "Accuracy")
    ax.legend(frameon=False, labelcolor=TEXT)

    if has_auc:
        ax = axes[2]
        _plot_band(ax, epochs, stats, "val_auc", VAL_COLOR, "Validation", "--")
        _style_axis(ax, "Validation AUC", "AUC")
        best = int(np.nanargmax(stats["val_auc_mean"]))
        best_auc = stats["val_auc_mean"][best]
        ax.scatter([epochs[best]], [best_auc], s=64, color=VAL_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3)
        near_right = best > 0.6 * (len(epochs) - 1)
        ax.annotate(f"Best mean AUC {best_auc:.3f}\nepoch {epochs[best]}", (epochs[best], best_auc),
                    xytext=(-10 if near_right else 10, -72), textcoords="offset points",
                    ha="right" if near_right else "left", fontsize=9, color=TEXT,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor=GRID),
                    arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, linewidth=1))

    suptitle = f"{title} - " if title else ""
    fig.suptitle(f"{suptitle}mean ± 1 SD across {stats['n_folds']} folds", fontsize=13, color=TEXT,
                 x=0.01, ha="left")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    return fig


def plot_kfold_test_results(
    results: Union[str, List[Dict]],
    metrics: Sequence[str] = ("test_acc", "test_balanced_acc", "test_auc", "test_f1"),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Real test-set metrics per fold: one dot per fold, with the mean ± 1 SD drawn as a line and whiskers.

    Metrics stay on their 0-1 scale so accuracy, AUC and F1 share one axis.
    """
    folds = _as_folds(results)
    metrics = [m for m in metrics if any(f.get(m) is not None for f in folds)]

    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 2, 4.2), facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for i, m in enumerate(metrics):
        vals = np.array([f[m] for f in folds if f.get(m) is not None], dtype=float)
        vals = vals[~np.isnan(vals)]
        mean = vals.mean()
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(i + jitter, vals, s=40, color=TRAIN_COLOR, alpha=0.55, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.errorbar(i + 0.25, mean, yerr=std, fmt="o", markersize=8, color=TEXT, ecolor=TEXT_MUTED,
                    elinewidth=2, capsize=4, zorder=4)
        ax.annotate(f"{mean:.3f} ± {std:.3f}", (i + 0.25, mean), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics])
    ax.set_xlim(-0.5, len(metrics) - 0.1)
    ax.set_ylim(0, 1.02)
    _style_axis(ax, "", "Score on the test set")
    ax.set_xlabel("")
    suptitle = f"{title} - " if title else ""
    ax.set_title(f"{suptitle}test metrics per fold (dots) and mean ± 1 SD", fontsize=12, color=TEXT, loc="left")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    return fig
