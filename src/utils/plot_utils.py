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


def _style_axis(ax, title: str, ylabel: str, xlabel: str = "Epoch", integer_x: bool = True):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=12, color=TEXT, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_MUTED)
    ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if integer_x:
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
    metrics: Sequence[str] = ("test_auc", "test_balanced_acc", "test_f1", "test_acc"),
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


# One hue per architecture (validated categorical palette). Medical models are solid with filled markers, general
# ones dashed with hollow markers, and the ImageNet DenseNet a gray dotted reference, so identity never rests on color.
FAMILY_COLORS = {"siglip": "#1f9e6e", "dinov2": "#2a78d6", "clip": "#eb6834", "densenet": TEXT_MUTED}
FAMILY_TITLES = {"siglip": "SigLIP-400M", "dinov2": "DINOv2 ViT-B/14", "clip": "CLIP ViT-B/16"}
MODEL_STYLES = {  # model: (family, medical, marker, label)
    "B_imagenet_probe": ("densenet", False, "o", "ImageNet DenseNet-121"),
    "C0_siglip_probe": ("siglip", False, "s", "SigLIP"),
    "C_medsiglip_probe": ("siglip", True, "s", "MedSigLIP"),
    "D0_dinov2_probe": ("dinov2", False, "D", "DINOv2"),
    "D_raddino_probe": ("dinov2", True, "D", "RAD-DINO"),
    "E0_clip_probe": ("clip", False, "^", "CLIP"),
    "E_biomedclip_probe": ("clip", True, "^", "BiomedCLIP"),
}
OTHER_PAIR_LINESTYLES = ("--", ":", "-.")


def _style_of(m):
    return MODEL_STYLES.get(m, ("densenet", False, "o", m))


def _size_axis(ax, n):
    """Log x axis labelled with the training-set sizes themselves."""
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{v:,}" for v in n], fontsize=8)
    ax.minorticks_off()


def _plot_model(ax, n, sizes, m, metric, band=True):
    family, medical, marker, label = _style_of(m)
    color = FAMILY_COLORS[family]
    rs = [s["models"][m][metric] for s in sizes]
    if band:
        lo, hi = np.array([r["ci_total"] for r in rs]).T
        ax.fill_between(n, lo, hi, color=color, alpha=0.12, linewidth=0)
    ls = "-" if medical else (":" if family == "densenet" else "--")
    ax.plot(n, [r["mean"] for r in rs], color=color, linestyle=ls, linewidth=2, marker=marker, markersize=6,
            markerfacecolor=color if medical else SURFACE, markeredgecolor=SURFACE if medical else color,
            markeredgewidth=1.5, label=label, solid_capstyle="round")


def plot_data_efficiency(
    curves: Dict,
    metric: str = "test_auc",
    pairs: Optional[Sequence] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Test score against training-set size (probe_curves.py output), mean over draws with the 95% CI band
    (test set + draws). Top row: one panel per architecture-matched medical / general pair, with the ImageNet
    DenseNet as reference. Bottom row: the paired medical - general difference with its CI, and how often a single
    draw orders each compared pair as the mean over draws does.
    """
    sizes = list(curves["sizes"].values())
    n = np.array([s["n_train"] for s in sizes])
    models = curves["models"]
    comps = {(c["a"], c["b"]): c for c in sizes[0]["comparisons"][metric]}
    pairs = [p for p in (pairs or comps) if tuple(p) in comps]
    matched = [(a, b) for a, b in pairs if _style_of(a)[0] == _style_of(b)[0]]
    label = METRIC_LABELS.get(metric, metric)

    k = max(len(matched), 1)
    fig = plt.figure(figsize=(4.6 * max(k, 2), 8.4), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2 * k)
    top = [fig.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(len(matched))]
    for ax in top[1:]:
        ax.sharey(top[0])
    diff_ax, agree_ax = fig.add_subplot(grid[1, :k]), fig.add_subplot(grid[1, k:])

    for ax, (a, b) in zip(top, matched):
        for m in (a, b):
            _plot_model(ax, n, sizes, m, metric)
        if "B_imagenet_probe" in models:
            _plot_model(ax, n, sizes, "B_imagenet_probe", metric, band=False)
        _size_axis(ax, n)
        _style_axis(ax, FAMILY_TITLES.get(_style_of(a)[0], a), label, xlabel="Training images", integer_x=False)
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, handlelength=3.2, loc="lower right")

    for a, b in matched:
        color = FAMILY_COLORS[_style_of(a)[0]]
        cs = [next(c for c in s["comparisons"][metric] if (c["a"], c["b"]) == (a, b)) for s in sizes]
        lo, hi = np.array([c["ci_total"] for c in cs]).T
        diff_ax.fill_between(n, lo, hi, color=color, alpha=0.12, linewidth=0)
        diff_ax.plot(n, [c["mean_diff"] for c in cs], color=color, linewidth=2, marker=_style_of(a)[2], markersize=6,
                     markeredgecolor=SURFACE, label=f"{_style_of(a)[3]} − {_style_of(b)[3]}")
    diff_ax.axhline(0, color=TEXT_MUTED, linewidth=1, linestyle=(0, (2, 3)))
    _size_axis(diff_ax, n)
    _style_axis(diff_ax, f"Medical − general pretraining ({label})", f"Paired difference in {label}",
                xlabel="Training images", integer_x=False)
    diff_ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, handlelength=3.2)

    others = iter(OTHER_PAIR_LINESTYLES)
    for a, b in pairs:
        is_matched = (a, b) in matched
        color = FAMILY_COLORS[_style_of(a)[0]] if is_matched else TEXT_MUTED
        agree = [next(c for c in s["comparisons"][metric] if (c["a"], c["b"]) == (a, b))["draw_agreement"]
                 for s in sizes]
        agree_ax.plot(n, agree, color=color, linestyle="-" if is_matched else next(others, "--"), linewidth=2,
                      marker=_style_of(a)[2] if is_matched else "o", markersize=5, markeredgecolor=SURFACE,
                      label=f"{_style_of(a)[3]} vs {_style_of(b)[3]}")
    agree_ax.axhline(0.5, color=TEXT_MUTED, linewidth=1, linestyle=(0, (2, 3)))
    _size_axis(agree_ax, n)
    agree_ax.set_ylim(0, 1.05)
    _style_axis(agree_ax, f"Single-draw agreement ({label})", "Share of draws agreeing with the mean",
                xlabel="Training images", integer_x=False)
    agree_ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, handlelength=3.2, loc="lower right")

    suptitle = f"{title} - " if title else ""
    fig.suptitle(f"{suptitle}linear probes on frozen features, mean and 95% CI over subset draws", fontsize=13,
                 color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    return fig


DATASET_STYLES = {"pneumonia": ("#2a78d6", "o", "Pneumonia"), "ham10000": ("#eb6834", "s", "HAM10000"),
                  "chexpert": ("#1f9e6e", "^", "CheXpert")}


def plot_reliability(rows: List[Dict], metric: str = "test_auc", save_prefix: Optional[str] = None):
    """
    reliability.py output. Variance figure: per dataset, the between-draw SD of one study's paired difference
    (training subset) and its test-set bootstrap SE, median over model pairs with the interquartile range.
    Agreement figure: measured agreement of a k-draw mean with the held-out mean against z = |mean diff| / SD,
    with the normal-model curve, for the sizes with 20 draws.
    """
    from scipy import stats
    label = METRIC_LABELS.get(metric, metric)
    datasets = [d for d in DATASET_STYLES if any(r["dataset"] == d for r in rows)]
    figs = {}

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.0), facecolor=SURFACE, squeeze=False)
    for ax, d in zip(axes[0], datasets):
        sub = [r for r in rows if r["dataset"] == d and r["n_draws"] > 1]
        n = np.array(sorted({r["n_train"] for r in sub}))
        for key, color, ls, name in (("sd_subset", "#2a78d6", "-", "Training subset (SD between draws)"),
                                     ("se_test", "#eb6834", "--", "Test set (bootstrap SE)")):
            q = np.array([np.percentile([r[key] for r in sub if r["n_train"] == v], [25, 50, 75]) for v in n])
            ax.fill_between(n, q[:, 0], q[:, 2], color=color, alpha=0.12, linewidth=0)
            ax.plot(n, q[:, 1], color=color, linestyle=ls, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, label=name)
        _size_axis(ax, n)
        ax.set_ylim(bottom=0)
        _style_axis(ax, DATASET_STYLES[d][2], f"Spread of one study's {label} difference",
                    xlabel="Training images", integer_x=False)
    axes[0][0].legend(frameon=False, labelcolor=TEXT, fontsize=9, handlelength=3.2)
    fig.suptitle(f"Where a single study's uncertainty comes from (median and IQR over model pairs)", fontsize=13,
                 color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    figs["variance"] = fig

    ks = (1, 5)
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.4), facecolor=SURFACE)
    for ax, k in zip(axes, ks):
        z_grid = np.linspace(0, 3, 200)
        a, b = np.sqrt(k) * z_grid, np.sqrt(20 - k) * z_grid
        model = stats.norm.cdf(a) * stats.norm.cdf(b) + stats.norm.cdf(-a) * stats.norm.cdf(-b)
        ax.plot(z_grid, model, color=TEXT_MUTED, linewidth=1.5, linestyle=(0, (4, 2)), label="Normal model")
        for d in datasets:
            color, marker, name = DATASET_STYLES[d]
            sub = [r for r in rows if r["dataset"] == d and r["n_draws"] == 20 and r["agree_k"].get(k) is not None]
            ax.scatter([min(r["z"], 3) for r in sub], [r["agree_k"][k] for r in sub], color=color, marker=marker,
                       s=28, alpha=0.8, edgecolor=SURFACE, linewidth=0.8, label=name, zorder=3)
        ax.axhline(0.5, color=TEXT_MUTED, linewidth=1, linestyle=(0, (2, 3)))
        ax.set_xlim(-0.05, 3.05)
        ax.set_ylim(0, 1.03)
        _style_axis(ax, f"{'One study' if k == 1 else f'Mean of {k} studies'} vs the held-out mean",
                    "Share ordering the pair as the reference does",
                    xlabel="z = |mean difference| / SD between draws (capped at 3)", integer_x=False)
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="lower right", handlelength=3.2)
    fig.suptitle(f"How often a low-data comparison gets the order right ({label}; every model pair and size "
                 f"with 20 draws)", fontsize=13, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    figs["agreement"] = fig

    if save_prefix:
        for name, f in figs.items():
            f.savefig(f"{save_prefix}_{name}.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    return figs
