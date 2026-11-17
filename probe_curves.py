'''Data-efficiency curves from cached features (extract_features.py): one logistic-regression probe per model,
training-set size and subset draw, with bootstrap CIs, paired comparisons and how often single draws agree.

For each size n and draw r, the training subset is drawn by run_experiment.py's sampler with subset seed r, so all
models see the same subsets, and the n=300 draws 0-4 are exactly the subsets of the probe runs in --ref_dir.
Each probe standardizes the features and fits an L2 logistic regression, with C picked by stratified 5-fold CV on
the subset by log-loss (AUC alone favours over-regularized probes whose probabilities all sit near the class prior,
which ruins thresholded metrics such as balanced accuracy), class-weighted where DATASET_PROTOCOL says so. Size "all" is the whole training pool (one image
per lesion / patient where the dataset has them).

Examples:
    python probe_curves.py --dataset pneumonia
    python probe_curves.py --dataset chexpert --sizes 25 50 100 300 1000 --n_draws 20 --n_jobs 32

Outputs in --out_dir: <dataset>_<model>_n<size>_curve_predictions.npz (labels, probs (draws, N, C)),
<dataset>_curves.json (per-size results and comparisons) and <dataset>_curves.png.
'''
import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

from compare_models import load_predictions
from extract_features import PROBE_MODELS
from run_experiment import add_data_args
from src.protocol import DATASET_PROTOCOL, PROTOCOL
from src.utils.plot_utils import plot_data_efficiency
from src.utils.stats_utils import bootstrap_test_metric, compare_models_on_test, holm_adjust

# medical vs general pretraining (same SigLIP-400M architecture), and each SigLIP vs the ImageNet DenseNet
PAIRS = (("C_medsiglip_probe", "C0_siglip_probe"),
         ("C_medsiglip_probe", "B_imagenet_probe"),
         ("C0_siglip_probe", "B_imagenet_probe"))
CURVE_METRICS = ("test_auc", "test_balanced_acc")
C_GRID = np.logspace(-4, 1, 11)   # inverse L2 strength on standardized features
LARGE = 3000                      # sizes above this, and "all", get --n_draws_large draws


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--models", nargs="+", choices=PROBE_MODELS, default=PROBE_MODELS)
    p.add_argument("--features_dir", default="features")
    p.add_argument("--out_dir", default="experiments_curves")
    p.add_argument("--sizes", nargs="+", default=["25", "50", "100", "300", "1000", "3000", "10000", "all"],
                   help="training-set sizes; sizes the dataset can't fill are skipped")
    p.add_argument("--n_draws", type=int, default=20, help="subset draws per size (seeds 0..n_draws-1)")
    p.add_argument("--n_draws_large", type=int, default=5, help=f"draws for sizes above {LARGE} and 'all'")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--ref_dir", default="experiments", help="run_experiment.py outputs, compared at n=300")
    p.add_argument("--smoke_test", action="store_true", help="the *_smoke_features.npz caches")
    return p.parse_args()


class SubsetSampler:
    """Rows of the cached training pool for a size-n subset with subset seed r, from run_experiment.py's sampler."""

    def __init__(self, args, pool_paths, tmp_dir):
        self.args, self.tmp_dir = args, Path(tmp_dir)
        self.row = {p: i for i, p in enumerate(pool_paths)}
        self.min_per_class = max(5, PROTOCOL["n_folds"])
        if args.dataset == "pneumonia":
            from src.data.chest_xray_dataset import ChestXrayDataset
            with contextlib.redirect_stdout(io.StringIO()):
                self.pneumonia = ChestXrayDataset(args.pneumonia_root, split="train", augment=False)

    def __call__(self, n, seed):
        """n=None: the whole pool. Raises ValueError when the pool can't fill the class quotas."""
        a = self.args
        with contextlib.redirect_stdout(io.StringIO()):
            if a.dataset == "pneumonia":
                from src.utils.subset_utils import create_subset_indices
                ds = self.pneumonia
                info = create_subset_indices(ds, subset_size=n or len(ds.labels), seed=seed,
                                             save_path=str(self.tmp_dir / "pneumonia_subset.json"))
                paths = [ds.images[i] for i in info["indices"]]
            elif a.dataset == "ham10000":
                from src.data.HAM10000_dataset import make_ham10000_splits
                paths = make_ham10000_splits(a.ham_metadata, a.ham_image_dirs, subset_size=n,
                                             min_per_class=self.min_per_class, seed=PROTOCOL["seed"],
                                             subset_seed=seed)["subset_images"]
            else:
                from src.data.chexpert_dataset import make_chexpert_splits
                paths = make_chexpert_splits(a.chexpert_root, task=a.chexpert_task, subset_size=n,
                                             min_per_class=self.min_per_class, seed=seed)["subset_images"]
        return np.array([self.row[p] for p in paths])


def fit_probe(X, y, X_test, n_classes, class_weight, seed):
    """Standardize, then L2 logistic regression with C chosen by stratified CV on the training subset (log-loss)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    n_splits = min(5, int(np.bincount(y, minlength=n_classes).min()))
    if n_splits < 2:
        raise ValueError("every class needs at least 2 training images")
    with threadpool_limits(1), warnings.catch_warnings():  # one thread per fit; joblib runs fits in parallel
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")  # 1.8+ API transition notices
        scaler = StandardScaler().fit(X)
        clf = LogisticRegressionCV(Cs=C_GRID, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
                                   scoring="neg_log_loss",
                                   class_weight=class_weight, max_iter=2000)
        clf.fit(scaler.transform(X), y)
        return clf.predict_proba(scaler.transform(X_test)).astype(np.float32), float(np.ravel(clf.C_)[0])


def as_draws(probs, labels, seeds):
    """{seed: predictions} in the stats_utils format, one 'fold' model per draw."""
    return {int(s): {"labels": labels, "probs": probs[i][None]} for i, s in enumerate(seeds)}


def summarize_size(size, preds, labels, seeds, models, n_boot):
    """Per-model CIs and the paired comparisons (Holm over PAIRS) at one training-set size."""
    out = {"models": {}, "comparisons": {}}
    draws = {m: as_draws(preds[m], labels, seeds) for m in models}
    for m in models:
        out["models"][m] = {}
        for metric in CURVE_METRICS:
            r = bootstrap_test_metric(draws[m], metric, n_boot=n_boot)
            per = np.array(list(r["per_subset"].values()))
            r["sd_between_draws"] = float(per.std(ddof=1)) if len(per) > 1 else 0.0
            out["models"][m][metric] = r
    pairs = [(a, b) for a, b in PAIRS if a in models and b in models]
    for metric in CURVE_METRICS:
        comps = [dict(a=a, b=b, **compare_models_on_test(draws[a], draws[b], metric, n_boot=n_boot)) for a, b in pairs]
        for c, p_holm in zip(comps, holm_adjust([c["p_value"] for c in comps])):
            c["p_holm"] = p_holm
            # how often one draw alone would order the two models as the mean over draws does
            per = np.array(c["per_subset_diff"])
            c["draw_agreement"] = float(np.mean(np.sign(per) == np.sign(c["mean_diff"])))
            if "delong_per_subset" in c:
                d = [(v["p_value"] < 0.05, np.sign(v["diff"])) for v in c["delong_per_subset"].values()]
                c["delong_sig_same_direction"] = float(np.mean([s and sg == np.sign(c["mean_diff"]) for s, sg in d]))
                c["delong_sig_opposite_direction"] = float(np.mean([s and sg == -np.sign(c["mean_diff"]) for s, sg in d]))
        out["comparisons"][metric] = comps
    return size, out


def reference_check(args, preds, labels, seeds, models):
    """n=300: cached-feature logistic regression vs the run_experiment.py probes (MLP head, augmentation)."""
    print(f"\nn=300 check: logistic regression on cached features vs the --ref_dir probe runs (test_auc)")
    for m in models:
        ref = load_predictions(args.ref_dir, args.dataset, m, "")
        if not ref:
            print(f"  {m}: no runs in {args.ref_dir}")
            continue
        lr = {s: p for s, p in as_draws(preds[m], labels, seeds).items() if s in ref}
        if not lr:
            continue
        try:
            c = compare_models_on_test(lr, ref, "test_auc", n_boot=args.n_boot)
        except ValueError as e:
            print(f"  {m}: {e}")
            continue
        a, b = bootstrap_test_metric(lr, "test_auc", n_boot=args.n_boot), bootstrap_test_metric(
            {s: ref[s] for s in lr}, "test_auc", n_boot=args.n_boot)
        print(f"  {m:20s} cached LR {a['mean']:.3f}   MLP probe {b['mean']:.3f}   LR - MLP {c['mean_diff']:+.4f} "
              f"[{c['ci_total'][0]:+.4f}, {c['ci_total'][1]:+.4f}]  ({len(lr)} shared draws)")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke_test else ""

    caches = {m: np.load(Path(args.features_dir) / f"{args.dataset}_{m}{suffix}_features.npz") for m in args.models}
    first = caches[args.models[0]]
    for m, c in caches.items():
        if not (np.array_equal(c["pool_paths"], first["pool_paths"]) and np.array_equal(c["test_paths"], first["test_paths"])):
            raise ValueError(f"{m} was cached on different images than {args.models[0]}; re-run extract_features.py")
    pool_paths, pool_y, test_y = first["pool_paths"].tolist(), first["pool_y"], first["test_y"]
    n_classes = int(max(pool_y.max(), test_y.max())) + 1
    pool_X = {m: c["pool_X"] for m, c in caches.items()}
    test_X = {m: c["test_X"].astype(np.float32) for m, c in caches.items()}
    class_weight = DATASET_PROTOCOL[args.dataset].get("class_weights")

    sampler = SubsetSampler(args, pool_paths, out_dir)
    subsets = {}
    for s in args.sizes:
        n = None if s == "all" else int(s)
        if n is not None and n >= len(pool_paths):
            print(f"[probe_curves] n={s}: skipped, the training pool has {len(pool_paths)} images")
            continue
        if n is None and args.dataset == "pneumonia":
            n_draws = 1  # no grouping, so every draw of the whole pool is the same set
        else:
            n_draws = args.n_draws_large if n is None or n > LARGE else args.n_draws
        try:
            subsets[s] = [(r, sampler(n, r)) for r in range(n_draws)]
        except ValueError as e:
            print(f"[probe_curves] n={s}: skipped ({e})")
            continue
        print(f"[probe_curves] n={s}: {n_draws} draws of {len(subsets[s][0][1])} images", flush=True)

    tasks = [(m, s, r, rows) for s, draws in subsets.items() for r, rows in draws for m in args.models]
    fits = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(fit_probe)(pool_X[m][rows].astype(np.float32), pool_y[rows], test_X[m], n_classes, class_weight, r)
        for m, s, r, rows in tasks)

    preds, chosen_C = {s: {} for s in subsets}, {s: {} for s in subsets}
    for (m, s, r, rows), (probs, C) in zip(tasks, fits):
        preds[s].setdefault(m, []).append(probs)
        chosen_C[s].setdefault(m, []).append(C)
    seeds = {s: [r for r, _ in draws] for s, draws in subsets.items()}
    for s in subsets:
        for m in args.models:
            preds[s][m] = np.stack(preds[s][m])
            np.savez(out_dir / f"{args.dataset}_{m}{suffix}_n{s}_curve_predictions.npz", labels=test_y,
                     probs=preds[s][m], seeds=np.array(seeds[s]), C=np.array(chosen_C[s][m]))

    summaries = dict(Parallel(n_jobs=min(len(subsets), args.n_jobs if args.n_jobs > 0 else len(subsets)))(
        delayed(summarize_size)(s, preds[s], test_y, seeds[s], args.models, args.n_boot) for s in subsets))
    curves = {"dataset": args.dataset, "models": list(args.models), "probe": "logistic_regression_cv_log_loss",
              "C_grid": C_GRID.tolist(), "class_weight": class_weight, "sizes": {}}
    for s in subsets:
        curves["sizes"][s] = {"n_train": int(len(subsets[s][0][1])), "seeds": seeds[s],
                              "C": chosen_C[s], **summaries[s]}

    print(f"\n{args.dataset}: test AUC by training-set size, mean over draws [95% CI test set + draws]")
    print(f"{'n':>7s}" + "".join(f"{m:>30s}" for m in args.models))
    for s, c in curves["sizes"].items():
        cells = [f"{r['mean']:.3f} [{r['ci_total'][0]:.3f}, {r['ci_total'][1]:.3f}]"
                 for r in (c["models"][m]["test_auc"] for m in args.models)]
        print(f"{c['n_train']:>7d}" + "".join(f"{x:>30s}" for x in cells))
    print(f"\nPaired differences in test AUC (Holm over {len(PAIRS)} per size); agree = share of single draws "
          f"that order the pair as the mean does")
    for s, c in curves["sizes"].items():
        for comp in c["comparisons"]["test_auc"]:
            print(f"{c['n_train']:>7d}  {comp['a']:>18s} - {comp['b']:<18s} {comp['mean_diff']:+.4f} "
                  f"[{comp['ci_total'][0]:+.4f}, {comp['ci_total'][1]:+.4f}]  p_holm={comp['p_holm']:.4f}  "
                  f"agree={comp['draw_agreement']:.2f}")

    if "300" in subsets:
        reference_check(args, preds["300"], test_y, seeds["300"], args.models)

    with open(out_dir / f"{args.dataset}{suffix}_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    plot_data_efficiency(curves, title=args.dataset, save_path=out_dir / f"{args.dataset}{suffix}_curves.png")
    print(f"\nSaved {args.dataset}{suffix}_curves.json / .png and the per-size predictions to {out_dir}")


if __name__ == "__main__":
    main()
