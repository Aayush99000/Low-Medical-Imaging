'''How reliable is a low-data comparison of two models? Computed from the probe_curves.py outputs.

A "study" trains both models on one training subset of size n and compares them on the test set. For every pair of
probe models, dataset and size (all draws use the same subsets for both models and the same test set):
  - the variance of one study's paired difference, split into the training-subset part (spread between draws on
    the fixed test set) and the test-set part (stratified bootstrap of the test cases within each draw);
  - how often one study, or the mean of k studies, orders the pair as the reference does, where the reference is
    the mean over the other draws (held out, so a study is never compared with itself);
  - how often one study's own test-set bootstrap calls the difference significant in the reference's direction,
    in the opposite direction, or not significant.
If per-draw differences are normal with mean mu and SD sigma, the mean of k draws agrees with the mean of the other
R - k with probability Phi(a)Phi(b) + Phi(-a)Phi(-b), a = sqrt(k) z, b = sqrt(R - k) z, z = |mu| / sigma; the figure
checks that model against the measured agreement.

Examples:
    python reliability.py                                    # every dataset with curves in experiments_curves/
    python reliability.py --datasets chexpert --metric test_balanced_acc

Outputs in --curves_dir: reliability[_<metric>].json / .csv, reliability[_<metric>]_variance.png and
reliability[_<metric>]_agreement.png.
'''
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats

from src.utils.plot_utils import METRIC_LABELS, plot_reliability
from src.utils.stats_utils import _bootstrap_p, _stratified_indices, batch_metric

K_DRAWS = (1, 2, 3, 5, 10)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--curves_dir", default="experiments_curves")
    p.add_argument("--datasets", nargs="+", default=["pneumonia", "ham10000", "chexpert"],
                   help="datasets without <dataset>_curves.json are skipped")
    p.add_argument("--metric", default="test_auc", choices=["test_auc", "test_balanced_acc"])
    p.add_argument("--n_boot", type=int, default=500, help="test-set bootstrap resamples per draw")
    p.add_argument("--n_splits", type=int, default=2000, help="random study / reference splits per k")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def draw_scores(curves_dir, dataset, model, size, metric, boot_idx):
    """Per-draw test score (R,) and its test-set bootstrap (R, B), with the same resamples for every model."""
    z = np.load(Path(curves_dir) / f"{dataset}_{model}_n{size}_curve_predictions.npz")
    labels, probs = z["labels"], z["probs"]
    order = np.argsort(labels, kind="stable")  # boot_idx rows list the cases class by class, as order does
    point = batch_metric(probs, labels, metric)
    boot = np.stack([batch_metric(p[boot_idx], labels[order], metric) for p in probs])
    return point, boot, z["seeds"]


def agreement(d, k, n_splits, rng):
    """Share of random splits where the mean of k draws has the sign of the mean of the other draws."""
    R = len(d)
    if R < 2 * k:
        return None
    hits = []
    for _ in range(n_splits):
        perm = rng.permutation(R)
        study, ref = d[perm[:k]].mean(), d[perm[k:]].mean()
        hits.append(np.sign(study) == np.sign(ref))
    return float(np.mean(hits))


def normal_agreement(z, k, R):
    """Agreement of a k-draw mean with the mean of the other R - k draws if per-draw differences are normal."""
    a, b = np.sqrt(k) * z, np.sqrt(R - k) * z
    return float(stats.norm.cdf(a) * stats.norm.cdf(b) + stats.norm.cdf(-a) * stats.norm.cdf(-b))


def pair_reliability(d, boot_d, alpha, n_splits, rng):
    """Reliability summary for one pair at one size: d (R,) per-draw differences, boot_d (R, B) their bootstraps."""
    R = len(d)
    var_subset = float(d.var(ddof=1)) if R > 1 else float("nan")
    var_test = float(boot_d.var(axis=1, ddof=1).mean())
    out = {"n_draws": R, "mean_diff": float(d.mean()), "sd_subset": float(np.sqrt(var_subset)),
           "se_test": float(np.sqrt(var_test)), "share_subset": var_subset / (var_subset + var_test)}
    if R < 2:
        return out
    out["z"] = abs(out["mean_diff"]) / out["sd_subset"] if out["sd_subset"] > 0 else float("inf")
    out["agree_k"] = {k: agreement(d, k, n_splits, rng) for k in K_DRAWS}
    out["agree_normal_k"] = {k: normal_agreement(out["z"], k, R) for k in K_DRAWS if R >= 2 * k}
    # one study's own verdict (test-set bootstrap) against the mean over the other draws
    right = wrong = 0
    for r in range(R):
        ref = np.sign(np.delete(d, r).mean())
        if _bootstrap_p(boot_d[r]) < alpha:
            right += np.sign(d[r]) == ref
            wrong += np.sign(d[r]) == -ref
    out.update(sig_right=right / R, sig_wrong=wrong / R, inconclusive=1 - (right + wrong) / R)
    return out


def analyse_dataset(args, dataset, rng):
    curves = json.load(open(Path(args.curves_dir) / f"{dataset}_curves.json"))
    models = curves["models"]
    first = np.load(Path(args.curves_dir) / f"{dataset}_{models[0]}_n{next(iter(curves['sizes']))}_curve_predictions.npz")
    labels = first["labels"]
    # stratified resamples keep every class count, so each resample's labels equal np.sort(labels)
    boot_idx = np.stack([_stratified_indices(labels, rng) for _ in range(args.n_boot)])
    assert np.array_equal(labels[boot_idx[0]], np.sort(labels))
    rows = []
    for size, info in curves["sizes"].items():
        scores = {m: draw_scores(args.curves_dir, dataset, m, size, args.metric, boot_idx) for m in models}
        for a, b in itertools.combinations(models, 2):
            (pa, ba, sa), (pb, bb, sb) = scores[a], scores[b]
            if not np.array_equal(sa, sb):
                raise ValueError(f"{dataset} n={size}: {a} and {b} were fitted on different draws")
            rows.append({"dataset": dataset, "size": size, "n_train": info["n_train"], "a": a, "b": b,
                         **pair_reliability(pa - pb, ba - bb, args.alpha, args.n_splits, rng)})
        print(f"[reliability] {dataset} n={info['n_train']}: {len(models) * (len(models) - 1) // 2} pairs", flush=True)
    return rows


def main():
    args = parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for dataset in args.datasets:
        if not (Path(args.curves_dir) / f"{dataset}_curves.json").exists():
            print(f"[reliability] {dataset}: no curves, skipped")
            continue
        rows += analyse_dataset(args, dataset, rng)

    tag = "" if args.metric == "test_auc" else f"_{args.metric.removeprefix('test_')}"
    out = Path(args.curves_dir) / f"reliability{tag}"
    with open(f"{out}.json", "w") as f:
        json.dump({"metric": args.metric, "n_boot": args.n_boot, "alpha": args.alpha, "k_draws": K_DRAWS,
                   "rows": rows}, f, indent=2)
    fields = (["dataset", "n_train", "a", "b", "n_draws", "mean_diff", "sd_subset", "se_test", "share_subset", "z"]
              + [f"agree_k{k}" for k in K_DRAWS] + [f"agree_normal_k{k}" for k in K_DRAWS]
              + ["sig_right", "sig_wrong", "inconclusive"])
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{f"agree_k{k}": v for k, v in r.get("agree_k", {}).items()},
                        **{f"agree_normal_k{k}": v for k, v in r.get("agree_normal_k", {}).items()}})

    label = METRIC_LABELS.get(args.metric, args.metric)
    print(f"\n{label}: median over model pairs per size (sd_subset / se_test in {label} units)")
    print(f"{'dataset':>10s} {'n':>7s} {'pairs':>5s} {'sd_subset':>9s} {'se_test':>8s} {'share':>6s} "
          f"{'agree1':>6s} {'agree5':>6s} {'sig_wrong':>9s}  pairs with single-draw agreement < 0.8")
    for (dataset, n), group in itertools.groupby(rows, key=lambda r: (r["dataset"], r["n_train"])):
        g = [r for r in group if r["n_draws"] > 1]
        if not g:
            continue
        med = lambda key: np.nanmedian([r[key] for r in g])
        a5 = [r["agree_k"][5] for r in g if r["agree_k"][5] is not None]
        low = sum(r["agree_k"][1] < 0.8 for r in g)
        print(f"{dataset:>10s} {n:>7d} {len(g):>5d} {med('sd_subset'):>9.4f} {med('se_test'):>8.4f} "
              f"{med('share_subset'):>6.2f} {np.median([r['agree_k'][1] for r in g]):>6.2f} "
              f"{(np.median(a5) if a5 else float('nan')):>6.2f} {np.mean([r['sig_wrong'] for r in g]):>9.3f}  "
              f"{low}/{len(g)}")

    plot_reliability(rows, metric=args.metric, save_prefix=str(out))
    print(f"\nSaved {out}.json / .csv and the figures")


if __name__ == "__main__":
    main()
