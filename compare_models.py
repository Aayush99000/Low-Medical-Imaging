'''Results table with bootstrap CIs and the planned model comparisons (paired bootstrap, DeLong, Holm).

Reads the files run_experiment.py writes to --out_dir (<dataset>_<model>_subsets_summary.json and the
test-prediction files it lists), so every model must have been run with the same subset seeds.

Examples:
    python compare_models.py --dataset pneumonia
    python compare_models.py --dataset ham10000 --metric test_balanced_acc
    python compare_models.py --dataset chexpert --pairs C_medsiglip_probe:B_imagenet_probe
'''
import argparse
import json
from pathlib import Path

from src.protocol import COMPARISONS, MODEL_ZOO
from src.utils.stats_utils import BOOTSTRAP_METRICS, bootstrap_test_metric, compare_models_on_test, holm_adjust


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["pneumonia", "ham10000", "chexpert"], required=True)
    p.add_argument("--out_dir", default="experiments")
    p.add_argument("--metric", choices=BOOTSTRAP_METRICS, default="test_auc", help="metric the comparisons test")
    p.add_argument("--pairs", nargs="+", help="comparisons as a:b (tested as a - b); default: COMPARISONS")
    p.add_argument("--suffix", default="", help="run-name suffix, e.g. _smoke or _blr3e-05")
    p.add_argument("--n_boot", type=int, default=2000)
    return p.parse_args()


def load_predictions(out_dir, dataset, model, suffix):
    path = Path(out_dir) / f"{dataset}_{model}{suffix}_subsets_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        files = json.load(f)["predictions_files"]
    return {int(s): Path(out_dir) / name for s, name in files.items()}


def main():
    args = parse_args()
    preds = {m: load_predictions(args.out_dir, args.dataset, m, args.suffix) for m in MODEL_ZOO}
    preds = {m: p for m, p in preds.items() if p is not None}
    if not preds:
        raise SystemExit(f"No {args.dataset} runs found in {args.out_dir}.")

    table = {}
    print(f"\n{args.dataset}: mean over subset draws, 95% CI (test set + draws)")
    print(f"{'model':24s}" + "".join(f"{m:>28s}" for m in BOOTSTRAP_METRICS))
    for model, p in preds.items():
        table[model] = {m: bootstrap_test_metric(p, m, n_boot=args.n_boot) for m in BOOTSTRAP_METRICS}
        cells = [f"{r['mean']:.3f} [{r['ci_total'][0]:.3f}, {r['ci_total'][1]:.3f}]" for r in table[model].values()]
        print(f"{model:24s}" + "".join(f"{c:>28s}" for c in cells))

    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else list(COMPARISONS)
    pairs = [(a, b) for a, b in pairs if a in preds and b in preds]
    comparisons = [dict(a=a, b=b, **compare_models_on_test(preds[a], preds[b], args.metric, n_boot=args.n_boot))
                   for a, b in pairs]
    for c, p_holm in zip(comparisons, holm_adjust([c["p_value"] for c in comparisons])):
        c["p_holm"] = p_holm

    if comparisons:
        print(f"\nPlanned comparisons on {args.metric} (a - b), paired bootstrap, Holm-corrected over {len(comparisons)}")
        for c in comparisons:
            line = (f"{c['a']:>22s} - {c['b']:<22s} {c['mean_diff']:+.4f}  95% CI [{c['ci_total'][0]:+.4f}, "
                    f"{c['ci_total'][1]:+.4f}]  p={c['p_value']:.4f}  p_holm={c['p_holm']:.4f}")
            if "delong_per_subset" in c:
                line += "  DeLong p per draw: " + ", ".join(f"{d['p_value']:.3f}" for d in c["delong_per_subset"].values())
            print(line)

    out = Path(args.out_dir) / f"{args.dataset}{args.suffix}_comparison_{args.metric}.json"
    with open(out, "w") as f:
        json.dump({"table": table, "comparisons": comparisons}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
