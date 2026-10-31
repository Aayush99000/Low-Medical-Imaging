'''Results table with bootstrap CIs and the planned model comparisons (paired bootstrap, DeLong, Holm).

Reads the test-prediction files run_experiment.py writes to --out_dir (<dataset>_<model>_sub<seed>_test_predictions.npz),
whether the draws ran in one job or as a job array. Comparisons use the subset seeds both models have.

Examples:
    python compare_models.py --dataset pneumonia
    python compare_models.py --dataset ham10000 --metric test_balanced_acc
    python compare_models.py --dataset chexpert --pairs C_medsiglip_probe:B_imagenet_probe
'''
import argparse
import json
import re
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
    """{subset_seed: predictions file} for every draw of this run, including draws run as separate jobs."""
    pattern = re.compile(rf"^{re.escape(f'{dataset}_{model}{suffix}')}_sub(\d+)_test_predictions\.npz$")
    files = {int(m.group(1)): p for p in Path(out_dir).glob("*_test_predictions.npz") if (m := pattern.match(p.name))}
    return files or None


def main():
    args = parse_args()
    preds = {m: load_predictions(args.out_dir, args.dataset, m, args.suffix) for m in MODEL_ZOO}
    preds = {m: p for m, p in preds.items() if p is not None}
    if not preds:
        raise SystemExit(f"No {args.dataset} runs found in {args.out_dir}.")

    table = {}
    print(f"\n{args.dataset}: mean over subset draws, 95% CI (test set + draws)")
    print(f"{'model':32s}" + "".join(f"{m:>28s}" for m in BOOTSTRAP_METRICS))
    for model, p in preds.items():
        table[model] = {m: bootstrap_test_metric(p, m, n_boot=args.n_boot) for m in BOOTSTRAP_METRICS}
        cells = [f"{r['mean']:.3f} [{r['ci_total'][0]:.3f}, {r['ci_total'][1]:.3f}]" for r in table[model].values()]
        print(f"{f'{model} ({len(p)} draws)':32s}" + "".join(f"{c:>28s}" for c in cells))

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
