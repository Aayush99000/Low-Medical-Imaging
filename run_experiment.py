'''Run one model from the comparison grid (src/protocol.py MODEL_ZOO) on one dataset under the shared protocol.

Examples:
    python run_experiment.py --dataset pneumonia --model B_imagenet_probe
    python run_experiment.py --dataset ham10000 --model C_medsiglip_finetune --batch_size 4 --grad_accum_steps 4
    python run_experiment.py --dataset ham10000 --model B_imagenet_probe --class_weights balanced --monitor_metric val_balanced_acc
    python run_experiment.py --dataset chexpert --model C_medsiglip_probe     # Pleural Effusion, frontal views
    python run_experiment.py --dataset pneumonia --model A_scratch --smoke_test     # tiny run to check the pipeline
    python run_experiment.py --dataset pneumonia --model B_imagenet_probe --subset_seeds 0   # one subset draw only

Each model runs k-fold CV on every subset draw in --subset_seeds (default PROTOCOL['subset_seeds']), with the
test set fixed. Results: <run>_sub<seed>_kfold_results.json per draw, <run>_subsets_summary.json across draws.
'''
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.protocol import MODEL_ZOO, PROTOCOL
from src.models.backbones import build_model, get_preprocessing, make_optimizer, count_parameters
from src.utils.kfold_utils import summarize_repeated_subsets, train_with_kfold
from src.utils.subset_utils import create_subset_indices


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["pneumonia", "ham10000", "chexpert"], required=True)
    p.add_argument("--model", choices=sorted(MODEL_ZOO), required=True)
    p.add_argument("--pneumonia_root", default="chest_xray_pheumonia/chest_xray")
    p.add_argument("--ham_metadata", default="HAM10000_Dataset/metadata/HAM10000_metadata.csv")
    p.add_argument("--ham_image_dirs", nargs="+", default=["HAM10000_Dataset/train_dir", "HAM10000_Dataset/val_dir"])
    p.add_argument("--chexpert_root", default="CheXpert_Dataset", help="folder with train.csv, valid.csv, train/, valid/")
    p.add_argument("--chexpert_task", default="Pleural Effusion", help="one CheXpert finding (a train.csv column)")
    p.add_argument("--out_dir", default="experiments")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    p.add_argument("--num_workers", type=int, default=2)
    # Protocol overrides. Change these for every model in a comparison, never for one model only
    # (batch_size x grad_accum_steps is the one exception: keep their product fixed).
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--grad_accum_steps", type=int)
    p.add_argument("--backbone_lr", type=float, help="for the BACKBONE_LR_GRID sweep")
    p.add_argument("--n_folds", type=int)
    p.add_argument("--subset_size", type=int)
    p.add_argument("--per_class", type=int, help="balanced subset with this many images per class")
    p.add_argument("--seed", type=int, help="fold split, model init and data order (and the HAM10000 test split)")
    p.add_argument("--subset_seeds", type=int, nargs="+", help="training-subset draws, e.g. 0 1 2 3 4")
    p.add_argument("--class_weights", choices=["none", "balanced"], help="balanced = class-weighted cross-entropy")
    p.add_argument("--monitor_metric", choices=["val_auc", "val_balanced_acc", "val_acc", "val_loss"],
                   help="metric that picks the best epoch")
    p.add_argument("--smoke_test", action="store_true", help="tiny subset / test set, 2 folds, 1 epoch, 64px")
    return p.parse_args()


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_pneumonia(args, cfg, prep, subset_seed):
    from src.data.chest_xray_dataset import ChestXrayDataset
    kw = dict(augmentation_strength=cfg["augmentation_strength"], **prep)
    train_aug = ChestXrayDataset(args.pneumonia_root, split="train", augment=True, **kw)
    train_plain = ChestXrayDataset(args.pneumonia_root, split="train", augment=False, **kw)
    test_ds = ChestXrayDataset(args.pneumonia_root, split="test", augment=False, **kw)
    info = create_subset_indices(
        train_plain, subset_size=None if cfg["per_class"] else cfg["subset_size"], per_class=cfg["per_class"],
        seed=subset_seed, save_path=str(Path(args.out_dir) / f"pneumonia_subset_seed{subset_seed}.json"))
    idx = info["indices"]
    labels = [train_plain.labels[i] for i in idx]
    return Subset(train_aug, idx), Subset(train_plain, idx), labels, test_ds, 2


def build_ham10000(args, cfg, prep, subset_seed):
    from src.data.HAM10000_dataset import HAM10000Dataset, make_ham10000_splits
    sp = make_ham10000_splits(
        args.ham_metadata, args.ham_image_dirs, subset_size=cfg["subset_size"], per_class=cfg["per_class"],
        min_per_class=max(5, cfg["n_folds"]), seed=cfg["seed"], subset_seed=subset_seed,
        save_path=str(Path(args.out_dir) / f"ham10000_split_seed{cfg['seed']}_sub{subset_seed}.json"))
    kw = dict(augmentation_strength=cfg["augmentation_strength"], **prep)
    train_aug = HAM10000Dataset(sp["subset_images"], sp["subset_labels"], augment=True, **kw)
    train_plain = HAM10000Dataset(sp["subset_images"], sp["subset_labels"], augment=False, **kw)
    test_ds = HAM10000Dataset(sp["test_images"], sp["test_labels"], augment=False, **kw)
    return train_aug, train_plain, sp["subset_labels"], test_ds, len(sp["classes"])


def build_chexpert(args, cfg, prep, subset_seed):
    from src.data.chexpert_dataset import CheXpertBinaryDataset, make_chexpert_splits
    task_tag = args.chexpert_task.lower().replace(" ", "_")
    sp = make_chexpert_splits(
        args.chexpert_root, task=args.chexpert_task, subset_size=cfg["subset_size"], per_class=cfg["per_class"],
        min_per_class=max(5, cfg["n_folds"]), seed=subset_seed,
        save_path=str(Path(args.out_dir) / f"chexpert_{task_tag}_split_seed{subset_seed}.json"))
    kw = dict(augmentation_strength=cfg["augmentation_strength"], **prep)
    train_aug = CheXpertBinaryDataset(sp["subset_images"], sp["subset_labels"], augment=True, **kw)
    train_plain = CheXpertBinaryDataset(sp["subset_images"], sp["subset_labels"], augment=False, **kw)
    test_ds = CheXpertBinaryDataset(sp["test_images"], sp["test_labels"], augment=False, **kw)
    return train_aug, train_plain, sp["subset_labels"], test_ds, 2


def main():
    args = parse_args()
    cfg = dict(PROTOCOL, per_class=None)
    if args.smoke_test:  # applied first, so explicit arguments still win
        cfg.update(num_epochs=1, n_folds=2, image_size=(64, 64), batch_size=4, per_class=6, subset_seeds=(0, 1))
    for key, arg in [("num_epochs", "epochs"), ("batch_size", "batch_size"), ("grad_accum_steps", "grad_accum_steps"),
                     ("backbone_lr", "backbone_lr"), ("n_folds", "n_folds"), ("subset_size", "subset_size"),
                     ("per_class", "per_class"), ("seed", "seed"), ("subset_seeds", "subset_seeds"),
                     ("monitor_metric", "monitor_metric")]:
        if getattr(args, arg) is not None:
            cfg[key] = getattr(args, arg)
    if args.class_weights is not None:
        cfg["class_weights"] = None if args.class_weights == "none" else args.class_weights

    spec = MODEL_ZOO[args.model]
    device = pick_device(args.device)
    prep = get_preprocessing(spec["backbone"], cfg["image_size"])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    builder = {"pneumonia": build_pneumonia, "ham10000": build_ham10000, "chexpert": build_chexpert}[args.dataset]

    def model_fn():
        return build_model(spec["backbone"], spec["weights"], spec["regime"], num_classes,
                           image_size=cfg["image_size"], dropout=cfg["head_dropout"])

    def optimizer_fn(model):
        return make_optimizer(model, spec["weights"], cfg["head_lr"], cfg["backbone_lr"],
                              cfg["scratch_lr"], cfg["weight_decay"])

    def scheduler_fn(optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])

    base_name = f"{args.dataset}_{args.model}" + (f"_blr{cfg['backbone_lr']:g}" if args.backbone_lr else "") \
        + ("_smoke" if args.smoke_test else "")
    results_by_subset = {}
    for subset_seed in cfg["subset_seeds"]:
        train_aug, train_plain, labels, test_ds, num_classes = builder(args, cfg, prep, subset_seed)
        if args.smoke_test:
            rng = np.random.default_rng(0)
            test_ds = Subset(test_ds, rng.choice(len(test_ds), size=min(16, len(test_ds)), replace=False).tolist())
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                                 num_workers=args.num_workers, pin_memory=device.type == "cuda")

        run_name = f"{base_name}_sub{subset_seed}"
        print(f"[run_experiment] {run_name} on {device} | {spec} | {count_parameters(model_fn())}")
        results_by_subset[subset_seed] = train_with_kfold(
            model_fn=model_fn, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn,
            train_dataset=train_aug, eval_dataset=train_plain, labels=labels, test_loader=test_loader,
            run_name=run_name, n_folds=cfg["n_folds"], num_epochs=cfg["num_epochs"], batch_size=cfg["batch_size"],
            num_workers=args.num_workers, seed=cfg["seed"], early_stop_patience=cfg["early_stop_patience"],
            class_weights=cfg["class_weights"], num_classes=num_classes,
            output_dir=args.out_dir, device=device,
            use_amp=device.type == "cuda", clip_grad_norm=cfg["clip_grad_norm"],
            monitor_metric=cfg["monitor_metric"], grad_accum_steps=cfg["grad_accum_steps"],
        )

    summary = summarize_repeated_subsets(results_by_subset, title=base_name)
    with open(Path(args.out_dir) / f"{base_name}_subsets_summary.json", "w") as f:
        json.dump({"subset_seeds": list(cfg["subset_seeds"]),
                   "results_files": {s: f"{base_name}_sub{s}_kfold_results.json" for s in cfg["subset_seeds"]},
                   "summary": summary}, f, indent=2)

    run_config = {"dataset": args.dataset, "model": args.model, **spec, "protocol": cfg,
                  "preprocessing": prep, "device": str(device), "n_test": len(test_ds),
                  "parameters": count_parameters(model_fn())}
    with open(Path(args.out_dir) / f"{base_name}_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    return results_by_subset

if __name__ == "__main__":
    main()
