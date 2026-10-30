'''Run one model from the comparison grid (src/protocol.py MODEL_ZOO) on one dataset under the shared protocol.

Examples:
    python run_experiment.py --dataset pneumonia --model B_imagenet_probe
    python run_experiment.py --dataset ham10000 --model C_medsiglip_finetune --batch_size 4 --grad_accum_steps 4
    python run_experiment.py --dataset pneumonia --model A_scratch --smoke_test     # tiny run to check the pipeline
'''
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.protocol import MODEL_ZOO, PROTOCOL
from src.models.backbones import build_model, get_preprocessing, make_optimizer, count_parameters
from src.utils.kfold_utils import train_with_kfold
from src.utils.subset_utils import create_subset_indices


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["pneumonia", "ham10000"], required=True)
    p.add_argument("--model", choices=sorted(MODEL_ZOO), required=True)
    p.add_argument("--pneumonia_root", default="chest_xray_pheumonia/chest_xray")
    p.add_argument("--ham_metadata", default="HAM10000_Dataset/metadata/HAM10000_metadata.csv")
    p.add_argument("--ham_image_dirs", nargs="+", default=["HAM10000_Dataset/train_dir", "HAM10000_Dataset/val_dir"])
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
    p.add_argument("--seed", type=int)
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


def build_pneumonia(args, cfg, prep):
    from src.data.chest_xray_dataset import ChestXrayDataset
    kw = dict(augmentation_strength=cfg["augmentation_strength"], **prep)
    train_aug = ChestXrayDataset(args.pneumonia_root, split="train", augment=True, **kw)
    train_plain = ChestXrayDataset(args.pneumonia_root, split="train", augment=False, **kw)
    test_ds = ChestXrayDataset(args.pneumonia_root, split="test", augment=False, **kw)
    info = create_subset_indices(
        train_plain, subset_size=None if cfg["per_class"] else cfg["subset_size"], per_class=cfg["per_class"],
        seed=cfg["seed"], save_path=str(Path(args.out_dir) / f"pneumonia_subset_seed{cfg['seed']}.json"))
    idx = info["indices"]
    labels = [train_plain.labels[i] for i in idx]
    return Subset(train_aug, idx), Subset(train_plain, idx), labels, test_ds, 2


def build_ham10000(args, cfg, prep):
    from src.data.HAM10000_dataset import HAM10000Dataset, make_ham10000_splits
    sp = make_ham10000_splits(
        args.ham_metadata, args.ham_image_dirs, subset_size=cfg["subset_size"], per_class=cfg["per_class"],
        min_per_class=max(5, cfg["n_folds"]), seed=cfg["seed"],
        save_path=str(Path(args.out_dir) / f"ham10000_split_seed{cfg['seed']}.json"))
    kw = dict(augmentation_strength=cfg["augmentation_strength"], **prep)
    train_aug = HAM10000Dataset(sp["subset_images"], sp["subset_labels"], augment=True, **kw)
    train_plain = HAM10000Dataset(sp["subset_images"], sp["subset_labels"], augment=False, **kw)
    test_ds = HAM10000Dataset(sp["test_images"], sp["test_labels"], augment=False, **kw)
    return train_aug, train_plain, sp["subset_labels"], test_ds, len(sp["classes"])


def main():
    args = parse_args()
    cfg = dict(PROTOCOL, per_class=None)
    if args.smoke_test:  # applied first, so explicit arguments still win
        cfg.update(num_epochs=1, n_folds=2, image_size=(64, 64), batch_size=4, per_class=6)
    for key, arg in [("num_epochs", "epochs"), ("batch_size", "batch_size"), ("grad_accum_steps", "grad_accum_steps"),
                     ("backbone_lr", "backbone_lr"), ("n_folds", "n_folds"), ("subset_size", "subset_size"),
                     ("per_class", "per_class"), ("seed", "seed")]:
        if getattr(args, arg) is not None:
            cfg[key] = getattr(args, arg)

    spec = MODEL_ZOO[args.model]
    device = pick_device(args.device)
    prep = get_preprocessing(spec["backbone"], cfg["image_size"])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    builder = build_pneumonia if args.dataset == "pneumonia" else build_ham10000
    train_aug, train_plain, labels, test_ds, num_classes = builder(args, cfg, prep)
    if args.smoke_test:
        rng = np.random.default_rng(0)
        test_ds = Subset(test_ds, rng.choice(len(test_ds), size=min(16, len(test_ds)), replace=False).tolist())
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    def model_fn():
        return build_model(spec["backbone"], spec["weights"], spec["regime"], num_classes,
                           image_size=cfg["image_size"], dropout=cfg["head_dropout"])

    def optimizer_fn(model):
        return make_optimizer(model, spec["weights"], cfg["head_lr"], cfg["backbone_lr"],
                              cfg["scratch_lr"], cfg["weight_decay"])

    def scheduler_fn(optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])

    run_name = f"{args.dataset}_{args.model}" + (f"_blr{cfg['backbone_lr']:g}" if args.backbone_lr else "") \
        + ("_smoke" if args.smoke_test else "")
    print(f"[run_experiment] {run_name} on {device} | {spec} | {count_parameters(model_fn())}")

    results = train_with_kfold(
        model_fn=model_fn, optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn,
        train_dataset=train_aug, eval_dataset=train_plain, labels=labels, test_loader=test_loader,
        run_name=run_name, n_folds=cfg["n_folds"], num_epochs=cfg["num_epochs"], batch_size=cfg["batch_size"],
        num_workers=args.num_workers, seed=cfg["seed"], early_stop_patience=cfg["early_stop_patience"],
        output_dir=args.out_dir, device=device,
        use_amp=device.type == "cuda", clip_grad_norm=cfg["clip_grad_norm"],
        monitor_metric=cfg["monitor_metric"], grad_accum_steps=cfg["grad_accum_steps"],
    )

    run_config = {"dataset": args.dataset, "model": args.model, **spec, "protocol": cfg,
                  "preprocessing": prep, "device": str(device), "n_test": len(test_ds),
                  "parameters": count_parameters(model_fn())}
    with open(Path(args.out_dir) / f"{run_name}_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    main()
