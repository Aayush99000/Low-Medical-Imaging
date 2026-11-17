'''Cache frozen-backbone features for every training-pool and test image, for the probe data-efficiency curves.

Runs the frozen backbone of one linear-probe model (src/protocol.py MODEL_ZOO) once over every image a training
subset can be drawn from and over the test set, at the protocol resolution with the same preprocessing as
run_experiment.py (no augmentation). probe_curves.py then fits probes on the cache in seconds.

Examples:
    python extract_features.py --dataset pneumonia --model C_medsiglip_probe
    python extract_features.py --dataset chexpert --model B_imagenet_probe --batch_size 128
    python extract_features.py --dataset pneumonia --model B_imagenet_probe --smoke_test   # 64px pipeline check

Output: <out_dir>/<dataset>_<model>_features.npz with pool_X, pool_y, pool_paths (the training pool) and
test_X, test_y, test_paths. Features are stored as float16.
'''
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from run_experiment import add_data_args, pick_device
from src.protocol import MODEL_ZOO, PROTOCOL
from src.models.backbones import build_model, get_preprocessing

PROBE_MODELS = sorted(m for m, spec in MODEL_ZOO.items() if spec["regime"] == "linear_probe")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--model", choices=PROBE_MODELS, required=True)
    p.add_argument("--out_dir", default="features")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--smoke_test", action="store_true", help="64px images, to check the pipeline quickly")
    return p.parse_args()


def dataset_pools(args):
    """Training pool and test set as (paths, labels) lists: every image a run_experiment.py subset can draw from."""
    if args.dataset == "pneumonia":
        from src.data.chest_xray_dataset import ChestXrayDataset
        train = ChestXrayDataset(args.pneumonia_root, split="train", augment=False)
        test = ChestXrayDataset(args.pneumonia_root, split="test", augment=False)
        return (train.images, train.labels), (test.images, test.labels)
    if args.dataset == "ham10000":
        from src.data.HAM10000_dataset import make_ham10000_splits
        sp = make_ham10000_splits(args.ham_metadata, args.ham_image_dirs, subset_size=None, seed=PROTOCOL["seed"])
    else:
        from src.data.chexpert_dataset import make_chexpert_splits
        sp = make_chexpert_splits(args.chexpert_root, task=args.chexpert_task, subset_size=None, seed=PROTOCOL["seed"])
    return (sp["train_pool_images"], sp["train_pool_labels"]), (sp["test_images"], sp["test_labels"])


def image_dataset(dataset_name, paths, labels, prep):
    """Unaugmented dataset over the given images, loaded exactly like the run_experiment.py datasets."""
    if dataset_name == "ham10000":
        from src.data.HAM10000_dataset import HAM10000Dataset
        return HAM10000Dataset(paths, labels, augment=False, **prep)
    from src.data.chexpert_dataset import CheXpertBinaryDataset  # ChestXrayDataset loading on an image list
    return CheXpertBinaryDataset(paths, labels, augment=False, **prep)


@torch.no_grad()
def extract(backbone, dataset, batch_size, num_workers, device, name):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    feats, start, report_every = [], time.time(), max(1, len(loader) // 10)
    for i, (x, _) in enumerate(loader, 1):
        # same mixed precision as the probe training runs
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            f = backbone(x.to(device, non_blocking=True))
        feats.append(f.float().cpu().numpy())
        if i % report_every == 0 or i == len(loader):
            done = min(i * batch_size, len(dataset))
            print(f"[extract_features] {name}: {done}/{len(dataset)} images, {done / (time.time() - start):.0f} img/s",
                  flush=True)
    return np.concatenate(feats).astype(np.float16)


def main():
    args = parse_args()
    spec = MODEL_ZOO[args.model]
    device = pick_device(args.device)
    image_size = (64, 64) if args.smoke_test else PROTOCOL["image_size"]
    prep = get_preprocessing(spec["backbone"], image_size)

    (pool_paths, pool_y), (test_paths, test_y) = dataset_pools(args)

    backbone = build_model(spec["backbone"], spec["weights"], spec["regime"], num_classes=2,
                           image_size=image_size).backbone.eval().to(device)
    print(f"[extract_features] {args.dataset} {args.model} on {device}: {len(pool_paths)} pool + "
          f"{len(test_paths)} test images at {image_size}, {backbone.out_dim}-d features")
    pool_X = extract(backbone, image_dataset(args.dataset, pool_paths, pool_y, prep),
                     args.batch_size, args.num_workers, device, "pool")
    test_X = extract(backbone, image_dataset(args.dataset, test_paths, test_y, prep),
                     args.batch_size, args.num_workers, device, "test")

    out = Path(args.out_dir) / f"{args.dataset}_{args.model}{'_smoke' if args.smoke_test else ''}_features.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": args.dataset, "model": args.model, **spec, "image_size": list(image_size),
            "preprocessing": prep, "device": str(device)}
    np.savez(out, pool_X=pool_X, pool_y=np.asarray(pool_y, dtype=np.int64), pool_paths=np.asarray(pool_paths),
             test_X=test_X, test_y=np.asarray(test_y, dtype=np.int64), test_paths=np.asarray(test_paths),
             meta=json.dumps(meta, default=str))
    print(f"[extract_features] Saved {pool_X.shape} pool / {test_X.shape} test features to {out}")


if __name__ == "__main__":
    main()
