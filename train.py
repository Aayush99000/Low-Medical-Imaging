"""PyTorch training script for chest X-ray binary classification."""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple

# from ModelB import build_model, get_model_summary
# from image_preprocessing import get_data_loaders, get_dataset_statistics

from src.data.chest_xray_dataset import ChestXrayDataset
from src.utils.subset_utils import load_subset
from torch.utils.data import 

# try to import dataset + utils from src; if not available, raise clear error
try:
    from src.data.chest_xray_dataset import ChestXrayDataset
except Exception as e:
    raise ImportError("Could not import ChestXrayDataset from src.data. Make sure src/data/chest_xray_dataset.py exists and src/ is on PYTHONPATH.") from e

try:
    from src.utils.subset_utils import create_subset_indices, load_subset
except Exception:
    # fallback minimal versions if src.utils not present
    def create_subset_indices(*args, **kwargs):
        raise ImportError("create_subset_indices not found in src.utils.subset_utils")
    def load_subset(*args, **kwargs):
        raise ImportError("load_subset not found in src.utils.subset_utils")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cudnn may slow training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_modelA():

    full_train = ChestXrayDataset(
        dataset_root="path",
        split="train",
        augment=True,
        normalize=True
    )

    train_subset = load_subset(
        full_train,
        "data/processed/subset_indices_300.json"
    )

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)


def train_epoch(model: nn.Module, 
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: PyTorch model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
    
    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Validate the model.
    
    Args:
        model: PyTorch model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device to validate on.
    
    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_loop(
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str,
    epochs: int,
    lr: float,
    out_dir: Path,
    start_epoch: int = 0,
    save_every: int = 1
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
        else:
            val_acc = 0.0

        print(f"[Epoch {epoch+1}/{epochs}] Train loss: {train_loss:.4f} Train acc: {train_acc:.4f} Val acc: {val_acc:.4f}")

        # save checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = out_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            }, ckpt_path)

        # update best
        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = out_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)

    return best_val_acc


def build_dataloaders(
    dataset_root: str,
    subset_json: Optional[str],
    subset_size: Optional[int],
    batch_size: int,
    num_workers: int,
    seed: int,
    create_subset_if_missing: bool = False
):
    """
    Loads full train dataset, loads (or creates) subset JSON, constructs DataLoaders.
    Returns: train_loader, val_loader, test_loader, full_train_dataset (useful later)
    """
    # instantiate datasets
    full_train = ChestXrayDataset(dataset_root, split="train", augment=True, normalize=True)
    val_dataset = ChestXrayDataset(dataset_root, split="val", augment=False, normalize=True)
    test_dataset = ChestXrayDataset(dataset_root, split="test", augment=False, normalize=True)

    # ensure reproducibility
    set_seed(seed)

    # load or create subset
    if subset_json is None:
        # don't use subset -> use full train
        train_ds = full_train
    else:
        subset_path = Path(subset_json)
        if not subset_path.exists():
            if create_subset_if_missing:
                print(f"Subset JSON not found at {subset_path}. Creating a new subset with size={subset_size} (seed={seed}).")
                create_subset_indices(full_train, subset_size=subset_size, seed=seed, save_path=str(subset_path))
            else:
                raise FileNotFoundError(f"Subset json not found: {subset_path}. Pass --create_subset to auto-create.")
        # load as a Subset
        train_ds = load_subset(full_train, str(subset_path))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_train


def parse_args():
    p = argparse.ArgumentParser(description="Train DenseNet experiments (Model A scratch / Model B ImageNet TL)")
    p.add_argument("--model", choices=["A", "B"], required=True, help="A: DenseNet scratch, B: DenseNet ImageNet-pretrained")
    p.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root (contains train/val/test folders)")
    p.add_argument("--subset_json", type=str, default="data/processed/subset_indices_300.json", help="Path to subset JSON")
    p.add_argument("--subset_size", type=int, default=300, help="When creating subset, total images to pick")
    p.add_argument("--create_subset", action="store_true", help="If set and subset JSON missing, create it from full train set")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="experiments/run")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Arguments:", args)

    # set seeds once
    set_seed(args.seed)

    # build dataloaders and datasets (this will create the subset JSON if requested)
    train_loader, val_loader, test_loader, full_train_ds = build_dataloaders(
        dataset_root=args.dataset_root,
        subset_json=args.subset_json,
        subset_size=args.subset_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        create_subset_if_missing=args.create_subset
    )

    # build model
    if args.model == "A":
        print("Building Model A (DenseNet from scratch)")
        model = build_densenet(num_classes=2, pretrained=False, device=args.device)
    else:
        print("Building Model B (DenseNet pretrained on ImageNet)")
        model = build_densenet(num_classes=2, pretrained=True, device=args.device)

    # quick sanity check on one batch
    xb, yb = next(iter(train_loader))
    print("Sanity batch shapes:", xb.shape, yb.shape)

    # train
    best_val = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=out_dir,
        save_every=1
    )

    # final evaluation on test set using best model if exists
    best_model_path = out_dir / "best_model.pt"
    if best_model_path.exists():
        print("Loading best model for final test evaluation.")
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    test_acc = evaluate(model, test_loader, device=args.device)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # write metadata about run
    run_info = {
        "args": vars(args),
        "best_val_acc": best_val,
        "final_test_acc": float(test_acc)
    }
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run metadata saved to {out_dir / 'run_info.json'}")


if __name__ == "__main__":
    main()