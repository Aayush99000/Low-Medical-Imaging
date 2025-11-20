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

from models import build_model, get_model_summary
from image_preprocessing import get_data_loaders, get_dataset_statistics


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 10, verbose: bool = True, delta: float = 0.0):
        """Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs with no improvement after which training stops.
            verbose: Whether to print early stopping messages.
            delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Validation loss value.
        
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        
        return self.early_stop


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


def train_model(model_name: str,
                dataset_root: str,
                epochs: int = 50,
                batch_size: int = 32,
                learning_rate: float = 1e-4,
                freeze_base: bool = True,
                fine_tune_epochs: int = 10,
                fine_tune_layers: int = 50,
                checkpoint_dir: str = "./checkpoints",
                device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict:
    """Train a transfer learning model on chest X-ray data.
    
    Args:
        model_name: Name of model ("vgg19", "resnet50", "inceptionv3", "densenet121").
        dataset_root: Root path to dataset.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        freeze_base: Whether to freeze base model initially.
        fine_tune_epochs: Number of fine-tuning epochs.
        fine_tune_layers: Number of layers to unfreeze for fine-tuning.
        checkpoint_dir: Directory to save checkpoints.
        device: Device to train on ("cuda" or "cpu").
    
    Returns:
        Dictionary with training results.
    """
    # Setup
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Device: {device}")
    print(f"Dataset root: {dataset_root}")
    
    # Dataset statistics
    stats = get_dataset_statistics(dataset_root)
    print(f"Dataset statistics: {stats}")
    
    # Build model
    print(f"\nBuilding {model_name}...")
    model = build_model(
        model_name=model_name,
        num_classes=2,
        pretrained=True,
        freeze_base=freeze_base,
        hidden_size=512,
        dropout_rate=0.5
    )
    model = model.to(device)
    get_model_summary(model)
    
    # Data loaders
    print(f"\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        image_size=(224, 224),
        num_workers=4,
        augmentation_strength=0.5
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    results = {
        "model_name": model_name,
        "dataset_stats": stats,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "freeze_base": freeze_base
        }
    }
    
    # Initial training phase
    print(f"\n{'='*70}")
    print(f"Initial Training Phase ({epochs} epochs)")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            scheduler.step(val_loss)
            
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    results["initial_training"] = {
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_train_acc": float(history["train_acc"][-1]),
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        "final_val_acc": float(history["val_acc"][-1]) if history["val_acc"] else None
    }
    
    # Fine-tuning phase
    if fine_tune_epochs > 0 and freeze_base:
        print(f"\n{'='*70}")
        print(f"Fine-tuning Phase ({fine_tune_epochs} epochs)")
        print(f"{'='*70}")
        
        model.unfreeze_last_n_layers(fine_tune_layers)
        
        # Use lower learning rate for fine-tuning
        fine_tune_lr = learning_rate / 10
        optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-8)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        
        ft_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        for epoch in range(fine_tune_epochs):
            print(f"\nFT Epoch [{epoch+1}/{fine_tune_epochs}]")
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            ft_history["train_loss"].append(train_loss)
            ft_history["train_acc"].append(train_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            if val_loader:
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                ft_history["val_loss"].append(val_loss)
                ft_history["val_acc"].append(val_acc)
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                scheduler.step(val_loss)
                
                if early_stopping(val_loss):
                    print(f"Early stopping at fine-tune epoch {epoch+1}")
                    break
        
        results["fine_tuning"] = {
            "epochs": fine_tune_epochs,
            "layers_unfrozen": fine_tune_layers,
            "learning_rate": fine_tune_lr,
            "epochs_trained": len(ft_history["train_loss"]),
            "final_train_loss": float(ft_history["train_loss"][-1]),
            "final_train_acc": float(ft_history["train_acc"][-1]),
            "final_val_loss": float(ft_history["val_loss"][-1]) if ft_history["val_loss"] else None,
            "final_val_acc": float(ft_history["val_acc"][-1]) if ft_history["val_acc"] else None
        }
    
    # Save model
    model_path = os.path.join(checkpoint_dir, f"{model_name}_{timestamp}_final.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    results["model_path"] = model_path
    
    # Save results
    results_path = os.path.join(checkpoint_dir, f"{model_name}_{timestamp}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    dataset_root = "./dataset"  # Update with actual dataset path
    
    models_to_train = ["vgg19", "resnet50", "densenet121"]
    
    for model_name in models_to_train:
        print(f"\n{'#'*70}")
        print(f"# Training {model_name.upper()}")
        print(f"{'#'*70}\n")
        
        results = train_model(
            model_name=model_name,
            dataset_root=dataset_root,
            epochs=30,
            batch_size=32,
            learning_rate=1e-4,
            freeze_base=True,
            fine_tune_epochs=10,
            fine_tune_layers=50,
            checkpoint_dir="./checkpoints"
        )
        
        print(f"\nResults for {model_name}:")
        print(json.dumps(results, indent=2))