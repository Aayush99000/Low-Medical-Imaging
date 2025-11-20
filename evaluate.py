"""Evaluation script for chest X-ray classification model."""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve, auc)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import build_model
from image_preprocessing import get_data_loaders


class ModelEvaluator:
    """Evaluate PyTorch model on classification tasks."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device = None):
        """Initialize ModelEvaluator.
        
        Args:
            model: PyTorch model to evaluate.
            device: Device to evaluate on.
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader for evaluation.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating", leave=True)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        
        return metrics
    
    def _compute_metrics(self, y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        y_probs: np.ndarray) -> Dict:
        """Compute evaluation metrics.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_probs: Predicted probabilities.
        
        Returns:
            Dictionary with computed metrics.
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        # AUC-ROC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                auc_roc = roc_auc_score(y_true, y_probs[:, 1])
                metrics["auc_roc"] = float(auc_roc)
            except Exception as e:
                print(f"Error computing AUC-ROC: {e}")
                metrics["auc_roc"] = None
        
        return metrics
    
    def predict_batch(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on a batch of images.
        
        Args:
            images: Batch of images with shape (B, 3, H, W).
        
        Returns:
            Tuple of (predicted_labels, probabilities).
        """
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy(), probs.cpu().numpy()


def evaluate_model(model_path: str,
                   dataset_root: str,
                   model_name: str = "vgg19",
                   batch_size: int = 32,
                   split: str = "test",
                   device: str = "cuda" if torch.cuda.is_available() else "cpu",
                   output_dir: str = "./results") -> Dict:
    """Evaluate a trained model on a dataset split.
    
    Args:
        model_path: Path to saved model weights.
        dataset_root: Root directory of dataset.
        model_name: Name of model architecture.
        batch_size: Batch size for evaluation.
        split: Data split to evaluate ("test", "val", or "train").
        device: Device to use for evaluation.
        output_dir: Directory to save evaluation results.
    
    Returns:
        Dictionary with evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    
    print(f"Loading model from {model_path}...")
    model = build_model(
        model_name=model_name,
        num_classes=2,
        pretrained=False,
        freeze_base=False
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    print(f"Loading {split} data...")
    
    # Create single dataloader for the split
    from image_preprocessing import ChestXrayDataset
    dataset = ChestXrayDataset(
        dataset_root=dataset_root,
        split=split,
        image_size=(224, 224),
        augment=False,
        normalize=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluating on {split} set ({len(dataset)} images)...")
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(dataloader)
    
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "split": split,
        "num_samples": len(dataset),
        "metrics": metrics
    }
    
    # Save results
    results_path = os.path.join(output_dir, f"{model_name}_{split}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Print metrics
    print(f"\n{'='*70}")
    print(f"Evaluation Results on {split} set")
    print(f"{'='*70}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if metrics.get('auc_roc'):
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    print(f"{'='*70}\n")
    
    return results


def compare_models(model_paths: Dict[str, str],
                   dataset_root: str,
                   batch_size: int = 32,
                   split: str = "test",
                   device: str = "cuda" if torch.cuda.is_available() else "cpu",
                   output_dir: str = "./results") -> Dict:
    """Compare multiple trained models on the same dataset.
    
    Args:
        model_paths: Dictionary mapping model names to model paths.
        dataset_root: Root directory of dataset.
        batch_size: Batch size for evaluation.
        split: Data split to evaluate.
        device: Device to use for evaluation.
        output_dir: Directory to save comparison results.
    
    Returns:
        Dictionary with comparison results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    comparison = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n{'#'*70}")
        print(f"# Evaluating {model_name.upper()}")
        print(f"{'#'*70}\n")
        
        results = evaluate_model(
            model_path=model_path,
            dataset_root=dataset_root,
            model_name=model_name,
            batch_size=batch_size,
            split=split,
            device=device,
            output_dir=output_dir
        )
        
        comparison[model_name] = results["metrics"]
    
    # Save comparison
    comparison_path = os.path.join(output_dir, f"model_comparison_{split}.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {comparison_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Model Comparison Summary ({split} set)")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*70}")
    for model_name, metrics in comparison.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    print(f"{'='*70}\n")
    
    return comparison


if __name__ == "__main__":
    # Example: Evaluate single model
    model_path = "./checkpoints/vgg19_20251120_120000_final.pt"
    dataset_root = "./dataset"
    
    results = evaluate_model(
        model_path=model_path,
        dataset_root=dataset_root,
        model_name="vgg19",
        batch_size=32,
        split="test"
    )
    
    # Example: Compare multiple models
    # model_paths = {
    #     "vgg19": "./checkpoints/vgg19_20251120_120000_final.pt",
    #     "resnet50": "./checkpoints/resnet50_20251120_120000_final.pt",
    #     "densenet121": "./checkpoints/densenet121_20251120_120000_final.pt"
    # }
    # comparison = compare_models(model_paths, dataset_root, split="test")
