import time
import csv
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

# Optional AUC metric
try:
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Metrics that can pick the best epoch (all higher-is-better; val_loss is negated internally)
MONITOR_METRICS = ("val_auc", "val_balanced_acc", "val_acc", "val_loss")


def compute_auc(y_true: np.ndarray, probs: np.ndarray) -> Optional[float]:
    """
    Binary: ROC AUC of the positive class. Multiclass: macro one-vs-rest AUC over the classes that have
    both positives and negatives in y_true (small CV folds can miss a rare class). probs: (N, C), float32.
    """
    if not _HAS_SKLEARN or len(y_true) == 0:
        return None
    n_classes = probs.shape[1]
    if n_classes == 2:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, probs[:, 1]))
    aucs = []
    for c in range(n_classes):
        positives = y_true == c
        if 0 < positives.sum() < len(y_true):
            aucs.append(roc_auc_score(positives.astype(int), probs[:, c]))
    return float(np.mean(aucs)) if aucs else None


def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    """'Balanced' class weights n / (C * n_c), as in sklearn; a class missing from labels gets weight 0."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    weights = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        output_dir: str = "experiments",
        experiment_name: str = "default",
        use_amp: bool = True,
        clip_grad_norm: Optional[float] = 1.0,
        log_interval: int = 50,
        save_interval: int = 1,
        monitor_metric: str = "val_auc",  # or "val_acc"
        grad_accum_steps: int = 1  # optimizer step every N batches (effective batch = batch_size * N)
    ):
        self.model = model
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        self.save_interval = save_interval
        if monitor_metric not in MONITOR_METRICS:
            raise ValueError(f"monitor_metric must be one of {MONITOR_METRICS}, got {monitor_metric}")
        self.monitor_metric = monitor_metric
        self._warned_auc_fallback = False
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        # output paths
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path = self.output_dir / "training_log.csv"
        self.best_model_path = self.output_dir / "best_model.pt"

        # training state
        self.current_epoch = 0
        self.best_metric = -float("inf")
        self.best_epoch = None
        self.history = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [],
            "val_balanced_acc": [], "lrs": [], "epoch_times": []
        }

        # init csv header
        self._init_csv()

        # move model to device
        self.model.to(self.device)

    def _init_csv(self):
        if not self.log_csv_path.exists():
            with open(self.log_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc",
                    "val_balanced_acc", "lr", "epoch_time_s"
                ])

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, Optional[float]]:
        """
        Return (accuracy, auc_or_none) computed on CPU tensors.
        """
        probs = torch.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).sum().item() / labels.size(0)
        return acc, compute_auc(labels.cpu().numpy(), probs.cpu().numpy())

    def _select_metric(self, val_loss: float, val_acc: float, val_auc: Optional[float],
                       val_balanced_acc: Optional[float]) -> Optional[float]:
        """Value of the monitored metric (higher is better). AUC falls back to balanced accuracy, never plain accuracy."""
        if self.monitor_metric == "val_auc":
            if val_auc is not None:
                return val_auc
            if not self._warned_auc_fallback:
                print("[ModelTrainer] val AUC unavailable this epoch; selecting by val_balanced_acc instead")
                self._warned_auc_fallback = True
            return val_balanced_acc
        if self.monitor_metric == "val_balanced_acc":
            return val_balanced_acc
        if self.monitor_metric == "val_loss":
            return -val_loss
        return val_acc

    def _train_one_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        n_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}", leave=False)
        self.optimizer.zero_grad()
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # step after every grad_accum_steps batches, and on the last (possibly partial) group
            step_now = (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == n_batches
            group_size = self.grad_accum_steps if (batch_idx // self.grad_accum_steps + 1) * self.grad_accum_steps <= n_batches \
                else n_batches % self.grad_accum_steps

            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss / group_size).backward()
                if step_now:
                    if self.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                (loss / group_size).backward()
                if step_now:
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % self.log_interval == 0 or (batch_idx + 1) == len(self.train_loader):
                avg_loss = running_loss / total
                acc = correct / total
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.3f}", "lr": f"{lr:.2e}"})

        avg_loss = running_loss / max(1, total)
        acc = correct / max(1, total)
        return avg_loss, acc

    def _eval_loader(self, loader) -> Tuple[float, float, Optional[float], Optional[float]]:
        """
        Evaluate on loader and return (loss, acc, auc_or_none, balanced_acc_or_none).
        AUC is binary or macro one-vs-rest multiclass, from float32 softmax probabilities.
        """
        self.model.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        all_probs = []
        all_labels = []
        all_preds = []

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Eval (epoch {self.current_epoch})", leave=False)
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images).float()  # float32 so probabilities sum to 1 (AMP logits may be fp16)
                loss = self.criterion(logits, labels)
                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        avg_loss = running_loss / max(1, total)
        acc = correct / max(1, total)
        auc, balanced_acc = None, None
        if _HAS_SKLEARN and total > 0:
            y_true = np.concatenate(all_labels)
            auc = compute_auc(y_true, np.concatenate(all_probs).astype(np.float32))
            balanced_acc = float(balanced_accuracy_score(y_true, np.concatenate(all_preds)))
        return avg_loss, acc, auc, balanced_acc

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ck = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "history": self.history,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(ck, path)
        if is_best:
            torch.save(self.model.state_dict(), self.best_model_path)
        return path

    def load_checkpoint(self, path: str) -> int:
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck and ck["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in ck and ck["scheduler_state_dict"] is not None:
            try:
                self.scheduler.load_state_dict(ck["scheduler_state_dict"])
            except Exception:
                pass
        if self.use_amp and "scaler_state_dict" in ck and ck["scaler_state_dict"] is not None:
            try:
                self.scaler.load_state_dict(ck["scaler_state_dict"])
            except Exception:
                pass
        self.best_metric = ck.get("best_metric", self.best_metric)
        return ck.get("epoch", 0)

    def train(self, num_epochs: int, resume_from: Optional[str] = None,
              early_stop_patience: Optional[int] = None):
        """
        Train the model for num_epochs. Optionally resume from checkpoint.
        If early_stop_patience is provided, stop when metric hasn't improved for that many epochs.
        """
        start_epoch = 1
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"[resume] starting from epoch {start_epoch}")

        no_improve = 0
        early_stop_patience = early_stop_patience or 99999

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            t0 = time.time()

            # Train
            train_loss, train_acc = self._train_one_epoch()

            # Validate
            val_loss, val_acc, val_auc, val_bal = self._eval_loader(self.val_loader)

            # Choose metric to monitor
            metric_val = self._select_metric(val_loss, val_acc, val_auc, val_bal)

            # Scheduler step
            if self.scheduler is not None:
                # ReduceLROnPlateau expects metric, others step per epoch
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metric_val)
                else:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

            epoch_time = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            # Log CSV and history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc if val_auc is not None else float("nan"))
            self.history["val_balanced_acc"].append(val_bal if val_bal is not None else float("nan"))
            self.history["lrs"].append(lr)
            self.history["epoch_times"].append(epoch_time)

            with open(self.log_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                                 f"{val_loss:.6f}", f"{val_acc:.4f}",
                                 f"{val_auc if val_auc is not None else 'nan'}",
                                 f"{val_bal if val_bal is not None else 'nan'}",
                                 f"{lr:.6e}", f"{epoch_time:.1f}"])

            print(f"[Epoch {epoch}/{num_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc} val_bal_acc={val_bal} "
                  f"lr={lr:.2e} time={epoch_time:.1f}s")

            # checkpoint & best model logic
            is_best = (metric_val is not None) and (metric_val > self.best_metric)
            if is_best:
                self.best_metric = metric_val
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                no_improve = 0
            else:
                # still save periodic checkpoints
                if epoch % self.save_interval == 0:
                    self.save_checkpoint(epoch, is_best=False)
                no_improve += 1

            # early stop
            if no_improve >= early_stop_patience:
                print(f"Early stopping triggered (no improvement for {early_stop_patience} epochs).")
                break

        # after training, evaluate test using best model if exists
        if self.best_model_path.exists():
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        test_loss, test_acc, test_auc, test_bal = self._eval_loader(self.test_loader)
        print(f"Test results -- loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc}, balanced_acc: {test_bal}")
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auc": test_auc,
            "test_balanced_acc": test_bal
        }
