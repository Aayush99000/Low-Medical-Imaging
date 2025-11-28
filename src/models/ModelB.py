"""DenseNet121 Transfer Learning (ImageNet) wrapper for chest X-ray binary classification."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict


class TransferLearningModel(nn.Module):

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int = 2,
        in_features: Optional[int] = None,
        hidden_size: int = 512,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self._in_features = in_features

        # If in_features not provided, attempt to infer it (best-effort)
        if self._in_features is None:
            self._get_input_features()

        if self._in_features is None:
            raise RuntimeError("Could not determine classifier input features. Pass in_features explicitly.")

        # Build classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self._in_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def _get_input_features(self):
        # ResNet-like
        if hasattr(self.base_model, "fc"):
            try:
                self._in_features = self.base_model.fc.in_features
                return
            except Exception:
                pass

        # DenseNet-like or models with `classifier`
        if hasattr(self.base_model, "classifier"):
            cls = self.base_model.classifier
            # If classifier is a single Linear
            if isinstance(cls, nn.Linear):
                self._in_features = cls.in_features
                return
            # If classifier is Sequential and final layer is Linear
            if isinstance(cls, nn.Sequential) and len(cls) > 0:
                last = cls[-1]
                if isinstance(last, nn.Linear):
                    self._in_features = last.in_features
                    return

        # If nothing found, leave None (caller must supply)
        self._in_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        logits = self.classifier(features)
        return logits

    def freeze_base_model(self):
        """Freeze backbone parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("Base model parameters frozen.")

    def unfreeze_base_model(self):
        """Unfreeze backbone parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True
        print("Base model parameters unfrozen.")

    def unfreeze_last_n_layers(self, n_layers: int):
        """
        Freeze all base params then unfreeze the last `n_layers` parameter tensors.
        `n_layers` counts parameter tensors (not logical layers).
        """
        self.freeze_base_model()
        params_list = list(self.base_model.named_parameters())
        if n_layers <= 0 or n_layers > len(params_list):
            raise ValueError(f"n_layers must be between 1 and {len(params_list)}")
        for name, param in params_list[-n_layers:]:
            param.requires_grad = True
        print(f"Unfroze last {n_layers} parameter tensors of base model.")


def build_densenet121(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_base: bool = True,
    hidden_size: int = 512,
    dropout_rate: float = 0.5,
) -> TransferLearningModel:
    base_model = models.densenet121(pretrained=pretrained)
    # read in_features BEFORE removing classifier
    if not hasattr(base_model, "classifier") or not hasattr(base_model.classifier, "in_features"):
        # fallback: try to inspect final classifier sequentially
        try:
            if isinstance(base_model.classifier, nn.Sequential) and len(base_model.classifier) > 0:
                last = base_model.classifier[-1]
                in_feats = last.in_features if isinstance(last, nn.Linear) else None
            else:
                in_feats = None
        except Exception:
            in_feats = None
    else:
        in_feats = base_model.classifier.in_features

    if in_feats is None:
        raise RuntimeError("Could not determine DenseNet121 classifier in_features.")

    # remove original classifier so base_model(x) returns feature maps (B, C, H, W)
    base_model.classifier = nn.Identity()

    tl_model = TransferLearningModel(
        base_model=base_model,
        num_classes=num_classes,
        in_features=in_feats,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
    )

    if freeze_base:
        tl_model.freeze_base_model()

    return tl_model


def get_model_summary(model: nn.Module) -> None:
    """Print parameter counts and model summary (concise)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 70)
    print(f"Model class: {model.__class__.__name__}")
    print("=" * 70)
    # print architecture summary (avoid very long prints in notebooks; optional)
    print(model)
    print("=" * 70)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters:    {total_params - trainable_params:,}")
    print("=" * 70)


# Example usage:
# modelB = build_densenet121(pretrained=True, freeze_base=True)
# get_model_summary(modelB)
