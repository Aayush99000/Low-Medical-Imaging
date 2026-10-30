'''One builder for every model, so models differ only in the backbone, its pretrained weights and the regime'''
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.models.ModelC import ClassificationHead, MEDSIGLIP_MODEL_ID, get_medsiglip_preprocessing

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# General-domain SigLIP with the same SigLIP-400M architecture as MedSigLIP (hidden 1152, 27 layers, patch 14)
SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
HF_MODEL_IDS = {"siglip": SIGLIP_MODEL_ID, "medsiglip": MEDSIGLIP_MODEL_ID}
REGIMES = ("linear_probe", "full_finetune")


class DenseNetFeatures(nn.Module):
    """DenseNet-121 trunk -> 1024-d pooled features."""
    def __init__(self, weights: Optional[str]):
        super().__init__()
        tv_weights = models.DenseNet121_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
        self.features = models.densenet121(weights=tv_weights).features
        self.out_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.features(x), inplace=True)
        return torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), 1)


class SiglipFeatures(nn.Module):
    """SigLIP / MedSigLIP vision tower -> pooled (MAP head) features."""
    def __init__(self, model_id: str, image_size: Tuple[int, int]):
        super().__init__()
        from transformers import SiglipVisionModel
        self.vision = SiglipVisionModel.from_pretrained(model_id)
        cfg = self.vision.config
        self.out_dim = cfg.hidden_size
        # run at the protocol resolution even when it differs from the checkpoint's (e.g. 384 -> 448)
        self.interpolate_pos_encoding = tuple(image_size) != (cfg.image_size, cfg.image_size)
        self.arch = {k: getattr(cfg, k) for k in ("hidden_size", "num_hidden_layers", "patch_size", "intermediate_size")}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision(pixel_values=x, interpolate_pos_encoding=self.interpolate_pos_encoding).pooler_output


class ClassifierModel(nn.Module):
    """Backbone + the shared classification head, with a linear-probe or full fine-tuning regime."""
    def __init__(self, backbone: nn.Module, num_classes: int, regime: str, dropout: float = 0.3):
        super().__init__()
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {regime}")
        self.backbone = backbone
        self.head = ClassificationHead(input_dim=backbone.out_dim, num_classes=num_classes, dropout_rate=dropout)
        self.regime = regime
        if regime == "linear_probe":
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def frozen(self) -> bool:
        return self.regime == "linear_probe"

    def train(self, mode: bool = True):
        # a frozen backbone stays in eval mode, so BatchNorm running stats / dropout don't change it either
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features)


def build_model(backbone: str, weights: Optional[str], regime: str, num_classes: int,
                image_size: Tuple[int, int] = (448, 448), dropout: float = 0.3) -> ClassifierModel:
    """
    backbone: "densenet121" (weights None or "imagenet"), "siglip" (general) or "medsiglip" (medical).
    regime: "linear_probe" (frozen backbone, train the head) or "full_finetune" (train everything).
    """
    if backbone == "densenet121":
        trunk = DenseNetFeatures(weights)
    elif backbone in HF_MODEL_IDS:
        if weights is None:
            raise ValueError("SigLIP backbones are only compared with pretrained weights.")
        trunk = SiglipFeatures(HF_MODEL_IDS[backbone], image_size)
    else:
        raise ValueError(f"Unknown backbone {backbone}")
    return ClassifierModel(trunk, num_classes=num_classes, regime=regime, dropout=dropout)


def get_preprocessing(backbone: str, image_size: Tuple[int, int] = (448, 448)) -> Dict:
    """Dataset kwargs: the protocol resolution for every model, normalization from each backbone's pretraining."""
    if backbone == "densenet121":
        return {"image_size": tuple(image_size), "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    if backbone in HF_MODEL_IDS:
        prep = get_medsiglip_preprocessing(HF_MODEL_IDS[backbone])
        return {"image_size": tuple(image_size), "mean": prep["mean"], "std": prep["std"]}
    raise ValueError(f"Unknown backbone {backbone}")


def make_optimizer(model: ClassifierModel, weights: Optional[str], head_lr: float, backbone_lr: float,
                   scratch_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with the same parameter groups for every model: head at head_lr, backbone at backbone_lr."""
    groups: List[Dict] = [{"params": list(model.head.parameters()), "lr": head_lr}]
    if not model.frozen:
        lr = scratch_lr if weights is None else backbone_lr
        groups.append({"params": list(model.backbone.parameters()), "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
