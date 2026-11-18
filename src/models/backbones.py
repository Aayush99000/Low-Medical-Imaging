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
# Two more architecture-matched general / medical pairs, for the linear-probe curves. RAD-DINO is DINOv2-B
# fine-tuned on chest X-rays; BiomedCLIP is a CLIP ViT-B/16 trained on PubMed Central figure-caption pairs.
DINO_MODEL_IDS = {"dinov2": "facebook/dinov2-base", "raddino": "microsoft/rad-dino"}
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"   # open_clip
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)   # both CLIP models' normalization
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
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


class Dinov2Features(nn.Module):
    """DINOv2 / RAD-DINO ViT-B/14 -> CLS token after the final layer norm (position embeddings interpolate to any size)."""
    def __init__(self, model_id: str):
        super().__init__()
        from transformers import Dinov2Model
        self.vit = Dinov2Model.from_pretrained(model_id)
        self.out_dim = self.vit.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=x).pooler_output


class ClipFeatures(nn.Module):
    """OpenAI CLIP image tower -> projected image embedding (the one CLIP training aligns with text)."""
    def __init__(self, model_id: str, image_size: Tuple[int, int]):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection
        self.vision = CLIPVisionModelWithProjection.from_pretrained(model_id)
        cfg = self.vision.config
        self.out_dim = cfg.projection_dim
        self.interpolate_pos_encoding = tuple(image_size) != (cfg.image_size, cfg.image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision(pixel_values=x, interpolate_pos_encoding=self.interpolate_pos_encoding).image_embeds


class OpenClipFeatures(nn.Module):
    """BiomedCLIP image tower (open_clip around a timm ViT-B/16) -> projected image embedding."""
    def __init__(self, model_id: str, image_size: Tuple[int, int]):
        super().__init__()
        import open_clip
        model, _ = open_clip.create_model_from_pretrained(model_id)
        self.visual = model.visual
        trunk = self.visual.trunk
        if tuple(image_size) != tuple(trunk.patch_embed.img_size):
            trunk.set_input_size(img_size=tuple(image_size))  # resample the position embeddings, as CLIP / SigLIP do
        with torch.no_grad():
            self.out_dim = self.visual(torch.zeros(1, 3, *image_size)).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


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
    backbone: "densenet121" (weights None or "imagenet"), or a pretrained general / medical pair:
        "siglip" / "medsiglip", "dinov2" / "raddino", "clip" / "biomedclip".
    regime: "linear_probe" (frozen backbone, train the head) or "full_finetune" (train everything).
    """
    if backbone == "densenet121":
        trunk = DenseNetFeatures(weights)
    else:
        if weights is None:
            raise ValueError(f"{backbone} is only compared with pretrained weights.")
        if backbone in HF_MODEL_IDS:
            trunk = SiglipFeatures(HF_MODEL_IDS[backbone], image_size)
        elif backbone in DINO_MODEL_IDS:
            trunk = Dinov2Features(DINO_MODEL_IDS[backbone])
        elif backbone == "clip":
            trunk = ClipFeatures(CLIP_MODEL_ID, image_size)
        elif backbone == "biomedclip":
            trunk = OpenClipFeatures(BIOMEDCLIP_MODEL_ID, image_size)
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
    if backbone in DINO_MODEL_IDS:
        # DINOv2 uses ImageNet statistics, RAD-DINO its own chest X-ray ones; read from each image processor
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(DINO_MODEL_IDS[backbone])
        return {"image_size": tuple(image_size), "mean": tuple(proc.image_mean), "std": tuple(proc.image_std)}
    if backbone in ("clip", "biomedclip"):
        return {"image_size": tuple(image_size), "mean": OPENAI_CLIP_MEAN, "std": OPENAI_CLIP_STD}
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
