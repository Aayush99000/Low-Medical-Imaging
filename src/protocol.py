'''The shared experimental protocol: every model is trained and evaluated with these settings'''

# Which models are compared. Each pretraining comparison keeps the architecture fixed:
#   DenseNet-121:     random init (A)        vs ImageNet (B)
#   SigLIP-400M ViT:  general SigLIP (C0)    vs MedSigLIP (C)   <- domain-specific vs general pretraining
# Every pretrained backbone is run in both regimes, so the regime is never mixed up with the pretraining.
MODEL_ZOO = {
    "A_scratch":            {"backbone": "densenet121", "weights": None,       "regime": "full_finetune"},
    "B_imagenet_probe":     {"backbone": "densenet121", "weights": "imagenet", "regime": "linear_probe"},
    "B_imagenet_finetune":  {"backbone": "densenet121", "weights": "imagenet", "regime": "full_finetune"},
    "C0_siglip_probe":      {"backbone": "siglip",      "weights": "general",  "regime": "linear_probe"},
    "C0_siglip_finetune":   {"backbone": "siglip",      "weights": "general",  "regime": "full_finetune"},
    "C_medsiglip_probe":    {"backbone": "medsiglip",   "weights": "medical",  "regime": "linear_probe"},
    "C_medsiglip_finetune": {"backbone": "medsiglip",   "weights": "medical",  "regime": "full_finetune"},
}

# Identical for every model. Per-model differences are limited to the backbone and its pretrained weights.
PROTOCOL = {
    "image_size": (448, 448),        # every backbone sees the same resolution
    "batch_size": 16,                # effective batch = batch_size * grad_accum_steps; keep it fixed across models
    "grad_accum_steps": 1,           # raise this (and lower batch_size) when a model doesn't fit in memory
    "num_epochs": 30,
    "n_folds": 5,
    "seed": 42,                      # fold split, model init, data order (and the HAM10000 test split)
    "subset_seeds": (0, 1, 2, 3, 4), # independent draws of the training subset; the test set stays fixed
    "subset_size": 300,
    "augmentation_strength": 0.5,
    "head_dropout": 0.3,             # same classification head (src.models.ModelC.ClassificationHead) for all
    "weight_decay": 1e-4,
    "head_lr": 1e-3,                 # classification head, every model
    "backbone_lr": 1e-4,             # pretrained backbone in full fine-tuning; sweep the same grid for every model
    "scratch_lr": 1e-3,              # whole network when training from random init (Model A)
    "scheduler": "cosine",           # CosineAnnealingLR over num_epochs, stepped per epoch
    "early_stop_patience": None,     # None = train all epochs, select the best epoch on validation
    "monitor_metric": "val_auc",     # binary or macro one-vs-rest AUC; also "val_balanced_acc", "val_acc", "val_loss"
    "class_weights": None,           # None or "balanced" (per-fold weighted cross-entropy); same for every model
    "clip_grad_norm": 1.0,
}

# Same small learning-rate grid for every fine-tuned model, selected on the CV validation folds,
# so no model gets more tuning than another.
BACKBONE_LR_GRID = (1e-5, 3e-5, 1e-4)
