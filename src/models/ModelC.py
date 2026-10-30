import torch
import torch.nn as nn
# transformers is imported lazily so DenseNet-only runs (and ClassificationHead) don't need it

MEDSIGLIP_MODEL_ID = "google/medsiglip-448"
# SigLIP preprocessing: resize to 448x448, scale to [0, 1], normalize with mean = std = 0.5 (NOT ImageNet stats)
MEDSIGLIP_DEFAULTS = {"image_size": (448, 448), "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}


def get_medsiglip_preprocessing(model_id: str = MEDSIGLIP_MODEL_ID) -> dict:
    """
    Read image size and normalization stats from the model's own image processor, so the dataset
    feeds MedSigLIP exactly what it was pretrained on. Returns kwargs for the dataset classes:
        ChestXrayDataset(root, split="train", **get_medsiglip_preprocessing())
    Falls back to the SigLIP defaults if the processor can't be loaded (e.g. no Hugging Face access).
    Works for any SigLIP checkpoint (e.g. the general-domain google/siglip-so400m-patch14-384).
    """
    try:
        # image processor only: the full AutoProcessor also loads the text tokenizer (needs sentencepiece)
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        size = image_processor.size
        return {
            "image_size": (size["height"], size["width"]),
            "mean": tuple(image_processor.image_mean),
            "std": tuple(image_processor.image_std),
        }
    except Exception as e:
        print(f"[ModelC] Could not load the {model_id} processor ({e}); using SigLIP defaults {MEDSIGLIP_DEFAULTS}")
        return dict(MEDSIGLIP_DEFAULTS)


class ClassificationHead(nn.Module):
    """
    A small Multi-Layer Perceptron (MLP) to classify MedSigLIP features.
    """
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        
        # Define the MLP layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, features):
        return self.classifier(features)
    
class ModelC_MedSigLIP(nn.Module):
    def __init__(self, model_id, num_classes, dropout_rate=0.3):
        super().__init__()
        
        print(f"Loading {model_id}...")
        from transformers import AutoModel
        self.medsiglip = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        
        for param in self.medsiglip.parameters():
            param.requires_grad = False
            
        vision_hidden_size = self.medsiglip.config.vision_config.hidden_size
        print(f"Vision hidden size: {vision_hidden_size}")
        # Initialize the Trainable Classification Head 
        self.classification_head = ClassificationHead(
            input_dim=vision_hidden_size, 
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

    def forward(self, pixel_values):
    # Method 1: Use vision_model directly
        vision_outputs = self.medsiglip.vision_model(pixel_values)
        
        # Extract pooled features
        if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
            image_features = vision_outputs.pooler_output
        else:
            # Fallback: use CLS token
            image_features = vision_outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classification_head(image_features)
        return logits
    


if __name__ == "__main__":
    # Model summary. Run `huggingface-cli login` (or set HF_TOKEN) first: medsiglip-448 is a gated model.
    model_id = MEDSIGLIP_MODEL_ID
    num_classes = 2
    model = ModelC_MedSigLIP(
        model_id=model_id,
        num_classes=num_classes
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\n{'='*80}")
    print(f"Model C Pretrained on Medical Data")
    print(f"{'='*80}")
    print(f"Model ID: {model_id}")
    print(f"Number of Classes: {num_classes}")
    print(f"Device: {device}")
    print(f"Embedding Dimension: {model.medsiglip.config.vision_config.hidden_size}")
    print(f"Preprocessing: {get_medsiglip_preprocessing(model_id)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nParameter Count:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {frozen_params:,}")
    print(f"  Trainable Percentage: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*80}\n")