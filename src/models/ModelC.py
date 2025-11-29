import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

class ClassificationHead(nn.Module):
    """
    A small Multi-Layer Perceptron (MLP) to classify MedSigLIP features.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        # Define the MLP layers
        self.classifier = nn.Sequential(
            # Dense Layer 1: Reduces dimensionality (e.g., from 1152 to 512)
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Recommended for regularization
            
            # Output Dense Layer: Maps to the number of classes (N)
            nn.Linear(512, num_classes)
            # Softmax is applied in the loss function or separately for prediction
        )

    def forward(self, features):
        return self.classifier(features)
    

class ModelC_MedSigLIP(nn.Module):
    def __init__(self, model_id, num_classes):
        super().__init__()
        
        # --- 1. Load MedSigLIP Vision Encoder ---
        # MedSigLIP is a multi-modal model, but we'll only use its Vision Encoder.
        # The 'google/medsiglip-448' variant is common.
        self.medsiglip = AutoModel.from_pretrained("google/medsiglip-448")
        
        # --- 2. Freeze the Encoder (Crucial for Feature Extraction TL) ---
        for param in self.medsiglip.parameters():
            param.requires_grad = False
        # Determine the embedding dimension from the model's configuration
        # MedSigLIP (SigLIP-400M) typically outputs an embedding of 1152 dimensions.
        vision_hidden_size = self.medsiglip.config.vision_config.hidden_size
        
        # --- 3. Initialize the Trainable Classification Head ---
        self.classification_head = ClassificationHead(
            input_dim=vision_hidden_size, 
            num_classes=num_classes
        )

    def forward(self, pixel_values):
        # 1. Get the Vision Encoder Output
        # We only pass the image data (pixel_values) to the vision component.
        # The 'vision_model' is the Vision Transformer (ViT) backbone.
        vision_output = self.medsiglip.get_vision_features(pixel_values)
        
        # 2. Extract the Global Feature Vector
        # SigLIP models typically use the last feature vector (CLS token replacement)
        # for global representation.
        # For simplicity, we can use the final pooled output or the last layer output.
        # Assuming the model returns a pooled output or we take the mean/CLS token:
        
        image_features = vision_output.image_embeds 
        
        # 3. Pass features to the Classification Head
        logits = self.classification_head(image_features)
        
        return logits
    

# For HAM10000 (Skin Lesions) - 7 classes
model_ham = ModelC_MedSigLIP(
    model_id="google/medsiglip-448",
    num_classes=7
).to(device)

# For CheXpert (Chest X-rays) - 14 classes
model_chexpert = ModelC_MedSigLIP(
    model_id="google/medsiglip-448",
    num_classes=14
).to(device)

# For Chest X-ray (Pneumonia) - 2 classes
model_pneumonia = ModelC_MedSigLIP(
    model_id="google/medsiglip-448",
    num_classes=2
).to(device)

print(f"HAM10000 Model: {sum(p.numel() for p in model_ham.parameters() if p.requires_grad):,} trainable params")
print(f"CheXpert Model: {sum(p.numel() for p in model_chexpert.parameters() if p.requires_grad):,} trainable params")
print(f"Pneumonia Model: {sum(p.numel() for p in model_pneumonia.parameters() if p.requires_grad):,} trainable params")