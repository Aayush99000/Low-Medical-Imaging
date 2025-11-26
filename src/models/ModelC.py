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
        self.medsiglip = AutoModel.from_pretrained(model_id)
        
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
        
        # Note: The exact feature to take depends on the specific MedSigLIP implementation.
        # In a typical ViT-based model, you might take the first token (CLS) or average.
        # For a clean approach, we'll try to rely on the model's built-in feature/pooled output.
        # If the model returns a structured output, we access the pooled features:
        # For MedSigLIP, let's use the default image embedding output:
        
        image_features = vision_output.image_embeds 
        
        # 3. Pass features to the Classification Head
        logits = self.classification_head(image_features)
        
        return logits