"""PyTorch transfer learning models for chest X-ray binary classification."""
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional


class TransferLearningModel(nn.Module):
    """Base class for transfer learning models with flexible architecture."""
    
    def __init__(self, 
                 base_model: nn.Module,
                 num_classes: int = 2,
                 hidden_size: int = 512,
                 dropout_rate: float = 0.5):
        """Initialize TransferLearningModel.
        
        Args:
            base_model: Pretrained base model.
            num_classes: Number of output classes.
            hidden_size: Size of hidden layers in classification head.
            dropout_rate: Dropout rate for regularization.
        """
        super(TransferLearningModel, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # Get the input size for the classification head
        # by doing a dummy forward pass
        self._in_features = None
        self._get_input_features()
        
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
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def _get_input_features(self):
        """Infer input feature size from base model."""
        if hasattr(self.base_model, 'fc'):
            # ResNet, VGG
            self._in_features = self.base_model.fc.in_features
        elif hasattr(self.base_model, 'classifier'):
            # Inception, DenseNet
            if hasattr(self.base_model.classifier, 'in_features'):
                self._in_features = self.base_model.classifier.in_features
            else:
                # For models like Inception
                self._in_features = self.base_model.classifier[-1].in_features
        else:
            raise RuntimeError("Could not determine input features for base model")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor with shape (B, 3, H, W).
        
        Returns:
            Output logits with shape (B, num_classes).
        """
        x = self.base_model(x)
        x = self.classifier(x)
        return x
    
    def freeze_base_model(self):
        """Freeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("Base model parameters frozen.")
    
    def unfreeze_base_model(self):
        """Unfreeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True
        print("Base model parameters unfrozen.")
    
    def unfreeze_last_n_layers(self, n_layers: int):
        """Unfreeze the last N layers of the base model.
        
        Args:
            n_layers: Number of layers to unfreeze from the end.
        """
        # First freeze all
        self.freeze_base_model()
        
        # Get all named parameters and unfreeze the last n
        params_list = list(self.base_model.named_parameters())
        for name, param in params_list[-n_layers:]:
            param.requires_grad = True
        
        print(f"Unfroze last {n_layers} layers of base model.")


def build_vgg19(num_classes: int = 2, 
                pretrained: bool = True,
                freeze_base: bool = True,
                hidden_size: int = 512,
                dropout_rate: float = 0.5) -> TransferLearningModel:
    """Build VGG19 transfer learning model.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        freeze_base: Whether to freeze base model initially.
        hidden_size: Size of hidden layers.
        dropout_rate: Dropout rate.
    
    Returns:
        TransferLearningModel instance.
    """
    base_model = models.vgg19(pretrained=pretrained)
    
    # Remove the original classifier
    base_model.classifier = nn.Identity()
    
    model = TransferLearningModel(
        base_model=base_model,
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    )
    
    if freeze_base:
        model.freeze_base_model()
    
    return model


def build_resnet50(num_classes: int = 2, 
                   pretrained: bool = True,
                   freeze_base: bool = True,
                   hidden_size: int = 512,
                   dropout_rate: float = 0.5) -> TransferLearningModel:
    """Build ResNet50 transfer learning model.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        freeze_base: Whether to freeze base model initially.
        hidden_size: Size of hidden layers.
        dropout_rate: Dropout rate.
    
    Returns:
        TransferLearningModel instance.
    """
    base_model = models.resnet50(pretrained=pretrained)
    
    # Remove the original classifier
    base_model.fc = nn.Identity()
    
    model = TransferLearningModel(
        base_model=base_model,
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    )
    
    if freeze_base:
        model.freeze_base_model()
    
    return model


def build_inceptionv3(num_classes: int = 2, 
                      pretrained: bool = True,
                      freeze_base: bool = True,
                      hidden_size: int = 512,
                      dropout_rate: float = 0.5) -> TransferLearningModel:
    """Build InceptionV3 transfer learning model.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        freeze_base: Whether to freeze base model initially.
        hidden_size: Size of hidden layers.
        dropout_rate: Dropout rate.
    
    Returns:
        TransferLearningModel instance.
    """
    base_model = models.inception_v3(pretrained=pretrained, aux_logits=False)
    
    # Remove the original classifier
    base_model.fc = nn.Identity()
    
    model = TransferLearningModel(
        base_model=base_model,
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    )
    
    if freeze_base:
        model.freeze_base_model()
    
    return model


def build_densenet121(num_classes: int = 2, 
                      pretrained: bool = True,
                      freeze_base: bool = True,
                      hidden_size: int = 512,
                      dropout_rate: float = 0.5) -> TransferLearningModel:
    """Build DenseNet121 transfer learning model.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        freeze_base: Whether to freeze base model initially.
        hidden_size: Size of hidden layers.
        dropout_rate: Dropout rate.
    
    Returns:
        TransferLearningModel instance.
    """
    base_model = models.densenet121(pretrained=pretrained)
    
    # Remove the original classifier
    base_model.classifier = nn.Identity()
    
    model = TransferLearningModel(
        base_model=base_model,
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    )
    
    if freeze_base:
        model.freeze_base_model()
    
    return model


def build_model(model_name: str = "vgg19",
                num_classes: int = 2,
                pretrained: bool = True,
                freeze_base: bool = True,
                hidden_size: int = 512,
                dropout_rate: float = 0.5) -> TransferLearningModel:
    """Factory function to build any available transfer learning model.
    
    Args:
        model_name: Name of model ("vgg19", "resnet50", "inceptionv3", "densenet121").
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        freeze_base: Whether to freeze base model initially.
        hidden_size: Size of hidden layers.
        dropout_rate: Dropout rate.
    
    Returns:
        TransferLearningModel instance.
    
    Raises:
        ValueError: If model_name is not recognized.
    """
    model_name = model_name.lower()
    
    models_dict = {
        "vgg19": build_vgg19,
        "resnet50": build_resnet50,
        "inceptionv3": build_inceptionv3,
        "densenet121": build_densenet121
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}")
    
    return models_dict[model_name](
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_base=freeze_base,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    )


def get_model_summary(model: nn.Module):
    """Print a summary of model architecture and parameters.
    
    Args:
        model: PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*70)
    print(f"Model: {model.__class__.__name__}")
    print("="*70)
    print(model)
    print("="*70)
    print(f"Total Parameters:    {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters:    {total_params - trainable_params:,}")
    print("="*70)