import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class AudioEfficientNet(nn.Module):
    """
    EfficientNet-B0 backbone adapted for audio classification.

    Replaces the final classifier layer to output `num_classes` 
    (default: 2 for Drone/No Drone).
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)  # weights=None - avoid downloading pretrained weights
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)