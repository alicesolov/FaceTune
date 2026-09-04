"""Model constructors used in controlled baseline comparisons."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """Construct the same ResNet-50 for RGB and FFT inputs.

    The controlled H1-N RGB/FFT comparison uses ``pretrained=False`` for both representations.
    ImageNet initialisation remains available only for an explicitly labelled practical ablation:
    it is a natural RGB prior but not a semantically equivalent prior for FFT magnitude images.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
