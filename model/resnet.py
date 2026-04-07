"""ResNet-50 model construction for editorial image authenticity screening.

This module defines the baseline classifier architecture used by the project.
It only constructs the network; training and checkpoint production are handled
elsewhere.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from torchvision.models import ResNet50_Weights, resnet50
except ImportError:  # pragma: no cover - depends on optional dependency.
    ResNet50_Weights = None
    resnet50 = None


NUM_CLASSES = 2


def create_resnet50_binary_classifier(pretrained: bool = False) -> nn.Module:
    """Build the baseline ResNet-50 classifier.

    Purpose:
    Construct the documented baseline backbone with a binary classification head
    for `real` and `ai_generated` outputs.

    Inputs:
    - pretrained: Whether to initialize from torchvision ImageNet weights.

    Outputs:
    - Torch module ready for training or inference state loading.

    Key assumptions:
    - Input tensors already use the project's 3-channel FFT representation.
    - Binary classification is represented with two output logits.

    Failure modes:
    - Raises `ImportError` if torchvision is unavailable.
    - May trigger a weight download if `pretrained=True` and weights are absent.
    """

    if resnet50 is None:
        raise ImportError(
            "The 'torchvision' package is required to build the ResNet-50 model."
        )

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_binary_classifier_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """Load checkpoint weights into the baseline classifier.

    Purpose:
    Support inference from a saved local checkpoint while keeping loading logic
    separate from predictor behavior.

    Inputs:
    - model: Instantiated baseline architecture.
    - weights_path: Filesystem path to a PyTorch checkpoint.

    Outputs:
    - The same model instance with weights loaded.

    Key assumptions:
    - The checkpoint contains either a plain state dict or a dictionary with a
      `state_dict` key.

    Failure modes:
    - Propagates deserialization and key mismatch errors from `torch.load` and
      `load_state_dict`.
    """

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model
