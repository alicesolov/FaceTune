"""MobileNetV3-Large classifier for fast CPU baseline training.

Use this instead of ResNet-50 when GPU is unavailable. MobileNetV3-Large is
~5x faster on CPU than ResNet-50 with comparable accuracy on binary tasks.

Train and compare val F1 against ResNet-50 before choosing the production backbone.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
except ImportError:  # pragma: no cover
    MobileNet_V3_Large_Weights = None  # type: ignore[assignment,misc]
    mobilenet_v3_large = None  # type: ignore[assignment]


NUM_CLASSES = 2


def create_mobilenet_v3_classifier(pretrained: bool = True) -> nn.Module:
    """Build a MobileNetV3-Large binary classifier.

    Purpose:
    Provide a fast CPU-trainable backbone for quick baseline runs and iterative
    threshold calibration. MobileNetV3-Large has 5.4M parameters vs 25M for
    ResNet-50, making it ~5x faster per batch on CPU.

    Inputs:
    - pretrained: Whether to initialise from torchvision ImageNet weights.

    Outputs:
    - Torch module with a two-class linear head, ready for training.

    Key assumptions:
    - Input tensors use the project's 3-channel FFT representation (3, 256, 256).
    - Binary classification is represented with two output logits.

    Failure modes:
    - Raises ImportError if torchvision is unavailable or too old.
    """
    if mobilenet_v3_large is None:
        raise ImportError(
            "torchvision >= 0.15 is required to build the MobileNetV3-Large model."
        )

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model


def load_mobilenet_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """Load checkpoint weights into a MobileNetV3-Large classifier."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model
