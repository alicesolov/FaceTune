"""EfficientNet-B4 classifier for editorial image authenticity screening.

Drop-in alternative to the ResNet-50 baseline. Use after the baseline run
establishes a performance floor, then compare val F1 to decide whether to
promote this architecture.

Usage (compare to baseline):
    python -m training.train --weights-path model/efficientnet.pth
    # then swap create_resnet50_binary_classifier for create_efficientnet_b4_classifier
    # in training/train.py and re-run
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4
except ImportError:  # pragma: no cover
    EfficientNet_B4_Weights = None  # type: ignore[assignment,misc]
    efficientnet_b4 = None  # type: ignore[assignment]


NUM_CLASSES = 2


def create_efficientnet_b4_classifier(pretrained: bool = True) -> nn.Module:
    """Build an EfficientNet-B4 binary classifier.

    Purpose:
    Provide a stronger backbone candidate to compare against the ResNet-50
    baseline. EfficientNet-B4 offers better accuracy-per-parameter than
    ResNet-50 on standard benchmarks, at the cost of higher memory usage.

    Inputs:
    - pretrained: Whether to initialise from torchvision ImageNet weights.

    Outputs:
    - Torch module with a two-class linear head, ready for training.

    Key assumptions:
    - Input tensors use the project's 3-channel FFT representation.
    - Binary classification is represented with two output logits.

    Failure modes:
    - Raises ImportError if torchvision is unavailable or too old.
    - May trigger a weight download if pretrained=True and weights are absent.
    """
    if efficientnet_b4 is None:
        raise ImportError(
            "torchvision >= 0.15 is required to build the EfficientNet-B4 model."
        )

    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def load_efficientnet_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """Load checkpoint weights into an EfficientNet-B4 classifier.

    Purpose:
    Support inference from a saved local checkpoint.

    Inputs:
    - model: Instantiated EfficientNet-B4 architecture.
    - weights_path: Filesystem path to a PyTorch checkpoint.

    Outputs:
    - The same model instance with weights loaded.

    Key assumptions:
    - The checkpoint contains either a plain state dict or a dict with a
      `state_dict` key (same convention as the ResNet-50 checkpoint saver).

    Failure modes:
    - Propagates deserialization and key mismatch errors.
    """
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model
