"""Central model factory — build any detector by name or config dict.

Supported model types:
  resnet50   — FFT-only ResNet-50 baseline (default)
  mobilenet  — FFT-only MobileNetV3-Large (5× faster on CPU)
  hybrid     — FFT + CLIP dual-branch detector

Usage:
    model = build_model("hybrid")
    model = build_model_from_config(config_dict)
    model, meta = load_checkpoint_any(weights_path)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

SUPPORTED_MODELS = ("resnet50", "mobilenet", "hybrid")


# ── Builders ──────────────────────────────────────────────────────────────────

def build_model(
    model_type: str = "resnet50",
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Instantiate a detector by name.

    Args:
        model_type:  One of SUPPORTED_MODELS.
        pretrained:  Load ImageNet weights for backbone (ignored for hybrid,
                     which always loads pretrained ResNet-50 and CLIP).
        **kwargs:    Passed to the model constructor (e.g. fusion_hidden=512).

    Returns:
        Initialised nn.Module with a ._model_type attribute for serialization.
    """
    model_type = model_type.lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model type {model_type!r}. Choose from: {SUPPORTED_MODELS}"
        )

    if model_type == "hybrid":
        from model.hybrid import HybridDetector
        logging.info("Building HybridDetector (FFT+CLIP) …")
        model = HybridDetector(**kwargs)

    elif model_type == "mobilenet":
        from model.mobilenet import create_mobilenet_v3_classifier
        logging.info("Building MobileNetV3-Large (pretrained=%s) …", pretrained)
        model = create_mobilenet_v3_classifier(pretrained=pretrained)

    else:  # resnet50
        from model.resnet import create_resnet50_binary_classifier
        logging.info("Building ResNet-50 (pretrained=%s) …", pretrained)
        model = create_resnet50_binary_classifier(pretrained=pretrained)

    model._model_type = model_type
    return model


def build_model_from_config(config: dict) -> nn.Module:
    """Build a model from a config dict (as loaded from a YAML config file).

    Expected config structure:
        model:
          type: hybrid | resnet50 | mobilenet
          # hybrid-specific:
          clip_model: openai/clip-vit-base-patch32
          freeze_clip: true
          freeze_fft: true
          fusion_hidden: 512
          dropout: 0.3
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "resnet50")

    kwargs: dict = {}
    if model_type == "hybrid":
        for key in ("clip_model_name", "fusion_hidden", "dropout",
                    "freeze_clip", "freeze_fft"):
            # YAML key "clip_model" → kwarg "clip_model_name"
            yaml_key = "clip_model" if key == "clip_model_name" else key
            if yaml_key in model_cfg:
                kwargs[key] = model_cfg[yaml_key]

    return build_model(
        model_type=model_type,
        pretrained=model_cfg.get("pretrained", True),
        **kwargs,
    )


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint_any(
    weights_path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    """Load a checkpoint from any model type — auto-detects architecture.

    Returns:
        (model, metadata_dict)  where metadata includes epoch, val_f1, val_fpr.
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(weights_path, map_location="cpu")
    model_type = checkpoint.get("model_name", "resnet50")

    model = build_model(model_type, pretrained=False)

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    meta = {
        "model_type": model_type,
        "epoch": checkpoint.get("epoch"),
        "val_f1": checkpoint.get("val_f1"),
        "val_fpr": checkpoint.get("val_fpr"),
        "val_accuracy": checkpoint.get("val_accuracy"),
    }
    logging.info(
        "Loaded %s checkpoint (epoch=%s  val_f1=%.4f  val_fpr=%s)",
        model_type,
        meta["epoch"],
        meta["val_f1"] or 0.0,
        meta["val_fpr"],
    )
    return model, meta


def save_checkpoint_any(
    model: nn.Module,
    epoch: int,
    val_metrics: dict[str, float],
    path: Path,
) -> None:
    """Serialize any model type with its metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model_type = getattr(model, "_model_type", "resnet50")
    torch.save(
        {
            "model_name": model_type,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_accuracy": val_metrics.get("accuracy", 0.0),
            "val_precision": val_metrics.get("precision", 0.0),
            "val_recall": val_metrics.get("recall", 0.0),
            "val_f1": val_metrics.get("f1", 0.0),
            "val_fpr": val_metrics.get("fpr", 1.0),
        },
        path,
    )
    logging.info(
        "  ✓ Saved %s checkpoint → %s  (f1=%.4f  fpr=%.4f)",
        model_type, path,
        val_metrics.get("f1", 0.0),
        val_metrics.get("fpr", 1.0),
    )
