"""Reproducible real-versus-synthetic image-detection experiments."""

from .reproducibility import get_device, seed_everything

__all__ = ["get_device", "seed_everything"]
