"""Image augmentation pipeline for training robustness.

Simulates real-world degradation: social media recompression, mobile photos,
editor exports, and compressed uploads — the key FP categories from the audit.

Usage:
    pipeline = build_augmentation_pipeline("medium")
    augmented_pil = pipeline(original_pil)
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageFilter

try:
    import torchvision.transforms as T
    _HAS_TV = True
except ImportError:
    _HAS_TV = False


# ── PIL-level transforms ──────────────────────────────────────────────────────

class RandomJPEGCompression:
    """Simulate JPEG recompression at a random quality level."""

    def __init__(self, min_quality: int = 60, max_quality: int = 92) -> None:
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.min_quality, self.max_quality)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


class RandomResizeDegradation:
    """Downscale then upscale to simulate low-res upload / screenshot."""

    def __init__(
        self,
        min_scale: float = 0.5,
        max_scale: float = 0.85,
        target_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h = img.size
        small = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.BILINEAR,
        )
        return small.resize(self.target_size, Image.BILINEAR)


class RandomMildBlur:
    """Apply a slight Gaussian blur, simulating mobile camera softness."""

    def __init__(self, max_radius: float = 1.5) -> None:
        self.max_radius = max_radius

    def __call__(self, img: Image.Image) -> Image.Image:
        radius = random.uniform(0.0, self.max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomDoubleCompression:
    """Two JPEG compression passes — common for screenshots of compressed images."""

    def __init__(
        self,
        first_range: tuple[int, int] = (70, 90),
        second_range: tuple[int, int] = (60, 85),
    ) -> None:
        self.first = RandomJPEGCompression(*first_range)
        self.second = RandomJPEGCompression(*second_range)

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.second(self.first(img))


# ── torchvision tensor transforms ────────────────────────────────────────────

def _tv_augments(strength: str) -> list:
    """Return torchvision tensor-level transforms by strength."""
    if not _HAS_TV:
        return []
    if strength == "light":
        return [T.RandomHorizontalFlip(p=0.5)]
    if strength == "medium":
        return [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.RandomCrop(size=224, padding=16),
        ]
    if strength == "heavy":
        return [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
            T.RandomCrop(size=224, padding=32),
            T.RandomGrayscale(p=0.05),
        ]
    return []


# ── Composed pipelines ────────────────────────────────────────────────────────

class AugmentationPipeline:
    """PIL-level augmentation pipeline applied before FFT preprocessing."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img: Image.Image) -> Image.Image:
        for t in self.transforms:
            img = t(img)
        return img


def build_augmentation_pipeline(strength: str = "medium") -> AugmentationPipeline:
    """Build a PIL augmentation pipeline for the given strength.

    Args:
        strength: "light" | "medium" | "heavy"

    Returns:
        AugmentationPipeline callable that takes and returns a PIL image.

    Notes:
        - "light":  horizontal flip only — safe for all scene types.
        - "medium": flip + mild color jitter + JPEG(70-92) — default for training.
        - "heavy":  all medium + double JPEG + resize degradation — for hard neg mining.
    """
    if strength == "light":
        transforms = [
            RandomJPEGCompression(min_quality=80, max_quality=95),
            RandomMildBlur(max_radius=0.8),
        ]
    elif strength == "medium":
        transforms = [
            RandomJPEGCompression(min_quality=70, max_quality=92),
            RandomMildBlur(max_radius=1.2),
        ]
    elif strength == "heavy":
        transforms = [
            RandomDoubleCompression(
                first_range=(70, 90),
                second_range=(60, 85),
            ),
            RandomResizeDegradation(min_scale=0.5, max_scale=0.85),
            RandomMildBlur(max_radius=1.5),
        ]
    else:
        raise ValueError(f"Unknown augmentation strength: {strength!r}. "
                         "Choose 'light', 'medium', or 'heavy'.")

    return AugmentationPipeline(transforms)
