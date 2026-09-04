"""Image-domain and frequency-domain feature transforms used by every experiment.

The project intentionally keeps two preprocessing protocols:

``legacy_resize_v1``
    The original direct rectangular resize, retained only so previously declared baselines can be
    reproduced.

``h1n_square_crop_128_v1``
    The controlled H1-N protocol.  It decodes to RGB, crops a square from the source raster, and
    makes one common 128 x 128 raster before either RGB features or an FFT are calculated.  This
    removes source aspect-ratio and interpolation policy as an accidental label cue.
"""

from __future__ import annotations

import random
from io import BytesIO
from typing import Final

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LEGACY_PREPROCESSING_PROTOCOL: Final = "legacy_resize_v1"
CONTROLLED_PREPROCESSING_PROTOCOL: Final = "h1n_square_crop_128_v1"
CONTROLLED_PREPROCESSING_VERSION: Final = "1.0"
CONTROLLED_IMAGE_SIZE: Final = 128
LEGACY_IMAGE_SIZE: Final = 256


def _validate_protocol(protocol: str) -> str:
    supported = {LEGACY_PREPROCESSING_PROTOCOL, CONTROLLED_PREPROCESSING_PROTOCOL}
    if protocol not in supported:
        raise ValueError(f"Unsupported preprocessing protocol {protocol!r}; expected one of {sorted(supported)}")
    return protocol


def preprocessing_metadata(protocol: str, image_size: int | None = None) -> dict[str, object]:
    """Return a serialisable description of the declared preprocessing protocol."""
    protocol = _validate_protocol(protocol)
    if protocol == CONTROLLED_PREPROCESSING_PROTOCOL:
        if image_size is not None and image_size != CONTROLLED_IMAGE_SIZE:
            raise ValueError(
                "The controlled H1-N protocol fixes the common raster at "
                f"{CONTROLLED_IMAGE_SIZE} x {CONTROLLED_IMAGE_SIZE}."
            )
        return {
            "protocol": protocol,
            "version": CONTROLLED_PREPROCESSING_VERSION,
            "image_size": CONTROLLED_IMAGE_SIZE,
            "train_crop": "seeded_random_square_crop",
            "eval_crop": "center_square_crop",
            "resize": "single_square_lanczos",
            "fft_input": "common_raster_only",
        }
    size = LEGACY_IMAGE_SIZE if image_size is None else image_size
    return {
        "protocol": protocol,
        "version": "1.0",
        "image_size": size,
        "resize": "legacy_direct_square_resize",
    }


def source_normalized_rasterize(
    image: Image.Image,
    *,
    size: int = CONTROLLED_IMAGE_SIZE,
    train: bool = False,
    rng: random.Random | None = None,
) -> Image.Image:
    """Decode RGB, crop a square in source coordinates, then resize it once.

    The crop happens *before* resizing.  Thus a 16:9 source and a portrait source are never
    stretched independently into the model raster or letterboxed with a class-correlated border.
    In training, a supplied per-sample RNG makes the crop reproducible from the experiment seed;
    evaluation always uses the centre crop.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    decoded = ImageOps.exif_transpose(image).convert("RGB")
    width, height = decoded.size
    if width <= 0 or height <= 0:
        raise ValueError("Cannot rasterize an image with an empty extent")
    edge = min(width, height)
    if train:
        generator = rng if rng is not None else random
        left = generator.randint(0, width - edge) if width > edge else 0
        top = generator.randint(0, height - edge) if height > edge else 0
    else:
        left = (width - edge) // 2
        top = (height - edge) // 2
    square = decoded.crop((left, top, left + edge, top + edge))
    return square.resize((size, size), resample=Image.Resampling.LANCZOS)


def fft_magnitude(image: Image.Image, size: int = 256) -> np.ndarray:
    """Return log FFT magnitude in [0, 1] without using image metadata.

    The normalization is intentionally per image to match the original project brief. It should be
    treated as an experimental choice: it may remove absolute-energy information.
    """
    grayscale = image.convert("L")
    # In the controlled protocol the input already is the common square raster, so this branch
    # deliberately performs no second resampling before calculating the FFT.
    if grayscale.size != (size, size):
        # Leave the legacy FFT interpolation unchanged.  Controlled H1-N reaches this line with
        # the requested common raster size and therefore does not perform a second resize.
        grayscale = grayscale.resize((size, size))
    gray = np.asarray(grayscale, dtype=np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))
    low, high = np.percentile(magnitude, (1, 99))
    if high <= low:
        return np.zeros_like(magnitude, dtype=np.float32)
    return np.clip((magnitude - low) / (high - low), 0.0, 1.0).astype(np.float32)


def radial_power_spectrum(image: Image.Image, size: int = 256, bins: int = 64) -> np.ndarray:
    """Compute a compact, interpretable radial log-power feature vector."""
    magnitude = fft_magnitude(image, size=size)
    yy, xx = np.indices(magnitude.shape)
    radius = np.sqrt((xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2)
    edges = np.linspace(0, radius.max(), bins + 1)
    values = np.empty(bins, dtype=np.float32)
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        values[index] = magnitude[mask].mean() if mask.any() else 0.0
    return values


class RGBTransform:
    """ImageNet-normalised RGB transform with legacy and controlled H1-N modes."""

    def __init__(
        self,
        size: int | None = None,
        train: bool = False,
        robust_augmentation: bool = False,
        preprocessing_protocol: str = LEGACY_PREPROCESSING_PROTOCOL,
    ):
        self.preprocessing_protocol = _validate_protocol(preprocessing_protocol)
        metadata = preprocessing_metadata(self.preprocessing_protocol, size)
        self.size = int(metadata["image_size"])
        self.train = train
        self.robust_augmentation = robust_augmentation
        # The dataset supplies a stable per-sample RNG only for this mode.  Keeping legacy random
        # behaviour untouched permits exact reruns of the originally declared baseline.
        self.uses_contextual_rng = (
            self.preprocessing_protocol == CONTROLLED_PREPROCESSING_PROTOCOL and self.train
        )
        transforms: list[object] = []
        if self.preprocessing_protocol == LEGACY_PREPROCESSING_PROTOCOL:
            transforms.append(v2.Resize((self.size, self.size), antialias=True))
            if train:
                transforms.append(v2.RandomHorizontalFlip())
        transforms.extend(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.transform = v2.Compose(transforms)

    def rasterize(self, image: Image.Image, *, rng: random.Random | None = None) -> Image.Image:
        """Return the raster immediately before representation-specific encoding."""
        if self.preprocessing_protocol == CONTROLLED_PREPROCESSING_PROTOCOL:
            image = source_normalized_rasterize(image, size=self.size, train=self.train, rng=rng)
            image = _apply_train_augmentation(
                image,
                train=self.train,
                robust_augmentation=self.robust_augmentation,
                rng=rng,
            )
        else:
            image = _apply_legacy_train_augmentation(
                image, train=self.train, robust_augmentation=self.robust_augmentation
            )
        return image.convert("RGB")

    def encode_raster(self, raster: Image.Image) -> torch.Tensor:
        """Encode a prepared RGB raster without recropping or resizing it."""
        return self.transform(raster.convert("RGB"))

    def __call__(self, image: Image.Image, *, rng: random.Random | None = None) -> torch.Tensor:
        return self.encode_raster(self.rasterize(image, rng=rng))


class FFTTransform:
    """FFT representation with the same source-raster contract as ``RGBTransform``."""

    def __init__(
        self,
        size: int | None = None,
        train: bool = False,
        robust_augmentation: bool = False,
        preprocessing_protocol: str = LEGACY_PREPROCESSING_PROTOCOL,
    ):
        self.preprocessing_protocol = _validate_protocol(preprocessing_protocol)
        metadata = preprocessing_metadata(self.preprocessing_protocol, size)
        self.size = int(metadata["image_size"])
        self.train = train
        self.robust_augmentation = robust_augmentation
        self.uses_contextual_rng = (
            self.preprocessing_protocol == CONTROLLED_PREPROCESSING_PROTOCOL and self.train
        )
        self.normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def rasterize(self, image: Image.Image, *, rng: random.Random | None = None) -> Image.Image:
        """Return the common spatial raster before calculating the FFT magnitude."""
        if self.preprocessing_protocol == CONTROLLED_PREPROCESSING_PROTOCOL:
            image = source_normalized_rasterize(image, size=self.size, train=self.train, rng=rng)
            image = _apply_train_augmentation(
                image,
                train=self.train,
                robust_augmentation=self.robust_augmentation,
                rng=rng,
            )
        else:
            image = _apply_legacy_train_augmentation(
                image, train=self.train, robust_augmentation=self.robust_augmentation
            )
        return image.convert("RGB")

    def encode_raster(self, raster: Image.Image) -> torch.Tensor:
        """Calculate FFT features from a prepared raster without a second source resize."""
        # For controlled H1-N, ``image`` has exactly ``self.size`` pixels on each side here.
        magnitude = fft_magnitude(raster, size=self.size)
        tensor = torch.from_numpy(np.repeat(magnitude[None, :, :], 3, axis=0))
        return self.normalize(tensor)

    def __call__(self, image: Image.Image, *, rng: random.Random | None = None) -> torch.Tensor:
        return self.encode_raster(self.rasterize(image, rng=rng))


def _apply_legacy_train_augmentation(
    image: Image.Image, *, train: bool, robust_augmentation: bool
) -> Image.Image:
    """Keep the project's original optional augmentation behaviour byte-for-byte in spirit."""
    if train and robust_augmentation:
        roll = random.random()
        if roll < 0.25:
            return apply_degradation(image, "jpeg", jpeg_quality=random.choice((75, 85, 95)))
        if roll < 0.4:
            return apply_degradation(image, "resize", resize_scale=random.choice((0.5, 0.75)))
        if roll < 0.5:
            return apply_degradation(image, "blur")
    return image


def _apply_train_augmentation(
    image: Image.Image,
    *,
    train: bool,
    robust_augmentation: bool,
    rng: random.Random | None,
) -> Image.Image:
    """Apply post-raster train-time variation using a supplied deterministic RNG when available."""
    if not train:
        return image
    generator = rng if rng is not None else random
    if generator.random() < 0.5:
        image = ImageOps.mirror(image)
    if not robust_augmentation:
        return image
    roll = generator.random()
    if roll < 0.25:
        return apply_degradation(image, "jpeg", jpeg_quality=generator.choice((75, 85, 95)))
    if roll < 0.4:
        return apply_degradation(image, "resize", resize_scale=generator.choice((0.5, 0.75)))
    if roll < 0.5:
        return apply_degradation(image, "blur")
    return image


class DegradedTransform:
    """Wrap any representation transform with a fixed evaluation degradation."""

    def __init__(self, base_transform: object, kind: str, **parameters: object):
        self.base_transform = base_transform
        self.kind = kind
        self.parameters = parameters

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply H1-N test degradation after its common rasterisation step.

        In legacy mode the historical order is retained.  In the controlled protocol degrading a
        raw image before crop/resize would couple the operation to source geometry and recreate
        the confound that H1-N was introduced to remove.
        """
        base = self.base_transform
        if (
            getattr(base, "preprocessing_protocol", None) == CONTROLLED_PREPROCESSING_PROTOCOL
            and hasattr(base, "rasterize")
            and hasattr(base, "encode_raster")
        ):
            raster = base.rasterize(image)  # type: ignore[union-attr]
            degraded = apply_degradation(raster, self.kind, **self.parameters)
            return base.encode_raster(degraded)  # type: ignore[union-attr]
        return base(apply_degradation(image, self.kind, **self.parameters))  # type: ignore[operator]


def apply_degradation(
    image: Image.Image,
    kind: str = "clean",
    jpeg_quality: int = 95,
    resize_scale: float = 0.5,
) -> Image.Image:
    """Apply a deterministic, documented test-time degradation."""
    image = image.convert("RGB")
    if kind == "clean":
        return image
    if kind == "jpeg":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, subsampling=0)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if kind == "resize":
        width, height = image.size
        small = image.resize(
            (max(1, round(width * resize_scale)), max(1, round(height * resize_scale)))
        )
        return small.resize((width, height))
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    raise ValueError(f"Unsupported degradation: {kind}")
