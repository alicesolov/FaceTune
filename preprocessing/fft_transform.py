"""Deterministic FFT preprocessing for editorial image screening.

The baseline pipeline converts an input image into a normalized log-magnitude
frequency representation suitable for model input.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from PIL import Image


DEFAULT_IMAGE_SIZE = (256, 256)


def _load_image(image_input: Image.Image | bytes | bytearray) -> Image.Image:
    """Convert supported image inputs into a PIL image.

    Purpose:
    Support PIL images and raw uploaded bytes with one internal loader.

    Inputs:
    - image_input: PIL image object or raw image bytes.

    Outputs:
    - PIL image in memory.

    Key assumptions:
    - Raw bytes represent a valid image format that Pillow can decode.

    Failure modes:
    - Raises `TypeError` for unsupported input types.
    - Propagates Pillow decoding errors for malformed image bytes.
    """

    if isinstance(image_input, Image.Image):
        return image_input
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(BytesIO(image_input))
    raise TypeError("image_input must be a PIL image or raw image bytes")


def fft_preprocess(
    image_input: Image.Image | bytes | bytearray,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """Convert an image into the baseline FFT tensor representation.

    Purpose:
    Implement the documented preprocessing pipeline for inference and future
    training work.

    Inputs:
    - image_input: PIL image or raw image bytes.
    - image_size: Target spatial size for preprocessing.

    Outputs:
    - Tensor of shape `(3, H, W)` with float32 values in `[0, 1]`.

    Key assumptions:
    - Images should be resized to `256x256` unless intentionally overridden.
    - The model expects a 3-channel tensor, so the grayscale spectrum is repeated.

    Failure modes:
    - Propagates image decoding errors for corrupted uploads.
    - May raise numeric errors if upstream image content is invalid.
    """

    image = _load_image(image_input).convert("L")
    resized = image.resize(image_size, Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32)

    fft = np.fft.fft2(image_array)
    shifted = np.fft.fftshift(fft)
    magnitude = np.abs(shifted)
    log_magnitude = np.log1p(magnitude)

    min_value = float(log_magnitude.min())
    max_value = float(log_magnitude.max())
    if max_value > min_value:
        normalized = (log_magnitude - min_value) / (max_value - min_value)
    else:
        normalized = np.zeros_like(log_magnitude, dtype=np.float32)

    spectrum = normalized.astype(np.float32)
    repeated = np.repeat(spectrum[np.newaxis, :, :], 3, axis=0)
    return torch.from_numpy(repeated)
