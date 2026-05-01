"""Noise residual preprocessing for AI image forensics.

Extracts the high-frequency noise residual by subtracting a denoised version
of the image from the original. Real cameras produce sensor noise with specific
statistical properties; AI generators produce different residual patterns.

This can be used as a standalone input or combined with the FFT spectrum
(concatenated along the channel dimension) to give the model two
complementary views of the same image.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter


DEFAULT_IMAGE_SIZE = (256, 256)
# Gaussian blur radius that preserves scene structure but removes camera noise.
DENOISE_RADIUS = 1.5


def noise_residual_preprocess(
    image_input: Image.Image,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    denoise_radius: float = DENOISE_RADIUS,
) -> torch.Tensor:
    """Extract a normalised noise residual tensor from an image.

    Purpose:
    Isolate the high-frequency residual signal that differs between real camera
    images and AI-generated content. Real cameras introduce sensor noise with
    specific spatial correlations; AI models produce smoother or differently
    structured residuals.

    Pipeline:
    1. Resize to target size, convert to grayscale float.
    2. Apply Gaussian blur (denoised version = "scene without noise").
    3. Subtract: residual = original - denoised.
    4. Normalise to [0, 1] and replicate to 3 channels.

    Inputs:
    - image_input: PIL image.
    - image_size: Target spatial size.
    - denoise_radius: Gaussian blur radius. Larger values remove more signal
      before subtraction, leaving only fine-grained noise.

    Outputs:
    - Tensor of shape (3, H, W) float32 in [0, 1].

    Key assumptions:
    - The output is compatible with the same model head as fft_preprocess —
      both produce (3, H, W) tensors in [0, 1].
    - The three channels are identical (grayscale residual repeated).

    Failure modes:
    - Propagates PIL errors for malformed images.
    """
    img = image_input.convert("L").resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)

    blurred_pil = img.filter(ImageFilter.GaussianBlur(radius=denoise_radius))
    blurred = np.asarray(blurred_pil, dtype=np.float32)

    residual = arr - blurred  # signed residual, range roughly [-255, 255]

    # Shift to [0, 1]: map [-255, 255] → [0, 1]
    residual = (residual + 255.0) / 510.0
    residual = residual.clip(0.0, 1.0).astype(np.float32)

    repeated = np.repeat(residual[np.newaxis, :, :], 3, axis=0)
    return torch.from_numpy(repeated)


def combined_preprocess(
    image_input: Image.Image,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """Concatenate FFT spectrum and noise residual into a 6-channel tensor.

    Purpose:
    Give the model both the frequency-domain view (good for catching diffusion
    model upsampling artefacts) and the spatial noise-domain view (good for
    catching camera vs synthetic texture differences) in a single forward pass.

    Inputs:
    - image_input: PIL image.
    - image_size: Target spatial size.

    Outputs:
    - Tensor of shape (6, H, W) float32 in [0, 1].
      Channels 0–2: FFT log-magnitude spectrum (3 identical).
      Channels 3–5: Noise residual (3 identical).

    Key assumptions:
    - The classifier head must accept 6-channel input instead of 3. Use
      create_*_classifier(in_channels=6) when training with combined features.

    Failure modes:
    - Propagates errors from either preprocessing step.
    """
    from preprocessing.fft_transform import fft_preprocess
    fft = fft_preprocess(image_input, image_size)
    noise = noise_residual_preprocess(image_input, image_size)
    return torch.cat([fft, noise], dim=0)
