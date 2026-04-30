"""Visual diagnostics for the FFT preprocessing pipeline.

Provides colorized spectrum rendering and statistical summaries for use in
the Streamlit app diagnostics tab.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from preprocessing.fft_transform import fft_preprocess


def render_fft_colormap(image: Image.Image) -> np.ndarray:
    """Render the FFT log-magnitude spectrum as a colorized uint8 RGB array.

    Purpose:
    Produce a visually interpretable frequency-domain image for the FFT
    Visualization tab. The viridis colormap highlights spectrum energy
    distribution in a way that is perceptually uniform and accessible.

    Inputs:
    - image: Uploaded PIL image (RGB or any mode accepted by fft_preprocess).

    Outputs:
    - uint8 NumPy array of shape (H, W, 3) suitable for st.image.

    Key assumptions:
    - Falls back to grayscale if matplotlib is unavailable.
    - The three FFT channels are identical, so only channel 0 is used.

    Failure modes:
    - Propagates FFT preprocessing errors for malformed images.
    """
    tensor = fft_preprocess(image)
    spectrum = tensor[0].detach().cpu().numpy()  # (H, W) in [0, 1]

    try:
        import matplotlib.cm as cm  # type: ignore[import]
        colorized = cm.viridis(spectrum)[:, :, :3]  # drop alpha channel
        return (colorized * 255).astype(np.uint8)
    except ImportError:
        gray = (spectrum * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def render_fft_grayscale(image: Image.Image) -> np.ndarray:
    """Render the FFT log-magnitude spectrum as a plain grayscale uint8 array.

    Purpose:
    Provide a lightweight fallback visualization that does not depend on
    matplotlib.

    Inputs:
    - image: Uploaded PIL image.

    Outputs:
    - uint8 NumPy array of shape (H, W) in [0, 255].

    Key assumptions:
    - The three FFT channels are identical; channel 0 is used.

    Failure modes:
    - Propagates FFT preprocessing errors for malformed images.
    """
    tensor = fft_preprocess(image)
    spectrum = tensor[0].detach().cpu().numpy()
    return (spectrum * 255).astype(np.uint8)


def compute_power_spectrum_stats(image: Image.Image) -> dict[str, float]:
    """Compute summary statistics over the FFT log-magnitude spectrum.

    Purpose:
    Provide a compact numerical summary of the frequency content that can
    be shown in the diagnostics tab alongside the verdict.

    Inputs:
    - image: Uploaded PIL image.

    Outputs:
    - Dictionary with mean, std, min, max, center_energy, and edge_energy.
      center_energy is the mean of the central 32x32 region (low-frequency
      content). edge_energy is the mean of the outermost 16-pixel border
      rows (high-frequency content).

    Key assumptions:
    - Image size is 256x256 after preprocessing. Center region is [112:144].

    Failure modes:
    - Propagates FFT preprocessing errors for malformed images.
    """
    tensor = fft_preprocess(image)
    spectrum = tensor[0].detach().cpu().numpy()

    center = spectrum[112:144, 112:144]
    top_border = spectrum[:16, :]
    bottom_border = spectrum[-16:, :]
    edges = np.concatenate([top_border.ravel(), bottom_border.ravel()])

    return {
        "mean": float(spectrum.mean()),
        "std": float(spectrum.std()),
        "min": float(spectrum.min()),
        "max": float(spectrum.max()),
        "center_energy": float(center.mean()),
        "edge_energy": float(edges.mean()),
    }
