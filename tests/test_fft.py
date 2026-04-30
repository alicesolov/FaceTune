"""Smoke tests for the FFT preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from preprocessing.fft_transform import fft_preprocess


def _random_image(width: int = 64, height: int = 64) -> Image.Image:
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _solid_image(value: int = 128, width: int = 64, height: int = 64) -> Image.Image:
    arr = np.full((height, width, 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


class TestOutputShape:
    def test_default_size(self):
        tensor = fft_preprocess(_random_image())
        assert tensor.shape == (3, 256, 256)

    def test_custom_size(self):
        tensor = fft_preprocess(_random_image(), image_size=(128, 128))
        assert tensor.shape == (3, 128, 128)

    def test_non_square_input(self):
        tensor = fft_preprocess(_random_image(width=320, height=240))
        assert tensor.shape == (3, 256, 256)


class TestOutputValues:
    def test_dtype_is_float32(self):
        tensor = fft_preprocess(_random_image())
        assert tensor.dtype == torch.float32

    def test_values_in_unit_range(self):
        tensor = fft_preprocess(_random_image())
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0

    def test_solid_image_returns_zeros(self):
        # Solid image has no frequency variation; spectrum normalises to zero.
        tensor = fft_preprocess(_solid_image(128))
        assert torch.allclose(tensor, torch.zeros_like(tensor))


class TestChannelBehavior:
    def test_three_channels_identical(self):
        tensor = fft_preprocess(_random_image())
        assert torch.allclose(tensor[0], tensor[1])
        assert torch.allclose(tensor[1], tensor[2])


class TestDeterminism:
    def test_same_image_same_output(self):
        img = _random_image()
        t1 = fft_preprocess(img)
        t2 = fft_preprocess(img)
        assert torch.allclose(t1, t2)

    def test_different_images_different_outputs(self):
        np.random.seed(0)
        t1 = fft_preprocess(_random_image())
        np.random.seed(1)
        t2 = fft_preprocess(_random_image())
        assert not torch.allclose(t1, t2)


class TestInputTypes:
    def test_accepts_rgb_pil_image(self):
        img = _random_image().convert("RGB")
        tensor = fft_preprocess(img)
        assert tensor.shape == (3, 256, 256)

    def test_accepts_bytes(self):
        import io
        img = _random_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        tensor = fft_preprocess(buf.getvalue())
        assert tensor.shape == (3, 256, 256)

    def test_rejects_invalid_type(self):
        with pytest.raises(TypeError):
            fft_preprocess(42)  # type: ignore[arg-type]
