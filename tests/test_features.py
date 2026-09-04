import pytest
from PIL import Image

from ai_image_detector.features import (
    CONTROLLED_PREPROCESSING_PROTOCOL,
    HIGHRES_CANONICAL_IMAGE_SIZE,
    HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL,
    DegradedTransform,
    FFTTransform,
    RGBTransform,
    apply_degradation,
    fft_magnitude,
    radial_power_spectrum,
    require_canonical_highres_raster,
)


def test_fft_features_have_expected_shape_and_range() -> None:
    image = Image.new("RGB", (40, 30), color=(120, 80, 20))
    spectrum = fft_magnitude(image, size=32)
    assert spectrum.shape == (32, 32)
    assert spectrum.min() >= 0.0
    assert spectrum.max() <= 1.0
    assert radial_power_spectrum(image, size=32, bins=8).shape == (8,)
    assert FFTTransform(size=32)(image).shape == (3, 32, 32)


def test_documented_degradations_preserve_image_size() -> None:
    image = Image.new("RGB", (100, 50), color="white")
    for kind in ("clean", "jpeg", "resize", "blur"):
        assert apply_degradation(image, kind=kind).size == image.size


def test_controlled_degradation_is_encoded_from_the_common_raster() -> None:
    image = Image.new("RGB", (200, 100), color="white")
    base = RGBTransform(train=False, preprocessing_protocol=CONTROLLED_PREPROCESSING_PROTOCOL)

    tensor = DegradedTransform(base, "jpeg", jpeg_quality=75)(image)

    assert tensor.shape == (3, 128, 128)


def test_highres_canonical_protocol_accepts_only_frozen_common_raster() -> None:
    image = Image.new(
        "RGB", (HIGHRES_CANONICAL_IMAGE_SIZE, HIGHRES_CANONICAL_IMAGE_SIZE), color="white"
    )
    transform = RGBTransform(
        train=False,
        preprocessing_protocol=HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL,
    )

    assert transform(image).shape == (3, HIGHRES_CANONICAL_IMAGE_SIZE, HIGHRES_CANONICAL_IMAGE_SIZE)
    assert transform.uses_contextual_rng is False

    with pytest.raises(ValueError, match="already materialised"):
        transform(Image.new("RGB", (383, 384), color="white"))
    with pytest.raises(ValueError, match="decoded RGB"):
        require_canonical_highres_raster(
            Image.new("P", (HIGHRES_CANONICAL_IMAGE_SIZE, HIGHRES_CANONICAL_IMAGE_SIZE)),
            size=HIGHRES_CANONICAL_IMAGE_SIZE,
        )

    train = RGBTransform(
        train=True,
        preprocessing_protocol=HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL,
    )
    assert train.uses_contextual_rng is True
