import pytest
from PIL import Image

from ai_image_detector.features import (
    CONTROLLED_PREPROCESSING_PROTOCOL,
    DANI_HIGHRES_IMAGE_SIZE,
    DANI_HIGHRES_PREPROCESSING_PROTOCOL,
    DANI_SOURCE_IMAGE_SIZE,
    HIGHRES_CANONICAL_IMAGE_SIZE,
    HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL,
    DegradedTransform,
    FFTTransform,
    RGBTransform,
    apply_degradation,
    fft_magnitude,
    preprocessing_metadata,
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


def test_dani_highres_protocol_downsamples_1024_to_384_without_accepting_substitutes() -> None:
    source = Image.new("RGB", (DANI_SOURCE_IMAGE_SIZE, DANI_SOURCE_IMAGE_SIZE), color="white")
    rgb = RGBTransform(train=False, preprocessing_protocol=DANI_HIGHRES_PREPROCESSING_PROTOCOL)
    fft = FFTTransform(train=False, preprocessing_protocol=DANI_HIGHRES_PREPROCESSING_PROTOCOL)

    assert rgb(source).shape == (3, DANI_HIGHRES_IMAGE_SIZE, DANI_HIGHRES_IMAGE_SIZE)
    assert fft(source).shape == (3, DANI_HIGHRES_IMAGE_SIZE, DANI_HIGHRES_IMAGE_SIZE)
    assert rgb.uses_contextual_rng is False
    assert preprocessing_metadata(DANI_HIGHRES_PREPROCESSING_PROTOCOL) == {
        "protocol": DANI_HIGHRES_PREPROCESSING_PROTOCOL,
        "version": "1.0",
        "image_size": DANI_HIGHRES_IMAGE_SIZE,
        "input_contract": "audited_exact_1024_square_source",
        "crop_policy": "no_crop_exact_square_source",
        "resize": "single_square_lanczos_downsample",
        "upsampling_permitted": False,
        "fft_input": "common_raster_only",
        "neural_train_augmentation": {
            "horizontal_flip": {
                "probability": 0.5,
                "applies_to": "train_only",
                "order": "after_common_raster",
            }
        },
    }

    with pytest.raises(ValueError, match="requires an audited 1024 x 1024"):
        rgb(Image.new("RGB", (512, 512), color="white"))

    train = RGBTransform(train=True, preprocessing_protocol=DANI_HIGHRES_PREPROCESSING_PROTOCOL)
    assert train.uses_contextual_rng is True
