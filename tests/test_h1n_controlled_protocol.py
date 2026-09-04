from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ai_image_detector.dataset import ManifestImageDataset
from ai_image_detector.features import (
    CONTROLLED_IMAGE_SIZE,
    CONTROLLED_PREPROCESSING_PROTOCOL,
    FFTTransform,
    RGBTransform,
    preprocessing_metadata,
    source_normalized_rasterize,
)
from ai_image_detector.training import PairedGroupSampler, predict


def _three_band_image(size: tuple[int, int], horizontal: bool) -> Image.Image:
    """Make a red/blue/green source where only the central band should survive a centre crop."""
    image = Image.new("RGB", size, color="blue")
    draw = ImageDraw.Draw(image)
    width, height = size
    if horizontal:
        band = (width - height) // 2
        draw.rectangle((0, 0, band - 1, height - 1), fill="red")
        draw.rectangle((width - band, 0, width - 1, height - 1), fill="green")
    else:
        band = (height - width) // 2
        draw.rectangle((0, 0, width - 1, band - 1), fill="red")
        draw.rectangle((0, height - band, width - 1, height - 1), fill="green")
    return image


def test_square_crop_standardizes_wide_tall_and_square_sources() -> None:
    sources = (
        _three_band_image((300, 100), horizontal=True),
        _three_band_image((100, 300), horizontal=False),
        Image.new("RGB", (100, 100), color="blue"),
    )
    for source in sources:
        raster = source_normalized_rasterize(source, train=False)
        assert raster.mode == "RGB"
        assert raster.size == (CONTROLLED_IMAGE_SIZE, CONTROLLED_IMAGE_SIZE)
        # A direct non-isotropic 300x100 -> 128x128 resize would retain red/green side bands.
        assert raster.getpixel((64, 64)) == (0, 0, 255)


def test_controlled_eval_is_deterministic_and_train_rng_is_reproducible() -> None:
    pixels = np.arange(300 * 100 * 3, dtype=np.uint8).reshape(100, 300, 3)
    image = Image.fromarray(pixels, mode="RGB")
    rgb_eval = RGBTransform(train=False, preprocessing_protocol=CONTROLLED_PREPROCESSING_PROTOCOL)
    fft_eval = FFTTransform(train=False, preprocessing_protocol=CONTROLLED_PREPROCESSING_PROTOCOL)
    assert rgb_eval(image).shape == (3, CONTROLLED_IMAGE_SIZE, CONTROLLED_IMAGE_SIZE)
    assert torch.equal(rgb_eval(image), rgb_eval(image))
    assert torch.equal(fft_eval(image), fft_eval(image))

    rgb_train = RGBTransform(train=True, preprocessing_protocol=CONTROLLED_PREPROCESSING_PROTOCOL)
    assert torch.equal(rgb_train(image, rng=random.Random(17)), rgb_train(image, rng=random.Random(17)))


def test_dataset_seeded_crop_and_group_metadata_are_reproducible(tmp_path) -> None:
    path = tmp_path / "wide.png"
    _three_band_image((300, 100), horizontal=True).save(path)
    frame = pd.DataFrame(
        {
            "path": [str(path)],
            "label": [0],
            "generator": ["real"],
            "split": ["train"],
            "source_id": ["source-1"],
            "group_id": ["legacy-group-1"],
            "leakage_group": ["leakage-group-1"],
        }
    )
    transform = RGBTransform(train=True, preprocessing_protocol=CONTROLLED_PREPROCESSING_PROTOCOL)
    dataset = ManifestImageDataset(frame, transform, seed=91)
    dataset.set_epoch(3)
    first_tensor, _, metadata = dataset[0]
    second_tensor, _, _ = dataset[0]
    assert torch.equal(first_tensor, second_tensor)
    assert metadata["group_id"] == "legacy-group-1"
    assert metadata["leakage_group"] == "leakage-group-1"


def test_dataset_seeded_crop_uses_source_id_not_absolute_path(tmp_path) -> None:
    width, height = 1_000, 100
    columns = np.arange(width, dtype=np.uint16)
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = columns % 251
    pixels[..., 1] = (columns // 251) % 251
    pixels[..., 2] = columns // (251 * 251)
    image = Image.fromarray(pixels, mode="RGB")
    first_path = tmp_path / "first" / "wide.png"
    second_path = tmp_path / "second" / "wide.png"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    image.save(first_path)
    image.save(second_path)
    transform = RGBTransform(train=True, preprocessing_protocol=CONTROLLED_PREPROCESSING_PROTOCOL)

    def tensor_for(path, source_id: str) -> torch.Tensor:
        frame = pd.DataFrame(
            {
                "path": [str(path)],
                "label": [0],
                "generator": ["real"],
                "split": ["train"],
                "source_id": [source_id],
            }
        )
        dataset = ManifestImageDataset(frame, transform, seed=91)
        dataset.set_epoch(3)
        return dataset[0][0]

    assert torch.equal(
        tensor_for(first_path, "stable-source"), tensor_for(second_path, "stable-source")
    )
    assert not torch.equal(
        tensor_for(first_path, "stable-source-a"), tensor_for(first_path, "stable-source-b")
    )


@pytest.mark.parametrize("source_id", [None, "   "])
def test_dataset_controlled_rng_rejects_missing_or_empty_source_id(tmp_path, source_id) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), color="blue").save(path)
    frame = pd.DataFrame(
        {
            "path": [str(path)],
            "label": [0],
            "generator": ["real"],
            "split": ["train"],
            "source_id": [source_id],
        }
    )
    transform = RGBTransform(train=True, preprocessing_protocol=CONTROLLED_PREPROCESSING_PROTOCOL)

    with pytest.raises(ValueError, match="non-empty 'source_id'"):
        ManifestImageDataset(frame, transform, seed=91)


def test_dataset_controlled_rng_rejects_absent_source_id_column(tmp_path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), color="blue").save(path)
    frame = pd.DataFrame(
        {
            "path": [str(path)],
            "label": [0],
            "generator": ["real"],
            "split": ["train"],
        }
    )
    transform = RGBTransform(train=True, preprocessing_protocol=CONTROLLED_PREPROCESSING_PROTOCOL)

    with pytest.raises(ValueError, match="non-empty 'source_id'"):
        ManifestImageDataset(frame, transform, seed=91)


def test_controlled_metadata_records_train_only_stochastic_flip() -> None:
    metadata = preprocessing_metadata(CONTROLLED_PREPROCESSING_PROTOCOL)

    assert metadata["crop_policy"] == "seeded_random_square_train_center_square_eval"
    assert metadata["neural_train_augmentation"] == {
        "horizontal_flip": {
            "probability": 0.5,
            "applies_to": "train_only",
            "order": "after_common_raster",
        }
    }


def test_prediction_output_preserves_group_identifiers(tmp_path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), color="blue").save(path)
    frame = pd.DataFrame(
        {
            "path": [str(path)],
            "label": [1],
            "generator": ["sdxl"],
            "split": ["test"],
            "source_id": ["source-2"],
            "group_id": ["legacy-group-2"],
            "leakage_group": ["leakage-group-2"],
        }
    )
    dataset = ManifestImageDataset(frame, lambda _: torch.zeros((3, 4, 4)))
    predictions = predict(
        torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2)),
        DataLoader(dataset, batch_size=1),
        torch.device("cpu"),
    )
    assert predictions.loc[0, "group_id"] == "legacy-group-2"
    assert predictions.loc[0, "leakage_group"] == "leakage-group-2"


def _paired_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fake_generators = ("dalle3", "midjourney_v6", "sd21", "sd3", "sdxl")
    for group_number in range(10):
        group = f"group-{group_number}"
        # The final group is deliberately much larger; it must still contribute one pair only.
        variants = 4 if group_number == 9 else 1
        for variant in range(variants):
            rows.append(
                {
                    "path": f"{group}/real-{variant}.png",
                    "label": 0,
                    "generator": "real",
                    "leakage_group": group,
                }
            )
            for generator in fake_generators:
                rows.append(
                    {
                        "path": f"{group}/{generator}-{variant}.png",
                        "label": 1,
                        "generator": generator,
                        "leakage_group": group,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_group_sampler_keeps_pairs_in_group_and_balances_generators() -> None:
    frame = _paired_frame()
    sampler = PairedGroupSampler(frame, seed=17)
    indices = list(sampler)
    assert indices == list(PairedGroupSampler(frame, seed=17))
    assert len(indices) == 2 * frame["leakage_group"].nunique()

    paired_groups: list[str] = []
    fake_generators: list[str] = []
    for real_index, fake_index in zip(indices[::2], indices[1::2], strict=True):
        real = frame.iloc[real_index]
        fake = frame.iloc[fake_index]
        assert real["label"] == 0
        assert fake["label"] == 1
        assert real["leakage_group"] == fake["leakage_group"]
        paired_groups.append(str(real["leakage_group"]))
        fake_generators.append(str(fake["generator"]))

    assert Counter(paired_groups) == {f"group-{index}": 1 for index in range(10)}
    assert Counter(fake_generators) == {
        "dalle3": 2,
        "midjourney_v6": 2,
        "sd21": 2,
        "sd3": 2,
        "sdxl": 2,
    }


def test_paired_group_sampler_falls_back_to_group_id_when_needed() -> None:
    frame = _paired_frame().rename(columns={"leakage_group": "group_id"})
    sampler = PairedGroupSampler(frame, seed=17)
    assert sampler.metadata()["group_column"] == "group_id"
