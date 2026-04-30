"""Dataset loading helpers and PyTorch Dataset wrapper for the Defactify dataset.

This module provides reproducible loading of the
`Rajarshi-Roy-research/Defactify_Image_Dataset` splits and a PyTorch-compatible
Dataset class that applies the project FFT preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset

try:
    from datasets import Dataset, load_dataset
except ImportError:  # pragma: no cover - depends on optional dependency.
    Dataset = Any  # type: ignore[misc,assignment]
    load_dataset = None

from preprocessing.fft_transform import fft_preprocess


DEFAULT_DATASET_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
DEFAULT_DATASET_CONFIG = "default"
SUPPORTED_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for loading a dataset split.

    Purpose:
    Make dataset source and split configuration explicit and reproducible.

    Inputs:
    - dataset_id: Hugging Face dataset identifier.
    - config_name: Hugging Face dataset config name.
    - split: Requested split name.
    - cache_dir: Optional local datasets cache directory.

    Outputs:
    - Immutable configuration object used by loading helpers.

    Key assumptions:
    - The dataset follows the documented `Caption`, `Image`, `Label_A`,
      and `Label_B` schema.

    Failure modes:
    - Invalid split names raise `ValueError`.
    """

    dataset_id: str = DEFAULT_DATASET_ID
    config_name: str = DEFAULT_DATASET_CONFIG
    split: str = "train"
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split '{self.split}'. Expected one of {SUPPORTED_SPLITS}."
            )


def load_defactify_split(config: DatasetConfig) -> Dataset:
    """Load one split of the planned Hugging Face dataset.

    Purpose:
    Provide a thin wrapper around `datasets.load_dataset` for the project's
    documented dataset choice.

    Inputs:
    - config: Dataset loading configuration.

    Outputs:
    - A Hugging Face `Dataset` object for the requested split.

    Key assumptions:
    - The `datasets` package is installed.
    - Network access or a populated cache is available.

    Failure modes:
    - Raises `ImportError` if the `datasets` package is unavailable.
    - Propagates dataset download and loading exceptions from Hugging Face.
    """

    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required to load the Hugging Face dataset."
        )

    return load_dataset(
        path=config.dataset_id,
        name=config.config_name,
        split=config.split,
        cache_dir=config.cache_dir,
    )


def normalize_label_a(value: int) -> str:
    """Map the documented binary label into project class names.

    Purpose:
    Normalize `Label_A` values into stable text labels for later training and
    inspection code.

    Inputs:
    - value: Integer value from the dataset's `Label_A` field.

    Outputs:
    - `real` for `0` and `ai_generated` for `1`.

    Key assumptions:
    - The dataset follows the published binary label mapping.

    Failure modes:
    - Raises `ValueError` for unsupported label values.
    """

    mapping = {0: "real", 1: "ai_generated"}
    if value not in mapping:
        raise ValueError(f"Unexpected Label_A value: {value}")
    return mapping[value]


def validate_dataset_columns(dataset: Dataset) -> None:
    """Check that the expected dataset columns are present.

    Purpose:
    Catch schema mismatches early before training or evaluation code is added.

    Inputs:
    - dataset: Loaded Hugging Face dataset split.

    Outputs:
    - None. The function validates in place and raises on mismatch.

    Key assumptions:
    - The dataset exposes a `column_names` attribute.

    Failure modes:
    - Raises `ValueError` if one or more expected columns are missing.
    """

    expected = {"Caption", "Image", "Label_A", "Label_B"}
    actual = set(dataset.column_names)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def summarize_split(dataset: Dataset) -> dict[str, Any]:
    """Return a small summary for inspection and sanity checks.

    Purpose:
    Expose a lightweight split summary that later scripts can print or log.

    Inputs:
    - dataset: Loaded Hugging Face dataset split.

    Outputs:
    - Dictionary with row count and available columns.

    Key assumptions:
    - The dataset is already loaded successfully.

    Failure modes:
    - Propagates dataset attribute errors if an unexpected dataset object is used.
    """

    return {
        "num_rows": len(dataset),
        "columns": list(dataset.column_names),
    }


class DefactifyTorchDataset(TorchDataset):
    """PyTorch Dataset wrapper around a loaded Defactify HuggingFace split.

    Purpose:
    Bridge the Hugging Face dataset API with PyTorch DataLoader by applying
    the project FFT preprocessing pipeline to each image on access.

    Inputs:
    - hf_dataset: A loaded Hugging Face `Dataset` split with the expected
      `Image` and `Label_A` columns.

    Outputs:
    - (tensor, label) tuples where tensor is shape `(3, 256, 256)` float32
      and label is an int (0 = real, 1 = ai_generated).

    Key assumptions:
    - The `Image` column contains PIL images (decoded by Hugging Face).
    - `Label_A` values are 0 or 1 as documented.

    Failure modes:
    - Propagates FFT preprocessing errors for malformed images.
    - Raises `KeyError` if required columns are missing.
    """

    def __init__(self, hf_dataset: Dataset) -> None:
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        image = item["Image"]
        label = int(item["Label_A"])
        tensor = fft_preprocess(image)
        return tensor, label

    def get_all_labels(self) -> list[int]:
        """Return all Label_A values as a flat list for sampler construction.

        Purpose:
        Allow the training script to build a class-balanced sampler without
        iterating through the full dataset via __getitem__.

        Inputs:
        - None.

        Outputs:
        - List of integer labels in dataset order.

        Key assumptions:
        - `Label_A` is present and integer-valued.

        Failure modes:
        - Propagates errors from the underlying dataset if schema is unexpected.
        """
        return [int(item["Label_A"]) for item in self.dataset]
