"""Tests for dataset configuration, label normalization, and schema validation."""

from __future__ import annotations

import pytest
import torch

from training.dataset import (
    DatasetConfig,
    DefactifyTorchDataset,
    normalize_label_a,
    summarize_split,
    validate_dataset_columns,
    SUPPORTED_SPLITS,
)


# ── DatasetConfig ─────────────────────────────────────────────────────────────

class TestDatasetConfig:
    def test_valid_splits_are_accepted(self):
        for split in SUPPORTED_SPLITS:
            config = DatasetConfig(split=split)
            assert config.split == split

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="Unsupported split"):
            DatasetConfig(split="garbage")

    def test_default_values(self):
        config = DatasetConfig()
        assert config.split == "train"
        assert config.cache_dir is None


# ── normalize_label_a ─────────────────────────────────────────────────────────

class TestNormalizeLabelA:
    def test_zero_maps_to_real(self):
        assert normalize_label_a(0) == "real"

    def test_one_maps_to_ai_generated(self):
        assert normalize_label_a(1) == "ai_generated"

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="Unexpected Label_A value"):
            normalize_label_a(99)


# ── validate_dataset_columns ──────────────────────────────────────────────────

class _MockDataset:
    def __init__(self, columns):
        self.column_names = columns


class TestValidateDatasetColumns:
    def test_all_expected_columns_pass(self):
        ds = _MockDataset(["Caption", "Image", "Label_A", "Label_B"])
        validate_dataset_columns(ds)  # must not raise

    def test_missing_column_raises(self):
        ds = _MockDataset(["Caption", "Image", "Label_A"])  # Label_B missing
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataset_columns(ds)

    def test_extra_columns_are_ignored(self):
        ds = _MockDataset(["Caption", "Image", "Label_A", "Label_B", "extra"])
        validate_dataset_columns(ds)  # extra columns do not cause failure


# ── summarize_split ───────────────────────────────────────────────────────────

class TestSummarizeSplit:
    def test_returns_num_rows_and_columns(self):
        ds = _MockDataset(["Caption", "Image", "Label_A", "Label_B"])
        ds.__len__ = lambda self: 42  # type: ignore[assignment]

        class _DS(_MockDataset):
            def __len__(self):
                return 42

        summary = summarize_split(_DS(["Caption", "Image", "Label_A", "Label_B"]))
        assert summary["num_rows"] == 42
        assert "columns" in summary


# ── DefactifyTorchDataset ─────────────────────────────────────────────────────

class _FakeHFItem:
    def __init__(self, label: int):
        import numpy as np
        from PIL import Image
        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        self._image = Image.fromarray(arr)
        self._label = label

    def __getitem__(self, key: str):
        if key == "Image":
            return self._image
        if key == "Label_A":
            return self._label
        raise KeyError(key)


class _FakeHFDataset:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def __iter__(self):
        return iter(self._items)


class TestDefactifyTorchDataset:
    def _make_dataset(self, labels=(0, 1, 0, 1)):
        items = [_FakeHFItem(l) for l in labels]
        hf_ds = _FakeHFDataset(items)
        return DefactifyTorchDataset(hf_ds)

    def test_len(self):
        ds = self._make_dataset()
        assert len(ds) == 4

    def test_getitem_returns_tensor_and_label(self):
        ds = self._make_dataset()
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 256, 256)
        assert isinstance(label, int)
        assert label in (0, 1)

    def test_label_values_preserved(self):
        ds = self._make_dataset(labels=[0, 1])
        _, label0 = ds[0]
        _, label1 = ds[1]
        assert label0 == 0
        assert label1 == 1

    def test_get_all_labels(self):
        ds = self._make_dataset(labels=[0, 1, 0])
        all_labels = ds.get_all_labels()
        assert all_labels == [0, 1, 0]
