"""Hybrid dataset and hard negative support.

Provides:
  DefactifyHybridDataset — wraps an HF dataset, returns (fft, clip_pv, label).
  HardNegativeDataset    — loads FP real images from diagnose.py fp.json.
  build_hybrid_loaders   — convenience function for train/val DataLoaders.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from preprocessing.fft_transform import fft_preprocess
from training.augment import AugmentationPipeline, build_augmentation_pipeline

try:
    from transformers import CLIPImageProcessor
    _HAS_CLIP_PROC = True
except ImportError:
    _HAS_CLIP_PROC = False


# ── Hybrid dataset ─────────────────────────────────────────────────────────────

class DefactifyHybridDataset(Dataset):
    """Returns (fft_tensor, clip_pixel_values, label) for each sample.

    The clip_pixel_values tensor is sized (3, 224, 224) as expected by
    CLIPVisionModel. The fft_tensor is (3, 256, 256) as in the baseline.

    Args:
        hf_dataset:        HuggingFace dataset split.
        clip_processor:    CLIPImageProcessor instance (from transformers).
        augment_pipeline:  Optional PIL augmentation applied BEFORE both
                           FFT and CLIP preprocessing.
    """

    def __init__(
        self,
        hf_dataset,
        clip_processor,
        augment_pipeline: AugmentationPipeline | None = None,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.clip_processor = clip_processor
        self.augment = augment_pipeline

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        item = self.hf_dataset[idx]
        image = item["Image"]          # PIL image
        label = int(item["Label_A"])

        if self.augment is not None:
            image = self.augment(image)

        fft_tensor = fft_preprocess(image)                        # (3, 256, 256)
        clip_inputs = self.clip_processor(
            images=image, return_tensors="pt"
        )
        clip_pv = clip_inputs["pixel_values"][0]                  # (3, 224, 224)

        return fft_tensor, clip_pv, label

    def get_all_labels(self) -> list[int]:
        return [int(self.hf_dataset[i]["Label_A"]) for i in range(len(self))]


# ── Hard negative dataset ──────────────────────────────────────────────────────

class HardNegativeDataset(Dataset):
    """Real images (label=0) identified as false positives by diagnose.py.

    Can be used standalone or concatenated into a hybrid/standard dataset.
    Set is_hybrid=True to return (fft, clip_pv, 0) instead of (fft, 0).
    """

    def __init__(
        self,
        hf_dataset,
        indices: list[int],
        clip_processor=None,
        augment_pipeline: AugmentationPipeline | None = None,
        is_hybrid: bool = True,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.clip_processor = clip_processor
        self.augment = augment_pipeline
        self.is_hybrid = is_hybrid

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        item = self.hf_dataset[self.indices[idx]]
        image = item["Image"]

        if self.augment is not None:
            image = self.augment(image)

        fft_tensor = fft_preprocess(image)

        if self.is_hybrid and self.clip_processor is not None:
            clip_pv = self.clip_processor(
                images=image, return_tensors="pt"
            )["pixel_values"][0]
            return fft_tensor, clip_pv, 0

        return fft_tensor, 0


def load_fp_indices(fp_json: Path) -> list[int]:
    """Parse diagnose.py output → list of dataset_index values."""
    with fp_json.open() as f:
        data = json.load(f)
    top_key = next(
        (k for k in data if k.startswith("top_") and "false_positive" in k),
        None,
    )
    if top_key is None:
        raise ValueError(
            f"No 'top_N_false_positives' key found in {fp_json}.\n"
            "Re-run: python -m training.diagnose --top-n 2000 --save-json fp.json"
        )
    return [entry["dataset_index"] for entry in data[top_key]]


# ── Weighted sampler ───────────────────────────────────────────────────────────

def make_weighted_sampler_hybrid(
    labels: list[int],
    hn_count: int = 0,
    hn_weight: float = 2.5,
) -> WeightedRandomSampler:
    """Build a sampler that up-weights minority class AND hard negatives.

    The first (len(labels) - hn_count) samples are the main dataset.
    The last hn_count samples are hard negatives and get an extra boost.
    """
    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[l] for l in labels]

    # Boost hard negatives (they are real, label=0, appended at the end)
    if hn_count > 0:
        main_n = len(labels) - hn_count
        for i in range(main_n, len(labels)):
            sample_weights[i] *= hn_weight

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Convenience builder ────────────────────────────────────────────────────────

def build_hybrid_loaders(
    train_hf,
    val_hf,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    augmentation: bool = False,
    augmentation_strength: str = "medium",
    hard_negatives: bool = False,
    fp_json: Path | None = None,
    hn_weight: float = 2.5,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for hybrid training.

    Returns:
        (train_loader, val_loader)
    """
    if not _HAS_CLIP_PROC:
        raise ImportError(
            "transformers is required for hybrid training.\n"
            "Run: pip install transformers"
        )

    clip_proc = CLIPImageProcessor.from_pretrained(clip_model_name)
    aug = build_augmentation_pipeline(augmentation_strength) if augmentation else None

    train_ds: Dataset = DefactifyHybridDataset(train_hf, clip_proc, augment_pipeline=aug)
    val_ds: Dataset = DefactifyHybridDataset(val_hf, clip_proc)

    train_labels = train_ds.get_all_labels()
    hn_count = 0

    if hard_negatives and fp_json is not None and Path(fp_json).exists():
        hn_indices = load_fp_indices(Path(fp_json))
        hn_aug = build_augmentation_pipeline("heavy")  # heavier aug on hard negs
        hn_ds = HardNegativeDataset(
            train_hf, hn_indices, clip_proc, augment_pipeline=hn_aug, is_hybrid=True
        )
        train_ds = ConcatDataset([train_ds, hn_ds])
        hn_labels = [0] * len(hn_ds)
        train_labels = train_labels + hn_labels
        hn_count = len(hn_ds)
        logging.info("Hard negatives: +%d real FP images (weight=%.1f)", hn_count, hn_weight)

    sampler = make_weighted_sampler_hybrid(train_labels, hn_count=hn_count, hn_weight=hn_weight)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logging.info(
        "Hybrid loaders: train=%d batches  val=%d batches",
        len(train_loader), len(val_loader),
    )
    return train_loader, val_loader
