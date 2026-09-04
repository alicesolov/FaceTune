"""Torch datasets backed by an auditable CSV manifest."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .features import CONTROLLED_PREPROCESSING_PROTOCOL


class ManifestImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, str]]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Callable[..., torch.Tensor],
        *,
        seed: int | None = None,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic augmentation epoch before a training loader is iterated."""
        self.epoch = epoch

    def _sample_rng(self, row: pd.Series) -> random.Random:
        """Derive a stable RNG independent of worker ordering or Python's salted ``hash``."""
        if self.seed is None:
            raise RuntimeError("A deterministic sample RNG was requested without a seed")
        key = f"{self.seed}|{self.epoch}|{row['path']}|{row.get('source_id', '')}".encode()
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), byteorder="little")
        return random.Random(value)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        row = self.frame.iloc[index]
        with Image.open(Path(row.path)) as opened:
            if getattr(self.transform, "preprocessing_protocol", None) == CONTROLLED_PREPROCESSING_PROTOCOL:
                # ``copy`` fully decodes the source raster while retaining orientation information
                # for the controlled transform to normalise.
                image = opened.copy()
            else:
                # Preserve the original legacy input conversion exactly for baseline reruns.
                image = opened.convert("RGB")
        leakage_group = row.get("leakage_group", row.get("group_id", ""))
        metadata = {
            "path": str(row.path),
            "generator": str(row.generator),
            "split": str(row.split),
            "source_id": str(row.source_id),
            "group_id": str(row.get("group_id", "")),
            "leakage_group": str(leakage_group),
        }
        if self.seed is not None and bool(getattr(self.transform, "uses_contextual_rng", False)):
            tensor = self.transform(image, rng=self._sample_rng(row))
        else:
            tensor = self.transform(image)
        return tensor, torch.tensor(int(row.label), dtype=torch.long), metadata
