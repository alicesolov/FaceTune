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
        self._uses_contextual_rng = bool(getattr(self.transform, "uses_contextual_rng", False))
        if self.seed is not None and self._uses_contextual_rng:
            self._validate_stable_manifest_identities()

    def __len__(self) -> int:
        return len(self.frame)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic augmentation epoch before a training loader is iterated."""
        self.epoch = epoch

    def _validate_stable_manifest_identities(self) -> None:
        """Require the portable manifest identity used by controlled train-time RNGs."""
        if "source_id" not in self.frame.columns:
            raise ValueError(
                "Controlled H1-N stochastic training requires a non-empty 'source_id' "
                "column as its stable manifest identity."
            )
        source_ids = self.frame["source_id"]
        invalid = source_ids.isna() | source_ids.astype("string").str.strip().eq("")
        invalid_count = int(invalid.sum())
        if invalid_count:
            raise ValueError(
                "Controlled H1-N stochastic training requires non-empty 'source_id' values "
                f"for every row; found {invalid_count} missing or empty value(s)."
            )

    @staticmethod
    def _stable_manifest_identity(row: pd.Series) -> str:
        """Return the source ID used in a portable train-time augmentation key."""
        if "source_id" not in row.index:
            raise ValueError(
                "Controlled H1-N stochastic training requires a non-empty 'source_id' "
                "column as its stable manifest identity."
            )
        source_id = row["source_id"]
        if pd.isna(source_id) or not str(source_id).strip():
            raise ValueError(
                "Controlled H1-N stochastic training requires a non-empty 'source_id' "
                "value as its stable manifest identity."
            )
        return str(source_id).strip()

    def _sample_rng(self, row: pd.Series) -> random.Random:
        """Derive a portable RNG independent of paths, worker ordering, or salted ``hash``."""
        if self.seed is None:
            raise RuntimeError("A deterministic sample RNG was requested without a seed")
        source_id = self._stable_manifest_identity(row)
        key = f"{self.seed}|{self.epoch}|{source_id}".encode()
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
        if self.seed is not None and self._uses_contextual_rng:
            tensor = self.transform(image, rng=self._sample_rng(row))
        else:
            tensor = self.transform(image)
        return tensor, torch.tensor(int(row.label), dtype=torch.long), metadata
