"""Training and prediction loops with explicit artifacts and no hidden test selection."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm.auto import tqdm

from .dataset import ManifestImageDataset
from .metrics import binary_metrics, choose_threshold
from .reproducibility import environment_snapshot, save_json, seed_everything

LEGACY_LABEL_WEIGHTED_SAMPLER = "legacy_label_weighted_v1"
PAIRED_GROUP_BALANCED_SAMPLER = "paired_group_balanced_v1"


def resolve_group_column(frame: pd.DataFrame, requested: str = "leakage_group") -> str:
    """Resolve the controlled grouping column without silently changing the manifest.

    Grouped Defactify manifests normally expose ``leakage_group``.  Earlier prepared manifests
    expose the equivalent leakage-safe component as ``group_id`` only, so that is an explicit
    compatibility fallback.  The resolved value is written into run metadata by the CLI.
    """
    if requested in frame.columns:
        return requested
    if requested == "leakage_group" and "group_id" in frame.columns:
        return "group_id"
    raise ValueError(
        f"Paired group sampler requires {requested!r}; available group columns are "
        f"{[column for column in ('leakage_group', 'group_id') if column in frame.columns]}"
    )


class PairedGroupSampler(Sampler[int]):
    """Sample one matched real/fake pair from every leakage group per epoch.

    A group's number of near-duplicate images cannot increase its contribution: each group emits
    exactly two indices.  The fake generator is assigned from a shuffled, globally balanced cycle
    (five generators in Defactify), then one available real and fake raster is drawn within that
    group.  Consecutive sampler indices are always ``[real, fake]`` pairs.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        seed: int,
        group_column: str = "leakage_group",
        expected_fake_generators: int = 5,
    ) -> None:
        group_column = resolve_group_column(frame, group_column)
        required = {"label", "generator", group_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Paired group sampler requires columns: {sorted(missing)}")
        if frame[group_column].isna().any():
            raise ValueError(f"Paired group sampler requires non-null {group_column!r} values")
        self.frame = frame.reset_index(drop=True)
        self.seed = seed
        self.group_column = group_column
        self.epoch = 0
        self.fake_generators = tuple(
            sorted(self.frame.loc[self.frame["label"] == 1, "generator"].astype(str).unique())
        )
        if len(self.fake_generators) != expected_fake_generators:
            raise ValueError(
                f"Expected {expected_fake_generators} fake generators for paired sampling, found "
                f"{len(self.fake_generators)}: {list(self.fake_generators)}"
            )
        if self.frame.loc[self.frame["label"] == 0].empty:
            raise ValueError("Paired group sampler requires real (label=0) images")

        grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        real_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in self.frame.iterrows():
            group = str(row[group_column])
            if int(row["label"]) == 0:
                real_indices[group].append(index)
            elif int(row["label"]) == 1:
                grouped[group][str(row["generator"])].append(index)
            else:
                raise ValueError("Paired group sampler accepts only binary labels 0 and 1")

        self._groups: list[tuple[str, tuple[int, ...], dict[str, tuple[int, ...]]]] = []
        for group in sorted(set(real_indices) | set(grouped)):
            missing_fakes = [
                generator for generator in self.fake_generators if not grouped[group].get(generator)
            ]
            if not real_indices[group] or missing_fakes:
                raise ValueError(
                    f"Group {group!r} cannot form a real/fake pair; "
                    f"missing real={not bool(real_indices[group])}, missing fakes={missing_fakes}"
                )
            fake_indices = {
                generator: tuple(grouped[group][generator]) for generator in self.fake_generators
            }
            self._groups.append((group, tuple(real_indices[group]), fake_indices))
        if not self._groups:
            raise ValueError("Paired group sampler received no groups")

    def __len__(self) -> int:
        return 2 * len(self._groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def metadata(self) -> dict[str, object]:
        return {
            "choice": PAIRED_GROUP_BALANCED_SAMPLER,
            "group_column": self.group_column,
            "groups_per_epoch": len(self._groups),
            "paired_samples_per_epoch": len(self),
            "fake_generators": list(self.fake_generators),
            "fake_generator_assignment": "balanced_uniform_cycle",
        }

    def __iter__(self) -> Iterator[int]:
        # The multiplier makes epochs independent while keeping the exact sequence reproducible.
        generator = random.Random(self.seed + self.epoch * 1_000_003)
        groups = self._groups.copy()
        generator.shuffle(groups)
        # Every fake generator appears floor(n/5) or ceil(n/5) times, rather than allowing class
        # frequency or the largest groups to decide the pairing distribution.
        choices = [
            fake_generator
            for _ in range((len(groups) + len(self.fake_generators) - 1) // len(self.fake_generators))
            for fake_generator in self.fake_generators
        ][: len(groups)]
        generator.shuffle(choices)
        for (_, real_indices, fake_indices), fake_generator in zip(groups, choices, strict=True):
            yield generator.choice(real_indices)
            yield generator.choice(fake_indices[fake_generator])


@dataclass(frozen=True)
class TrainConfig:
    experiment_name: str
    representation: str
    seed: int = 7
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 4
    workers: int = 0
    preprocessing_protocol: str = "legacy_resize_v1"
    preprocessing_version: str = "1.0"
    image_size: int = 256
    train_sampler: str = LEGACY_LABEL_WEIGHTED_SAMPLER
    paired_group_column: str | None = None


def make_loader(
    frame: pd.DataFrame,
    transform: object,
    batch_size: int,
    train: bool,
    workers: int = 0,
    *,
    sampler_protocol: str = LEGACY_LABEL_WEIGHTED_SAMPLER,
    seed: int | None = None,
    group_column: str = "leakage_group",
) -> DataLoader:
    dataset = ManifestImageDataset(frame, transform, seed=seed)  # type: ignore[arg-type]
    if train:
        if sampler_protocol == PAIRED_GROUP_BALANCED_SAMPLER:
            if batch_size % 2:
                raise ValueError("Paired group sampling requires an even batch_size to keep pairs intact")
            if seed is None:
                raise ValueError("Paired group sampling requires an explicit experiment seed")
            sampler = PairedGroupSampler(frame, seed=seed, group_column=group_column)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
        if sampler_protocol != LEGACY_LABEL_WEIGHTED_SAMPLER:
            raise ValueError(f"Unsupported train sampler {sampler_protocol!r}")
        label_count = frame["label"].value_counts()
        weights = frame["label"].map(lambda label: 1.0 / label_count[label]).to_numpy(copy=True)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def train_sampler_metadata(loader: DataLoader) -> dict[str, object]:
    """Expose the selected train sampler in the run artifact without serialising implementation."""
    sampler = loader.sampler
    if isinstance(sampler, PairedGroupSampler):
        return sampler.metadata()
    if isinstance(sampler, WeightedRandomSampler):
        return {"choice": LEGACY_LABEL_WEIGHTED_SAMPLER, "replacement": True}
    return {"choice": type(sampler).__name__}


def _set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Synchronise deterministic augmentation and sampling before each training epoch."""
    dataset = loader.dataset
    if isinstance(dataset, ManifestImageDataset):
        dataset.set_epoch(epoch)
    sampler = loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for images, labels, metadata in tqdm(loader, desc="predict", leave=False):
            logits = model(images.to(device))
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            for index, score in enumerate(scores):
                rows.append(
                    {
                        "path": metadata["path"][index],
                        "generator": metadata["generator"][index],
                        "split": metadata["split"][index],
                        "source_id": metadata["source_id"][index],
                        "group_id": metadata.get("group_id", [""] * len(labels))[index],
                        "leakage_group": metadata.get("leakage_group", [""] * len(labels))[index],
                        "label": int(labels[index]),
                        "ai_score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    output_dir: str | Path,
) -> tuple[nn.Module, pd.DataFrame, float]:
    """Fit using validation loss only; returns validation predictions and its frozen threshold."""
    seed_everything(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.3)
    # The selected train loader controls its class and group balance.  Do not also weight the loss:
    # combining the two mechanisms would make the predeclared comparison harder to interpret.
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    no_improvement = 0
    checkpoint = output / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        _set_loader_epoch(train_loader, epoch)
        model.train()
        loss_sum, examples = 0.0, 0
        for images, labels, _ in tqdm(
            train_loader, desc=f"epoch {epoch}/{config.epochs}", leave=False
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            examples += len(labels)
        val_predictions = predict(model, val_loader, device)
        val_loss_proxy = float(
            np.mean(
                -(
                    val_predictions["label"]
                    * np.log(np.clip(val_predictions["ai_score"], 1e-7, 1 - 1e-7))
                    + (1 - val_predictions["label"])
                    * np.log(np.clip(1 - val_predictions["ai_score"], 1e-7, 1 - 1e-7))
                )
            )
        )
        scheduler.step(val_loss_proxy)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(examples, 1),
                "val_log_loss": val_loss_proxy,
            }
        )
        if val_loss_proxy < best_loss:
            best_loss = val_loss_proxy
            no_improvement = 0
            torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, checkpoint)
        else:
            no_improvement += 1
            if no_improvement >= config.patience:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    val_predictions = predict(model, val_loader, device)
    threshold = choose_threshold(
        val_predictions["label"].to_numpy(), val_predictions["ai_score"].to_numpy()
    )
    validation_metrics = binary_metrics(
        val_predictions["label"].to_numpy(), val_predictions["ai_score"].to_numpy(), threshold
    )
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    val_predictions.assign(threshold=threshold).to_csv(
        output / "validation_predictions.csv", index=False
    )
    save_json(validation_metrics, output / "validation_metrics.json")
    save_json(
        {
            "config": asdict(config),
            "preprocessing": {
                "protocol": config.preprocessing_protocol,
                "version": config.preprocessing_version,
                "image_size": config.image_size,
            },
            "train_sampler": {
                "choice": config.train_sampler,
                "paired_group_column": config.paired_group_column,
            },
            "environment": environment_snapshot(),
            "threshold": threshold,
        },
        output / "run.json",
    )
    return model, val_predictions, threshold


def evaluate_and_save(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    output_dir: str | Path,
    name: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    predictions = predict(model, loader, device)
    metrics = binary_metrics(
        predictions["label"].to_numpy(), predictions["ai_score"].to_numpy(), threshold
    )
    metrics["seconds"] = perf_counter() - started
    predictions.assign(threshold=threshold).to_csv(output / f"{name}_predictions.csv", index=False)
    save_json(metrics, output / f"{name}_metrics.json")
    return predictions, metrics
