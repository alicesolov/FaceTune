from __future__ import annotations

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ai_image_detector import training
from ai_image_detector.training import TrainConfig, fit


class _TinyImageDataset(Dataset[tuple[torch.Tensor, int, dict[str, str]]]):
    """Minimal labelled batches with the metadata contract used by ``predict``."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        return (
            torch.full((3, 4, 4), float(index)),
            index,
            {
                "path": f"sample-{index}.png",
                "generator": "real" if index == 0 else "sdxl",
                "split": "val",
                "source_id": f"source-{index}",
                "group_id": f"group-{index}",
                "leakage_group": f"group-{index}",
            },
        )


def test_fit_persists_history_before_first_checkpoint(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    original_save = training.torch.save
    observed_history: list[pd.DataFrame] = []
    fallback_environment = {"git_revision": "at-fit-entry", "python": "test"}

    def save_after_asserting_history(*args, **kwargs) -> None:
        observed_history.append(pd.read_csv(output / "history.csv"))
        original_save(*args, **kwargs)

    monkeypatch.setattr(training.torch, "save", save_after_asserting_history)
    monkeypatch.setattr(training, "environment_snapshot", lambda: fallback_environment)
    loader = DataLoader(_TinyImageDataset(), batch_size=2)
    config = TrainConfig(experiment_name="tiny", representation="rgb", epochs=1)
    fit(
        torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2)),
        loader,
        loader,
        config,
        torch.device("cpu"),
        output,
    )

    assert len(observed_history) == 1
    assert observed_history[0]["epoch"].tolist() == [1]
    assert (output / "history.csv").is_file()
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["environment_at_launch"] == fallback_environment
    assert "environment" not in run


def test_fit_preserves_cli_injected_launch_environment(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    cli_environment = {"git_revision": "captured-before-data-load", "python": "test"}

    def unexpected_snapshot() -> dict[str, str]:
        raise AssertionError("fit must not replace the CLI launch snapshot")

    monkeypatch.setattr(training, "environment_snapshot", unexpected_snapshot)
    loader = DataLoader(_TinyImageDataset(), batch_size=2)
    config = TrainConfig(experiment_name="tiny", representation="fft", epochs=1)
    fit(
        torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2)),
        loader,
        loader,
        config,
        torch.device("cpu"),
        output,
        environment_at_launch=cli_environment,
    )

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["environment_at_launch"] == cli_environment
