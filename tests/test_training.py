from __future__ import annotations

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

    def save_after_asserting_history(*args, **kwargs) -> None:
        observed_history.append(pd.read_csv(output / "history.csv"))
        original_save(*args, **kwargs)

    monkeypatch.setattr(training.torch, "save", save_after_asserting_history)
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
