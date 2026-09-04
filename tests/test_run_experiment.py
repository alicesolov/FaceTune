from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
run_experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_experiment)


def test_experiment_output_directory_refuses_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "rgb_seed7"
    run_experiment.require_fresh_output_dir(output_dir)
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_experiment.require_fresh_output_dir(output_dir)


def test_model_launch_metadata_records_manifest_and_requested_options(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    frame = pd.DataFrame(
        {
            "path": ["a.png", "b.png", "c.png", "d.png"],
            "split": ["train", "train", "val", "test"],
        }
    )
    frame.to_csv(manifest_path, index=False)
    args = argparse.Namespace(
        representation="fft",
        seed=17,
        epochs=9,
        batch_size=16,
        learning_rate=0.0003,
        patience=3,
        preprocessing_protocol="h1n_square_crop_128_v1",
        train_sampler=None,
        paired_group_column="leakage_group",
        from_scratch=True,
        robust_augmentation=True,
        device="auto",
    )
    preprocessing = {"protocol": "h1n_square_crop_128_v1", "version": "1.0", "image_size": 128}
    sampler = {"choice": "paired_group_balanced_v1", "group_column": "leakage_group"}
    metadata = run_experiment.build_model_launch_metadata(
        manifest=run_experiment.manifest_launch_metadata(manifest_path, frame),
        requested_options=run_experiment.requested_launch_options(args),
        resolved_train_sampler="paired_group_balanced_v1",
        resolved_group_column="leakage_group",
        resolved_device="mps",
        environment_at_launch={"git_revision": "launch-revision"},
        preprocessing=preprocessing,
        train_sampler=sampler,
        trainable_parameters=23,
    )

    assert metadata["manifest"] == {
        "resolved_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "row_counts": {"total": 4, "by_split": {"test": 1, "train": 2, "val": 1}},
    }
    assert metadata["launch_options"] == {
        "requested": {
            "representation": "fft",
            "seed": 17,
            "epochs": 9,
            "batch_size": 16,
            "learning_rate": 0.0003,
            "patience": 3,
            "preprocessing_protocol": "h1n_square_crop_128_v1",
            "train_sampler": None,
            "paired_group_column": "leakage_group",
            "from_scratch": True,
            "robust_augmentation": True,
            "device": "auto",
        },
        "resolved": {
            "train_sampler": "paired_group_balanced_v1",
            "paired_group_column": "leakage_group",
            "device": "mps",
        },
    }
    assert metadata["model"] == {
        "architecture": "resnet50",
        "pretrained": False,
        "trainable_parameters": 23,
    }
    assert metadata["environment_at_launch"] == {"git_revision": "launch-revision"}
    assert metadata["device"] == "mps"
    assert metadata["trainable_parameters"] == 23
    assert metadata["preprocessing"] == preprocessing
    assert metadata["train_sampler"] == sampler
