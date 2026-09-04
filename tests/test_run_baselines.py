from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_baselines.py"
SPEC = importlib.util.spec_from_file_location("run_baselines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
run_baselines = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_baselines)


def test_baseline_launch_metadata_preserves_existing_fields_and_input_trace(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    frame = pd.DataFrame(
        {
            "path": ["a.png", "b.png", "c.png", "d.png"],
            "split": ["train", "train", "val", "test"],
        }
    )
    frame.to_csv(manifest_path, index=False)
    args = argparse.Namespace(
        seed=17,
        only=None,
        preprocessing_protocol="h1n_square_crop_128_v1",
    )
    resolved_path, launch_sha256 = run_baselines.manifest_path_and_hash_at_launch(manifest_path)
    requested = run_baselines.requested_launch_options(args)
    baselines = run_baselines.selected_baselines(args.only)
    metadata = run_baselines.build_baseline_run_metadata(
        name="radial_fft_logistic",
        seed=args.seed,
        threshold=0.42,
        preprocessing={"protocol": args.preprocessing_protocol},
        manifest=run_baselines.manifest_launch_metadata(resolved_path, launch_sha256, frame),
        environment_at_launch={"git_revision": "launch-revision"},
        requested_options=requested,
        resolved_options=run_baselines.resolved_launch_options(requested, baselines),
    )

    assert metadata["name"] == "radial_fft_logistic"
    assert metadata["seed"] == 17
    assert metadata["threshold"] == 0.42
    assert metadata["preprocessing"] == {"protocol": "h1n_square_crop_128_v1"}
    assert metadata["role"] == "pixel_baseline"
    assert metadata["manifest"] == {
        "resolved_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "row_counts": {"total": 4, "by_split": {"test": 1, "train": 2, "val": 1}},
    }
    assert metadata["environment_at_launch"] == {"git_revision": "launch-revision"}
    assert metadata["launch_options"] == {
        "requested": {
            "seed": 17,
            "selected_baseline": None,
            "preprocessing_protocol": "h1n_square_crop_128_v1",
        },
        "resolved": {
            "seed": 17,
            "selected_baselines": ["radial_fft_logistic", "file_metadata_control"],
            "preprocessing_protocol": "h1n_square_crop_128_v1",
        },
    }


def test_selected_baselines_keeps_an_explicit_single_baseline() -> None:
    assert run_baselines.selected_baselines("file_metadata_control") == ("file_metadata_control",)


def test_baseline_output_paths_are_protocol_specific_and_refuse_overwrite(tmp_path: Path) -> None:
    outputs = run_baselines.require_fresh_output_dirs(
        tmp_path, ("radial_fft_logistic", "file_metadata_control"), 17, "h1n_square_crop_128_v1"
    )
    assert outputs == {
        "radial_fft_logistic": tmp_path / "radial_fft_logistic_h1n_square_crop_128_v1_seed17",
        "file_metadata_control": tmp_path / "file_metadata_control_seed17",
    }

    outputs["radial_fft_logistic"].mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_baselines.require_fresh_output_dirs(
            tmp_path, ("radial_fft_logistic", "file_metadata_control"), 17, "h1n_square_crop_128_v1"
        )


def test_run_one_writes_launch_provenance_to_its_existing_run_json(tmp_path: Path, monkeypatch) -> None:
    sentinel_model = object()
    monkeypatch.setattr(run_baselines, "fit_radial_logistic", lambda *args, **kwargs: sentinel_model)
    monkeypatch.setattr(
        run_baselines,
        "radial_predict",
        lambda model, frame, **kwargs: np.linspace(0.2, 0.8, len(frame)),
    )
    monkeypatch.setattr(run_baselines, "choose_threshold", lambda *args: 0.5)
    monkeypatch.setattr(run_baselines, "binary_metrics", lambda *args: {"accuracy": 0.5})
    frame = pd.DataFrame(
        {
            "path": ["a.png", "b.png"],
            "label": [0, 1],
            "generator": ["real", "synthetic"],
            "split": ["test", "test"],
        }
    )
    requested = {
        "seed": 17,
        "selected_baseline": "radial_fft_logistic",
        "preprocessing_protocol": "h1n_square_crop_128_v1",
    }
    resolved = {
        "seed": 17,
        "selected_baselines": ["radial_fft_logistic"],
        "preprocessing_protocol": "h1n_square_crop_128_v1",
    }
    output = tmp_path / "radial"

    run_baselines.run_one(
        "radial_fft_logistic",
        frame,
        frame,
        frame,
        output,
        17,
        "h1n_square_crop_128_v1",
        {"resolved_path": "/input/manifest.csv", "manifest_sha256": "abc"},
        {"git_revision": "launch-revision"},
        requested,
        resolved,
    )

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["manifest"] == {"resolved_path": "/input/manifest.csv", "manifest_sha256": "abc"}
    assert run["environment_at_launch"] == {"git_revision": "launch-revision"}
    assert run["launch_options"] == {"requested": requested, "resolved": resolved}
    assert run["name"] == "radial_fft_logistic"
    assert run["threshold"] == 0.5
