from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
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


@pytest.mark.parametrize(
    ("protocol", "expected_sampler"),
    (
        (
            run_experiment.CONTROLLED_PREPROCESSING_PROTOCOL,
            run_experiment.PAIRED_GROUP_BALANCED_SAMPLER,
        ),
        (
            run_experiment.DANI_HIGHRES_PREPROCESSING_PROTOCOL,
            run_experiment.PAIRED_COMPONENT_BINARY_SAMPLER,
        ),
        (
            run_experiment.HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL,
            run_experiment.PAIRED_COMPONENT_BINARY_SAMPLER,
        ),
        (
            run_experiment.LEGACY_PREPROCESSING_PROTOCOL,
            run_experiment.LEGACY_LABEL_WEIGHTED_SAMPLER,
        ),
    ),
)
def test_protocol_default_sampler_is_explicit(protocol: str, expected_sampler: str) -> None:
    assert run_experiment.resolve_train_sampler(protocol, None) == expected_sampler


def test_requested_sampler_overrides_protocol_default() -> None:
    assert (
        run_experiment.resolve_train_sampler(
            run_experiment.HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL,
            run_experiment.LEGACY_LABEL_WEIGHTED_SAMPLER,
        )
        == run_experiment.LEGACY_LABEL_WEIGHTED_SAMPLER
    )


def test_defactify_exploratory_integrity_blocks_neural_training() -> None:
    with pytest.raises(ValueError, match="not eligible for primary HighRes training"):
        run_experiment.require_primary_highres_training_eligibility(
            {
                "eligibility": {
                    "eligible_for_exploratory_sensitivity_training": True,
                    "eligible_for_primary_highres_training": False,
                    "eligible_for_model_selection": False,
                    "eligible_for_external_evaluation": False,
                }
            }
        )


def test_paired_group_default_preserves_highres_caption_pairs() -> None:
    frame = _isolated_manifest_frame()

    assert (
        run_experiment.resolve_paired_group_column(
            frame, run_experiment.DANI_HIGHRES_PREPROCESSING_PROTOCOL, None
        )
        == "group_id"
    )
    assert (
        run_experiment.resolve_paired_group_column(
            frame, run_experiment.HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL, None
        )
        == "group_id"
    )
    assert (
        run_experiment.resolve_paired_group_column(
            frame, run_experiment.CONTROLLED_PREPROCESSING_PROTOCOL, None
        )
        == "leakage_group"
    )
    assert (
        run_experiment.resolve_paired_group_column(
            frame, run_experiment.HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL, "leakage_group"
        )
        == "leakage_group"
    )


def _isolated_manifest_frame(include_caption: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "path": ["train.png", "val.png", "test.png"],
            "label": [0, 1, 0],
            "split": ["train", "val", "test"],
            "generator": ["real", "sdxl", "real"],
            "leakage_group": ["component-train", "component-val", "component-test"],
            "group_id": ["group-train", "group-val", "group-test"],
            "source_id": ["source-train", "source-val", "source-test"],
            "sha256": ["sha-train", "sha-val", "sha-test"],
            "phash": ["phash-train", "phash-val", "phash-test"],
        }
    )
    if include_caption:
        frame["caption"] = ["caption-train", "caption-val", "caption-test"]
    return frame


def test_split_isolation_gate_allows_clean_manifest_without_caption() -> None:
    run_experiment.validate_split_isolation(_isolated_manifest_frame(include_caption=False))


def test_split_isolation_gate_ignores_blank_optional_captions() -> None:
    frame = _isolated_manifest_frame()
    frame.loc[[0, 1], "caption"] = " "

    run_experiment.validate_split_isolation(frame)


@pytest.mark.parametrize(
    "column",
    ("leakage_group", "group_id", "source_id", "sha256", "phash", "caption"),
)
def test_split_isolation_gate_rejects_each_cross_split_exact_key(column: str) -> None:
    frame = _isolated_manifest_frame()
    frame.loc[1, column] = frame.loc[0, column]

    with pytest.raises(ValueError, match="split-isolation gate") as error:
        run_experiment.validate_split_isolation(frame)

    assert column in str(error.value)
    assert "refusing to begin training or write model artifacts" in str(error.value).lower()


def test_split_isolation_gate_fails_closed_when_required_key_is_missing() -> None:
    frame = _isolated_manifest_frame().drop(columns="sha256")

    with pytest.raises(ValueError, match="Cannot verify manifest split isolation") as error:
        run_experiment.validate_split_isolation(frame)

    assert "sha256" in str(error.value)


def test_split_isolation_gate_fails_closed_when_required_key_is_blank() -> None:
    frame = _isolated_manifest_frame()
    frame.loc[0, "phash"] = ""

    with pytest.raises(ValueError, match="blank values") as error:
        run_experiment.validate_split_isolation(frame)

    assert "phash" in str(error.value)


def test_highres_split_isolation_requires_source_level_hashes() -> None:
    frame = _isolated_manifest_frame()

    with pytest.raises(ValueError, match="source_sha256"):
        run_experiment.validate_split_isolation(frame, require_highres_source_keys=True)

    frame["source_sha256"] = ["source-byte-train", "source-byte-val", "source-byte-test"]
    frame["source_pixel_sha256"] = ["source-pixel-train", "source-pixel-val", "source-pixel-test"]
    frame["source_phash"] = ["source-phash-train", "source-phash-val", "source-phash-test"]
    run_experiment.validate_split_isolation(frame, require_highres_source_keys=True)


def test_main_blocks_leakage_before_creating_model_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = _isolated_manifest_frame()
    frame.loc[1, "leakage_group"] = frame.loc[0, "leakage_group"]
    output_dir = tmp_path / "blocked-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--manifest",
            str(tmp_path / "invalid.csv"),
            "--representation",
            "rgb",
            "--output-dir",
            str(output_dir),
        ],
    )
    (tmp_path / "invalid.csv").write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(run_experiment, "environment_snapshot", dict)
    monkeypatch.setattr(run_experiment, "seed_everything", lambda seed: None)
    monkeypatch.setattr(run_experiment, "load_manifest", lambda *args, **kwargs: frame)

    with pytest.raises(ValueError, match="split-isolation gate"):
        run_experiment.main()

    assert not output_dir.exists()


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
        skip_internal_test=True,
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
            "skip_internal_test": True,
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
