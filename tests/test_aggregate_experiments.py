from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "aggregate_experiments.py"
SPEC = importlib.util.spec_from_file_location("aggregate_experiments", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
aggregate_experiments = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate_experiments)


def metric_payload(balanced_accuracy: float, roc_auc: float | None = None) -> dict[str, float]:
    roc = balanced_accuracy if roc_auc is None else roc_auc
    return {
        "roc_auc": roc,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": balanced_accuracy,
        "fpr_at_tpr_95": 1.0 - balanced_accuracy,
    }


def write_experiment(
    root: Path,
    representation: str,
    seed: int,
    *,
    internal_score: float,
    validation_score: float,
    validation_roc_auc: float | None = None,
    validation_threshold: float | None = 0.5,
    write_validation: bool = True,
    write_analysis: bool = True,
    git_revision: str = "training-pipeline-revision",
) -> Path:
    experiment = root / f"{representation}_seed{seed}"
    experiment.mkdir(parents=True)
    run = {
        "threshold": validation_threshold,
        "config": {
            "experiment_name": experiment.name,
            "representation": representation,
            "seed": seed,
            "epochs": 15,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "patience": 4,
            "preprocessing_protocol": "h1n_square_crop_128_v1",
            "preprocessing_version": "1.0",
            "image_size": 128,
            "train_sampler": "paired_group_balanced_v1",
            "paired_group_column": "leakage_group",
        },
        "preprocessing": {
            "protocol": "h1n_square_crop_128_v1",
            "version": "1.0",
            "image_size": 128,
        },
        "train_sampler": {
            "choice": "paired_group_balanced_v1",
            "group_column": "leakage_group",
            "fake_generator_assignment": "balanced_uniform_cycle",
        },
    }
    model = {
        "device": "mps",
        "environment_at_launch": {"git_revision": git_revision},
        "manifest": {
            "resolved_path": "/relocated/project/data/manifest.csv",
            "manifest_sha256": "manifest-sha256",
            "row_counts": {"total": 96000, "by_split": {"train": 67200, "val": 14400, "test": 14400}},
        },
        "preprocessing": {
            "protocol": "h1n_square_crop_128_v1",
            "version": "1.0",
            "image_size": 128,
            "train_crop": "seeded_random_square_crop",
            "eval_crop": "center_square_crop",
        },
        "train_sampler": {
            "choice": "paired_group_balanced_v1",
            "group_column": "leakage_group",
            "fake_generator_assignment": "balanced_uniform_cycle",
        },
        "launch_options": {
            "requested": {
                "representation": representation,
                "seed": seed,
                "epochs": 15,
                "batch_size": 32,
                "learning_rate": 0.0001,
                "patience": 4,
                "preprocessing_protocol": "h1n_square_crop_128_v1",
                "train_sampler": None,
                "paired_group_column": "leakage_group",
                "from_scratch": True,
                "robust_augmentation": False,
                "device": "mps",
            },
            "resolved": {
                "train_sampler": "paired_group_balanced_v1",
                "paired_group_column": "leakage_group",
                "device": "mps",
            },
        },
        "model": {
            "architecture": "resnet50",
            "pretrained": False,
            "trainable_parameters": 23512130,
        },
    }
    (experiment / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (experiment / "model.json").write_text(json.dumps(model), encoding="utf-8")
    (experiment / "best_model.pt").write_bytes(f"checkpoint-{representation}-{seed}".encode())
    if write_validation:
        (experiment / "validation_metrics.json").write_text(
            json.dumps(metric_payload(validation_score, validation_roc_auc)), encoding="utf-8"
        )
    if write_analysis:
        analysis = experiment / "analysis"
        analysis.mkdir()
        (analysis / "analysis_summary.json").write_text(
            json.dumps({"metrics": metric_payload(internal_score)}), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "generator": ["synthetic"],
                **{metric: [value] for metric, value in metric_payload(internal_score).items()},
            }
        ).to_csv(analysis / "per_generator_metrics.csv", index=False)
    return experiment


def run_aggregate(monkeypatch, output: Path, *experiments: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_experiments.py",
            *[
                value
                for experiment in experiments
                for value in ("--experiment-dir", str(experiment))
            ],
            "--output-dir",
            str(output),
        ],
    )
    aggregate_experiments.main()


def selection_record(output: Path) -> dict[str, object]:
    return json.loads((output / "prototype_selection_record.json").read_text(encoding="utf-8"))


def test_single_representation_emits_validation_aggregate_without_decision(
    tmp_path: Path, monkeypatch
) -> None:
    first = write_experiment(
        tmp_path, "rgb", 7, internal_score=0.7, validation_score=0.6, validation_roc_auc=0.7
    )
    second = write_experiment(
        tmp_path, "rgb", 17, internal_score=0.9, validation_score=0.8, validation_roc_auc=0.9
    )
    output = tmp_path / "aggregate"

    run_aggregate(monkeypatch, output, first, second)

    payload = json.loads((output / "aggregate_summary.json").read_text(encoding="utf-8"))
    record = selection_record(output)
    per_seed_validation = pd.read_csv(output / "per_seed_validation_metrics.csv")
    validation_summary = pd.read_csv(output / "validation_metric_mean_std.csv")

    assert payload["metrics"]["balanced_accuracy"]["mean"] == 0.8
    assert payload["validation_metrics_by_representation"]["rgb"]["balanced_accuracy"]["mean"] == 0.7
    assert set(per_seed_validation["seed"]) == {7, 17}
    assert validation_summary.loc[0, "balanced_accuracy_mean"] == pytest.approx(0.7)
    assert record["selection_status"] == "incomplete_protocol"
    assert record["decision"]["made"] is False
    assert "exactly representations" in record["decision"]["reason"]
    assert record["selection_rule"]["kind"] == "prototype_selection_rule_not_external_validation"
    assert "no best seed" in payload["selection_rule"].lower()
    assert (output / "generator_metric_mean_std.csv").is_file()


def test_revision_equivalence_is_verified_only_for_training_paths(
    tmp_path: Path, monkeypatch
) -> None:
    first = write_experiment(
        tmp_path,
        "rgb",
        7,
        internal_score=0.7,
        validation_score=0.6,
        git_revision="revision-a",
    )
    second = write_experiment(
        tmp_path,
        "rgb",
        17,
        internal_score=0.8,
        validation_score=0.7,
        git_revision="revision-b",
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> object:
        calls.append(command)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(aggregate_experiments.subprocess, "run", fake_run)
    output = tmp_path / "aggregate"

    run_aggregate(monkeypatch, output, first, second)

    payload = json.loads((output / "aggregate_summary.json").read_text(encoding="utf-8"))
    equivalence = payload["signature"]["revision_equivalence"]
    assert equivalence["launch_git_revisions"] == ["revision-a", "revision-b"]
    assert equivalence["checked_paths"] == list(aggregate_experiments.TRAINING_CODE_PATHS)
    assert equivalence["training_code_equal_across_revisions"] is True
    assert calls == [
        [
            "git",
            "diff",
            "--quiet",
            "revision-a",
            "revision-b",
            "--",
            *aggregate_experiments.TRAINING_CODE_PATHS,
        ]
    ]


def test_multiple_representations_select_by_validation_not_internal_test(
    tmp_path: Path, monkeypatch
) -> None:
    rgb7 = write_experiment(tmp_path, "rgb", 7, internal_score=0.1, validation_score=0.8)
    rgb17 = write_experiment(tmp_path, "rgb", 17, internal_score=0.1, validation_score=0.8)
    rgb42 = write_experiment(tmp_path, "rgb", 42, internal_score=0.1, validation_score=0.8)
    fft7 = write_experiment(tmp_path, "fft", 7, internal_score=0.99, validation_score=0.7)
    fft17 = write_experiment(tmp_path, "fft", 17, internal_score=0.99, validation_score=0.7)
    fft42 = write_experiment(tmp_path, "fft", 42, internal_score=0.99, validation_score=0.7)
    output = tmp_path / "aggregate"

    run_aggregate(monkeypatch, output, fft7, rgb17, rgb7, fft42, rgb42, fft17)

    record = selection_record(output)
    decision = record["decision"]
    payload = json.loads((output / "aggregate_summary.json").read_text(encoding="utf-8"))
    per_seed_validation = pd.read_csv(output / "per_seed_validation_metrics.csv")
    assert record["selection_status"] == "validation_selected_pending_external_validation"
    assert record["selection_rule"]["internal_test_metrics_used_for_selection"] is False
    assert decision["made"] is True
    assert decision["selected_representation"] == "rgb"
    assert decision["representative_seed"] == 17
    assert decision["experiment"] == "rgb_seed17"
    assert decision["checkpoint_sha256"] == hashlib.sha256(b"checkpoint-rgb-17").hexdigest()
    assert decision["validation_threshold"] == pytest.approx(0.5)
    assert decision["validation_metrics"]["balanced_accuracy"] == pytest.approx(0.8)
    assert set(per_seed_validation["validation_threshold"]) == {0.5}
    assert "metrics" not in payload


@pytest.mark.parametrize(
    ("fft_roc_auc", "rgb_roc_auc", "expected"),
    [(0.8, 0.9, "rgb"), (0.9, 0.9, "fft")],
)
def test_cross_representation_tie_breaks_use_validation_roc_auc_then_name(
    tmp_path: Path, monkeypatch, fft_roc_auc: float, rgb_roc_auc: float, expected: str
) -> None:
    experiments = [
        write_experiment(
            tmp_path,
            representation,
            seed,
            internal_score=0.99 if representation == "rgb" else 0.01,
            validation_score=0.8,
            validation_roc_auc=roc_auc,
        )
        for representation, roc_auc in (("fft", fft_roc_auc), ("rgb", rgb_roc_auc))
        for seed in (7, 17, 42)
    ]
    output = tmp_path / "aggregate"

    run_aggregate(monkeypatch, output, *experiments)

    assert selection_record(output)["decision"]["selected_representation"] == expected


def test_aggregate_rejects_missing_or_malformed_validation_metrics(tmp_path: Path, monkeypatch) -> None:
    first = write_experiment(
        tmp_path, "rgb", 7, internal_score=0.7, validation_score=0.7, write_validation=False
    )
    second = write_experiment(tmp_path, "rgb", 17, internal_score=0.8, validation_score=0.8)
    with pytest.raises(SystemExit, match="validation metrics"):
        run_aggregate(monkeypatch, tmp_path / "missing", first, second)

    (first / "validation_metrics.json").write_text(
        json.dumps({"roc_auc": 0.7, "balanced_accuracy": "bad"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="balanced_accuracy"):
        run_aggregate(monkeypatch, tmp_path / "malformed", first, second)


def test_aggregate_rejects_missing_or_nonfinite_validation_threshold(tmp_path: Path, monkeypatch) -> None:
    missing = write_experiment(
        tmp_path / "missing_threshold",
        "rgb",
        7,
        internal_score=0.7,
        validation_score=0.7,
        validation_threshold=None,
    )
    valid = write_experiment(
        tmp_path / "missing_threshold",
        "rgb",
        17,
        internal_score=0.8,
        validation_score=0.8,
    )
    with pytest.raises(SystemExit, match="threshold is missing"):
        run_aggregate(monkeypatch, tmp_path / "missing_threshold_aggregate", missing, valid)

    nonfinite = write_experiment(
        tmp_path / "nonfinite_threshold",
        "rgb",
        7,
        internal_score=0.7,
        validation_score=0.7,
        validation_threshold=float("nan"),
    )
    valid = write_experiment(
        tmp_path / "nonfinite_threshold",
        "rgb",
        17,
        internal_score=0.8,
        validation_score=0.8,
    )
    with pytest.raises(SystemExit, match="threshold must be finite"):
        run_aggregate(monkeypatch, tmp_path / "nonfinite_threshold_aggregate", nonfinite, valid)


def test_incomplete_seed_protocol_is_reported_without_fallback_selection(
    tmp_path: Path, monkeypatch
) -> None:
    rgb7 = write_experiment(tmp_path, "rgb", 7, internal_score=0.7, validation_score=0.7)
    rgb42 = write_experiment(tmp_path, "rgb", 42, internal_score=0.8, validation_score=0.8)
    fft7 = write_experiment(tmp_path, "fft", 7, internal_score=0.7, validation_score=0.7)
    fft42 = write_experiment(tmp_path, "fft", 42, internal_score=0.8, validation_score=0.8)
    output = tmp_path / "aggregate"

    run_aggregate(monkeypatch, output, rgb7, rgb42, fft7, fft42)

    decision = selection_record(output)["decision"]
    assert decision["made"] is False
    assert decision["selected_representation"] is None
    assert "requires exactly predeclared seeds [7, 17, 42]" in decision["reason"]


def test_selection_does_not_require_or_read_internal_test_analysis(tmp_path: Path, monkeypatch) -> None:
    rgb7 = write_experiment(
        tmp_path, "rgb", 7, internal_score=0.1, validation_score=0.8, write_analysis=False
    )
    rgb17 = write_experiment(
        tmp_path, "rgb", 17, internal_score=0.1, validation_score=0.8, write_analysis=False
    )
    rgb42 = write_experiment(
        tmp_path, "rgb", 42, internal_score=0.1, validation_score=0.8, write_analysis=False
    )
    fft7 = write_experiment(
        tmp_path, "fft", 7, internal_score=0.99, validation_score=0.7, write_analysis=False
    )
    fft17 = write_experiment(
        tmp_path, "fft", 17, internal_score=0.99, validation_score=0.7, write_analysis=False
    )
    fft42 = write_experiment(
        tmp_path, "fft", 42, internal_score=0.99, validation_score=0.7, write_analysis=False
    )
    output = tmp_path / "aggregate"

    run_aggregate(monkeypatch, output, rgb7, rgb17, rgb42, fft7, fft17, fft42)

    payload = json.loads((output / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert selection_record(output)["decision"]["selected_representation"] == "rgb"
    assert payload["internal_test_reporting"]["status"] == "not_available_for_all_runs"
    assert "metrics" not in payload


def test_aggregate_rejects_duplicate_seed_and_incompatible_signature(tmp_path: Path, monkeypatch) -> None:
    first = write_experiment(tmp_path / "duplicate", "rgb", 17, internal_score=0.7, validation_score=0.7)
    second = write_experiment(tmp_path / "duplicate_other", "rgb", 17, internal_score=0.8, validation_score=0.8)
    with pytest.raises(SystemExit, match="Duplicate seed"):
        run_aggregate(monkeypatch, tmp_path / "duplicate_aggregate", first, second)

    compatible = write_experiment(
        tmp_path / "signature", "rgb", 7, internal_score=0.7, validation_score=0.7
    )
    incompatible = write_experiment(
        tmp_path / "signature", "rgb", 17, internal_score=0.8, validation_score=0.8
    )
    run = json.loads((incompatible / "run.json").read_text(encoding="utf-8"))
    run["config"]["batch_size"] = 64
    (incompatible / "run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(SystemExit, match="incompatible controlled-protocol"):
        run_aggregate(monkeypatch, tmp_path / "signature_aggregate", compatible, incompatible)


def test_aggregate_rejects_mismatched_model_launch_contract(tmp_path: Path, monkeypatch) -> None:
    first = write_experiment(tmp_path, "rgb", 7, internal_score=0.7, validation_score=0.7)
    second = write_experiment(tmp_path, "rgb", 17, internal_score=0.8, validation_score=0.8)
    model = json.loads((second / "model.json").read_text(encoding="utf-8"))
    model["launch_options"]["requested"]["robust_augmentation"] = True
    (second / "model.json").write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(SystemExit, match="incompatible controlled-protocol"):
        run_aggregate(monkeypatch, tmp_path / "model_signature_aggregate", first, second)


def test_aggregate_refuses_existing_output_directory(tmp_path: Path, monkeypatch) -> None:
    first = write_experiment(tmp_path, "rgb", 7, internal_score=0.7, validation_score=0.7)
    second = write_experiment(tmp_path, "rgb", 17, internal_score=0.8, validation_score=0.8)
    output = tmp_path / "existing_aggregate"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_aggregate(monkeypatch, output, first, second)
