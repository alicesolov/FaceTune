from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

from ai_image_detector.reproducibility import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_internal_selection.py"
SPEC = importlib.util.spec_from_file_location("evaluate_internal_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluate_internal_selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_internal_selection)


class PixelTransform:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        red = float(image.convert("RGB").getpixel((0, 0))[0]) / 255.0
        return torch.full((3, 2, 2), red, dtype=torch.float32)


class PixelScoreModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        score = images[:, 0, 0, 0]
        return torch.stack((1.0 - score, score), dim=1) * 8.0


def write_image(path: Path, red: int) -> None:
    Image.new("RGB", (4, 4), (red, 0, 0)).save(path)


def selection_payload(experiment_dir: Path) -> dict[str, object]:
    return {
        "schema_version": evaluate_internal_selection.SELECTION_SCHEMA,
        "selection_status": evaluate_internal_selection.SELECTION_STATUS,
        "selection_rule": {
            "expected_representations": ["fft", "rgb"],
            "expected_seeds": [7, 17, 42],
            "external_validation_completed": False,
            "internal_test_metrics_used_for_selection": False,
            "representative_seed": 17,
        },
        "decision": {
            "made": True,
            "selected_representation": "fft",
            "representative_seed": 17,
            "experiment_dir": str(experiment_dir),
            "checkpoint_sha256": "a" * 64,
            "validation_threshold": 0.5,
        },
    }


def write_selection(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def manifest_frame(tmp_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_index in range(4):
        for generator, label, red in (("real", 0, 0), ("fake_a", 1, 255), ("fake_b", 1, 255)):
            path = tmp_path / f"{group_index}-{generator}.png"
            write_image(path, red)
            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "split": "test",
                    "generator": generator,
                    "group_id": f"group-{group_index}",
                    "source_id": f"{group_index}-{generator}",
                }
            )
    return pd.DataFrame(rows)


def frozen_bundle(experiment_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model=PixelScoreModel(),
        device=torch.device("cpu"),
        threshold=0.5,
        transform=PixelTransform(),
        preprocessing={"protocol": "highres_square_crop_384_v1"},
        representation="fft",
        experiment_dir=experiment_dir.resolve(),
        checkpoint_sha256="a" * 64,
        metrics={},
    )


def test_selection_contract_rejects_test_informed_or_incomplete_decision(tmp_path: Path) -> None:
    payload = selection_payload(tmp_path / "experiment")
    payload["selection_rule"]["internal_test_metrics_used_for_selection"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="exclude internal test"):
        evaluate_internal_selection.load_validation_selection(write_selection(tmp_path, payload))

    payload = selection_payload(tmp_path / "experiment")
    payload["selection_rule"]["expected_seeds"] = [7, 17]  # type: ignore[index]
    with pytest.raises(ValueError, match="complete 7/17/42"):
        evaluate_internal_selection.load_validation_selection(write_selection(tmp_path, payload))


def test_frozen_input_validation_pins_checkpoint_threshold_and_manifest(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    manifest = manifest_frame(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (experiment_dir / "model.json").write_text(
        json.dumps({"manifest": {"manifest_sha256": sha256_file(manifest_path)}}),
        encoding="utf-8",
    )
    selection = evaluate_internal_selection.load_validation_selection(
        write_selection(tmp_path, selection_payload(experiment_dir))
    )
    bundle = frozen_bundle(experiment_dir)
    test = evaluate_internal_selection.validate_frozen_inputs(
        selection, bundle, manifest_path, manifest
    )
    assert len(test) == 12

    bundle.checkpoint_sha256 = "b" * 64
    with pytest.raises(ValueError, match="Checkpoint SHA-256"):
        evaluate_internal_selection.validate_frozen_inputs(
            selection, bundle, manifest_path, manifest
        )


def test_clean_evaluation_preserves_provenance_and_bootstraps_by_group(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    manifest = manifest_frame(tmp_path)
    predictions = evaluate_internal_selection.evaluate_condition_rows(
        frozen_bundle(experiment_dir),
        manifest,
        condition="clean",
        parameters={},
        batch_size=3,
    )
    assert predictions["source_id"].tolist() == manifest["source_id"].tolist()
    assert predictions.loc[predictions["label"] == 0, "predicted_label"].eq(0).all()
    assert predictions.loc[predictions["label"] == 1, "predicted_label"].eq(1).all()

    metrics = evaluate_internal_selection.internal_metric_table(
        predictions,
        threshold=0.5,
        bootstrap_repeats=25,
        bootstrap_seed=17,
    )
    assert set(metrics["scope"]) == {"all", "real_vs_generator"}
    assert set(metrics.loc[metrics["scope"] == "real_vs_generator", "generator"]) == {
        "fake_a",
        "fake_b",
    }
    assert metrics["bootstrap_unit"].eq("group_id").all()
    assert metrics["bootstrap_groups"].eq(4).all()
    assert metrics["roc_auc_ci_lower"].eq(1.0).all()


def test_artifact_writer_refuses_to_overwrite_test_evidence(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    manifest = manifest_frame(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    selection_path = write_selection(tmp_path, selection_payload(experiment_dir))
    selection = evaluate_internal_selection.load_validation_selection(selection_path)
    bundle = frozen_bundle(experiment_dir)
    predictions = evaluate_internal_selection.evaluate_condition_rows(
        bundle,
        manifest,
        condition="clean",
        parameters={},
        batch_size=4,
    )
    predictions = pd.concat(
        [predictions.assign(condition=name) for name, _ in evaluate_internal_selection.CONDITIONS],
        ignore_index=True,
    )
    metrics = evaluate_internal_selection.internal_metric_table(
        predictions,
        threshold=0.5,
        bootstrap_repeats=10,
    )
    output = tmp_path / "evaluation"
    paths = evaluate_internal_selection.write_internal_artifacts(
        output_dir=output,
        manifest_path=manifest_path,
        selection=selection,
        bundle=bundle,
        predictions=predictions,
        metrics=metrics,
        bootstrap_repeats=10,
        bootstrap_seed=17,
        bootstrap_confidence=0.95,
    )
    assert all(path.is_file() for path in paths.values())
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert config["evaluation_status"] == "completed_no_post_test_tuning"
    assert "threshold selection" in config["protocol"]["prohibited_operations"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluate_internal_selection.write_internal_artifacts(
            output_dir=output,
            manifest_path=manifest_path,
            selection=selection,
            bundle=bundle,
            predictions=predictions,
            metrics=metrics,
            bootstrap_repeats=10,
            bootstrap_seed=17,
            bootstrap_confidence=0.95,
        )
