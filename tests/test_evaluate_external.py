from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_external.py"
SPEC = importlib.util.spec_from_file_location("evaluate_external", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluate_external = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_external)


class PixelTransform:
    """Tiny deterministic stand-in for the frozen RGB/FFT preprocessing."""

    preprocessing_protocol = "test_frozen_evaluation_v1"

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return torch.tensor([image.convert("RGB").getpixel((0, 0))[0] / 255.0])


class PixelScoreModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        score = images[:, 0] * 12.0 - 6.0
        return torch.stack((-score, score), dim=1)


def write_image(path: Path, red: int) -> None:
    Image.new("RGB", (12, 8), color=(red, 0, 0)).save(path)


def prepared_manifest(tmp_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(10):
        red = 0
        path = tmp_path / f"real-{index}.png"
        write_image(path, red)
        rows.append(
            {
                "path": str(path),
                "label": 0,
                "split": "external",
                "generator": "real",
                "group_id": f"raise-{index}",
                "source_id": f"raise-{index}",
                "source_dataset": "raise1k",
                "source_relative_path": path.name,
                "generator_family": "camera",
                "defactify_train_relation": "new_real_domain",
                "pixel_sha256": f"real-{index}",
            }
        )
    for generator, relation, red in (
        ("dalle3", "same_named_generator", 255),
        ("firefly", "unseen_family", 0),
    ):
        for index in range(10):
            path = tmp_path / f"{generator}-{index}.png"
            write_image(path, red)
            rows.append(
                {
                    "path": str(path),
                    "label": 1,
                    "split": "external",
                    "generator": generator,
                    "group_id": f"{generator}-{index}",
                    "source_id": f"{generator}-{index}",
                    "source_dataset": "synthbuster",
                    "source_relative_path": path.name,
                    "generator_family": generator,
                    "defactify_train_relation": relation,
                    "pixel_sha256": f"{generator}-{index}",
                }
            )
    return pd.DataFrame(rows)


def frozen_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        model=PixelScoreModel(),
        device=torch.device("cpu"),
        threshold=0.5,
        transform=PixelTransform(),
        preprocessing={"protocol": "test_frozen_evaluation_v1"},
        representation="rgb",
        experiment_dir=Path("/tmp/frozen-experiment"),
        checkpoint_sha256="a" * 64,
    )


def test_external_evaluation_preserves_manifest_metadata_and_relationships(tmp_path: Path) -> None:
    manifest = prepared_manifest(tmp_path)
    predictions = evaluate_external.evaluate_external_rows(
        frozen_bundle(), manifest, batch_size=2
    )

    assert len(predictions) == len(manifest)
    assert predictions["source_relative_path"].tolist() == manifest["source_relative_path"].tolist()
    assert predictions["pixel_sha256"].tolist() == manifest["pixel_sha256"].tolist()
    assert set(predictions["model_decision"]) == {"ai_like", "not_ai_like"}
    assert predictions.loc[predictions["generator"] == "dalle3", "predicted_label"].eq(1).all()
    assert predictions.loc[predictions["generator"] == "firefly", "predicted_label"].eq(0).all()

    metrics = evaluate_external.external_metric_table(
        predictions,
        threshold=0.5,
        bootstrap_repeats=31,
        bootstrap_seed=17,
    )
    generator_rows = metrics.loc[metrics["scope"] == "real_vs_generator"].set_index("generator")
    assert generator_rows.loc["dalle3", "defactify_train_relation"] == "same_named_generator"
    assert generator_rows.loc["firefly", "defactify_train_relation"] == "unseen_family"
    assert generator_rows.loc["dalle3", "n_real"] == 10
    assert generator_rows.loc["dalle3", "n_ai"] == 10
    assert generator_rows.loc["dalle3", "bootstrap_unit"] == "group_id"
    assert generator_rows.loc["dalle3", "bootstrap_groups"] == 20
    assert generator_rows.loc["dalle3", "bootstrap_repeats_requested"] == 31
    for metric in evaluate_external.EXTERNAL_CI_METRICS:
        assert f"{metric}_ci_lower" in generator_rows.columns
        assert f"{metric}_ci_upper" in generator_rows.columns
        assert generator_rows.loc["dalle3", f"{metric}_bootstrap_repeats"] > 0
    worst = metrics.loc[metrics["scope"] == "worst_generator_by_balanced_accuracy"].iloc[0]
    assert worst["generator"] == "firefly"
    assert worst["worst_selection_metric"] == "balanced_accuracy"
    assert "macro_average_across_generators" in set(metrics["scope"])


def test_external_manifest_rejects_internal_split_and_missing_relation(tmp_path: Path) -> None:
    manifest = prepared_manifest(tmp_path)
    manifest.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="forbidden splits"):
        evaluate_external.validate_external_manifest(manifest)

    manifest = prepared_manifest(tmp_path)
    manifest = manifest.drop(columns=["defactify_train_relation"])
    with pytest.raises(ValueError, match="prepare_synthbuster_external"):
        evaluate_external.validate_external_manifest(manifest)

    manifest = prepared_manifest(tmp_path)
    manifest.loc[1, "group_id"] = manifest.loc[0, "group_id"]
    with pytest.raises(ValueError, match="source-unique"):
        evaluate_external.validate_external_manifest(manifest)


def test_dry_run_and_artifact_writing_do_not_require_the_full_external_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = prepared_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    plan = evaluate_external.dry_run_plan(
        manifest_path,
        manifest,
        bootstrap_repeats=23,
        bootstrap_seed=41,
    )
    assert plan["mode"] == "dry_run_manifest_validation_only"
    assert {row["defactify_train_relation"] for row in plan["generator_relationships_from_manifest"]} == {
        "same_named_generator",
        "unseen_family",
    }
    assert plan["bootstrap"] == {
        "scope": "per-generator real_vs_generator rows",
        "unit": "group_id",
        "unit_interpretation": "source-unique external manifest row",
        "repeats_requested": 23,
        "seed": 41,
        "confidence": 0.95,
        "metrics": list(evaluate_external.EXTERNAL_CI_METRICS),
    }
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_external.py", "--manifest", str(manifest_path), "--dry-run"],
    )
    evaluate_external.main()
    assert "dry_run_manifest_validation_only" in capsys.readouterr().out

    bundle = frozen_bundle()
    predictions = evaluate_external.evaluate_external_rows(bundle, manifest, batch_size=3)
    metrics = evaluate_external.external_metric_table(
        predictions,
        threshold=float(bundle.threshold),
        bootstrap_repeats=23,
        bootstrap_seed=41,
    )
    paths = evaluate_external.write_external_artifacts(
        output_dir=tmp_path / "output",
        manifest_path=manifest_path,
        bundle=bundle,
        predictions=predictions,
        metrics=metrics,
        bootstrap_repeats=23,
        bootstrap_seed=41,
    )
    assert paths["predictions"].is_file()
    assert paths["metrics"].is_file()
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert "threshold selection" in config["protocol"]["prohibited_operations"]
    assert config["bootstrap"]["unit"] == "group_id"
    assert config["bootstrap"]["repeats_requested"] == 23
    assert config["bootstrap"]["seed"] == 41
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluate_external.write_external_artifacts(
            output_dir=tmp_path / "output",
            manifest_path=manifest_path,
            bundle=bundle,
            predictions=predictions,
            metrics=metrics,
        )
