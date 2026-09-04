"""Open the locked DANI test split for one frozen validation-selected checkpoint.

This is a deliberately narrow evaluation entry point.  It validates the complete
validation-only selection record, pins the checkpoint and manifest by SHA-256, refuses to
overwrite an existing result, and has no training or threshold-selection controls.  Clean and
predeclared robustness conditions are evaluated in one terminal stage after model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ai_image_detector.dataset import ManifestImageDataset
from ai_image_detector.features import DANI_HIGHRES_PREPROCESSING_PROTOCOL, DegradedTransform
from ai_image_detector.inference import ModelBundle
from ai_image_detector.manifest import load_manifest
from ai_image_detector.metrics import binary_metrics, cluster_bootstrap_intervals
from ai_image_detector.reproducibility import save_json, sha256_file

SELECTION_SCHEMA = "ai_image_detector_validation_selection_v1"
SELECTION_STATUS = "validation_selected_pending_external_validation"
EXPECTED_REPRESENTATIONS = {"fft", "rgb"}
EXPECTED_SEEDS = {7, 17, 42}
CI_METRICS = ("roc_auc", "balanced_accuracy", "macro_f1", "fpr_at_tpr_95")
DEFAULT_BOOTSTRAP_REPEATS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260829
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
PREDICTIONS_NAME = "internal_test_predictions.csv"
METRICS_NAME = "internal_test_metrics.csv"
CONFIG_NAME = "internal_test_evaluation.json"

CONDITIONS: tuple[tuple[str, dict[str, object]], ...] = (
    ("clean", {}),
    ("jpeg_q95", {"kind": "jpeg", "jpeg_quality": 95}),
    ("jpeg_q75", {"kind": "jpeg", "jpeg_quality": 75}),
    ("jpeg_q50", {"kind": "jpeg", "jpeg_quality": 50}),
    ("resize_075", {"kind": "resize", "resize_scale": 0.75}),
    ("resize_050", {"kind": "resize", "resize_scale": 0.50}),
    ("gaussian_blur_r1", {"kind": "blur"}),
)


def _json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must be valid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{description} must be a JSON object")
    return payload


def load_validation_selection(path: str | Path) -> dict[str, Any]:
    """Validate the validation-only decision that authorises opening the internal test."""
    record_path = Path(path).resolve()
    record = _json_object(record_path, "Selection record")
    if record.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("Unsupported validation selection record schema")
    if record.get("selection_status") != SELECTION_STATUS:
        raise ValueError(f"Selection status must be {SELECTION_STATUS!r}")

    rule = record.get("selection_rule")
    decision = record.get("decision")
    if not isinstance(rule, dict) or not isinstance(decision, dict):
        raise TypeError("Selection record must include selection_rule and decision objects")
    if rule.get("internal_test_metrics_used_for_selection") is not False:
        raise ValueError("Selection must explicitly exclude internal test metrics")
    if rule.get("external_validation_completed") is not False:
        raise ValueError("This evaluator requires the pre-external validation selection stage")
    if set(rule.get("expected_representations", [])) != EXPECTED_REPRESENTATIONS:
        raise ValueError("Selection record does not contain the complete RGB/FFT comparison")
    if set(rule.get("expected_seeds", [])) != EXPECTED_SEEDS:
        raise ValueError("Selection record does not contain the complete 7/17/42 seed queue")
    if decision.get("made") is not True:
        raise ValueError("Selection decision is not final")
    if decision.get("representative_seed") != rule.get("representative_seed"):
        raise ValueError("Decision does not use the predeclared representative seed")

    representation = decision.get("selected_representation")
    experiment_value = decision.get("experiment_dir")
    checkpoint_sha256 = decision.get("checkpoint_sha256")
    threshold = decision.get("validation_threshold")
    if representation not in EXPECTED_REPRESENTATIONS:
        raise ValueError("Decision has an unsupported selected representation")
    if not isinstance(experiment_value, str) or not experiment_value:
        raise ValueError("Decision must identify the selected experiment directory")
    if (
        not isinstance(checkpoint_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None
    ):
        raise ValueError("Decision has an invalid checkpoint SHA-256")
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Decision has an invalid frozen validation threshold")

    experiment_dir = Path(experiment_value)
    if not experiment_dir.is_absolute():
        experiment_dir = record_path.parent / experiment_dir
    return {
        "record_path": record_path,
        "record": record,
        "experiment_dir": experiment_dir.resolve(),
        "checkpoint_sha256": checkpoint_sha256,
        "representation": representation,
        "threshold": float(threshold),
    }


def validate_frozen_inputs(
    selection: dict[str, Any],
    bundle: ModelBundle,
    manifest_path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Pin checkpoint, transform, threshold, and DANI manifest before reading test bytes."""
    experiment_dir = Path(selection["experiment_dir"]).resolve()
    if bundle.experiment_dir.resolve() != experiment_dir:
        raise ValueError("Loaded experiment differs from the validation selection record")
    if bundle.checkpoint_sha256 != selection["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the validation selection record")
    if bundle.representation != selection["representation"]:
        raise ValueError("Checkpoint representation differs from the validation selection record")
    if not math.isclose(bundle.threshold, selection["threshold"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Checkpoint threshold differs from the frozen validation threshold")
    if bundle.preprocessing.get("protocol") != DANI_HIGHRES_PREPROCESSING_PROTOCOL:
        raise ValueError("Checkpoint does not use the controlled 384-pixel preprocessing protocol")
    if bundle.metrics:
        raise ValueError("Selected experiment already contains internal test metrics")

    model = _json_object(experiment_dir / "model.json", "Selected experiment model metadata")
    manifest_meta = model.get("manifest")
    if not isinstance(manifest_meta, dict):
        raise TypeError("Selected experiment model metadata has no manifest block")
    expected_manifest_sha = manifest_meta.get("manifest_sha256")
    actual_manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha != actual_manifest_sha:
        raise ValueError("Manifest SHA-256 differs from the training-time manifest")

    test = frame.loc[frame["split"] == "test"].reset_index(drop=True).copy()
    if test.empty:
        raise ValueError("Manifest has no locked test split")
    if test["group_id"].isna().any() or test["group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every test row must have a non-empty group_id")
    if set(test["label"].astype(int)) != {0, 1}:
        raise ValueError("Locked test split must contain both real and AI classes")
    return test


def condition_transform(base: object, parameters: dict[str, object]) -> object:
    if not parameters:
        return base
    values = dict(parameters)
    kind = str(values.pop("kind"))
    return DegradedTransform(base, kind, **values)


def evaluate_condition_rows(
    bundle: ModelBundle,
    test: pd.DataFrame,
    *,
    condition: str,
    parameters: dict[str, object],
    batch_size: int,
    workers: int = 0,
) -> pd.DataFrame:
    """Score one predeclared condition while retaining all row-level provenance."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if workers < 0:
        raise ValueError("workers cannot be negative")
    transform = condition_transform(bundle.transform, parameters)
    ordered = test.reset_index(drop=True).copy()
    loader = DataLoader(
        ManifestImageDataset(ordered, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=bundle.device.type == "cuda",
    )
    scores: list[float] = []
    bundle.model.eval()
    with torch.inference_mode():
        for images, _, _ in loader:
            logits = bundle.model(images.to(bundle.device))
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Frozen model must return binary logits with shape [batch, 2]")
            scores.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist())
    if len(scores) != len(ordered):
        raise RuntimeError(f"Scored {len(scores)} rows for a {len(ordered)}-row test split")
    ordered["condition"] = condition
    ordered["ai_score"] = scores
    ordered["predicted_label"] = (ordered["ai_score"] >= bundle.threshold).astype(int)
    ordered["model_decision"] = ordered["predicted_label"].map({0: "not_ai_like", 1: "ai_like"})
    ordered["threshold"] = bundle.threshold
    ordered["representation"] = bundle.representation
    ordered["experiment"] = bundle.experiment_dir.name
    ordered["checkpoint_sha256"] = bundle.checkpoint_sha256
    return ordered


def internal_metric_table(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
) -> pd.DataFrame:
    """Compute aggregate and real-vs-generator metrics; bootstrap clean results by parent."""
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be at least one")
    rows: list[dict[str, object]] = []
    for condition, condition_rows in predictions.groupby("condition", sort=False):
        comparisons: list[tuple[str, str, pd.DataFrame]] = [("all", "", condition_rows)]
        generators = sorted(condition_rows.loc[condition_rows["label"] == 1, "generator"].unique())
        for generator in generators:
            subset = condition_rows.loc[
                (condition_rows["label"] == 0) | (condition_rows["generator"] == generator)
            ]
            comparisons.append(("real_vs_generator", str(generator), subset))
        for scope, generator, comparison in comparisons:
            row: dict[str, object] = {
                "condition": str(condition),
                "scope": scope,
                "generator": generator,
                "n_real": int((comparison["label"] == 0).sum()),
                "n_ai": int((comparison["label"] == 1).sum()),
            }
            row.update(
                binary_metrics(
                    comparison["label"].to_numpy(dtype=int),
                    comparison["ai_score"].to_numpy(dtype=float),
                    threshold,
                )
            )
            if condition == "clean":
                intervals = cluster_bootstrap_intervals(
                    comparison,
                    threshold,
                    group_column="group_id",
                    metrics=CI_METRICS,
                    repeats=bootstrap_repeats,
                    seed=bootstrap_seed,
                    confidence=bootstrap_confidence,
                )
                row.update(
                    {
                        "bootstrap_unit": "group_id",
                        "bootstrap_groups": int(comparison["group_id"].nunique()),
                        "bootstrap_repeats_requested": bootstrap_repeats,
                        "bootstrap_seed": bootstrap_seed,
                        "bootstrap_confidence": bootstrap_confidence,
                    }
                )
                for metric, interval in intervals.items():
                    row[f"{metric}_ci_lower"] = interval["lower"]
                    row[f"{metric}_ci_upper"] = interval["upper"]
                    row[f"{metric}_bootstrap_repeats"] = interval["bootstrap_repeats"]
            rows.append(row)
    return pd.DataFrame(rows)


def write_internal_artifacts(
    *,
    output_dir: str | Path,
    manifest_path: str | Path,
    selection: dict[str, Any],
    bundle: ModelBundle,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    bootstrap_confidence: float,
) -> dict[str, Path]:
    """Atomically establish a new result directory; never overwrite test evidence."""
    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite existing internal evaluation: {output}"
        ) from error
    predictions_path = output / PREDICTIONS_NAME
    metrics_path = output / METRICS_NAME
    config_path = output / CONFIG_NAME
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    save_json(
        {
            "schema_version": "ai_image_detector_internal_test_evaluation_v1",
            "evaluation_status": "completed_no_post_test_tuning",
            "selection_record": {
                "path": str(selection["record_path"]),
                "sha256": sha256_file(selection["record_path"]),
                "status": selection["record"]["selection_status"],
                "internal_test_metrics_used_for_selection": False,
            },
            "manifest": {
                "path": str(Path(manifest_path).resolve()),
                "sha256": sha256_file(manifest_path),
                "test_rows": int(len(predictions) // len(CONDITIONS)),
            },
            "model": {
                "experiment_dir": str(bundle.experiment_dir),
                "representation": bundle.representation,
                "checkpoint_sha256": bundle.checkpoint_sha256,
                "validation_selected_threshold": bundle.threshold,
                "preprocessing": bundle.preprocessing,
                "device": str(bundle.device),
            },
            "protocol": {
                "stage": "single_terminal_internal_test_and_robustness_evaluation",
                "conditions": [name for name, _ in CONDITIONS],
                "degradation_order": "after_common_raster_before_model_transform",
                "prohibited_operations": [
                    "training",
                    "checkpoint replacement",
                    "threshold selection",
                    "representation selection",
                    "post-test hyperparameter tuning",
                ],
            },
            "bootstrap": {
                "scope": "clean aggregate and real-vs-generator rows",
                "unit": "group_id",
                "repeats_requested": bootstrap_repeats,
                "seed": bootstrap_seed,
                "confidence": bootstrap_confidence,
                "metrics": list(CI_METRICS),
            },
            "artifacts": {
                "predictions": PREDICTIONS_NAME,
                "metrics": METRICS_NAME,
            },
        },
        config_path,
    )
    return {"predictions": predictions_path, "metrics": metrics_path, "config": config_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-confidence", type=float, default=DEFAULT_BOOTSTRAP_CONFIDENCE)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing internal evaluation: {args.output_dir}")
    selection = load_validation_selection(args.selection_record)
    bundle = ModelBundle.load(selection["experiment_dir"], device_name=args.device)
    frame = load_manifest(args.manifest, check_paths=True)
    test = validate_frozen_inputs(selection, bundle, args.manifest, frame)

    condition_predictions = [
        evaluate_condition_rows(
            bundle,
            test,
            condition=condition,
            parameters=parameters,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        for condition, parameters in CONDITIONS
    ]
    predictions = pd.concat(condition_predictions, ignore_index=True)
    metrics = internal_metric_table(
        predictions,
        threshold=bundle.threshold,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence=args.bootstrap_confidence,
    )
    paths = write_internal_artifacts(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        selection=selection,
        bundle=bundle,
        predictions=predictions,
        metrics=metrics,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence=args.bootstrap_confidence,
    )
    print(metrics.to_string(index=False))
    print(f"Saved frozen internal evaluation to {paths['config'].parent}")


if __name__ == "__main__":
    main()
