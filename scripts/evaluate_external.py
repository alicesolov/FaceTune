"""Evaluate one frozen experiment on the locked Synthbuster + RAISE external manifest.

This script deliberately has no training, threshold selection, augmentation, or model-selection
arguments.  It loads the selected experiment through :class:`ModelBundle`, therefore uses the
checkpoint, validation-selected threshold, and exact evaluation preprocessing recorded by that
experiment.  The input must be the ``split=external`` manifest written by
``prepare_synthbuster_external.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ai_image_detector.dataset import ManifestImageDataset
from ai_image_detector.inference import ModelBundle
from ai_image_detector.manifest import load_manifest
from ai_image_detector.metrics import binary_metrics, cluster_bootstrap_intervals
from ai_image_detector.reproducibility import save_json, sha256_file

PREPARED_EXTERNAL_COLUMNS = frozenset(
    {
        "source_dataset",
        "generator_family",
        "defactify_train_relation",
    }
)
AVERAGED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "ai_precision",
    "ai_recall",
    "real_recall",
    "roc_auc",
    "pr_auc",
    "fpr_at_tpr_95",
    "ece_15",
    "brier",
)
EXTERNAL_CI_METRICS = (
    "roc_auc",
    "balanced_accuracy",
    "macro_f1",
    "fpr_at_tpr_95",
)
DEFAULT_BOOTSTRAP_REPEATS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260829
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
PREDICTIONS_NAME = "external_predictions.csv"
METRICS_NAME = "external_metrics.csv"
CONFIG_NAME = "external_evaluation.json"


def _single_nonempty_value(frame: pd.DataFrame, column: str, generator: str) -> str:
    """Read an explicit generator context, refusing an ambiguous external manifest."""
    values = frame[column]
    if values.isna().any():
        raise ValueError(f"Generator {generator!r} has a missing {column!r} relationship")
    unique = sorted({str(value).strip() for value in values})
    if len(unique) != 1 or not unique[0]:
        raise ValueError(
            f"Generator {generator!r} must have exactly one non-empty {column!r}; "
            f"found {unique!r}"
        )
    return unique[0]


def validate_external_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the prepared external-only contract and return generator relationships.

    The returned table is copied directly from the preparation manifest.  In particular, this
    evaluator never infers whether a generator is new, same-family, or same-named from a string
    heuristic.
    """
    if frame.empty:
        raise ValueError("External manifest contains no rows")
    non_external = sorted(set(frame.loc[frame["split"] != "external", "split"].astype(str)))
    if non_external:
        raise ValueError(
            "External evaluation accepts only split='external' rows; "
            f"found forbidden splits: {non_external}"
        )
    missing = sorted(PREPARED_EXTERNAL_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "External manifest must come from prepare_synthbuster_external.py and include "
            f"{missing}"
        )

    real = frame.loc[frame["label"] == 0]
    fake = frame.loc[frame["label"] == 1]
    if real.empty or fake.empty:
        raise ValueError("External evaluation requires both RAISE real and Synthbuster AI rows")
    if not real["generator"].eq("real").all():
        raise ValueError("All label=0 external rows must use generator='real'")
    if not real["source_dataset"].eq("raise1k").all():
        raise ValueError("All label=0 external rows must come from source_dataset='raise1k'")
    if fake["generator"].eq("real").any():
        raise ValueError("External AI rows cannot use generator='real'")
    if not fake["source_dataset"].eq("synthbuster").all():
        raise ValueError("All label=1 external rows must come from source_dataset='synthbuster'")
    if frame["group_id"].isna().any() or frame["group_id"].astype(str).str.strip().eq("").any():
        raise ValueError("External manifest requires a non-empty group_id for every source row")
    if frame["group_id"].duplicated().any():
        raise ValueError(
            "External group_id values must be source-unique so bootstrap units are independent"
        )

    contexts: list[dict[str, object]] = []
    for generator, rows in fake.groupby("generator", sort=True):
        contexts.append(
            {
                "generator": str(generator),
                "generator_family": _single_nonempty_value(
                    rows, "generator_family", str(generator)
                ),
                "defactify_train_relation": _single_nonempty_value(
                    rows, "defactify_train_relation", str(generator)
                ),
                "n_ai": len(rows),
            }
        )
    return pd.DataFrame(contexts)


def evaluate_external_rows(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    workers: int = 0,
) -> pd.DataFrame:
    """Score every external row with the frozen model and its exact evaluation transform."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if workers < 0:
        raise ValueError("workers cannot be negative")
    validate_external_manifest(frame)
    if not callable(bundle.transform):
        raise TypeError("ModelBundle.transform must be the callable frozen evaluation preprocessing")
    threshold = float(bundle.threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Frozen validation threshold must be in [0, 1], got {threshold}")

    ordered = frame.reset_index(drop=True).copy()
    dataset = ManifestImageDataset(ordered, bundle.transform)
    loader = DataLoader(
        dataset,
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
                raise ValueError("Frozen model must return binary class logits with shape [batch, 2]")
            scores.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist())
    if len(scores) != len(ordered):
        raise RuntimeError(
            f"External scoring produced {len(scores)} scores for {len(ordered)} manifest rows"
        )

    # Start with every manifest column, rather than reconstructing a small subset from a loader.
    # Thus row-level provenance (hashes, source path, and relationship) remains attached to score.
    ordered["ai_score"] = scores
    ordered["predicted_label"] = (ordered["ai_score"] >= threshold).astype(int)
    ordered["model_decision"] = ordered["predicted_label"].map(
        {0: "not_ai_like", 1: "ai_like"}
    )
    ordered["threshold"] = threshold
    ordered["representation"] = bundle.representation
    ordered["experiment"] = bundle.experiment_dir.name
    ordered["checkpoint_sha256"] = bundle.checkpoint_sha256
    return ordered


def _metric_row(
    predictions: pd.DataFrame,
    *,
    scope: str,
    threshold: float,
    generator: str,
    generator_family: str,
    relation: str,
    n_real: int,
    n_ai: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scope": scope,
        "generator": generator,
        "generator_family": generator_family,
        "defactify_train_relation": relation,
        "n_real": n_real,
        "n_ai": n_ai,
        "n_generators": 1 if generator else 0,
        "worst_selection_metric": "",
    }
    row.update(
        binary_metrics(
            predictions["label"].to_numpy(dtype=int),
            predictions["ai_score"].to_numpy(dtype=float),
            threshold,
        )
    )
    return row


def _validate_bootstrap_settings(repeats: int, confidence: float) -> None:
    if repeats < 1:
        raise ValueError("bootstrap_repeats must be at least one")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must be strictly between zero and one")


def _add_generator_intervals(
    row: dict[str, object],
    comparison: pd.DataFrame,
    *,
    threshold: float,
    repeats: int,
    seed: int,
    confidence: float,
) -> None:
    """Attach source-unit percentile CIs to one real-versus-generator comparison."""
    intervals = cluster_bootstrap_intervals(
        comparison,
        threshold,
        group_column="group_id",
        metrics=EXTERNAL_CI_METRICS,
        seed=seed,
        repeats=repeats,
        confidence=confidence,
    )
    row.update(
        {
            "bootstrap_unit": "group_id",
            "bootstrap_unit_interpretation": "source-unique external manifest row",
            "bootstrap_groups": int(comparison["group_id"].nunique()),
            "bootstrap_repeats_requested": repeats,
            "bootstrap_seed": seed,
            "bootstrap_confidence": confidence,
        }
    )
    for metric, interval in intervals.items():
        row[f"{metric}_ci_lower"] = interval["lower"]
        row[f"{metric}_ci_upper"] = interval["upper"]
        row[f"{metric}_bootstrap_repeats"] = interval["bootstrap_repeats"]


def external_metric_table(
    predictions: pd.DataFrame,
    threshold: float,
    *,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
) -> pd.DataFrame:
    """Create aggregate and comparable real-versus-each-generator metric rows."""
    _validate_bootstrap_settings(bootstrap_repeats, bootstrap_confidence)
    contexts = validate_external_manifest(predictions)
    real = predictions.loc[predictions["label"] == 0]
    fake = predictions.loc[predictions["label"] == 1]
    all_row = _metric_row(
        predictions,
        scope="all_external",
        threshold=threshold,
        generator="",
        generator_family="mixed",
        relation="mixed",
        n_real=len(real),
        n_ai=len(fake),
    )
    all_row["n_generators"] = len(contexts)

    per_generator: list[dict[str, object]] = []
    for context in contexts.itertuples(index=False):
        generator_rows = fake.loc[fake["generator"] == context.generator]
        # Each comparison deliberately has *all* RAISE real images and only one Synthbuster
        # generator, so every reported row remains a valid binary discrimination problem.
        comparison = pd.concat([real, generator_rows], ignore_index=True)
        row = _metric_row(
            comparison,
            scope="real_vs_generator",
            threshold=threshold,
            generator=str(context.generator),
            generator_family=str(context.generator_family),
            relation=str(context.defactify_train_relation),
            n_real=len(real),
            n_ai=len(generator_rows),
        )
        _add_generator_intervals(
            row,
            comparison,
            threshold=threshold,
            repeats=bootstrap_repeats,
            seed=bootstrap_seed,
            confidence=bootstrap_confidence,
        )
        per_generator.append(row)
    per_generator_frame = pd.DataFrame(per_generator)

    macro: dict[str, object] = {
        "scope": "macro_average_across_generators",
        "generator": "",
        "generator_family": "mixed",
        "defactify_train_relation": "mixed",
        "n": None,
        "threshold": threshold,
        "n_real": len(real),
        "n_ai": len(fake),
        "n_generators": len(per_generator_frame),
        "worst_selection_metric": "",
        "tn": None,
        "fp": None,
        "fn": None,
        "tp": None,
    }
    for metric in AVERAGED_METRICS:
        values = pd.to_numeric(per_generator_frame[metric], errors="coerce")
        macro[metric] = float(values.mean()) if values.notna().any() else float("nan")

    # "Worst" is predeclared as the smallest balanced accuracy, not retrospectively chosen from
    # whichever metric happens to make a model look weakest.  Generator name breaks exact ties.
    worst = per_generator_frame.sort_values(
        ["balanced_accuracy", "generator"], na_position="last"
    ).iloc[0].to_dict()
    worst["scope"] = "worst_generator_by_balanced_accuracy"
    worst["worst_selection_metric"] = "balanced_accuracy"
    return pd.DataFrame([all_row, *per_generator, macro, worst])


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy values to strict JSON values (NaN becomes null)."""
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def write_external_artifacts(
    *,
    output_dir: Path,
    manifest_path: Path,
    bundle: ModelBundle,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write scored rows, comparison metrics, and the frozen-evaluation provenance record."""
    _validate_bootstrap_settings(bootstrap_repeats, bootstrap_confidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_dir / PREDICTIONS_NAME,
        "metrics": output_dir / METRICS_NAME,
        "config": output_dir / CONFIG_NAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing external evaluation artifacts: {rendered}")

    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    contexts = validate_external_manifest(predictions)
    config = {
        "evaluation": "frozen external Synthbuster + RAISE-1k evaluation",
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "rows": len(predictions),
            "split": "external only",
        },
        "frozen_experiment": {
            "directory": str(bundle.experiment_dir),
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "representation": bundle.representation,
            "validation_threshold": float(bundle.threshold),
            "preprocessing": _json_value(bundle.preprocessing),
        },
        "protocol": {
            "allowed_operations": ["batched inference", "fixed-threshold evaluation"],
            "prohibited_operations": [
                "training",
                "early stopping",
                "threshold selection",
                "model selection",
                "augmentation selection",
                "external-result-driven tuning",
            ],
            "per_generator_comparison": "all RAISE real rows plus one Synthbuster generator",
            "macro_average": "unweighted mean of per-generator metric rows",
            "worst_generator": "smallest per-generator balanced_accuracy; name breaks ties",
        },
        "bootstrap": {
            "scope": "per-generator real_vs_generator rows",
            "unit": "group_id",
            "unit_interpretation": "source-unique external manifest row",
            "repeats_requested": bootstrap_repeats,
            "seed": bootstrap_seed,
            "confidence": bootstrap_confidence,
            "metrics": list(EXTERNAL_CI_METRICS),
        },
        "generator_relationships_from_manifest": _json_value(contexts.to_dict(orient="records")),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    save_json(_json_value(config), paths["config"])
    return paths


def dry_run_plan(
    manifest_path: Path,
    frame: pd.DataFrame,
    *,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
) -> dict[str, object]:
    """Return validation-only evidence without opening image files or a checkpoint."""
    _validate_bootstrap_settings(bootstrap_repeats, bootstrap_confidence)
    contexts = validate_external_manifest(frame)
    return {
        "mode": "dry_run_manifest_validation_only",
        "manifest": str(manifest_path.resolve()),
        "rows": len(frame),
        "real_rows": int((frame["label"] == 0).sum()),
        "ai_rows": int((frame["label"] == 1).sum()),
        "generator_relationships_from_manifest": _json_value(contexts.to_dict(orient="records")),
        "bootstrap": {
            "scope": "per-generator real_vs_generator rows",
            "unit": "group_id",
            "unit_interpretation": "source-unique external manifest row",
            "repeats_requested": bootstrap_repeats,
            "seed": bootstrap_seed,
            "confidence": bootstrap_confidence,
            "metrics": list(EXTERNAL_CI_METRICS),
        },
        "does_not_open": ["image files", "checkpoint"],
        "does_not_perform": ["training", "threshold selection", "model selection"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="Frozen completed experiment; required unless --dry-run is selected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for external artifacts; required unless --dry-run is selected.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-confidence", type=float, default=DEFAULT_BOOTSTRAP_CONFIDENCE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the external-manifest contract without reading images or loading a model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this evaluator's existing CSV/JSON artifacts in --output-dir.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least one")
    if args.workers < 0:
        raise SystemExit("--workers cannot be negative")
    try:
        _validate_bootstrap_settings(args.bootstrap_repeats, args.bootstrap_confidence)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    manifest_path = args.manifest.expanduser().resolve()
    frame = load_manifest(manifest_path, check_paths=not args.dry_run)
    if args.dry_run:
        print(
            json.dumps(
                dry_run_plan(
                    manifest_path,
                    frame,
                    bootstrap_repeats=args.bootstrap_repeats,
                    bootstrap_seed=args.bootstrap_seed,
                    bootstrap_confidence=args.bootstrap_confidence,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.experiment_dir is None:
        raise SystemExit("--experiment-dir is required unless --dry-run is selected")
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --dry-run is selected")

    bundle = ModelBundle.load(args.experiment_dir, device_name=args.device)
    predictions = evaluate_external_rows(
        bundle, frame, batch_size=args.batch_size, workers=args.workers
    )
    metrics = external_metric_table(
        predictions,
        float(bundle.threshold),
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence=args.bootstrap_confidence,
    )
    paths = write_external_artifacts(
        output_dir=args.output_dir.expanduser().resolve(),
        manifest_path=manifest_path,
        bundle=bundle,
        predictions=predictions,
        metrics=metrics,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence=args.bootstrap_confidence,
        overwrite=args.overwrite,
    )
    print(
        metrics[
            [
                "scope",
                "generator",
                "defactify_train_relation",
                "n",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote frozen external-evaluation artifacts to {paths['predictions'].parent}")


if __name__ == "__main__":
    main()
