"""Summarise a frozen experiment without using held-out data for selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ai_image_detector.metrics import (
    DEFAULT_CLUSTER_METRICS,
    binary_metrics,
    cluster_bootstrap_intervals,
    paired_cluster_bootstrap_difference,
    paired_group_ranking_accuracy,
)
from ai_image_detector.reproducibility import save_json


def load_experiment(experiment_dir: Path) -> tuple[pd.DataFrame, float]:
    prediction_path = experiment_dir / "internal_test_predictions.csv"
    run_path = experiment_dir / "run.json"
    if not prediction_path.is_file() or not run_path.is_file():
        raise SystemExit(
            f"Expected {prediction_path} and {run_path}; run a completed experiment first."
        )
    predictions = pd.read_csv(prediction_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    threshold = run.get("threshold")
    if not isinstance(threshold, (float, int)):
        raise SystemExit(f"{run_path} has no validation-selected threshold")
    if "leakage_group" not in predictions.columns:
        raise SystemExit(
            "Prediction CSV has no leakage_group. Re-run a controlled-protocol experiment; "
            "row-level bootstrap is not an acceptable substitute."
        )
    return predictions, float(threshold)


def generator_slices(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Evaluate each synthetic generator against its real siblings in the same groups."""
    rows: list[dict[str, object]] = []
    fakes = predictions[predictions["label"] == 1]
    for generator in sorted(fakes["generator"].unique()):
        fake = fakes[fakes["generator"] == generator]
        groups = set(fake["leakage_group"])
        real = predictions[
            (predictions["label"] == 0) & predictions["leakage_group"].isin(groups)
        ]
        subset = pd.concat([real, fake], ignore_index=True)
        metrics = binary_metrics(
            subset["label"].to_numpy(), subset["ai_score"].to_numpy(), threshold
        )
        rows.append({"generator": generator, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Completed matched experiment; reports experiment minus this experiment.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    predictions, threshold = load_experiment(args.experiment_dir)
    output = args.output_dir or args.experiment_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    overall = binary_metrics(
        predictions["label"].to_numpy(), predictions["ai_score"].to_numpy(), threshold
    )
    ranking = paired_group_ranking_accuracy(predictions)
    intervals = cluster_bootstrap_intervals(
        predictions,
        threshold,
        metrics=DEFAULT_CLUSTER_METRICS,
        seed=args.seed,
        repeats=args.bootstrap_repeats,
    )
    per_generator = generator_slices(predictions, threshold)
    per_generator.to_csv(output / "per_generator_metrics.csv", index=False)
    macro = {
        f"generator_macro_{metric}": float(per_generator[metric].mean())
        for metric in DEFAULT_CLUSTER_METRICS
    }
    worst = {
        f"generator_worst_{metric}": float(per_generator[metric].min())
        for metric in DEFAULT_CLUSTER_METRICS
    }
    summary: dict[str, object] = {
        "status": "exploratory_internal_stress_test",
        "experiment": args.experiment_dir.name,
        "threshold_selected_on": "validation",
        "metrics": overall,
        "paired_group_ranking": ranking,
        "generator_summary": {**macro, **worst},
        "bootstrap": {
            "unit": "leakage_group",
            "repeats_requested": args.bootstrap_repeats,
            "confidence": 0.95,
            "metrics": intervals,
        },
    }
    save_json(summary, output / "analysis_summary.json")
    if args.compare_to:
        reference, reference_threshold = load_experiment(args.compare_to)
        comparison = paired_cluster_bootstrap_difference(
            predictions,
            reference,
            left_threshold=threshold,
            right_threshold=reference_threshold,
            metrics=DEFAULT_CLUSTER_METRICS,
            seed=args.seed,
            repeats=args.bootstrap_repeats,
        )
        save_json(
            {
                "status": "exploratory_internal_stress_test",
                "difference": f"{args.experiment_dir.name} minus {args.compare_to.name}",
                "bootstrap_unit": "leakage_group",
                "metrics": comparison,
            },
            output / "paired_comparison.json",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
