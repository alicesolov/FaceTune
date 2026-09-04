"""Aggregate predeclared seed repeats without choosing a best test-set seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ai_image_detector.reproducibility import save_json

METRICS = ("roc_auc", "balanced_accuracy", "macro_f1", "fpr_at_tpr_95")


def load_one(experiment_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    run_path = experiment_dir / "run.json"
    summary_path = experiment_dir / "analysis" / "analysis_summary.json"
    if not run_path.is_file() or not summary_path.is_file():
        raise SystemExit(
            f"Expected completed run and analysis artifacts at {run_path} and {summary_path}."
        )
    return (
        json.loads(run_path.read_text(encoding="utf-8")),
        json.loads(summary_path.read_text(encoding="utf-8")),
    )


def run_signature(run: dict[str, object]) -> dict[str, object]:
    config = run.get("config")
    preprocessing = run.get("preprocessing")
    sampler = run.get("train_sampler")
    if not isinstance(config, dict) or not isinstance(preprocessing, dict) or not isinstance(sampler, dict):
        raise SystemExit("Every experiment must have current config/preprocessing/sampler metadata")
    return {
        "representation": config.get("representation"),
        "epochs": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "patience": config.get("patience"),
        "preprocessing_protocol": preprocessing.get("protocol"),
        "preprocessing_version": preprocessing.get("version"),
        "image_size": preprocessing.get("image_size"),
        "train_sampler": sampler.get("choice"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.experiment_dir) < 2:
        raise SystemExit("Aggregate at least two independently seeded experiments; never choose one best seed.")

    records: list[dict[str, object]] = []
    signature: dict[str, object] | None = None
    generator_frames: list[pd.DataFrame] = []
    for experiment_dir in args.experiment_dir:
        run, summary = load_one(experiment_dir)
        current_signature = run_signature(run)
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise SystemExit(
                "Experiments have incompatible controlled-protocol configuration: "
                f"expected {signature}, got {current_signature} for {experiment_dir}"
            )
        config = run["config"]
        metrics = summary.get("metrics")
        if not isinstance(config, dict) or not isinstance(metrics, dict):
            raise SystemExit(f"Malformed summary or config in {experiment_dir}")
        row = {
            "experiment": experiment_dir.name,
            "seed": config.get("seed"),
            **{metric: metrics.get(metric) for metric in METRICS},
        }
        records.append(row)
        per_generator_path = experiment_dir / "analysis" / "per_generator_metrics.csv"
        if per_generator_path.is_file():
            per_generator = pd.read_csv(per_generator_path)
            per_generator.insert(0, "seed", config.get("seed"))
            per_generator.insert(0, "experiment", experiment_dir.name)
            generator_frames.append(per_generator)

    per_seed = pd.DataFrame(records).sort_values("seed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "per_seed_internal_metrics.csv", index=False)
    aggregate = {
        metric: {
            "mean": float(pd.to_numeric(per_seed[metric], errors="coerce").mean()),
            "std": float(pd.to_numeric(per_seed[metric], errors="coerce").std(ddof=1)),
            "n_seeds": len(per_seed),
        }
        for metric in METRICS
    }
    payload: dict[str, object] = {
        "status": "exploratory_internal_stress_test",
        "selection_rule": "All predeclared seeds are aggregated; no best seed is selected.",
        "signature": signature,
        "experiments": per_seed.to_dict(orient="records"),
        "metrics": aggregate,
    }
    if generator_frames:
        combined_generators = pd.concat(generator_frames, ignore_index=True)
        combined_generators.to_csv(args.output_dir / "per_seed_generator_metrics.csv", index=False)
        numeric = [metric for metric in METRICS if metric in combined_generators]
        generator_aggregate = (
            combined_generators.groupby("generator", as_index=False)[numeric]
            .agg(["mean", "std"])
            .reset_index()
        )
        generator_aggregate.columns = [
            "_".join(part for part in column if part) if isinstance(column, tuple) else column
            for column in generator_aggregate.columns
        ]
        generator_aggregate.to_csv(args.output_dir / "generator_metric_mean_std.csv", index=False)
        payload["generator_aggregate_file"] = "generator_metric_mean_std.csv"
    save_json(payload, args.output_dir / "aggregate_summary.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
