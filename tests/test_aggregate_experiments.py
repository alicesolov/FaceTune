from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "aggregate_experiments.py"
SPEC = importlib.util.spec_from_file_location("aggregate_experiments", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
aggregate_experiments = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate_experiments)


def write_experiment(root: Path, seed: int, score: float) -> Path:
    experiment = root / f"seed{seed}"
    analysis = experiment / "analysis"
    analysis.mkdir(parents=True)
    run = {
        "config": {
            "representation": "rgb",
            "seed": seed,
            "epochs": 15,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "patience": 4,
        },
        "preprocessing": {"protocol": "h1n_square_crop_128_v1", "version": "1.0", "image_size": 128},
        "train_sampler": {"choice": "paired_group_balanced_v1"},
    }
    summary = {
        "metrics": {
            "roc_auc": score,
            "balanced_accuracy": score,
            "macro_f1": score,
            "fpr_at_tpr_95": 1.0 - score,
        }
    }
    (experiment / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (analysis / "analysis_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        {
            "generator": ["synthetic"],
            "roc_auc": [score],
            "balanced_accuracy": [score],
            "macro_f1": [score],
            "fpr_at_tpr_95": [1.0 - score],
        }
    ).to_csv(analysis / "per_generator_metrics.csv", index=False)
    return experiment


def test_aggregate_reports_all_seeds_without_best_seed_selection(
    tmp_path: Path, monkeypatch
) -> None:
    first = write_experiment(tmp_path, 7, 0.7)
    second = write_experiment(tmp_path, 17, 0.9)
    output = tmp_path / "aggregate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_experiments.py",
            "--experiment-dir",
            str(first),
            "--experiment-dir",
            str(second),
            "--output-dir",
            str(output),
        ],
    )

    aggregate_experiments.main()

    payload = json.loads((output / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["balanced_accuracy"]["mean"] == 0.8
    assert payload["metrics"]["balanced_accuracy"]["n_seeds"] == 2
    assert "no best seed" in payload["selection_rule"].lower()
    assert (output / "generator_metric_mean_std.csv").is_file()
