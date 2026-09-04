from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_predictions.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("analyze_predictions", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
analyze_predictions = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(analyze_predictions)


def test_analysis_summary_records_bootstrap_seed(tmp_path, monkeypatch) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "run.json").write_text('{"threshold": 0.5}\n', encoding="utf-8")
    pd.DataFrame(
        {
            "path": ["real-1", "fake-1", "real-2", "fake-2"],
            "label": [0, 1, 0, 1],
            "generator": ["real", "sd21", "real", "sd21"],
            "leakage_group": ["a", "a", "b", "b"],
            "ai_score": [0.1, 0.9, 0.2, 0.8],
        }
    ).to_csv(experiment / "internal_test_predictions.csv", index=False)
    output = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_predictions.py",
            "--experiment-dir",
            str(experiment),
            "--output-dir",
            str(output),
            "--bootstrap-repeats",
            "7",
            "--seed",
            "17",
        ],
    )

    analyze_predictions.main()

    summary = json.loads((output / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["bootstrap"]["seed"] == 17
    assert summary["bootstrap"]["repeats_requested"] == 7
