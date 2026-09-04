from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_report_figures.py"
SPEC = importlib.util.spec_from_file_location("build_report_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_report_figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_report_figures)


def test_validation_example_selection_is_complete_and_deterministic(tmp_path: Path) -> None:
    rows = []
    for parent_id in (20, 10):
        for cell in build_report_figures.EXPECTED_CELLS:
            rows.append(
                {
                    "split": "val",
                    "parent_group": f"coco-parent:{parent_id}",
                    "parent_coco_image_id": parent_id,
                    "cell": cell,
                    "path": f"missing-{parent_id}-{cell}.png",
                }
            )
    manifest = pd.DataFrame(rows)
    selected = build_report_figures.select_validation_parent(
        manifest, tmp_path, require_files=False
    )
    assert selected["parent_coco_image_id"].eq(10).all()
    assert selected["cell"].tolist() == list(build_report_figures.EXPECTED_CELLS)


def test_fft_log_magnitude_is_finite_and_normalised() -> None:
    pixels = np.tile(np.arange(32, dtype=np.uint8), (32, 1)) * 8
    image = Image.fromarray(pixels, mode="L")
    spectrum = build_report_figures.fft_log_magnitude(image, image_size=32)
    assert spectrum.shape == (32, 32)
    assert np.isfinite(spectrum).all()
    assert float(spectrum.min()) >= 0.0
    assert float(spectrum.max()) <= 1.0


def test_clean_metric_row_requires_one_aggregate_row() -> None:
    metrics = pd.DataFrame(
        [
            {"condition": "clean", "scope": "all", "tn": 2, "fp": 1, "fn": 1, "tp": 4},
            {
                "condition": "clean",
                "scope": "real_vs_generator",
                "tn": 2,
                "fp": 1,
                "fn": 0,
                "tp": 2,
            },
        ]
    )
    row = build_report_figures.clean_metric_row(metrics)
    assert int(row["tp"]) == 4
