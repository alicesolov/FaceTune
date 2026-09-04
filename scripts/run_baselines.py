"""Run predeclared non-neural controls on an audited, locked manifest."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import pandas as pd

from ai_image_detector.baselines import (
    file_metadata_predict,
    fit_file_metadata_logistic,
    fit_radial_logistic,
    radial_predict,
)
from ai_image_detector.features import (
    CONTROLLED_PREPROCESSING_PROTOCOL,
    LEGACY_PREPROCESSING_PROTOCOL,
    preprocessing_metadata,
)
from ai_image_detector.manifest import load_manifest
from ai_image_detector.metrics import binary_metrics, choose_threshold
from ai_image_detector.reproducibility import save_json


def run_one(
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output: Path,
    seed: int,
    preprocessing_protocol: str,
) -> None:
    if name == "radial_fft_logistic":
        model = fit_radial_logistic(
            train, seed=seed, preprocessing_protocol=preprocessing_protocol
        )
        predict = partial(radial_predict, preprocessing_protocol=preprocessing_protocol)
        run_preprocessing: dict[str, object] | None = preprocessing_metadata(preprocessing_protocol)
    elif name == "file_metadata_control":
        model = fit_file_metadata_logistic(train, seed=seed)
        predict = file_metadata_predict
        run_preprocessing = None
    else:
        raise ValueError(name)
    validation_scores = predict(model, val)
    threshold = choose_threshold(val.label.to_numpy(), validation_scores)
    test_scores = predict(model, test)
    metrics = binary_metrics(test.label.to_numpy(), test_scores, threshold)
    output.mkdir(parents=True, exist_ok=True)
    prediction_columns: dict[str, object] = {
        "path": test.path,
        "label": test.label,
        "generator": test.generator,
        "split": test.split,
        "ai_score": test_scores,
    }
    for column in ("source_id", "group_id", "leakage_group"):
        if column in test:
            prediction_columns[column] = test[column]
    pd.DataFrame(prediction_columns).to_csv(output / "internal_test_predictions.csv", index=False)
    save_json(
        {
            "name": name,
            "seed": seed,
            "threshold": threshold,
            "preprocessing": run_preprocessing,
            "role": "dataset_bias_control" if name == "file_metadata_control" else "pixel_baseline",
        },
        output / "run.json",
    )
    (output / "internal_test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"name": name, **metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", choices=("radial_fft_logistic", "file_metadata_control"))
    parser.add_argument(
        "--preprocessing-protocol",
        choices=(CONTROLLED_PREPROCESSING_PROTOCOL, LEGACY_PREPROCESSING_PROTOCOL),
        default=CONTROLLED_PREPROCESSING_PROTOCOL,
        help="Use legacy mode only to reproduce D0; H1-N controls default to source-normalised rasterisation.",
    )
    args = parser.parse_args()
    frame = load_manifest(args.manifest, check_paths=True)
    train, val, test = (frame[frame.split == split].copy() for split in ("train", "val", "test"))
    for name in ("radial_fft_logistic", "file_metadata_control"):
        if args.only is None or args.only == name:
            suffix = (
                f"_{args.preprocessing_protocol}" if name == "radial_fft_logistic" else ""
            )
            run_one(
                name,
                train,
                val,
                test,
                args.output_root / f"{name}{suffix}_seed{args.seed}",
                args.seed,
                args.preprocessing_protocol,
            )


if __name__ == "__main__":
    main()
