"""Evaluate a frozen checkpoint under predeclared image transformations.

The checkpoint and validation-selected threshold are inputs.  This script cannot fit, tune or
replace either one, so the locked internal test remains an evaluation set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_image_detector.features import DegradedTransform
from ai_image_detector.inference import ModelBundle
from ai_image_detector.manifest import load_manifest
from ai_image_detector.metrics import binary_metrics
from ai_image_detector.reproducibility import save_json
from ai_image_detector.training import make_loader, predict

CONDITIONS = (
    ("clean", {}),
    ("jpeg_q95", {"kind": "jpeg", "jpeg_quality": 95}),
    ("jpeg_q75", {"kind": "jpeg", "jpeg_quality": 75}),
    ("jpeg_q50", {"kind": "jpeg", "jpeg_quality": 50}),
    ("resize_075", {"kind": "resize", "resize_scale": 0.75}),
    ("resize_050", {"kind": "resize", "resize_scale": 0.50}),
    ("gaussian_blur_r1", {"kind": "blur"}),
)


def condition_transform(base: object, parameters: dict[str, object]) -> object:
    if not parameters:
        return base
    kind = str(parameters.pop("kind"))
    return DegradedTransform(base, kind, **parameters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    bundle = ModelBundle.load(args.experiment_dir, device_name=args.device)
    threshold = bundle.threshold
    frame = load_manifest(args.manifest, check_paths=True)
    test = frame[frame["split"] == "test"].copy()
    if test.empty:
        raise SystemExit("Manifest has no test split")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, object]] = []
    for condition, raw_parameters in CONDITIONS:
        transform = condition_transform(bundle.transform, dict(raw_parameters))
        predictions = predict(
            bundle.model,
            make_loader(test, transform, args.batch_size, train=False),
            bundle.device,
        )
        predictions["condition"] = condition
        predictions.to_csv(args.output_dir / f"{condition}_predictions.csv", index=False)
        aggregate = binary_metrics(
            predictions.label.to_numpy(), predictions.ai_score.to_numpy(), threshold
        )
        result_rows.append({"condition": condition, "slice": "all", **aggregate})
        # A generator-specific comparison contains all real test images plus one fake generator,
        # rather than reporting a meaningless one-class score for the generator alone.
        for generator in sorted(predictions.loc[predictions.label == 1, "generator"].unique()):
            subset = predictions[(predictions.label == 0) | (predictions.generator == generator)]
            metrics = binary_metrics(subset.label.to_numpy(), subset.ai_score.to_numpy(), threshold)
            result_rows.append({"condition": condition, "slice": f"real_vs_{generator}", **metrics})
    results = pd.DataFrame(result_rows)
    results.to_csv(args.output_dir / "robustness_metrics.csv", index=False)
    save_json(
        {
            "threshold": threshold,
            "representation": bundle.representation,
            "preprocessing": bundle.preprocessing,
            "degradation_order": "after_common_raster_for_h1n",
            "conditions": [name for name, _ in CONDITIONS],
        },
        args.output_dir / "evaluation_config.json",
    )
    print(
        results[
            ["condition", "slice", "n", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
