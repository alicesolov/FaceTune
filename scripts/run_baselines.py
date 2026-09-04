"""Run predeclared non-neural controls on an audited, locked manifest."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from ai_image_detector.baselines import (
    file_metadata_predict,
    fit_file_metadata_logistic,
    fit_radial_logistic,
    radial_predict,
    radial_preprocessing_metadata,
)
from ai_image_detector.canonical_integrity import validate_defactify_exploratory_corpus
from ai_image_detector.features import (
    CONTROLLED_PREPROCESSING_PROTOCOL,
    HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL,
    LEGACY_PREPROCESSING_PROTOCOL,
)
from ai_image_detector.manifest import load_manifest
from ai_image_detector.metrics import binary_metrics, choose_threshold
from ai_image_detector.reproducibility import (
    environment_snapshot,
    save_json,
    seed_everything,
    sha256_file,
)

BASELINE_NAMES = ("radial_fft_logistic", "file_metadata_control")


def selected_baselines(only: str | None) -> tuple[str, ...]:
    """Resolve the CLI selection to the exact baselines this launch will run."""
    return (only,) if only is not None else BASELINE_NAMES


def output_dir_for_baseline(
    output_root: Path, name: str, seed: int, preprocessing_protocol: str
) -> Path:
    """Return the one artifact directory owned by a concrete baseline launch."""
    suffix = f"_{preprocessing_protocol}" if name == "radial_fft_logistic" else ""
    return output_root / f"{name}{suffix}_seed{seed}"


def require_fresh_output_dirs(
    output_root: Path, baselines: tuple[str, ...], seed: int, preprocessing_protocol: str
) -> dict[str, Path]:
    """Reserve new artifact locations before a baseline can modify disk state.

    A result directory is an archival record. Refusing an existing directory prevents a notebook
    rerun from quietly replacing metrics or provenance; a deliberate reproduction must choose a
    new output root instead.
    """
    outputs = {
        name: output_dir_for_baseline(output_root, name, seed, preprocessing_protocol)
        for name in baselines
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing baseline artifact(s): {rendered}. "
            "Choose a new --output-root for a deliberate reproduction."
        )
    return outputs


def manifest_path_and_hash_at_launch(manifest_path: Path) -> tuple[Path, str]:
    """Capture an immutable manifest identity before the manifest is read."""
    resolved_path = manifest_path.resolve()
    return resolved_path, sha256_file(resolved_path)


def manifest_launch_metadata(
    manifest_path: Path, manifest_sha256: str, frame: pd.DataFrame
) -> dict[str, object]:
    """Describe the exact manifest rows parsed for a baseline launch."""
    split_counts = {
        str(split): int(count)
        for split, count in frame["split"].value_counts(sort=False).sort_index().items()
    }
    return {
        "resolved_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "row_counts": {"total": len(frame), "by_split": split_counts},
    }


def requested_launch_options(args: argparse.Namespace) -> dict[str, object]:
    """Retain user-supplied CLI values before defaults are resolved."""
    return {
        "seed": int(args.seed),
        "selected_baseline": args.only,
        "preprocessing_protocol": args.preprocessing_protocol,
    }


def resolved_launch_options(
    requested_options: dict[str, object], baselines: tuple[str, ...]
) -> dict[str, object]:
    """Record the concrete execution choices after expanding CLI defaults."""
    return {
        "seed": requested_options["seed"],
        "selected_baselines": list(baselines),
        "preprocessing_protocol": requested_options["preprocessing_protocol"],
    }


def build_baseline_run_metadata(
    *,
    name: str,
    seed: int,
    threshold: float,
    preprocessing: dict[str, object] | None,
    manifest: dict[str, object],
    environment_at_launch: dict[str, Any],
    requested_options: dict[str, object],
    resolved_options: dict[str, object],
    canonical_corpus_integrity: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Build the existing run.json payload plus immutable launch provenance."""
    metadata: dict[str, Any] = {
        # Keep the established fields unchanged for existing analysis tooling.
        "name": name,
        "seed": seed,
        "threshold": threshold,
        "preprocessing": preprocessing,
        "role": "dataset_bias_control" if name == "file_metadata_control" else "pixel_baseline",
        # Capture these values before training/prediction starts; do not infer them at write time.
        "manifest": manifest,
        "environment_at_launch": environment_at_launch,
        "launch_options": {
            "requested": requested_options,
            "resolved": resolved_options,
        },
    }
    if canonical_corpus_integrity is not None:
        metadata["canonical_corpus_integrity"] = canonical_corpus_integrity
    return metadata


def run_one(
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output: Path,
    seed: int,
    preprocessing_protocol: str,
    manifest: dict[str, object],
    environment_at_launch: dict[str, Any],
    requested_options: dict[str, object],
    resolved_options: dict[str, object],
    canonical_corpus_integrity: dict[str, object] | None = None,
) -> None:
    if name == "radial_fft_logistic":
        model = fit_radial_logistic(train, seed=seed, preprocessing_protocol=preprocessing_protocol)
        predict = partial(radial_predict, preprocessing_protocol=preprocessing_protocol)
        run_preprocessing: dict[str, object] | None = radial_preprocessing_metadata(
            preprocessing_protocol
        )
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
        build_baseline_run_metadata(
            name=name,
            seed=seed,
            threshold=threshold,
            preprocessing=run_preprocessing,
            manifest=manifest,
            environment_at_launch=environment_at_launch,
            requested_options=requested_options,
            resolved_options=resolved_options,
            canonical_corpus_integrity=canonical_corpus_integrity,
        ),
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
        choices=(
            CONTROLLED_PREPROCESSING_PROTOCOL,
            HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL,
            LEGACY_PREPROCESSING_PROTOCOL,
        ),
        default=CONTROLLED_PREPROCESSING_PROTOCOL,
        help=(
            "Use legacy mode only to reproduce D0; H1-N controls default to source-normalised "
            "rasterisation and Defactify-HR uses its frozen canonical 384px corpus."
        ),
    )
    args = parser.parse_args()
    baselines = selected_baselines(args.only)
    outputs = require_fresh_output_dirs(
        args.output_root, baselines, args.seed, args.preprocessing_protocol
    )
    # Capture process and input identities before the manifest or any image pixels are processed.
    environment_at_launch = environment_snapshot()
    requested_options = requested_launch_options(args)
    manifest_path, manifest_sha256 = manifest_path_and_hash_at_launch(args.manifest)
    seed_everything(args.seed)
    frame = load_manifest(manifest_path, check_paths=True)
    if sha256_file(manifest_path) != manifest_sha256:
        raise SystemExit("Manifest changed while the baseline launch was starting; rerun it.")
    canonical_corpus_integrity = (
        validate_defactify_exploratory_corpus(manifest_path, frame)
        if args.preprocessing_protocol == HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL
        else None
    )
    manifest = manifest_launch_metadata(manifest_path, manifest_sha256, frame)
    resolved_options = resolved_launch_options(requested_options, baselines)
    train, val, test = (frame[frame.split == split].copy() for split in ("train", "val", "test"))
    for name in baselines:
        run_one(
            name,
            train,
            val,
            test,
            outputs[name],
            args.seed,
            args.preprocessing_protocol,
            manifest,
            environment_at_launch,
            requested_options,
            resolved_options,
            canonical_corpus_integrity,
        )


if __name__ == "__main__":
    main()
