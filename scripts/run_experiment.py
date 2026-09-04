#!/usr/bin/env python3
"""Train one predeclared RGB or FFT experiment and save every result artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ai_image_detector.features import (
    CONTROLLED_PREPROCESSING_PROTOCOL,
    LEGACY_PREPROCESSING_PROTOCOL,
    FFTTransform,
    RGBTransform,
    preprocessing_metadata,
)
from ai_image_detector.manifest import load_manifest, split_overlap_report
from ai_image_detector.models import build_resnet50, trainable_parameter_count
from ai_image_detector.reproducibility import (
    environment_snapshot,
    get_device,
    save_json,
    seed_everything,
    sha256_file,
)
from ai_image_detector.training import (
    LEGACY_LABEL_WEIGHTED_SAMPLER,
    PAIRED_GROUP_BALANCED_SAMPLER,
    TrainConfig,
    evaluate_and_save,
    fit,
    make_loader,
    resolve_group_column,
    train_sampler_metadata,
)

MODEL_ARCHITECTURE = "resnet50"

# `group_id` is retained alongside the derived `leakage_group`: the latter is the connected
# component used for the grouped split, while the former is one of the original linking keys that
# must also remain partition-disjoint. Checking both makes a malformed or stale component column
# unable to silently bypass the launch gate.
SPLIT_ISOLATION_REQUIRED_COLUMNS = (
    "leakage_group",
    "group_id",
    "source_id",
    "sha256",
    "phash",
)
SPLIT_ISOLATION_OPTIONAL_COLUMNS = ("caption",)


def require_fresh_output_dir(output_dir: Path) -> None:
    """Refuse to replace an archived training artifact with a notebook rerun."""
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment artifact: {output_dir}. "
            "Choose a new --output-dir for a deliberate reproduction."
        )


def _blank_identifier_mask(values: pd.Series) -> pd.Series:
    """Return values that cannot be used to prove isolation for a required key."""
    return values.isna() | values.astype(str).str.strip().eq("")


def _overlap_description(key: str, report: pd.DataFrame) -> str:
    """Give a bounded, actionable summary without writing an unaudited artifact."""
    leaked_values = report[key].drop_duplicates().tolist()
    examples: list[str] = []
    for value in leaked_values[:3]:
        splits = sorted(report.loc[report[key] == value, "split"].astype(str).unique())
        examples.append(f"{value!r} in {splits}")
    rendered_examples = "; ".join(examples)
    return (
        f"{key}: {len(leaked_values)} shared value(s) across {len(report)} row(s)"
        f" (for example, {rendered_examples})"
    )


def validate_split_isolation(frame: pd.DataFrame) -> None:
    """Fail closed unless every exact leakage key is isolated to one manifest split.

    This guard deliberately checks the raw linking fields in addition to `leakage_group` before a
    model or artifact directory can be written. Captions are optional in supported manifests, but
    when supplied and nonblank, an exact repeated caption is still a split-isolation violation.
    """
    missing = [column for column in SPLIT_ISOLATION_REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            "Cannot verify manifest split isolation because required column(s) are missing: "
            f"{missing}. Use a leakage-audited grouped manifest with leakage_group, group_id, "
            "source_id, sha256, and phash; refusing to begin training or write model artifacts."
        )

    blank_columns = [
        column
        for column in SPLIT_ISOLATION_REQUIRED_COLUMNS
        if _blank_identifier_mask(frame[column]).any()
    ]
    if blank_columns:
        raise ValueError(
            "Cannot verify manifest split isolation because required key column(s) contain "
            f"blank values: {blank_columns}. Repair or regenerate the grouped manifest; refusing "
            "to begin training or write model artifacts."
        )

    keys = (*SPLIT_ISOLATION_REQUIRED_COLUMNS,)
    keys += tuple(column for column in SPLIT_ISOLATION_OPTIONAL_COLUMNS if column in frame)
    overlaps: dict[str, pd.DataFrame] = {}
    for key in keys:
        candidates = frame
        if key in SPLIT_ISOLATION_OPTIONAL_COLUMNS:
            candidates = frame.loc[~_blank_identifier_mask(frame[key])]
        report = split_overlap_report(candidates, key)
        if not report.empty:
            overlaps[key] = report
    if overlaps:
        details = "; ".join(
            _overlap_description(key, report) for key, report in overlaps.items()
        )
        raise ValueError(
            "Manifest failed the split-isolation gate: cross-split overlap(s) detected in "
            f"{details}. Rebuild the grouped manifest so every connected leakage component is "
            "assigned to exactly one split; refusing to begin training or write model artifacts."
        )


def manifest_path_and_hash_at_launch(manifest_path: Path) -> tuple[Path, str]:
    """Capture the immutable manifest identity before a long launch reads image pixels."""
    resolved_path = manifest_path.resolve()
    return resolved_path, sha256_file(resolved_path)


def manifest_launch_metadata(
    manifest_path: Path, frame: pd.DataFrame, manifest_sha256: str | None = None
) -> dict[str, object]:
    """Describe the immutable input manifest used by an experiment launch."""
    split_counts = {
        str(split): int(count)
        for split, count in frame["split"].value_counts(sort=False).sort_index().items()
    }
    return {
        "resolved_path": str(manifest_path.resolve()),
        "manifest_sha256": (
            sha256_file(manifest_path) if manifest_sha256 is None else manifest_sha256
        ),
        "row_counts": {"total": len(frame), "by_split": split_counts},
    }


def requested_launch_options(args: argparse.Namespace) -> dict[str, object]:
    """Return the CLI values requested before protocol defaults are resolved."""
    return {
        "representation": args.representation,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "patience": int(args.patience),
        "preprocessing_protocol": args.preprocessing_protocol,
        "train_sampler": args.train_sampler,
        "paired_group_column": args.paired_group_column,
        "from_scratch": bool(args.from_scratch),
        "robust_augmentation": bool(args.robust_augmentation),
        "device": args.device,
    }


def build_model_launch_metadata(
    *,
    manifest: dict[str, object],
    requested_options: dict[str, object],
    resolved_train_sampler: str,
    resolved_group_column: str | None,
    resolved_device: str,
    environment_at_launch: dict[str, Any],
    preprocessing: dict[str, object],
    train_sampler: dict[str, object],
    trainable_parameters: int,
) -> dict[str, Any]:
    """Build JSON-safe launch provenance without coupling it to training side effects."""
    pretrained = not bool(requested_options["from_scratch"])
    return {
        # Retain existing top-level facts so older local tooling can still read model.json.
        "device": resolved_device,
        "environment_at_launch": environment_at_launch,
        "trainable_parameters": int(trainable_parameters),
        "preprocessing": preprocessing,
        "train_sampler": train_sampler,
        # New launch contract: requested CLI values and the resolved protocol are both explicit.
        "manifest": manifest,
        "launch_options": {
            "requested": requested_options,
            "resolved": {
                "train_sampler": resolved_train_sampler,
                "paired_group_column": resolved_group_column,
                "device": resolved_device,
            },
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "pretrained": pretrained,
            "trainable_parameters": int(trainable_parameters),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--representation", choices=("rgb", "fft"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--robust-augmentation", action="store_true")
    parser.add_argument(
        "--preprocessing-protocol",
        choices=(CONTROLLED_PREPROCESSING_PROTOCOL, LEGACY_PREPROCESSING_PROTOCOL),
        default=CONTROLLED_PREPROCESSING_PROTOCOL,
        help=(
            "Controlled H1-N square-crop rasterization is the default. Select the legacy mode "
            "only to reproduce the pre-H1-N baseline."
        ),
    )
    parser.add_argument(
        "--train-sampler",
        choices=(PAIRED_GROUP_BALANCED_SAMPLER, LEGACY_LABEL_WEIGHTED_SAMPLER),
        default=None,
        help="Override the protocol's default sampler for a documented ablation.",
    )
    parser.add_argument(
        "--paired-group-column",
        default="leakage_group",
        help=(
            "Leakage-free group identifier used by the controlled paired sampler; the default "
            "falls back to group_id for compatible legacy manifests."
        ),
    )
    args = parser.parse_args()
    require_fresh_output_dir(args.output_dir)
    # Capture this before loading data or building a model: a long run must not report a later
    # repository/environment state simply because it finished after local files changed.
    environment_at_launch = environment_snapshot()
    launch_options = requested_launch_options(args)
    manifest_path, manifest_sha256 = manifest_path_and_hash_at_launch(args.manifest)
    seed_everything(args.seed)
    frame = load_manifest(manifest_path, check_paths=True)
    if sha256_file(manifest_path) != manifest_sha256:
        raise SystemExit("Manifest changed while the experiment launch was starting; rerun it.")
    required = {"train", "val", "test"}
    missing = required.difference(frame["split"])
    if missing:
        raise SystemExit(f"Manifest lacks required splits: {sorted(missing)}")
    validate_split_isolation(frame)
    manifest = manifest_launch_metadata(manifest_path, frame, manifest_sha256)
    preprocessing = preprocessing_metadata(args.preprocessing_protocol)
    train_sampler = args.train_sampler
    if train_sampler is None:
        train_sampler = (
            PAIRED_GROUP_BALANCED_SAMPLER
            if args.preprocessing_protocol == CONTROLLED_PREPROCESSING_PROTOCOL
            else LEGACY_LABEL_WEIGHTED_SAMPLER
        )
    paired_group_column = (
        resolve_group_column(frame, args.paired_group_column)
        if train_sampler == PAIRED_GROUP_BALANCED_SAMPLER
        else None
    )
    image_size = int(preprocessing["image_size"])
    if args.representation == "rgb":
        train_transform = RGBTransform(
            size=image_size,
            train=True,
            robust_augmentation=args.robust_augmentation,
            preprocessing_protocol=args.preprocessing_protocol,
        )
        eval_transform = RGBTransform(
            size=image_size, train=False, preprocessing_protocol=args.preprocessing_protocol
        )
    else:
        train_transform = FFTTransform(
            size=image_size,
            train=True,
            robust_augmentation=args.robust_augmentation,
            preprocessing_protocol=args.preprocessing_protocol,
        )
        eval_transform = FFTTransform(
            size=image_size, train=False, preprocessing_protocol=args.preprocessing_protocol
        )
    config = TrainConfig(
        experiment_name=args.output_dir.name,
        representation=args.representation,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        preprocessing_protocol=str(preprocessing["protocol"]),
        preprocessing_version=str(preprocessing["version"]),
        image_size=image_size,
        train_sampler=train_sampler,
        paired_group_column=paired_group_column,
    )
    train_loader = make_loader(
        frame[frame.split == "train"],
        train_transform,
        args.batch_size,
        train=True,
        sampler_protocol=train_sampler,
        seed=args.seed,
        group_column=paired_group_column or args.paired_group_column,
    )
    val_loader = make_loader(
        frame[frame.split == "val"], eval_transform, args.batch_size, train=False, seed=args.seed
    )
    test_loader = make_loader(
        frame[frame.split == "test"], eval_transform, args.batch_size, train=False, seed=args.seed
    )
    device = get_device(args.device)
    model = build_resnet50(pretrained=not args.from_scratch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler_metadata = train_sampler_metadata(train_loader)
    save_json(
        build_model_launch_metadata(
            manifest=manifest,
            requested_options=launch_options,
            resolved_train_sampler=train_sampler,
            resolved_group_column=paired_group_column,
            resolved_device=str(device),
            environment_at_launch=environment_at_launch,
            preprocessing=preprocessing,
            train_sampler=sampler_metadata,
            trainable_parameters=trainable_parameter_count(model),
        ),
        args.output_dir / "model.json",
    )
    model, _, threshold = fit(
        model,
        train_loader,
        val_loader,
        config,
        device,
        args.output_dir,
        environment_at_launch=environment_at_launch,
    )
    _, metrics = evaluate_and_save(
        model, test_loader, device, threshold, args.output_dir, "internal_test"
    )
    print(metrics)


if __name__ == "__main__":
    main()
