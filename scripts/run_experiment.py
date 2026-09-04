#!/usr/bin/env python3
"""Train one predeclared RGB or FFT experiment and save every result artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_image_detector.features import (
    CONTROLLED_PREPROCESSING_PROTOCOL,
    LEGACY_PREPROCESSING_PROTOCOL,
    FFTTransform,
    RGBTransform,
    preprocessing_metadata,
)
from ai_image_detector.manifest import load_manifest
from ai_image_detector.models import build_resnet50, trainable_parameter_count
from ai_image_detector.reproducibility import get_device, save_json, seed_everything
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
    seed_everything(args.seed)
    frame = load_manifest(args.manifest, check_paths=True)
    required = {"train", "val", "test"}
    missing = required.difference(frame["split"])
    if missing:
        raise SystemExit(f"Manifest lacks required splits: {sorted(missing)}")
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
    save_json(
        {
            "device": str(device),
            "trainable_parameters": trainable_parameter_count(model),
            "preprocessing": preprocessing,
            "train_sampler": train_sampler_metadata(train_loader),
        },
        args.output_dir / "model.json",
    )
    model, _, threshold = fit(model, train_loader, val_loader, config, device, args.output_dir)
    _, metrics = evaluate_and_save(
        model, test_loader, device, threshold, args.output_dir, "internal_test"
    )
    print(metrics)


if __name__ == "__main__":
    main()
