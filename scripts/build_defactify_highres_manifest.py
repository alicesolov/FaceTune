"""Build the canonical native-crop Defactify-384 corpus from pinned local source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_image_detector.highres_defactify import (
    MIN_SHORT_SIDE,
    PHASH_DISTANCE_THRESHOLD,
    TARGET_SIZE,
    build_highres_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Defactify source files, materialize a uniform native 384x384 RGB exploratory "
            "corpus, and preserve upstream roles after whole-component leakage checks."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/defactify_exploratory_native384_v2"),
        help=(
            "A new ignored directory for the Defactify exploratory corpus, its frozen manifest, "
            "and audit evidence."
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/raw/defactify/manifest.csv"),
    )
    parser.add_argument(
        "--source-provenance",
        type=Path,
        default=Path("data/raw/defactify/provenance.json"),
    )
    parser.add_argument(
        "--min-short-side",
        type=int,
        default=MIN_SHORT_SIDE,
        help="Native source-size gate. It cannot be below the fixed 384px crop.",
    )
    parser.add_argument(
        "--phash-distance-threshold",
        type=int,
        default=PHASH_DISTANCE_THRESHOLD,
        help=(
            "Over-inclusive cross-caption pHash candidate-link threshold used before preserving "
            "upstream split roles; it is not duplicate-identity evidence."
        ),
    )
    parser.add_argument("--crop-seed", type=int, default=20260829)
    parser.add_argument("--selection-seed", type=int, default=20260829)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_highres_manifest(
        args.output_dir,
        source_manifest=args.source_manifest,
        source_provenance=args.source_provenance,
        min_short_side=args.min_short_side,
        phash_distance_threshold=args.phash_distance_threshold,
        crop_size=TARGET_SIZE,
        crop_seed=args.crop_seed,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
