"""Audit materialised DANI duplicates, leakage components, and simple shortcuts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_image_detector.dani_integrity import DEFAULT_PHASH_THRESHOLD, audit_integrity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materialized_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phash-threshold", type=int, default=DEFAULT_PHASH_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_integrity(
        args.materialized_dir,
        args.output_dir,
        phash_threshold=args.phash_threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
