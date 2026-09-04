"""Freeze the metadata-only DANI HighRes-v1 selection before reading image bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_image_detector.dani_selection import DEFAULT_SELECTION_SEED, build_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lineage_scan_dir", type=Path)
    parser.add_argument("--lineage-audit-summary", type=Path, required=True)
    parser.add_argument("--coco-identity-summary", type=Path, required=True)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_selection(
        args.lineage_scan_dir,
        args.lineage_audit_summary,
        args.coco_identity_summary,
        args.annotations_zip,
        args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
