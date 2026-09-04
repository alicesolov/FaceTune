"""Freeze the bounded five-cell DANI HighRes core before downloading shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_image_detector.dani_core import build_core_plan
from ai_image_detector.dani_selection import MATERIALISATION_BYTE_BUDGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preselection_dir", type=Path)
    parser.add_argument("--lineage-scan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--byte-cap", type=int, default=MATERIALISATION_BYTE_BUDGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_core_plan(
        args.preselection_dir,
        args.lineage_scan_dir,
        args.output_dir,
        byte_cap=args.byte_cap,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
