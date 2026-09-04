#!/usr/bin/env python3
"""Run deterministic provenance and leakage checks before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_image_detector.manifest import audit_summary, load_manifest, split_overlap_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audit"))
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()
    frame = load_manifest(args.manifest, check_paths=args.check_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = audit_summary(frame)
    for name, value in summary.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(args.output_dir / f"{name}.csv")
        else:
            (args.output_dir / f"{name}.txt").write_text(f"{value}\n", encoding="utf-8")
    for key in ("source_id", "group_id", "caption", "phash"):
        if key in frame.columns:
            split_overlap_report(frame, key).to_csv(
                args.output_dir / f"cross_split_{key}.csv", index=False
            )
    print(f"Wrote audit artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
