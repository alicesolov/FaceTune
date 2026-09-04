"""Download and extract the frozen DANI core under its hard byte cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_image_detector.dani_materialize import materialize_core
from ai_image_detector.dani_selection import MATERIALISATION_BYTE_BUDGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core_dir", type=Path)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--byte-cap", type=int, default=MATERIALISATION_BYTE_BUDGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def progress(message: str) -> None:
        print(message, flush=True)

    report = materialize_core(
        args.core_dir,
        args.staging_dir,
        args.output_dir,
        byte_cap=args.byte_cap,
        progress=progress,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
