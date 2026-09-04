"""Build a DANI source-file manifest restricted to metadata shortcut diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_image_detector.dani_shortcut import build_source_shortcut_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--canonical-training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_source_shortcut_manifest(
        args.source_dir,
        args.canonical_training_manifest,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
