"""Create a source-preserving 1024px RGB PNG derivative of audited DANI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_image_detector.dani_canonical import canonicalize_dani


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--source-audit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = canonicalize_dani(
        args.source_dir,
        args.source_audit_summary,
        args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
