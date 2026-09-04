"""Build a path-only DANI lineage candidate catalogue without initialising Torch.

The scanner uses Arrow nested projection for image.path only. It never requests image.bytes or
the parent image field, so its output remains a blocked mapping-audit input rather than a corpus.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# This metadata-only process must not initialise Torch or its shared-memory machinery.
os.environ.setdefault("USE_TORCH", "0")

from ai_image_detector import dani, dani_lineage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project DANI image.path and non-binary metadata only. The output remains blocked "
            "from corpus selection pending an upstream mapping audit."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/dani_lineage_metadata"),
        help="A new directory for source_lock.json, lineage_catalog.csv, and provenance.json.",
    )
    parser.add_argument(
        "--revision",
        default=dani.PINNED_REVISION,
        help="Dataset ref to resolve to an immutable Hugging Face commit SHA before scanning.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help=(
            "Scan only the first N sorted shards as a partial scout. The result is ineligible "
            "for even the lineage audit and never becomes a training manifest."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Rows per Arrow metadata batch; image.bytes stays excluded at every batch.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=dani_lineage.LINEAGE_FSSPEC_BLOCK_SIZE,
        help=(
            "Bytes per explicit fsspec HTTP range request. The selected value is recorded in "
            "source_lock.json and provenance.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = dani_lineage.scan_lineage_metadata(
        args.output_dir,
        revision=args.revision,
        limit_shards=args.limit_shards,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
