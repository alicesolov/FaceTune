"""Create a metadata-only CommunityForensics catalogue without initialising Torch.

The process-local ``USE_TORCH=0`` setting is established before importing the scanner module. It
does not modify the environment used by model training or any other project command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Metadata-only Parquet streaming must not initialise Torch or its shared-memory machinery.
os.environ.setdefault("USE_TORCH", "0")
# Keep all Hugging Face lock/cache state inside the ignored project artifact directory. This must
# happen before importing the scanner, because the Hub and datasets packages read these settings
# during import.
DEFAULT_CACHE_DIR = Path("artifacts/cache/huggingface")
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_CACHE_DIR / "datasets"))

from ai_image_detector.community_forensics import PINNED_REVISION, scan_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Column-scan CommunityForensics-Small metadata only; image_data is never requested. "
            "This scanner process sets USE_TORCH=0 before importing datasets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/community_forensics_small_metadata"),
        help="A new directory for source_lock.json, source_catalog.csv, and provenance.json.",
    )
    parser.add_argument(
        "--revision",
        default=PINNED_REVISION,
        help="Dataset ref to resolve to an immutable Hugging Face commit SHA before scanning.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help=(
            "Scan only the first N sorted shards as a partial scout. The resulting catalogue is "
            "explicitly ineligible for candidate selection or training."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ["HF_DATASETS_CACHE"]),
        help=(
            "Ignored local cache for metadata-streaming locks. The default remains inside "
            "artifacts/cache/huggingface/datasets."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = scan_metadata(
        args.output_dir,
        revision=args.revision,
        limit_shards=args.limit_shards,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
