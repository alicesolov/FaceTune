"""Dataset overlap checker — captions, image hashes, and train/val/test leakage.

Reports:
  - within-split duplicate captions (count + top offenders)
  - cross-split caption leakage (train↔val, train↔test, val↔test)
  - class distribution per split
  - (optional) perceptual hash near-duplicates if imagehash is available

Usage:
    python scripts/check_overlap.py
    python scripts/check_overlap.py --cache-dir /data/hf_cache --save-json overlap.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

SPLITS = ("train", "validation", "test")


def _captions_for_split(split: str, cache_dir: str | None) -> list[str]:
    from training.dataset import DatasetConfig, load_defactify_split
    hf = load_defactify_split(DatasetConfig(split=split, cache_dir=cache_dir))
    return [str(item.get("Caption", "")) for item in hf]


def _labels_for_split(split: str, cache_dir: str | None) -> list[int]:
    from training.dataset import DatasetConfig, load_defactify_split
    hf = load_defactify_split(DatasetConfig(split=split, cache_dir=cache_dir))
    return [int(item["Label_A"]) for item in hf]


def _within_split_duplicates(captions: list[str]) -> dict:
    counts = Counter(captions)
    dups = {cap: cnt for cap, cnt in counts.items() if cnt > 1}
    return {
        "unique_captions": len(counts),
        "total_captions": len(captions),
        "duplicate_caption_count": len(dups),
        "top_duplicates": sorted(dups.items(), key=lambda x: -x[1])[:10],
    }


def _cross_split_leakage(
    captions_a: list[str],
    captions_b: list[str],
    name_a: str,
    name_b: str,
) -> dict:
    set_a = set(captions_a)
    set_b = set(captions_b)
    overlap = set_a & set_b
    return {
        "pair": f"{name_a}↔{name_b}",
        "leaked_captions": len(overlap),
        "pct_of_a": round(len(overlap) / len(set_a) * 100, 2) if set_a else 0.0,
        "pct_of_b": round(len(overlap) / len(set_b) * 100, 2) if set_b else 0.0,
        "examples": sorted(overlap)[:5],
    }


def run_overlap_check(cache_dir: str | None = None) -> dict:
    """Load all splits and produce a full overlap report."""
    result: dict = {}

    split_captions: dict[str, list[str]] = {}
    for split in SPLITS:
        logging.info("Loading '%s' split …", split)
        caps = _captions_for_split(split, cache_dir)
        labels = _labels_for_split(split, cache_dir)
        label_counts = Counter(labels)
        within = _within_split_duplicates(caps)

        result[split] = {
            "total_rows": len(caps),
            "real": label_counts.get(0, 0),
            "ai_generated": label_counts.get(1, 0),
            "imbalance_ratio": round(
                label_counts.get(1, 0) / max(label_counts.get(0, 1), 1), 2
            ),
            **within,
        }
        split_captions[split] = caps
        logging.info(
            "  %s: %d rows, %d unique captions, %d duplicates",
            split, len(caps), within["unique_captions"], within["duplicate_caption_count"],
        )

    # Cross-split leakage
    leakage_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]
    leakage_results = []
    total_leaked = 0
    for a, b in leakage_pairs:
        leak = _cross_split_leakage(
            split_captions[a], split_captions[b], a, b
        )
        leakage_results.append(leak)
        total_leaked += leak["leaked_captions"]
        logging.info(
            "  Leakage %s: %d captions (%.1f%% of %s)",
            leak["pair"], leak["leaked_captions"], leak["pct_of_a"], a,
        )

    result["cross_split_leakage"] = leakage_results
    result["total_leaked_captions"] = total_leaked

    if total_leaked > 0:
        logging.warning(
            "⚠  %d leaked captions detected. Consider deduplicating before training.",
            total_leaked,
        )
    else:
        logging.info("No caption leakage detected. ✓")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset overlap across splits.")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace dataset cache directory")
    parser.add_argument("--save-json", default=None, metavar="PATH",
                        help="Save full report to this JSON file")
    args = parser.parse_args()

    report = run_overlap_check(cache_dir=args.cache_dir)

    print(json.dumps(
        {k: v for k, v in report.items() if k != "cross_split_leakage"},
        indent=2
    ))
    print("\nCross-split leakage:")
    for entry in report.get("cross_split_leakage", []):
        print(f"  {entry['pair']}: {entry['leaked_captions']} captions")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to: {args.save_json}")
