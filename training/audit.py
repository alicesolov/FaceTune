"""Dataset audit utilities for class balance, duplicate detection, and coverage checks.

Run before training to validate that the dataset is suitable for the
editorial image authenticity screening task.

Usage:
    python -m training.audit --split train
    python -m training.audit --split validation --cache-dir /data/hf_cache
    python -m training.audit --all-splits
    python -m training.audit --all-splits --cache-dir /data/hf_cache
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

from training.dataset import DatasetConfig, load_defactify_split, validate_dataset_columns


# ── Individual audit checks ───────────────────────────────────────────────────

def audit_class_balance(dataset) -> dict:
    """Measure the real vs ai_generated class split.

    Purpose:
    Detect severe class imbalance that would require oversampling, undersampling,
    or a weighted loss during training.

    Inputs:
    - dataset: Loaded Hugging Face dataset split.

    Outputs:
    - Dictionary with total, per-class counts, percentages, and imbalance ratio.

    Key assumptions:
    - Label_A is 0 (real) or 1 (ai_generated).

    Failure modes:
    - Returns imbalance_ratio=inf if one class is absent.
    """
    labels = [int(item["Label_A"]) for item in dataset]
    counts = Counter(labels)
    total = len(labels)
    real = counts.get(0, 0)
    ai_gen = counts.get(1, 0)
    minority = min(real, ai_gen) if min(real, ai_gen) > 0 else None
    return {
        "total": total,
        "real": real,
        "ai_generated": ai_gen,
        "real_pct": real / total * 100 if total else 0.0,
        "ai_pct": ai_gen / total * 100 if total else 0.0,
        "imbalance_ratio": max(real, ai_gen) / minority if minority else float("inf"),
    }


def audit_duplicates_by_caption(dataset) -> dict:
    """Check for repeated captions that may indicate duplicate images.

    Purpose:
    Caption duplicates are a proxy for near-duplicate images. High duplication
    can cause train/test leakage or bias the model toward repeated content.

    Inputs:
    - dataset: Loaded Hugging Face dataset split.

    Outputs:
    - Dictionary with unique caption count, duplicate group count,
      and the 5 most-duplicated captions.

    Key assumptions:
    - The Caption field is a reliable proxy for image identity.
    - An empty Caption string is treated as a distinct value.

    Failure modes:
    - Returns zeros if the Caption column is absent or all-empty.
    """
    captions = [str(item.get("Caption", "")) for item in dataset]
    counter = Counter(captions)
    duplicates = {cap: n for cap, n in counter.items() if n > 1}
    return {
        "total": len(captions),
        "unique_captions": len(counter),
        "duplicate_groups": len(duplicates),
        "duplicate_rows": sum(duplicates.values()) - len(duplicates),
        "top5_duplicates": sorted(duplicates.items(), key=lambda x: -x[1])[:5],
    }


def audit_generator_distribution(dataset) -> dict[str, int]:
    """Report the Label_B generator-source distribution.

    Purpose:
    Understand which AI generators dominate the dataset, which matters for
    evaluating generalization to new generator families not seen in training.

    Inputs:
    - dataset: Loaded Hugging Face dataset split.

    Outputs:
    - Mapping from Label_B string value to row count, sorted descending.

    Key assumptions:
    - Label_B follows the documented mapping:
      0=Real, 1=SD21, 2=SDXL, 3=SD3, 4=DALLE3, 5=Midjourney.

    Failure modes:
    - Returns {"unknown": N} if Label_B is absent.
    """
    label_b_names = {
        "0": "Real",
        "1": "SD2.1",
        "2": "SDXL",
        "3": "SD3",
        "4": "DALL-E 3",
        "5": "Midjourney",
    }
    raw = [str(item.get("Label_B", "unknown")) for item in dataset]
    counter = Counter(raw)
    return {
        label_b_names.get(k, k): v
        for k, v in sorted(counter.items(), key=lambda x: -x[1])
    }


# ── Full audit ────────────────────────────────────────────────────────────────

def run_full_audit(split: str = "train", cache_dir: str | None = None) -> dict:
    """Load a dataset split and run all audit checks.

    Purpose:
    Produce a structured audit report before starting training. The report
    surfaces class imbalance, duplicate risk, and generator coverage so
    training decisions can be made with full dataset context.

    Inputs:
    - split: Dataset split name: 'train', 'validation', or 'test'.
    - cache_dir: Optional Hugging Face dataset cache directory.

    Outputs:
    - Dictionary containing split name, class_balance, duplicates,
      and generator_distribution sub-reports.

    Key assumptions:
    - Requires network access or a populated Hugging Face cache.

    Failure modes:
    - Raises ImportError if the datasets package is unavailable.
    - Propagates download errors when the cache is empty.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info("Loading '%s' split for audit …", split)
    config = DatasetConfig(split=split, cache_dir=cache_dir)
    dataset = load_defactify_split(config)
    validate_dataset_columns(dataset)
    logging.info("Loaded %d rows.", len(dataset))

    balance = audit_class_balance(dataset)
    logging.info(
        "Class balance → real=%d (%.1f%%)  ai_generated=%d (%.1f%%)  ratio=%.2fx",
        balance["real"], balance["real_pct"],
        balance["ai_generated"], balance["ai_pct"],
        balance["imbalance_ratio"],
    )
    if balance["imbalance_ratio"] > 1.5:
        logging.warning(
            "Imbalance ratio %.2fx exceeds 1.5 — "
            "class-balanced sampling is recommended during training.",
            balance["imbalance_ratio"],
        )

    duplicates = audit_duplicates_by_caption(dataset)
    logging.info(
        "Duplicates → %d groups  %d extra rows  (%d unique / %d total captions)",
        duplicates["duplicate_groups"],
        duplicates["duplicate_rows"],
        duplicates["unique_captions"],
        duplicates["total"],
    )
    if duplicates["duplicate_rows"] > 0:
        logging.warning(
            "Found %d duplicate-caption rows — inspect for train/test leakage.",
            duplicates["duplicate_rows"],
        )

    generators = audit_generator_distribution(dataset)
    logging.info("Generator distribution → %s", generators)

    return {
        "split": split,
        "class_balance": balance,
        "duplicates": duplicates,
        "generator_distribution": generators,
    }


# ── Cross-split leakage ───────────────────────────────────────────────────────

def audit_cross_split_leakage(splits: dict[str, object]) -> dict:
    """Detect captions that appear in more than one split.

    Purpose:
    Identify data leakage where the same (or near-duplicate) image appears in
    both train and test, which would inflate held-out metrics. Caption text is
    used as a proxy for image identity.

    Inputs:
    - splits: Mapping of split name → loaded Hugging Face Dataset.

    Outputs:
    - Dictionary with per-pair leakage counts and a combined leaked-caption set
      size. Keys: 'pairs' (dict of "A_vs_B" → count), 'total_leaked_captions'.

    Key assumptions:
    - Caption is a reliable proxy for image identity (same image → same caption).
    - An empty string caption is treated as a distinct value.

    Failure modes:
    - Returns zero counts when Caption column is absent.
    """
    caption_sets: dict[str, set[str]] = {
        name: {str(item.get("Caption", "")) for item in ds}
        for name, ds in splits.items()
    }

    pairs: dict[str, int] = {}
    all_leaked: set[str] = set()

    split_names = sorted(caption_sets)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = caption_sets[a] & caption_sets[b]
            pairs[f"{a}_vs_{b}"] = len(overlap)
            all_leaked |= overlap

    return {
        "pairs": pairs,
        "total_leaked_captions": len(all_leaked),
    }


# ── Full pipeline audit (all splits) ─────────────────────────────────────────

def run_pipeline_audit(cache_dir: str | None = None) -> dict:
    """Load all three splits and run a comprehensive cross-split audit.

    Purpose:
    Produce a single structured report covering class balance, duplicates,
    generator distribution, and cross-split leakage for every split. This
    must be run once before training to ensure evaluation is sound.

    Inputs:
    - cache_dir: Optional Hugging Face dataset cache directory.

    Outputs:
    - Dictionary keyed by split name (each containing the per-split report)
      plus a top-level 'cross_split_leakage' entry.

    Key assumptions:
    - Requires network access or a populated Hugging Face cache.

    Failure modes:
    - Raises ImportError if the datasets package is unavailable.
    - Propagates download errors when the cache is empty.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from training.dataset import SUPPORTED_SPLITS

    splits_data = {}
    per_split_reports: dict[str, dict] = {}

    for split_name in SUPPORTED_SPLITS:
        logging.info("══ Auditing '%s' split ══", split_name)
        config = DatasetConfig(split=split_name, cache_dir=cache_dir)
        ds = load_defactify_split(config)
        validate_dataset_columns(ds)
        logging.info("  Loaded %d rows.", len(ds))

        balance = audit_class_balance(ds)
        logging.info(
            "  Class balance → real=%d (%.1f%%)  ai_generated=%d (%.1f%%)  ratio=%.2fx",
            balance["real"], balance["real_pct"],
            balance["ai_generated"], balance["ai_pct"],
            balance["imbalance_ratio"],
        )
        if balance["imbalance_ratio"] > 1.5:
            logging.warning(
                "  Imbalance ratio %.2fx exceeds 1.5 — "
                "class-balanced sampling recommended.",
                balance["imbalance_ratio"],
            )

        duplicates = audit_duplicates_by_caption(ds)
        logging.info(
            "  Duplicates → %d groups  %d extra rows  (%d unique / %d total)",
            duplicates["duplicate_groups"],
            duplicates["duplicate_rows"],
            duplicates["unique_captions"],
            duplicates["total"],
        )

        generators = audit_generator_distribution(ds)
        logging.info("  Generator distribution → %s", generators)

        per_split_reports[split_name] = {
            "num_rows": len(ds),
            "class_balance": balance,
            "duplicates": duplicates,
            "generator_distribution": generators,
        }
        splits_data[split_name] = ds

    logging.info("══ Cross-split leakage check ══")
    leakage = audit_cross_split_leakage(splits_data)
    for pair, count in leakage["pairs"].items():
        level = logging.WARNING if count > 0 else logging.INFO
        logging.log(level, "  %s → %d overlapping captions", pair, count)
    if leakage["total_leaked_captions"] > 0:
        logging.warning(
            "  TOTAL leaked captions across all pairs: %d — "
            "inspect for train/test contamination before training.",
            leakage["total_leaked_captions"],
        )
    else:
        logging.info("  No cross-split caption overlap detected.")

    return {
        **per_split_reports,
        "cross_split_leakage": leakage,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit a Defactify dataset split.")
    parser.add_argument("--split", default="train",
                        help="Split to audit: train | validation | test")
    parser.add_argument("--all-splits", action="store_true",
                        help="Audit all three splits and check cross-split leakage")
    parser.add_argument("--cache-dir", default=None,
                        help="Hugging Face dataset cache directory")
    args = parser.parse_args()

    if args.all_splits:
        report = run_pipeline_audit(args.cache_dir)
    else:
        report = run_full_audit(args.split, args.cache_dir)
    print(json.dumps(report, indent=2, default=str))
