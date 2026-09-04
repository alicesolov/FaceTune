"""Deterministic, metadata-only DANI HighRes-v1 selection.

The selection is frozen before image bytes are requested. It keeps one caption-matched real row
and four synthetic model/protocol rows per official COCO parent, assigns whole parents to one split,
and remains ineligible for training until the selected bytes pass the materialisation audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from . import dani, dani_lineage

SELECTION_SCHEMA_VERSION: Final = "dani_highres_selection_v1"
SELECTION_CATALOG_NAME: Final = "selection_catalog.csv"
SELECTION_SPEC_NAME: Final = "selection_spec.json"
SELECTION_PROVENANCE_NAME: Final = "provenance.json"
LINEAGE_AUDIT_SCHEMA: Final = "dani_lineage_mapping_audit_v1"
COCO_IDENTITY_SCHEMA: Final = "dani_coco_identity_audit_v1"
DEFAULT_SELECTION_SEED: Final = 20_260_830
DECLARED_SOURCE_SIZE: Final = 1024
MATERIALISATION_BYTE_BUDGET: Final = 25_000_000_000
ALLOWED_COCO_LICENSE_IDS: Final = (2, 4)
SPLIT_RATIOS: Final = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_ORDER: Final = {"train": 0, "val": 1, "test": 2}
CELL_DEFINITIONS: Final = {
    "real_coco": ("0", "COCO", "T2I"),
    "fake_dalle3_t2i": ("1", "Dalle3", "T2I"),
    "fake_sdxl_i2i": ("1", "SD_XL", "I2I"),
    "fake_sdxl_t2i": ("1", "SD_XL", "T2I"),
    "fake_sdxl_ti2i": ("1", "SD_XL", "TI2I"),
}
COCO_CAPTION_MEMBERS: Final = (
    ("train2017", "annotations/captions_train2017.json"),
    ("val2017", "annotations/captions_val2017.json"),
)
SELECTION_COLUMNS: Final = (
    "selection_id",
    "split",
    "leakage_group",
    "parent_coco_image_id",
    "coco_caption_id",
    "official_coco_split",
    "official_coco_license_id",
    "official_coco_license_name",
    "official_coco_license_url",
    "cell",
    "label",
    "generator",
    "model",
    "gen_type",
    "declared_size",
    "locator",
    "repository_id",
    "revision",
    "shard_path",
    "row_index",
    "source_index",
    "source_index_hash",
    "image_path_basename",
    "category",
    "class_id",
)


def stable_hash(*parts: object) -> str:
    text = ":".join(str(part) for part in parts)
    return hashlib.sha256(text.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_audit_chain(
    *,
    lineage_scan_dir: Path,
    lineage_summary_path: Path,
    coco_summary_path: Path,
    lineage_summary: Mapping[str, object],
    coco_summary: Mapping[str, object],
    annotations_zip: Path,
) -> dict[str, str]:
    lineage_catalog_path = lineage_scan_dir / dani_lineage.LINEAGE_CATALOG_NAME
    lineage_provenance_path = lineage_scan_dir / dani_lineage.LINEAGE_PROVENANCE_NAME
    lineage_lock_path = lineage_scan_dir / dani_lineage.LINEAGE_SOURCE_LOCK_NAME
    required = (lineage_catalog_path, lineage_provenance_path, lineage_lock_path, annotations_zip)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Selection input is incomplete: " + ", ".join(missing))
    hashes = {
        "lineage_catalog_sha256": dani.sha256_file(lineage_catalog_path),
        "lineage_provenance_sha256": dani.sha256_file(lineage_provenance_path),
        "lineage_source_lock_sha256": dani.sha256_file(lineage_lock_path),
        "annotations_archive_sha256": dani.sha256_file(annotations_zip),
        "lineage_audit_summary_sha256": dani.sha256_file(lineage_summary_path),
        "coco_identity_summary_sha256": dani.sha256_file(coco_summary_path),
    }
    if lineage_summary.get("schema_version") != LINEAGE_AUDIT_SCHEMA:
        raise ValueError("lineage audit summary has an unsupported schema_version")
    lineage_source = lineage_summary.get("source")
    lineage_eligibility = lineage_summary.get("eligibility")
    if not isinstance(lineage_source, dict) or not isinstance(lineage_eligibility, dict):
        raise TypeError("lineage audit source and eligibility must be objects")
    if lineage_source.get("lineage_catalog_sha256") != hashes["lineage_catalog_sha256"]:
        raise ValueError("lineage catalogue SHA-256 differs from its audit summary")
    if lineage_source.get("provenance_sha256") != hashes["lineage_provenance_sha256"]:
        raise ValueError("lineage provenance SHA-256 differs from its audit summary")
    if lineage_source.get("source_lock_sha256") != hashes["lineage_source_lock_sha256"]:
        raise ValueError("lineage source lock SHA-256 differs from its audit summary")
    if (
        lineage_eligibility.get("candidate_parent_group_verified_against_pinned_djudge_mapping")
        is not True
        or lineage_eligibility.get("candidate_caption_pair_verified_against_pinned_djudge_mapping")
        is not True
        or lineage_eligibility.get("eligible_for_training") is not False
    ):
        raise ValueError("lineage audit is not in the required fail-closed state")
    if coco_summary.get("schema_version") != COCO_IDENTITY_SCHEMA:
        raise ValueError("COCO identity summary has an unsupported schema_version")
    coco_inputs = coco_summary.get("inputs")
    coco_eligibility = coco_summary.get("eligibility")
    verified_subset = coco_summary.get("verified_dani_subset")
    if (
        not isinstance(coco_inputs, dict)
        or not isinstance(coco_eligibility, dict)
        or not isinstance(verified_subset, dict)
    ):
        raise TypeError("COCO identity inputs, eligibility, and verified subset must be objects")
    if coco_inputs.get("annotations_archive_sha256") != hashes["annotations_archive_sha256"]:
        raise ValueError("COCO annotations SHA-256 differs from the identity audit")
    if coco_inputs.get("lineage_summary_sha256") != hashes["lineage_audit_summary_sha256"]:
        raise ValueError("lineage audit summary SHA-256 differs from the COCO identity audit")
    if (
        coco_eligibility.get("official_coco_parent_identity_verified") is not True
        or coco_eligibility.get("official_coco_caption_identity_verified") is not True
        or coco_eligibility.get("eligible_for_training") is not False
    ):
        raise ValueError("COCO identity audit is not in the required fail-closed state")
    if verified_subset.get("parent_count") != 5000:
        raise ValueError("COCO identity audit does not prove all 5,000 DANI parents")
    if verified_subset.get("parent_split_counts") != {"val2017": 5000}:
        raise ValueError("DANI parents are not exclusively official COCO val2017 records")
    return hashes


def load_coco_license_metadata(
    annotations_zip: str | Path,
) -> dict[int, dict[str, object]]:
    """Load official image split/licence metadata without extracting the COCO archive."""
    parents: dict[int, dict[str, object]] = {}
    licenses: dict[int, tuple[str, str]] = {}
    with zipfile.ZipFile(annotations_zip) as archive:
        for split, member in COCO_CAPTION_MEMBERS:
            if member not in archive.namelist():
                raise ValueError(f"COCO archive is missing {member}")
            payload = json.load(archive.open(member))
            if not isinstance(payload, dict):
                raise TypeError(f"{member} must contain a JSON object")
            raw_licenses = payload.get("licenses")
            raw_images = payload.get("images")
            if not isinstance(raw_licenses, list) or not isinstance(raw_images, list):
                raise TypeError(f"{member} has an invalid COCO schema")
            for record in raw_licenses:
                if not isinstance(record, dict):
                    raise TypeError(f"{member} contains a non-object licence")
                license_id = record.get("id")
                name = record.get("name")
                url = record.get("url")
                if (
                    isinstance(license_id, bool)
                    or not isinstance(license_id, int)
                    or not isinstance(name, str)
                    or not name
                    or not isinstance(url, str)
                    or not url
                ):
                    raise ValueError(f"{member} contains an invalid licence record")
                previous = licenses.setdefault(license_id, (name, url))
                if previous != (name, url):
                    raise ValueError(f"COCO licence {license_id} changes between splits")
            for record in raw_images:
                if not isinstance(record, dict):
                    raise TypeError(f"{member} contains a non-object image record")
                parent_id = record.get("id")
                license_id = record.get("license")
                file_name = record.get("file_name")
                if (
                    isinstance(parent_id, bool)
                    or not isinstance(parent_id, int)
                    or parent_id <= 0
                    or isinstance(license_id, bool)
                    or not isinstance(license_id, int)
                    or file_name != f"{parent_id:012d}.jpg"
                    or license_id not in licenses
                ):
                    raise ValueError(f"{member} contains an invalid image record")
                if parent_id in parents:
                    raise ValueError(f"COCO parent {parent_id} occurs in multiple splits")
                license_name, license_url = licenses[license_id]
                parents[parent_id] = {
                    "official_coco_split": split,
                    "official_coco_license_id": license_id,
                    "official_coco_license_name": license_name,
                    "official_coco_license_url": license_url,
                }
    return parents


def _cell_for_row(row: Mapping[str, str]) -> str | None:
    signature = (row["label"], row["model"], row["gen_type"])
    for cell, definition in CELL_DEFINITIONS.items():
        if signature == definition:
            return cell
    return None


def _split_counts(total: int) -> dict[str, int]:
    exact = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: int(value) for split, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        SPLIT_RATIOS,
        key=lambda split: (-(exact[split] - counts[split]), SPLIT_ORDER[split]),
    )
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def assign_parent_splits(
    parent_license_ids: Mapping[int, int],
    *,
    seed: int,
) -> dict[int, str]:
    """Assign whole parents within licence strata using stable largest-remainder quotas."""
    if seed < 0:
        raise ValueError("selection seed must be nonnegative")
    by_license: defaultdict[int, list[int]] = defaultdict(list)
    for parent_id, license_id in parent_license_ids.items():
        by_license[license_id].append(parent_id)
    result: dict[int, str] = {}
    for license_id, parent_ids in sorted(by_license.items()):
        ordered = sorted(
            parent_ids,
            key=lambda parent_id: stable_hash(
                SELECTION_SCHEMA_VERSION,
                "split",
                seed,
                license_id,
                parent_id,
            ),
        )
        counts = _split_counts(len(ordered))
        cursor = 0
        for split in ("train", "val", "test"):
            for parent_id in ordered[cursor : cursor + counts[split]]:
                result[parent_id] = split
            cursor += counts[split]
        if cursor != len(ordered):
            raise AssertionError("Split assignment did not consume its licence stratum")
    return result


def _choose_row(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int,
    parent_id: int,
    caption_id: int,
    cell: str,
) -> Mapping[str, str]:
    return min(
        rows,
        key=lambda row: stable_hash(
            SELECTION_SCHEMA_VERSION,
            "row",
            seed,
            parent_id,
            caption_id,
            cell,
            row["locator"],
        ),
    )


def build_selection(
    lineage_scan_dir: str | Path,
    lineage_audit_summary: str | Path,
    coco_identity_summary: str | Path,
    annotations_zip: str | Path,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SELECTION_SEED,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Freeze a caption-matched, parent-disjoint selection without reading image bytes."""
    source_dir = Path(lineage_scan_dir)
    lineage_summary_path = Path(lineage_audit_summary)
    coco_summary_path = Path(coco_identity_summary)
    archive_path = Path(annotations_zip)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing DANI selection: {destination}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"DANI lineage scan does not exist: {source_dir}")
    for path in (lineage_summary_path, coco_summary_path, archive_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required selection input does not exist: {path}")
    lineage_summary = _read_json(lineage_summary_path)
    coco_summary = _read_json(coco_summary_path)
    input_hashes = _validate_audit_chain(
        lineage_scan_dir=source_dir,
        lineage_summary_path=lineage_summary_path,
        coco_summary_path=coco_summary_path,
        lineage_summary=lineage_summary,
        coco_summary=coco_summary,
        annotations_zip=archive_path,
    )
    coco_parents = load_coco_license_metadata(archive_path)
    eligible_parents = {
        parent_id: metadata
        for parent_id, metadata in coco_parents.items()
        if metadata["official_coco_split"] == "val2017"
        and metadata["official_coco_license_id"] in ALLOWED_COCO_LICENSE_IDS
    }

    candidates: defaultdict[
        int,
        defaultdict[int, defaultdict[str, list[dict[str, str]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    catalog_path = source_dir / dani_lineage.LINEAGE_CATALOG_NAME
    with catalog_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != dani_lineage.LINEAGE_CANDIDATE_COLUMNS:
            raise ValueError("DANI lineage catalogue schema differs from the locked schema")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"lineage catalogue row {row_number} exceeds its schema")
            if row["declared_size"] != str(DECLARED_SOURCE_SIZE):
                continue
            parent_id = int(row["parent_coco_image_id"])
            if parent_id not in eligible_parents:
                continue
            cell = _cell_for_row(row)
            if cell is None:
                continue
            parsed_parent, caption_id, basename = dani_lineage.parse_lineage_basename(
                row["image_path_basename"]
            )
            if parsed_parent != parent_id or basename != row["image_path_basename"]:
                raise ValueError(f"lineage catalogue row {row_number} has inconsistent lineage")
            if caption_id != int(row["coco_caption_id"]):
                raise ValueError(f"lineage catalogue row {row_number} has inconsistent caption ID")
            candidates[parent_id][caption_id][cell].append(dict(row))

    selected_by_parent: dict[int, tuple[int, dict[str, Mapping[str, str]]]] = {}
    incomplete_parents: list[int] = []
    required_cells = set(CELL_DEFINITIONS)
    for parent_id in sorted(eligible_parents):
        complete_pairs = [
            caption_id
            for caption_id, by_cell in candidates[parent_id].items()
            if set(by_cell) == required_cells
        ]
        if not complete_pairs:
            incomplete_parents.append(parent_id)
            continue
        caption_id = min(
            complete_pairs,
            key=lambda candidate: stable_hash(
                SELECTION_SCHEMA_VERSION,
                "caption",
                seed,
                parent_id,
                candidate,
            ),
        )
        selected_by_parent[parent_id] = (
            caption_id,
            {
                cell: _choose_row(
                    candidates[parent_id][caption_id][cell],
                    seed=seed,
                    parent_id=parent_id,
                    caption_id=caption_id,
                    cell=cell,
                )
                for cell in CELL_DEFINITIONS
            },
        )
    parent_license_ids = {
        parent_id: int(eligible_parents[parent_id]["official_coco_license_id"])
        for parent_id in selected_by_parent
    }
    parent_splits = assign_parent_splits(parent_license_ids, seed=seed)

    selected_rows: list[dict[str, object]] = []
    selection_ids: set[str] = set()
    locators: set[str] = set()
    for parent_id, (caption_id, rows_by_cell) in selected_by_parent.items():
        split = parent_splits[parent_id]
        license_metadata = eligible_parents[parent_id]
        for cell, definition in CELL_DEFINITIONS.items():
            row = rows_by_cell[cell]
            locator = row["locator"]
            if locator in locators:
                raise ValueError("DANI selection duplicates a source locator")
            locators.add(locator)
            selection_id = "dani-selected:" + stable_hash(
                SELECTION_SCHEMA_VERSION,
                seed,
                parent_id,
                caption_id,
                cell,
                locator,
            )
            if selection_id in selection_ids:
                raise RuntimeError("Unexpected DANI selection ID collision")
            selection_ids.add(selection_id)
            label, model, gen_type = definition
            selected_rows.append(
                {
                    "selection_id": selection_id,
                    "split": split,
                    "leakage_group": f"coco-parent:{parent_id}",
                    "parent_coco_image_id": parent_id,
                    "coco_caption_id": caption_id,
                    **license_metadata,
                    "cell": cell,
                    "label": label,
                    "generator": "real" if label == "0" else f"{model}:{gen_type}",
                    "model": model,
                    "gen_type": gen_type,
                    "declared_size": row["declared_size"],
                    "locator": locator,
                    "repository_id": row["repository_id"],
                    "revision": row["revision"],
                    "shard_path": row["shard_path"],
                    "row_index": row["row_index"],
                    "source_index": row["source_index"],
                    "source_index_hash": row["source_index_hash"],
                    "image_path_basename": row["image_path_basename"],
                    "category": row["category"],
                    "class_id": row["class_id"],
                }
            )
    selected_rows.sort(
        key=lambda row: (
            SPLIT_ORDER[str(row["split"])],
            int(row["parent_coco_image_id"]),
            list(CELL_DEFINITIONS).index(str(row["cell"])),
        )
    )
    if len(selected_rows) != len(selected_by_parent) * len(CELL_DEFINITIONS):
        raise AssertionError("DANI selection does not have exactly one row per required cell")
    theoretical_rgb_bytes = len(selected_rows) * DECLARED_SOURCE_SIZE * DECLARED_SOURCE_SIZE * 3
    if theoretical_rgb_bytes > MATERIALISATION_BYTE_BUDGET:
        raise ValueError("Selected 1024 RGB upper-bound exceeds the 25 GB materialisation budget")

    hashes_after = {
        "lineage_catalog_sha256": dani.sha256_file(catalog_path),
        "lineage_provenance_sha256": dani.sha256_file(
            source_dir / dani_lineage.LINEAGE_PROVENANCE_NAME
        ),
        "lineage_source_lock_sha256": dani.sha256_file(
            source_dir / dani_lineage.LINEAGE_SOURCE_LOCK_NAME
        ),
        "annotations_archive_sha256": dani.sha256_file(archive_path),
        "lineage_audit_summary_sha256": dani.sha256_file(lineage_summary_path),
        "coco_identity_summary_sha256": dani.sha256_file(coco_summary_path),
    }
    if hashes_after != input_hashes:
        raise RuntimeError("Selection input changed while the manifest was being built")

    parent_split_counts = Counter(parent_splits.values())
    row_split_counts = Counter(str(row["split"]) for row in selected_rows)
    parent_license_counts = Counter(parent_license_ids.values())
    split_license_counts = Counter(
        (parent_splits[parent_id], license_id)
        for parent_id, license_id in parent_license_ids.items()
    )
    created = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    spec: dict[str, object] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "created_at_utc": created,
        "selection_seed": seed,
        "selection_kind": "metadata_only_frozen_selection_not_materialised_manifest",
        "declared_source_size": DECLARED_SOURCE_SIZE,
        "allowed_coco_license_ids": list(ALLOWED_COCO_LICENSE_IDS),
        "required_caption_matched_cells": {
            cell: {"label": int(label), "model": model, "gen_type": gen_type}
            for cell, (label, model, gen_type) in CELL_DEFINITIONS.items()
        },
        "parent_group_key": "parent_coco_image_id",
        "caption_pair_key": ["parent_coco_image_id", "coco_caption_id"],
        "split_ratios": SPLIT_RATIOS,
        "split_algorithm": (
            "stable_sha256_order_within_official_coco_license_id_then_largest_remainder_v1"
        ),
        "row_algorithm": "stable_sha256_minimum_within_parent_caption_cell_v1",
        "input_hashes": input_hashes,
        "budget": {
            "materialisation_byte_cap": MATERIALISATION_BYTE_BUDGET,
            "selected_count": len(selected_rows),
            "theoretical_1024_rgb_bytes": theoretical_rgb_bytes,
            "theoretical_1024_rgb_gib": theoretical_rgb_bytes / (1024**3),
            "note": (
                "The byte materialiser must enforce the cap on encoded input plus outputs; this "
                "metadata estimate is not observed storage."
            ),
        },
    }
    provenance: dict[str, object] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "created_at_utc": created,
        "selection_spec": SELECTION_SPEC_NAME,
        "selection_catalog": SELECTION_CATALOG_NAME,
        "image_bytes_requested": False,
        "image_bytes_read": False,
        "counts": {
            "official_coco_parent_count": len(coco_parents),
            "license_eligible_parent_count": len(eligible_parents),
            "incomplete_required_cell_parent_count": len(incomplete_parents),
            "selected_parent_count": len(selected_by_parent),
            "selected_row_count": len(selected_rows),
            "selected_parent_count_by_split": dict(sorted(parent_split_counts.items())),
            "selected_row_count_by_split": dict(sorted(row_split_counts.items())),
            "selected_parent_count_by_license_id": {
                str(key): value for key, value in sorted(parent_license_counts.items())
            },
            "selected_parent_count_by_split_license_id": {
                f"{split}:license_{license_id}": count
                for (split, license_id), count in sorted(split_license_counts.items())
            },
            "selected_row_count_by_cell": {
                cell: sum(row["cell"] == cell for row in selected_rows) for cell in CELL_DEFINITIONS
            },
        },
        "eligibility": {
            "eligible_for_selected_byte_materialisation": True,
            "eligible_for_split_assignment": True,
            "parent_group_disjoint_split_frozen": True,
            "eligible_for_training": False,
            "eligible_for_model_selection": False,
            "remaining_blocker": (
                "Selected bytes must pass geometry, mode/container, corruption, exact/perceptual "
                "duplicate, cross-split leakage, and shortcut audits."
            ),
        },
    }
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / SELECTION_SPEC_NAME, spec)
    with (destination / SELECTION_CATALOG_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(selected_rows)
    provenance["selection_spec_sha256"] = dani.sha256_file(destination / SELECTION_SPEC_NAME)
    provenance["selection_catalog_sha256"] = dani.sha256_file(destination / SELECTION_CATALOG_NAME)
    _write_json(destination / SELECTION_PROVENANCE_NAME, provenance)
    return provenance
