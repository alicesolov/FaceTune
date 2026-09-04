"""Verify DANI path-derived lineage against a pinned D-Judge mapping, offline.

This audit reads only the previously produced path/metadata catalogue and an explicitly supplied
JSON mapping. It does not access a network, read image bytes, emit caption text, select rows, assign
splits, or make the source trainable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_image_detector import dani, dani_lineage

AUDIT_SCHEMA_VERSION = "dani_lineage_mapping_audit_v1"
MAPPING_SCHEMA = "djudge_image_captions_dict_new_v1"
SOURCE_LOCK_SCHEMA = "dani_lineage_source_lock_v1"
SCAN_SCHEMA = "dani_lineage_metadata_scan_v1"
COMPLETE_SCAN_KIND = "complete_lineage_candidate_scan"
CATALOG_KIND = "path_derived_lineage_candidate_catalog_not_trainable_manifest"
FORBIDDEN_CATALOG_COLUMNS = frozenset(
    {"image", "image.bytes", "bytes", "image_data", "caption", "prompt"}
)
REMAINING_BLOCKERS = (
    (
        "The D-Judge mapping is not an official COCO annotation join; COCO parent and caption IDs "
        "still require verification against a revision-pinned official annotation source."
    ),
    (
        "The combined DANI, D-Judge mapping, and COCO licence/provenance chain has not been "
        "approved for the planned coursework use."
    ),
    (
        "Image bytes have not been materialised or audited for decoded geometry, container, mode, "
        "corruption, exact duplicates, perceptual duplicates, or metadata/file-size shortcuts."
    ),
    "A deterministic parent-grouped, class-balanced split and its leakage checks do not yet exist.",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _require_revision(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be an immutable lowercase 40-character commit SHA")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, field: str) -> int:
    text = "" if value is None else str(value)
    if not text.isdecimal() or int(text) <= 0:
        raise ValueError(f"{field} must be a positive decimal integer")
    return int(text)


def _nonnegative_int(value: object, *, field: str) -> int:
    text = "" if value is None else str(value)
    if not text.isdecimal():
        raise ValueError(f"{field} must be a nonnegative decimal integer")
    return int(text)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_counts(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _validate_source_lock(source_lock: Mapping[str, object]) -> tuple[str, list[str]]:
    if source_lock.get("schema_version") != SOURCE_LOCK_SCHEMA:
        raise ValueError("source_lock.json has an unsupported schema_version")
    if source_lock.get("repository_id") != dani.REPOSITORY_ID:
        raise ValueError("source_lock.json has an unexpected repository_id")
    revision = _require_revision(source_lock.get("revision"), field="source_lock.revision")
    if source_lock.get("license") != dani.SOURCE_LICENSE:
        raise ValueError("source_lock.json has an unexpected DANI license")
    if source_lock.get("repo_type") != "dataset" or source_lock.get("tree_recursive") is not True:
        raise ValueError("source_lock.json does not prove a recursive dataset tree lock")
    if source_lock.get("source_schema_columns") != list(dani.SOURCE_SCHEMA_COLUMNS):
        raise ValueError("source_lock.json source schema does not match DANI")
    if source_lock.get("projection_contract") != dani_lineage.lineage_projection_contract():
        raise ValueError("source_lock.json does not preserve the locked path-only projection")
    shards = source_lock.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("source_lock.json must contain a nonempty shard list")
    shard_paths: list[str] = []
    for position, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise TypeError(f"source_lock.shards[{position}] must be an object")
        shard_path = shard.get("path")
        if (
            not isinstance(shard_path, str)
            or shard_path.startswith("/")
            or not shard_path.endswith(".parquet")
        ):
            raise ValueError(f"source_lock.shards[{position}] has an invalid path")
        shard_paths.append(shard_path)
    if shard_paths != sorted(shard_paths) or len(shard_paths) != len(set(shard_paths)):
        raise ValueError("source_lock.json shard paths must be unique and sorted")
    if source_lock.get("shard_count") != len(shard_paths):
        raise ValueError("source_lock.json shard_count does not match its shard list")
    return revision, shard_paths


def _validate_provenance(
    provenance: Mapping[str, object],
    *,
    revision: str,
    shard_paths: list[str],
    source_hashes: Mapping[str, str],
) -> tuple[int, dict[str, int]]:
    if provenance.get("schema_version") != SCAN_SCHEMA:
        raise ValueError("provenance.json has an unsupported schema_version")
    if provenance.get("repository_id") != dani.REPOSITORY_ID:
        raise ValueError("provenance.json has an unexpected repository_id")
    if provenance.get("revision") != revision or provenance.get("resolved_revision") != revision:
        raise ValueError("source revision differs between provenance.json and source_lock.json")
    if provenance.get("source_lock") != dani_lineage.LINEAGE_SOURCE_LOCK_NAME:
        raise ValueError("provenance.json refers to an unexpected source lock")
    if provenance.get("lineage_catalog") != dani_lineage.LINEAGE_CATALOG_NAME:
        raise ValueError("provenance.json refers to an unexpected lineage catalogue")
    if (
        _require_sha256(provenance.get("source_lock_sha256"), field="source_lock_sha256")
        != source_hashes[dani_lineage.LINEAGE_SOURCE_LOCK_NAME]
    ):
        raise ValueError("source_lock.json SHA-256 does not match provenance.json")
    if (
        _require_sha256(provenance.get("lineage_catalog_sha256"), field="lineage_catalog_sha256")
        != source_hashes[dani_lineage.LINEAGE_CATALOG_NAME]
    ):
        raise ValueError("lineage_catalog.csv SHA-256 does not match provenance.json")
    if provenance.get("catalog_kind") != CATALOG_KIND:
        raise ValueError("provenance.json has an unexpected blocked catalog_kind")
    if provenance.get("projection_contract") != dani_lineage.lineage_projection_contract():
        raise ValueError("provenance.json path-only projection contract changed")
    observation = provenance.get("projection_observation")
    if not isinstance(observation, dict):
        raise TypeError("projection_observation must be an object")
    required_observation = {
        "image_path_requested": True,
        "image_path_materialised": True,
        "image_bytes_requested": False,
        "image_bytes_materialised": False,
        "image_bytes_decoded": False,
        "batch_image_struct_children_required": ["path"],
        "http_range_image_byte_disjointness_verified": False,
    }
    if observation != required_observation:
        raise ValueError("provenance.json does not preserve the no-image-byte observation")
    status = provenance.get("lineage_status")
    if not isinstance(status, dict) or status.get("upstream_mapping_join_performed") is not False:
        raise ValueError("lineage_status must describe an unaudited candidate mapping")
    scope = provenance.get("scan_scope")
    if not isinstance(scope, dict):
        raise TypeError("scan_scope must be an object")
    required_scope = {
        "kind": COMPLETE_SCAN_KIND,
        "partial": False,
        "complete": True,
        "eligible_for_lineage_audit": True,
        "eligible_for_candidate_selection": False,
        "eligible_for_training": False,
        "available_shard_count": len(shard_paths),
        "selected_shard_count": len(shard_paths),
        "limit_shards": None,
        "selected_shards": shard_paths,
    }
    bad_scope = [key for key, expected in required_scope.items() if scope.get(key) != expected]
    if bad_scope:
        raise ValueError(
            "Refusing partial or invalid DANI lineage scan; invalid scan_scope fields: "
            + ", ".join(bad_scope)
        )
    row_count = provenance.get("catalog_row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("provenance.json catalog_row_count must be positive")
    raw_by_shard = provenance.get("rows_scanned_by_shard")
    if not isinstance(raw_by_shard, dict) or list(raw_by_shard) != shard_paths:
        raise ValueError("provenance.json rows_scanned_by_shard does not match locked shards")
    by_shard: dict[str, int] = {}
    for shard_path in shard_paths:
        count = raw_by_shard[shard_path]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("provenance.json has an invalid per-shard row count")
        by_shard[shard_path] = count
    if sum(by_shard.values()) != row_count:
        raise ValueError("provenance.json per-shard counts do not sum to catalog_row_count")
    return row_count, by_shard


def _load_mapping(
    mapping_path: Path,
    *,
    mapping_url: str,
    mapping_revision: str,
) -> tuple[set[tuple[int, int]], set[int], dict[str, object]]:
    revision = _require_revision(mapping_revision, field="mapping_revision")
    if not mapping_url.startswith("https://") or f"/{revision}/" not in mapping_url:
        raise ValueError("mapping_url must be HTTPS and contain the immutable mapping revision")
    mapping = _read_json_object(mapping_path)
    pairs: set[tuple[int, int]] = set()
    parents: set[int] = set()
    caption_ids: set[int] = set()
    for parent_key, record in mapping.items():
        if not parent_key.isdecimal() or parent_key.startswith("0"):
            raise ValueError(f"mapping parent key {parent_key!r} is not canonical")
        parent_id = int(parent_key)
        if not isinstance(record, dict):
            raise TypeError(f"mapping parent {parent_key} must be an object")
        if set(record) != {"image_id", "image_name", "captions"}:
            raise ValueError(f"mapping parent {parent_key} has an unexpected schema")
        if record.get("image_id") != parent_id:
            raise ValueError(f"mapping parent {parent_key} disagrees with image_id")
        if record.get("image_name") != f"{parent_id:012d}.jpg":
            raise ValueError(f"mapping parent {parent_key} has an invalid COCO image_name")
        captions = record.get("captions")
        if not isinstance(captions, list) or not captions:
            raise ValueError(f"mapping parent {parent_key} has no captions")
        parents.add(parent_id)
        for position, caption in enumerate(captions):
            if not isinstance(caption, dict) or set(caption) != {"caption_id", "caption"}:
                raise ValueError(
                    f"mapping parent {parent_key} caption {position} has an unexpected schema"
                )
            caption_id = caption.get("caption_id")
            caption_text = caption.get("caption")
            if isinstance(caption_id, bool) or not isinstance(caption_id, int) or caption_id <= 0:
                raise ValueError(f"mapping parent {parent_key} has an invalid caption_id")
            if not isinstance(caption_text, str) or not caption_text.strip():
                raise ValueError(f"mapping parent {parent_key} has empty caption text")
            if caption_id in caption_ids:
                raise ValueError(f"mapping contains duplicate caption_id {caption_id}")
            caption_ids.add(caption_id)
            pairs.add((parent_id, caption_id))
    return (
        pairs,
        parents,
        {
            "schema": MAPPING_SCHEMA,
            "source_url": mapping_url,
            "revision": revision,
            "sha256": sha256_file(mapping_path),
            "parent_count": len(parents),
            "caption_pair_count": len(pairs),
            "caption_text_read_for_schema_validation_but_not_emitted": True,
        },
    )


def _validate_headers(headers: list[str] | None) -> None:
    if headers is None:
        raise ValueError("lineage_catalog.csv has no header")
    if len(headers) != len(set(headers)):
        raise ValueError("lineage_catalog.csv has duplicate headers")
    forbidden = sorted(value for value in headers if value.casefold() in FORBIDDEN_CATALOG_COLUMNS)
    if forbidden:
        raise ValueError(
            "lineage_catalog.csv contains forbidden raw fields: " + ", ".join(forbidden)
        )
    if tuple(headers) != dani_lineage.LINEAGE_CANDIDATE_COLUMNS:
        raise ValueError("lineage_catalog.csv schema differs from the locked lineage schema")


def audit_lineage(
    input_dir: str | Path,
    mapping_path: str | Path,
    output_dir: str | Path,
    *,
    mapping_url: str,
    mapping_revision: str,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Validate one complete lineage scan and pinned mapping without network or image access."""
    source = Path(input_dir)
    mapping_file = Path(mapping_path)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing DANI lineage audit: {destination}")
    if not source.is_dir():
        raise FileNotFoundError(f"DANI lineage input directory does not exist: {source}")
    if not mapping_file.is_file():
        raise FileNotFoundError(f"Pinned D-Judge mapping does not exist: {mapping_file}")
    required = {
        dani_lineage.LINEAGE_SOURCE_LOCK_NAME: source / dani_lineage.LINEAGE_SOURCE_LOCK_NAME,
        dani_lineage.LINEAGE_CATALOG_NAME: source / dani_lineage.LINEAGE_CATALOG_NAME,
        dani_lineage.LINEAGE_PROVENANCE_NAME: source / dani_lineage.LINEAGE_PROVENANCE_NAME,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("DANI lineage scan is incomplete; missing: " + ", ".join(missing))
    source_hashes_before = {name: sha256_file(path) for name, path in required.items()}
    mapping_hash_before = sha256_file(mapping_file)
    source_lock = _read_json_object(required[dani_lineage.LINEAGE_SOURCE_LOCK_NAME])
    provenance = _read_json_object(required[dani_lineage.LINEAGE_PROVENANCE_NAME])
    revision, shard_paths = _validate_source_lock(source_lock)
    declared_rows, declared_by_shard = _validate_provenance(
        provenance,
        revision=revision,
        shard_paths=shard_paths,
        source_hashes=source_hashes_before,
    )
    mapping_pairs, mapping_parents, mapping_evidence = _load_mapping(
        mapping_file,
        mapping_url=mapping_url,
        mapping_revision=mapping_revision,
    )
    if mapping_evidence["sha256"] != mapping_hash_before:
        raise RuntimeError("Pinned mapping changed while it was being loaded")

    rows_by_shard = Counter[str]()
    label_counts = Counter[str]()
    label_size_counts = Counter[tuple[str, int]]()
    model_label_counts = Counter[tuple[str, str]]()
    observed_parents: set[int] = set()
    observed_pairs: set[tuple[int, int]] = set()
    parent_labels: defaultdict[int, set[str]] = defaultdict(set)
    pair_labels: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    locators: set[str] = set()
    expected_row_index = {path: 0 for path in shard_paths}
    total_rows = 0
    catalog_path = required[dani_lineage.LINEAGE_CATALOG_NAME]
    with catalog_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_headers(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"lineage_catalog.csv row {row_number} exceeds its schema")
            if row.get("repository_id") != dani.REPOSITORY_ID or row.get("revision") != revision:
                raise ValueError(f"lineage_catalog.csv row {row_number} changes source identity")
            shard_path = row.get("shard_path")
            if shard_path not in expected_row_index:
                raise ValueError(
                    f"lineage_catalog.csv row {row_number} refers to an unlocked shard"
                )
            row_index = _nonnegative_int(row.get("row_index"), field=f"row {row_number} row_index")
            if row_index != expected_row_index[shard_path]:
                raise ValueError(
                    f"lineage_catalog.csv row {row_number} has nonsequential row_index"
                )
            expected_row_index[shard_path] += 1
            locator = f"{dani.REPOSITORY_ID}@{revision}:{shard_path}:{row_index}"
            if row.get("locator") != locator or locator in locators:
                raise ValueError(f"lineage_catalog.csv row {row_number} has invalid locator")
            locators.add(locator)
            source_index = row.get("source_index")
            if (
                not isinstance(source_index, str)
                or not source_index
                or source_index.strip() != source_index
            ):
                raise ValueError(f"lineage_catalog.csv row {row_number} has invalid source_index")
            if row.get("source_index_hash") != dani.source_index_hash(source_index):
                raise ValueError(
                    f"lineage_catalog.csv row {row_number} has invalid source_index_hash"
                )
            parent_id, caption_id, basename = dani_lineage.parse_lineage_basename(
                row.get("image_path_basename")
            )
            if row.get("image_path_basename") != basename:
                raise AssertionError("basename parser changed its input unexpectedly")
            if (
                _positive_int(
                    row.get("parent_coco_image_id"), field=f"row {row_number} parent_coco_image_id"
                )
                != parent_id
            ):
                raise ValueError(f"lineage_catalog.csv row {row_number} has inconsistent parent ID")
            if (
                _positive_int(row.get("coco_caption_id"), field=f"row {row_number} coco_caption_id")
                != caption_id
            ):
                raise ValueError(
                    f"lineage_catalog.csv row {row_number} has inconsistent caption ID"
                )
            pair = (parent_id, caption_id)
            if parent_id not in mapping_parents or pair not in mapping_pairs:
                raise ValueError(
                    f"lineage_catalog.csv row {row_number} has no exact pinned mapping pair"
                )
            declared_size = _positive_int(
                row.get("declared_size"), field=f"row {row_number} declared_size"
            )
            reference = row.get("reference")
            label = row.get("label")
            expected_label = "0" if reference == "True" else "1" if reference == "False" else None
            if expected_label is None or label != expected_label:
                raise ValueError(f"lineage_catalog.csv row {row_number} has inconsistent label")
            model = row.get("model") or "(missing)"
            total_rows += 1
            rows_by_shard[shard_path] += 1
            label_counts[label] += 1
            label_size_counts[(label, declared_size)] += 1
            model_label_counts[(model, label)] += 1
            observed_parents.add(parent_id)
            observed_pairs.add(pair)
            parent_labels[parent_id].add(label)
            pair_labels[pair].add(label)

    if total_rows != declared_rows:
        raise ValueError("lineage_catalog.csv row count does not match provenance.json")
    if {path: rows_by_shard[path] for path in shard_paths} != declared_by_shard:
        raise ValueError("lineage_catalog.csv per-shard counts do not match provenance.json")
    if observed_parents != mapping_parents or observed_pairs != mapping_pairs:
        raise ValueError("DANI lineage catalogue does not cover the pinned mapping exactly")
    source_hashes_after = {name: sha256_file(path) for name, path in required.items()}
    if (
        source_hashes_after != source_hashes_before
        or sha256_file(mapping_file) != mapping_hash_before
    ):
        raise RuntimeError("Audit input changed during verification")

    cross_label_parents = sum(labels == {"0", "1"} for labels in parent_labels.values())
    cross_label_pairs = sum(labels == {"0", "1"} for labels in pair_labels.values())
    if cross_label_parents != len(mapping_parents) or cross_label_pairs != len(mapping_pairs):
        raise ValueError("Not every verified DANI parent/caption pair crosses both labels")
    created = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    summary: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at_utc": created,
        "audit_kind": "offline_path_lineage_to_pinned_djudge_mapping_verification",
        "network_accessed": False,
        "image_bytes_requested": False,
        "image_bytes_read": False,
        "caption_text_emitted": False,
        "source": {
            "repository_id": dani.REPOSITORY_ID,
            "revision": revision,
            "catalog_row_count": total_rows,
            "source_lock_sha256": source_hashes_before[dani_lineage.LINEAGE_SOURCE_LOCK_NAME],
            "lineage_catalog_sha256": source_hashes_before[dani_lineage.LINEAGE_CATALOG_NAME],
            "provenance_sha256": source_hashes_before[dani_lineage.LINEAGE_PROVENANCE_NAME],
        },
        "mapping": mapping_evidence,
        "coverage": {
            "catalog_rows_exact_pair_joined": total_rows,
            "catalog_rows_unjoined": 0,
            "observed_parent_count": len(observed_parents),
            "observed_caption_pair_count": len(observed_pairs),
            "mapping_parent_coverage_fraction": 1.0,
            "mapping_caption_pair_coverage_fraction": 1.0,
            "cross_label_parent_count": cross_label_parents,
            "cross_label_caption_pair_count": cross_label_pairs,
            "all_verified_parents_cross_labels": True,
            "all_verified_caption_pairs_cross_labels": True,
        },
        "label_counts": dict(sorted(label_counts.items())),
        "eligibility": {
            "candidate_parent_group_verified_against_pinned_djudge_mapping": True,
            "candidate_caption_pair_verified_against_pinned_djudge_mapping": True,
            "eligible_for_coco_identity_claim": False,
            "eligible_for_candidate_selection": False,
            "eligible_for_split_assignment": False,
            "eligible_for_training": False,
            "eligible_for_model_selection": False,
            "remaining_blockers": list(REMAINING_BLOCKERS),
        },
        "table_files": {
            "label_declared_size": "label_declared_size_counts.csv",
            "model_label": "model_label_counts.csv",
        },
    }
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "summary.json", summary)
    _write_counts(
        destination / "label_declared_size_counts.csv",
        ("label", "declared_size", "row_count"),
        [
            {"label": label, "declared_size": size, "row_count": count}
            for (label, size), count in sorted(label_size_counts.items())
        ],
    )
    _write_counts(
        destination / "model_label_counts.csv",
        ("model", "label", "row_count"),
        [
            {"model": model, "label": label, "row_count": count}
            for (model, label), count in sorted(model_label_counts.items())
        ],
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a complete DANI path-lineage catalogue against a local pinned D-Judge "
            "mapping. No network or image-byte access occurs."
        )
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mapping-url", required=True)
    parser.add_argument("--mapping-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_lineage(
        args.input_dir,
        args.mapping,
        args.output_dir,
        mapping_url=args.mapping_url,
        mapping_revision=args.mapping_revision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
