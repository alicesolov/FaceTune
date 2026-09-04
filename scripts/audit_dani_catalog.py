"""Audit a complete DANI metadata-only catalogue without accessing image bytes.

The DANI public metadata schema does not expose a documented COCO parent/caption key. This audit
therefore verifies provenance and describes the source distribution, but it never promotes the
catalogue to an internal training, split-assignment, or model-selection manifest.
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

SOURCE_LOCK_NAME = "source_lock.json"
SOURCE_CATALOG_NAME = "source_catalog.csv"
PROVENANCE_NAME = "provenance.json"

REPOSITORY_ID = "Renyang/DANI"
SOURCE_LICENSE = "CC-BY-NC-4.0"
SOURCE_LOCK_SCHEMA_VERSION = "dani_source_lock_v1"
SCAN_SCHEMA_VERSION = "dani_metadata_scan_v1"
AUDIT_SCHEMA_VERSION = "dani_catalog_audit_v1"
COMPLETE_SCAN_KIND = "complete_source_scan"
CATALOG_KIND = "metadata_only_candidate_catalog_not_trainable_manifest"

IMAGE_COLUMN = "image"
SOURCE_SCHEMA_COLUMNS = (
    "index",
    IMAGE_COLUMN,
    "size",
    "category",
    "class_id",
    "model",
    "gen_type",
    "reference",
)
META_COLUMNS = tuple(column for column in SOURCE_SCHEMA_COLUMNS if column != IMAGE_COLUMN)
CATALOG_COLUMNS = (
    "locator",
    "repository_id",
    "revision",
    "shard_path",
    "row_index",
    "source_index",
    "source_index_hash",
    "source_index_group_id",
    "declared_size",
    "category",
    "class_id",
    "model",
    "gen_type",
    "reference",
    "label",
)
FORBIDDEN_RAW_COLUMNS = frozenset({"image", "image_data", "prompt"})
LABEL_NAMES = {"0": "real", "1": "ai_generated"}
MISSING_VALUE = "(missing)"
INTERNAL_SELECTION_BLOCKER = (
    "The public DANI metadata schema lacks a documented parent COCO/caption group; "
    "source_index is an image-level upstream identifier, not proven pair provenance."
)
AUDIT_BLOCKERS = (
    (
        "Metadata-only catalogue: image bytes, decoded geometry, format, mode, EXIF, byte/pixel "
        "hashes, and perceptual hashes have not been audited."
    ),
    INTERNAL_SELECTION_BLOCKER,
)


def sha256_file(path: str | Path) -> str:
    """Return a byte-exact SHA-256 digest without parsing a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_fresh_output_dir(output_dir: str | Path) -> Path:
    """Create an audit directory without mixing it with prior evidence."""
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing DANI catalogue audit: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_int(value: object, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    text = _require_text(value, field=field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def _require_revision(value: object, *, field: str) -> str:
    text = _require_text(value, field=field)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be an immutable 40-character lowercase commit SHA")
    return text


def _normalise_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text if text else MISSING_VALUE


def _parse_nonnegative_int(value: object, *, field: str) -> int:
    text = "" if value is None else str(value)
    if not text.isdecimal():
        raise ValueError(f"{field} must be a nonnegative decimal integer")
    return int(text)


def _parse_positive_int(value: object, *, field: str) -> int:
    parsed = _parse_nonnegative_int(value, field=field)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _parse_reference(value: object, *, row_number: int) -> tuple[bool, str]:
    if value == "True":
        return True, "0"
    if value == "False":
        return False, "1"
    raise ValueError(f"source_catalog.csv row {row_number} has invalid reference value")


def _source_index_hash(source_index: str) -> str:
    return hashlib.sha256(f"dani-source-index:{source_index}".encode()).hexdigest()


def _validate_catalog_headers(headers: list[str] | None) -> None:
    if headers is None:
        raise ValueError("source_catalog.csv has no header row")
    if any(not header for header in headers):
        raise ValueError("source_catalog.csv has an empty header name")
    if len(headers) != len(set(headers)):
        raise ValueError("source_catalog.csv has duplicate header names")
    forbidden = sorted(header for header in headers if header.casefold() in FORBIDDEN_RAW_COLUMNS)
    if forbidden:
        raise ValueError(
            "source_catalog.csv must not contain raw image/prompt columns: " + ", ".join(forbidden)
        )
    if tuple(headers) != CATALOG_COLUMNS:
        missing = sorted(set(CATALOG_COLUMNS).difference(headers))
        unexpected = sorted(set(headers).difference(CATALOG_COLUMNS))
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        if not details:
            details.append("column order differs from the locked schema")
        raise ValueError("source_catalog.csv schema mismatch: " + "; ".join(details))


def _validate_source_lock(source_lock: Mapping[str, object]) -> tuple[str, str, list[str]]:
    if source_lock.get("schema_version") != SOURCE_LOCK_SCHEMA_VERSION:
        raise ValueError("source_lock.json has an unsupported schema_version")
    if source_lock.get("repository_id") != REPOSITORY_ID:
        raise ValueError("source_lock.json has an unexpected repository_id")
    revision = _require_revision(source_lock.get("revision"), field="source_lock.revision")
    if source_lock.get("license") != SOURCE_LICENSE:
        raise ValueError("source_lock.json has an unexpected license")
    if source_lock.get("repo_type") != "dataset" or source_lock.get("tree_recursive") is not True:
        raise ValueError("source_lock.json must prove a recursive dataset tree lock")
    if source_lock.get("source_schema_columns") != list(SOURCE_SCHEMA_COLUMNS):
        raise ValueError("source_lock.json source schema does not match DANI")
    if source_lock.get("metadata_columns") != list(META_COLUMNS):
        raise ValueError("source_lock.json metadata projection does not match DANI")
    if source_lock.get("excluded_binary_columns") != [IMAGE_COLUMN]:
        raise ValueError("source_lock.json must explicitly exclude only the image field")

    raw_shards = source_lock.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("source_lock.json must contain a nonempty shard list")
    paths: list[str] = []
    for position, shard in enumerate(raw_shards):
        if not isinstance(shard, dict):
            raise TypeError(f"source_lock.shards[{position}] must be an object")
        path = _require_text(shard.get("path"), field=f"source_lock.shards[{position}].path")
        if path.startswith("/") or not path.endswith(".parquet"):
            raise ValueError(f"source_lock.shards[{position}] has an invalid Parquet path")
        size = _require_int(
            shard.get("size"), field=f"source_lock.shards[{position}].size", positive=True
        )
        lfs = shard.get("lfs")
        if not isinstance(lfs, dict):
            raise TypeError(f"source_lock.shards[{position}].lfs must be an object")
        lfs_size = _require_int(
            lfs.get("size"), field=f"source_lock.shards[{position}].lfs.size", positive=True
        )
        if lfs_size != size:
            raise ValueError(f"source_lock.shards[{position}] LFS size differs from shard size")
        _require_sha256(lfs.get("sha256"), field=f"source_lock.shards[{position}].lfs.sha256")
        _require_int(
            lfs.get("pointer_size"),
            field=f"source_lock.shards[{position}].lfs.pointer_size",
            positive=True,
        )
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("source_lock.json must contain uniquely sorted shard paths")
    if source_lock.get("shard_count") != len(paths):
        raise ValueError("source_lock.json shard_count does not match its shard list")
    return REPOSITORY_ID, revision, paths


def _validate_provenance(
    provenance: Mapping[str, object],
    *,
    revision: str,
    locked_paths: list[str],
    source_hashes: Mapping[str, str],
) -> tuple[int, dict[str, int], Mapping[str, object]]:
    if provenance.get("schema_version") != SCAN_SCHEMA_VERSION:
        raise ValueError("provenance.json has an unsupported schema_version")
    if provenance.get("repository_id") != REPOSITORY_ID:
        raise ValueError("provenance.json has an unexpected repository_id")
    if provenance.get("revision") != revision or provenance.get("resolved_revision") != revision:
        raise ValueError("resolved revision differs between source_lock.json and provenance.json")
    if provenance.get("source_lock") != SOURCE_LOCK_NAME:
        raise ValueError("provenance.json must refer to source_lock.json in the input directory")
    if provenance.get("source_catalog") != SOURCE_CATALOG_NAME:
        raise ValueError("provenance.json must refer to source_catalog.csv in the input directory")
    if (
        _require_sha256(provenance.get("source_lock_sha256"), field="source_lock_sha256")
        != source_hashes[SOURCE_LOCK_NAME]
    ):
        raise ValueError("source_lock.json SHA-256 does not match provenance.json")
    if (
        _require_sha256(provenance.get("source_catalog_sha256"), field="source_catalog_sha256")
        != source_hashes[SOURCE_CATALOG_NAME]
    ):
        raise ValueError("source_catalog.csv SHA-256 does not match provenance.json")
    if provenance.get("catalog_kind") != CATALOG_KIND:
        raise ValueError("provenance.json has an unexpected catalog_kind")
    if provenance.get("metadata_columns") != list(META_COLUMNS):
        raise ValueError("provenance.json metadata projection does not match DANI")
    if provenance.get("excluded_binary_columns") != [IMAGE_COLUMN]:
        raise ValueError("provenance.json must explicitly exclude only the image field")
    if provenance.get("image_materialised") is not False:
        raise ValueError("provenance.json does not prove image bytes were not materialised")
    if provenance.get("image_decoded") is not False:
        raise ValueError("provenance.json does not prove image bytes were not decoded")

    pairing = provenance.get("pairing_status")
    if not isinstance(pairing, dict):
        raise TypeError("provenance.json must contain pairing_status")
    if pairing.get("recoverable_from_catalog") is not False:
        raise ValueError("DANI catalogue must remain marked pairing_recoverable=false")
    if pairing.get("documented_parent_group_field") is not None:
        raise ValueError("DANI catalogue must not invent a documented parent group")
    if pairing.get("internal_selection_blocker") != INTERNAL_SELECTION_BLOCKER:
        raise ValueError("provenance.json has an unexpected internal selection blocker")

    scope = provenance.get("scan_scope")
    if not isinstance(scope, dict):
        raise TypeError("provenance.json must contain scan_scope")
    complete_conditions = {
        "kind": scope.get("kind") == COMPLETE_SCAN_KIND,
        "partial": scope.get("partial") is False,
        "complete": scope.get("complete") is True,
        "eligible_for_candidate_selection": scope.get("eligible_for_candidate_selection") is False,
        "eligible_for_training": scope.get("eligible_for_training") is False,
        "eligible_for_external_descriptive_evaluation": (
            scope.get("eligible_for_external_descriptive_evaluation") is True
        ),
        "internal_selection_blocker": scope.get("internal_selection_blocker")
        == INTERNAL_SELECTION_BLOCKER,
        "available_shard_count": scope.get("available_shard_count") == len(locked_paths),
        "selected_shard_count": scope.get("selected_shard_count") == len(locked_paths),
        "limit_shards": scope.get("limit_shards") is None,
        "selected_shards": scope.get("selected_shards") == locked_paths,
    }
    failed_conditions = [name for name, valid in complete_conditions.items() if not valid]
    if failed_conditions:
        raise ValueError(
            "Refusing partial or invalid DANI source scan; invalid scan_scope fields: "
            + ", ".join(failed_conditions)
        )

    declared_row_count = _require_int(
        provenance.get("catalog_row_count"), field="catalog_row_count"
    )
    declared_rows_by_shard = provenance.get("rows_scanned_by_shard")
    if not isinstance(declared_rows_by_shard, dict):
        raise TypeError("provenance.json must contain rows_scanned_by_shard")
    if set(declared_rows_by_shard) != set(locked_paths):
        raise ValueError("rows_scanned_by_shard does not cover exactly the locked source shards")
    parsed_rows_by_shard: dict[str, int] = {}
    for shard_path, count in declared_rows_by_shard.items():
        parsed_rows_by_shard[shard_path] = _require_int(
            count, field=f"rows_scanned_by_shard.{shard_path}"
        )
    if sum(parsed_rows_by_shard.values()) != declared_row_count:
        raise ValueError("rows_scanned_by_shard does not sum to catalog_row_count")
    return declared_row_count, parsed_rows_by_shard, scope


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_label_reference_counts(path: Path, counts: Counter[tuple[str, str]]) -> None:
    rows = [
        {
            "label": label,
            "label_name": LABEL_NAMES[label],
            "reference": reference,
            "count": count,
        }
        for (label, reference), count in sorted(counts.items())
    ]
    _write_rows(path, ("label", "label_name", "reference", "count"), rows)


def _write_declared_size_counts(path: Path, counts: Counter[tuple[int, str]]) -> None:
    rows = [
        {
            "declared_size": size,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "count": count,
        }
        for (size, label), count in sorted(
            counts.items(), key=lambda item: (item[0][0], item[0][1])
        )
    ]
    _write_rows(path, ("declared_size", "label", "label_name", "count"), rows)


def _write_cell_counts(path: Path, counts: Counter[tuple[int, str, str, str, str]]) -> None:
    rows = [
        {
            "declared_size": size,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "reference": reference,
            "model": model,
            "gen_type": gen_type,
            "count": count,
        }
        for (size, label, reference, model, gen_type), count in sorted(
            counts.items(), key=lambda item: (item[0][0], item[0][1], item[0][3], item[0][4])
        )
    ]
    _write_rows(
        path,
        ("declared_size", "label", "label_name", "reference", "model", "gen_type", "count"),
        rows,
    )


def _write_category_class_counts(path: Path, counts: Counter[tuple[str, str, str]]) -> None:
    rows = [
        {
            "category": category,
            "class_id": class_id,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "count": count,
        }
        for (category, class_id, label), count in sorted(
            counts.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
        )
    ]
    _write_rows(path, ("category", "class_id", "label", "label_name", "count"), rows)


def _write_shard_label_counts(path: Path, counts: Counter[tuple[str, str]]) -> None:
    rows = [
        {
            "shard_path": shard_path,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "count": count,
        }
        for (shard_path, label), count in sorted(counts.items())
    ]
    _write_rows(path, ("shard_path", "label", "label_name", "count"), rows)


def _write_metadata_completeness(path: Path, counts: Counter[tuple[str, str]]) -> None:
    rows = [
        {"field": field, "status": status, "count": count}
        for (field, status), count in sorted(counts.items())
    ]
    _write_rows(path, ("field", "status", "count"), rows)


def _duplicate_statistics(
    values: Counter[str],
    labels: Mapping[str, set[str]],
    *,
    total_rows: int,
) -> tuple[dict[str, int], list[str]]:
    duplicates = sorted(value for value, count in values.items() if count > 1)
    cross_label = [value for value in duplicates if {"0", "1"}.issubset(labels[value])]
    return (
        {
            "total_rows": total_rows,
            "distinct_source_indexes": len(values),
            "duplicate_value_count": len(duplicates),
            "duplicate_rows_beyond_first": sum(values[value] - 1 for value in duplicates),
            "rows_in_duplicate_values": sum(values[value] for value in duplicates),
            "largest_group_size": max(values.values(), default=0),
            "cross_label_duplicate_value_count": len(cross_label),
            "cross_label_duplicate_row_count": sum(values[value] for value in cross_label),
        },
        duplicates,
    )


def _write_duplicate_groups(
    path: Path,
    duplicate_hashes: list[str],
    *,
    values: Counter[str],
    labels: Mapping[str, set[str]],
    sizes: Mapping[str, set[int]],
    models: Mapping[str, set[str]],
    gen_types: Mapping[str, set[str]],
    category_class: Mapping[str, set[str]],
) -> None:
    rows: list[dict[str, object]] = []
    for source_hash in duplicate_hashes:
        rows.append(
            {
                "source_index_hash": source_hash,
                "count": values[source_hash],
                "labels": "|".join(sorted(labels[source_hash])),
                "cross_label": {"0", "1"}.issubset(labels[source_hash]),
                "declared_sizes": "|".join(str(value) for value in sorted(sizes[source_hash])),
                "models": "|".join(sorted(models[source_hash])),
                "gen_types": "|".join(sorted(gen_types[source_hash])),
                "category_class_values": "|".join(sorted(category_class[source_hash])),
            }
        )
    _write_rows(
        path,
        (
            "source_index_hash",
            "count",
            "labels",
            "cross_label",
            "declared_sizes",
            "models",
            "gen_types",
            "category_class_values",
        ),
        rows,
    )


def _write_duplicate_signature_counts(
    path: Path,
    duplicate_hashes: list[str],
    *,
    labels: Mapping[str, set[str]],
    sizes: Mapping[str, set[int]],
    models: Mapping[str, set[str]],
    gen_types: Mapping[str, set[str]],
) -> None:
    signatures = Counter[str]()
    for source_hash in duplicate_hashes:
        signature = ";".join(
            (
                "labels=" + "|".join(sorted(labels[source_hash])),
                "sizes=" + "|".join(str(value) for value in sorted(sizes[source_hash])),
                "models=" + "|".join(sorted(models[source_hash])),
                "gen_types=" + "|".join(sorted(gen_types[source_hash])),
            )
        )
        signatures[signature] += 1
    rows = [
        {"signature": signature, "duplicate_source_index_group_count": count}
        for signature, count in sorted(signatures.items(), key=lambda item: (-item[1], item[0]))
    ]
    _write_rows(path, ("signature", "duplicate_source_index_group_count"), rows)


def audit_catalog(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Audit one complete DANI scanner output without modifying it or accessing a network."""
    source_directory = Path(input_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing DANI catalogue audit: {destination}")
    if not source_directory.is_dir():
        raise FileNotFoundError(
            f"DANI source catalogue directory does not exist: {source_directory}"
        )
    try:
        destination.resolve().relative_to(source_directory.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(
            "Audit output directory must not be placed inside the immutable input directory"
        )

    paths = {
        SOURCE_LOCK_NAME: source_directory / SOURCE_LOCK_NAME,
        SOURCE_CATALOG_NAME: source_directory / SOURCE_CATALOG_NAME,
        PROVENANCE_NAME: source_directory / PROVENANCE_NAME,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DANI source catalogue is incomplete; missing required files: " + ", ".join(missing)
        )
    source_hashes_before = {name: sha256_file(path) for name, path in paths.items()}
    source_lock = _read_json(paths[SOURCE_LOCK_NAME])
    provenance = _read_json(paths[PROVENANCE_NAME])
    _, revision, locked_paths = _validate_source_lock(source_lock)
    declared_row_count, declared_rows_by_shard, scan_scope = _validate_provenance(
        provenance,
        revision=revision,
        locked_paths=locked_paths,
        source_hashes=source_hashes_before,
    )

    label_reference_counts = Counter[tuple[str, str]]()
    size_label_counts = Counter[tuple[int, str]]()
    cells = Counter[tuple[int, str, str, str, str]]()
    category_class_counts = Counter[tuple[str, str, str]]()
    shard_label_counts = Counter[tuple[str, str]]()
    metadata_completeness = Counter[tuple[str, str]]()
    rows_by_shard = Counter[str]()
    source_indexes = Counter[str]()
    source_index_labels: defaultdict[str, set[str]] = defaultdict(set)
    source_index_sizes: defaultdict[str, set[int]] = defaultdict(set)
    source_index_models: defaultdict[str, set[str]] = defaultdict(set)
    source_index_gen_types: defaultdict[str, set[str]] = defaultdict(set)
    source_index_category_class: defaultdict[str, set[str]] = defaultdict(set)
    source_index_values: dict[str, str] = {}
    locators: set[str] = set()
    expected_row_index = {path: 0 for path in locked_paths}
    total_rows = 0

    with paths[SOURCE_CATALOG_NAME].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_catalog_headers(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"source_catalog.csv row {row_number} has more values than its header"
                )
            if row.get("repository_id") != REPOSITORY_ID:
                raise ValueError(
                    f"source_catalog.csv row {row_number} has a different repository_id"
                )
            if row.get("revision") != revision:
                raise ValueError(f"source_catalog.csv row {row_number} has a different revision")
            shard_path = row.get("shard_path")
            if shard_path not in expected_row_index:
                raise ValueError(f"source_catalog.csv row {row_number} refers to an unlocked shard")
            row_index = _parse_nonnegative_int(
                row.get("row_index"), field=f"row {row_number} row_index"
            )
            if row_index != expected_row_index[shard_path]:
                raise ValueError(
                    f"source_catalog.csv row {row_number} has a nonsequential row_index for {shard_path}"
                )
            expected_row_index[shard_path] += 1
            locator = row.get("locator")
            expected_locator = f"{REPOSITORY_ID}@{revision}:{shard_path}:{row_index}"
            if locator != expected_locator:
                raise ValueError(
                    f"source_catalog.csv row {row_number} has an invalid stable locator"
                )
            if locator in locators:
                raise ValueError(f"source_catalog.csv row {row_number} duplicates a stable locator")
            locators.add(locator)

            source_index_raw = row.get("source_index")
            if (
                not isinstance(source_index_raw, str)
                or not source_index_raw
                or source_index_raw != source_index_raw.strip()
            ):
                raise ValueError(f"source_catalog.csv row {row_number} has an invalid source_index")
            source_hash = row.get("source_index_hash")
            if source_hash != _source_index_hash(source_index_raw):
                raise ValueError(
                    f"source_catalog.csv row {row_number} has an invalid source_index_hash"
                )
            if row.get("source_index_group_id") != f"upstream-index:{source_hash}":
                raise ValueError(
                    f"source_catalog.csv row {row_number} has an invalid source_index_group_id"
                )
            prior_index = source_index_values.setdefault(source_hash, source_index_raw)
            if prior_index != source_index_raw:
                raise RuntimeError("Unexpected source-index SHA-256 collision")

            declared_size = _parse_positive_int(
                row.get("declared_size"), field=f"row {row_number} declared_size"
            )
            reference, expected_label = _parse_reference(
                row.get("reference"), row_number=row_number
            )
            label = row.get("label")
            if label != expected_label:
                raise ValueError(
                    f"source_catalog.csv row {row_number} has inconsistent reference and label"
                )

            category = _normalise_text(row.get("category"))
            class_id = _normalise_text(row.get("class_id"))
            model = _normalise_text(row.get("model"))
            gen_type = _normalise_text(row.get("gen_type"))
            for field, value in (
                ("category", category),
                ("class_id", class_id),
                ("model", model),
                ("gen_type", gen_type),
            ):
                metadata_completeness[
                    (field, "missing" if value == MISSING_VALUE else "present")
                ] += 1

            total_rows += 1
            rows_by_shard[shard_path] += 1
            label_reference_counts[(label, "true" if reference else "false")] += 1
            size_label_counts[(declared_size, label)] += 1
            cells[(declared_size, label, "true" if reference else "false", model, gen_type)] += 1
            category_class_counts[(category, class_id, label)] += 1
            shard_label_counts[(shard_path, label)] += 1
            source_indexes[source_hash] += 1
            source_index_labels[source_hash].add(label)
            source_index_sizes[source_hash].add(declared_size)
            source_index_models[source_hash].add(model)
            source_index_gen_types[source_hash].add(gen_type)
            source_index_category_class[source_hash].add(f"{category}|{class_id}")

    if total_rows != declared_row_count:
        raise ValueError(
            "source_catalog.csv row count does not match provenance.json: "
            f"observed {total_rows}, declared {declared_row_count}"
        )
    observed_rows_by_shard = {path: rows_by_shard[path] for path in locked_paths}
    if observed_rows_by_shard != declared_rows_by_shard:
        raise ValueError("source_catalog.csv per-shard row counts do not match provenance.json")
    for shard_path, next_index in expected_row_index.items():
        if next_index != declared_rows_by_shard[shard_path]:
            raise ValueError(f"source_catalog.csv row indexes do not cover {shard_path} exactly")

    source_hashes_after = {name: sha256_file(path) for name, path in paths.items()}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError(
            "Source catalogue changed during audit; refusing to write derived results"
        )

    duplicate_stats, duplicate_hashes = _duplicate_statistics(
        source_indexes,
        source_index_labels,
        total_rows=total_rows,
    )
    timestamp = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    summary: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at_utc": timestamp,
        "audit_kind": "offline_dani_metadata_and_provenance_audit_not_candidate_selection",
        "source_catalog_provenance": {
            "input_directory": str(source_directory),
            "provenance_file": PROVENANCE_NAME,
            "provenance_sha256": source_hashes_before[PROVENANCE_NAME],
            "scanner_schema_version": provenance["schema_version"],
            "repository_id": REPOSITORY_ID,
            "requested_revision": provenance.get("requested_revision"),
            "resolved_revision": revision,
            "source_lock_file": SOURCE_LOCK_NAME,
            "source_lock_sha256": source_hashes_before[SOURCE_LOCK_NAME],
            "source_catalog_file": SOURCE_CATALOG_NAME,
            "source_catalog_sha256": source_hashes_before[SOURCE_CATALOG_NAME],
            "scan_scope": scan_scope,
            "catalog_row_count_declared": declared_row_count,
            "catalog_row_count_observed": total_rows,
            "rows_scanned_by_shard_observed": observed_rows_by_shard,
        },
        "label_reference_counts": {
            f"label_{label}_reference_{reference}": count
            for (label, reference), count in sorted(label_reference_counts.items())
        },
        "declared_size_label_counts": {
            f"{size}px_label_{label}": count
            for (size, label), count in sorted(size_label_counts.items())
        },
        "metadata_completeness": {
            field: {
                "present": metadata_completeness[(field, "present")],
                "missing": metadata_completeness[(field, "missing")],
            }
            for field in ("category", "class_id", "model", "gen_type")
        },
        "duplicates": {"source_index": duplicate_stats},
        "eligibility": {
            "eligible_for_descriptive_catalog_audit": True,
            "eligible_for_candidate_selection": False,
            "eligible_for_split_assignment": False,
            "eligible_for_training": False,
            "eligible_for_model_selection": False,
            "blockers": list(AUDIT_BLOCKERS),
            "note": (
                "A duplicate or cross-label source index is a potential linkage observation, not "
                "proof of a semantic pair and not permission to assign a split."
            ),
        },
        "table_files": {
            "label_reference": "label_reference_counts.csv",
            "declared_size_label": "declared_size_label_counts.csv",
            "declared_size_model_gen_type_reference": (
                "declared_size_model_gen_type_reference_counts.csv"
            ),
            "category_class_id_label": "category_class_id_label_counts.csv",
            "shard_label": "shard_label_counts.csv",
            "metadata_completeness": "metadata_completeness_counts.csv",
            "source_index_duplicate_groups": "duplicate_source_index_groups.csv",
            "source_index_duplicate_signatures": "source_index_group_signature_counts.csv",
        },
    }

    output = require_fresh_output_dir(destination)
    _write_json(output / "summary.json", summary)
    _write_label_reference_counts(output / "label_reference_counts.csv", label_reference_counts)
    _write_declared_size_counts(output / "declared_size_label_counts.csv", size_label_counts)
    _write_cell_counts(output / "declared_size_model_gen_type_reference_counts.csv", cells)
    _write_category_class_counts(
        output / "category_class_id_label_counts.csv", category_class_counts
    )
    _write_shard_label_counts(output / "shard_label_counts.csv", shard_label_counts)
    _write_metadata_completeness(output / "metadata_completeness_counts.csv", metadata_completeness)
    _write_rows(
        output / "source_index_duplicate_statistics.csv",
        ("identifier", "metric", "count"),
        [
            {"identifier": "source_index", "metric": metric, "count": count}
            for metric, count in sorted(duplicate_stats.items())
        ],
    )
    _write_duplicate_groups(
        output / "duplicate_source_index_groups.csv",
        duplicate_hashes,
        values=source_indexes,
        labels=source_index_labels,
        sizes=source_index_sizes,
        models=source_index_models,
        gen_types=source_index_gen_types,
        category_class=source_index_category_class,
    )
    _write_duplicate_signature_counts(
        output / "source_index_group_signature_counts.csv",
        duplicate_hashes,
        labels=source_index_labels,
        sizes=source_index_sizes,
        models=source_index_models,
        gen_types=source_index_gen_types,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a complete DANI metadata catalogue offline; no image download, selection, "
            "or network access occurs."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Complete scanner output with source_lock.json, source_catalog.csv, and provenance.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for immutable derived audit evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_catalog(args.input_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
