"""Audit a complete, metadata-only HighRes-v1 source catalogue offline.

This is deliberately a *catalogue* audit, not a candidate selector.  It refuses partial scans,
does not contact a remote service, and never reads image bytes.  The input is the immutable
three-file output of :mod:`ai_image_detector.community_forensics`; the audit is written to a
fresh sibling directory so the source evidence is not modified.
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

SOURCE_LOCK_SCHEMA_VERSION = "community_forensics_source_lock_v1"
SCAN_SCHEMA_VERSION = "community_forensics_metadata_scan_v1"
COMPLETE_SCAN_KIND = "complete_source_scan"

# This is the public, metadata-only schema emitted by community_forensics.py.  The explicit
# schema gives a future scanner change a fail-closed boundary rather than silently weakening the
# audit of a new catalogue shape.
CATALOG_COLUMNS = (
    "locator",
    "repository_id",
    "revision",
    "shard_path",
    "row_index",
    "image_name",
    "format",
    "source_width",
    "source_height",
    "mode",
    "model_name",
    "nsfw_flag",
    "prompt_hash",
    "prompt_present",
    "content_group_id",
    "real_source",
    "subset",
    "source_split",
    "label",
    "architecture",
)
FORBIDDEN_RAW_COLUMNS = frozenset({"image_data", "prompt"})
FREQUENCY_FIELDS = (
    "source_geometry",
    "format",
    "mode",
    "nsfw",
    "source_split",
    "architecture",
    "subset",
    "real_source",
)
QUALITY_GATES = (
    "exact_512x512",
    "png",
    "rgb",
    "explicit_non_nsfw",
    "valid_label",
)
LABEL_NAMES = {"0": "real", "1": "ai_generated", "invalid": "invalid"}
MISSING_VALUE = "(missing)"
INVALID_GEOMETRY = "(invalid)"


def sha256_file(path: str | Path) -> str:
    """Return a byte-exact SHA-256 digest without parsing the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_fresh_output_dir(output_dir: str | Path) -> Path:
    """Create a new output directory without mixing or overwriting audit evidence."""
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing HighRes catalogue audit: {path}")
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


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    text = _require_text(value, field=field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def _normalise_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text if text else MISSING_VALUE


def _canonical_text(value: object) -> str:
    return _normalise_text(value).casefold()


def _parse_positive_int(value: object) -> int | None:
    text = str(value).strip()
    if not text or not text.isdecimal():
        return None
    parsed = int(text)
    return parsed if parsed > 0 else None


def _parse_label(value: object) -> str:
    text = str(value).strip()
    return text if text in {"0", "1"} else "invalid"


def _parse_nsfw(value: object) -> bool | None:
    token = _canonical_text(value)
    if token in {"false", "0", "no"}:
        return False
    if token in {"true", "1", "yes"}:
        return True
    return None


def _nsfw_frequency_value(value: object) -> str:
    parsed = _parse_nsfw(value)
    if parsed is False:
        return "false"
    if parsed is True:
        return "true"
    return "missing_or_unrecognised"


def _geometry_value(width: object, height: object) -> tuple[str, bool]:
    parsed_width = _parse_positive_int(width)
    parsed_height = _parse_positive_int(height)
    if parsed_width is None or parsed_height is None:
        return INVALID_GEOMETRY, False
    return f"{parsed_width}x{parsed_height}", parsed_width == 512 and parsed_height == 512


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


def _validate_complete_source(
    source_lock: Mapping[str, object],
    provenance: Mapping[str, object],
    *,
    source_hashes: Mapping[str, str],
) -> tuple[str, str, set[str]]:
    """Fail closed unless the scanner produced one complete, internally consistent source lock."""
    if source_lock.get("schema_version") != SOURCE_LOCK_SCHEMA_VERSION:
        raise ValueError("source_lock.json has an unsupported schema_version")
    if provenance.get("schema_version") != SCAN_SCHEMA_VERSION:
        raise ValueError("provenance.json has an unsupported schema_version")
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

    repository_id = _require_text(
        source_lock.get("repository_id"), field="source_lock.repository_id"
    )
    revision = _require_text(source_lock.get("revision"), field="source_lock.revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(
            "source_lock.revision must be an immutable 40-character lowercase commit SHA"
        )
    if provenance.get("repository_id") != repository_id:
        raise ValueError("repository_id differs between source_lock.json and provenance.json")
    if provenance.get("revision") != revision or provenance.get("resolved_revision") != revision:
        raise ValueError("resolved revision differs between source_lock.json and provenance.json")
    if source_lock.get("excluded_binary_columns") != ["image_data"]:
        raise ValueError("source_lock.json must explicitly exclude only image_data")
    if provenance.get("excluded_binary_columns") != ["image_data"]:
        raise ValueError("provenance.json must explicitly exclude image_data")
    if provenance.get("image_data_materialised") is not False:
        raise ValueError("provenance.json does not prove image bytes were not materialised")
    if provenance.get("image_data_decoded") is not False:
        raise ValueError("provenance.json does not prove image bytes were not decoded")

    raw_shards = source_lock.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("source_lock.json must contain a nonempty shard list")
    locked_paths: list[str] = []
    for index, shard in enumerate(raw_shards):
        if not isinstance(shard, dict):
            raise TypeError(f"source_lock.shards[{index}] must be an object")
        locked_paths.append(
            _require_text(shard.get("path"), field=f"source_lock.shards[{index}].path")
        )
    if len(locked_paths) != len(set(locked_paths)):
        raise ValueError("source_lock.json contains duplicate shard paths")
    if source_lock.get("shard_count") != len(locked_paths):
        raise ValueError("source_lock.json shard_count does not match its shard list")

    scope = provenance.get("scan_scope")
    if not isinstance(scope, dict):
        raise TypeError("provenance.json must contain scan_scope")
    complete_conditions = {
        "kind": scope.get("kind") == COMPLETE_SCAN_KIND,
        "partial": scope.get("partial") is False,
        "complete": scope.get("complete") is True,
        "eligible_for_candidate_selection": scope.get("eligible_for_candidate_selection") is True,
        "available_shard_count": scope.get("available_shard_count") == len(locked_paths),
        "selected_shard_count": scope.get("selected_shard_count") == len(locked_paths),
        "limit_shards": scope.get("limit_shards") is None,
    }
    failed_conditions = [name for name, valid in complete_conditions.items() if not valid]
    if failed_conditions:
        raise ValueError(
            "Refusing partial or noncomplete source scan; invalid scan_scope fields: "
            + ", ".join(failed_conditions)
        )
    selected_paths = scope.get("selected_shards")
    if not isinstance(selected_paths, list) or set(selected_paths) != set(locked_paths):
        raise ValueError("scan_scope.selected_shards does not cover the complete source lock")
    return repository_id, revision, set(locked_paths)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_frequency_table(path: Path, counter: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("value", "count"))
        writer.writeheader()
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"value": value, "count": count})


def _write_model_counts(path: Path, counts: Counter[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("label", "label_name", "model_name", "count"))
        writer.writeheader()
        for (label, model_name), count in sorted(
            counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
        ):
            writer.writerow(
                {
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "model_name": model_name,
                    "count": count,
                }
            )


def _duplicate_summary(
    values: Counter[str], labels: Mapping[str, set[str]], *, blank_rows: int, total_rows: int
) -> tuple[dict[str, int], list[dict[str, object]]]:
    duplicate_groups = [(value, count) for value, count in values.items() if count > 1]
    cross_label_groups = [
        (value, count) for value, count in duplicate_groups if {"0", "1"}.issubset(labels[value])
    ]
    records = [
        {
            "value": value,
            "count": count,
            "labels": "|".join(sorted(labels[value])),
            "cross_label": {"0", "1"}.issubset(labels[value]),
        }
        for value, count in duplicate_groups
    ]
    records.sort(key=lambda item: (-int(item["count"]), str(item["value"])))
    return (
        {
            "total_rows": total_rows,
            "blank_identifier_rows": blank_rows,
            "nonblank_identifier_rows": total_rows - blank_rows,
            "distinct_nonblank_values": len(values),
            "duplicate_value_count": len(duplicate_groups),
            "duplicate_rows_beyond_first": sum(count - 1 for _, count in duplicate_groups),
            "rows_in_duplicate_values": sum(count for _, count in duplicate_groups),
            "largest_group_size": max(values.values(), default=0),
            "cross_label_duplicate_value_count": len(cross_label_groups),
            "cross_label_duplicate_row_count": sum(count for _, count in cross_label_groups),
        },
        records,
    )


def _write_duplicate_groups(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("value", "count", "labels", "cross_label"))
        writer.writeheader()
        writer.writerows(records)


def _write_duplicate_statistics(path: Path, statistics: Mapping[str, Mapping[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("identifier", "metric", "count"))
        writer.writeheader()
        for identifier in sorted(statistics):
            for metric, count in sorted(statistics[identifier].items()):
                writer.writerow({"identifier": identifier, "metric": metric, "count": count})


def _class_balance(label_counts: Mapping[str, int]) -> dict[str, int | float | None]:
    real_count = label_counts.get("0", 0)
    ai_count = label_counts.get("1", 0)
    valid_total = real_count + ai_count
    majority = max(real_count, ai_count)
    minority = min(real_count, ai_count)
    return {
        "real_label_0": real_count,
        "ai_generated_label_1": ai_count,
        "valid_label_total": valid_total,
        "invalid_label_rows": label_counts.get("invalid", 0),
        "real_fraction": None if valid_total == 0 else real_count / valid_total,
        "ai_generated_fraction": None if valid_total == 0 else ai_count / valid_total,
        "absolute_count_difference": abs(real_count - ai_count),
        "minority_to_majority_ratio": None if majority == 0 else minority / majority,
    }


def _validate_row_provenance(
    row: Mapping[str | None, str | None],
    *,
    row_number: int,
    repository_id: str,
    revision: str,
    locked_paths: set[str],
) -> tuple[str, int]:
    if None in row:
        raise ValueError(f"source_catalog.csv row {row_number} has more values than its header")
    if row.get("repository_id") != repository_id:
        raise ValueError(f"source_catalog.csv row {row_number} has a different repository_id")
    if row.get("revision") != revision:
        raise ValueError(f"source_catalog.csv row {row_number} has a different revision")
    shard_path = row.get("shard_path")
    if shard_path not in locked_paths:
        raise ValueError(f"source_catalog.csv row {row_number} refers to an unlocked shard")
    raw_row_index = row.get("row_index")
    parsed_index = _parse_positive_int(raw_row_index)
    if raw_row_index == "0":
        parsed_index = 0
    if parsed_index is None:
        raise ValueError(f"source_catalog.csv row {row_number} has an invalid row_index")
    expected_locator = f"{repository_id}@{revision}:{shard_path}:{parsed_index}"
    if row.get("locator") != expected_locator:
        raise ValueError(f"source_catalog.csv row {row_number} has an invalid stable locator")
    return shard_path, parsed_index


def audit_catalog(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Audit one complete scanner output directory without modifying it or accessing a network."""
    source_directory = Path(input_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing HighRes catalogue audit: {destination}"
        )
    if not source_directory.is_dir():
        raise FileNotFoundError(
            f"HighRes source catalogue directory does not exist: {source_directory}"
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
            "HighRes source catalogue is incomplete; missing required files: " + ", ".join(missing)
        )

    source_hashes_before = {name: sha256_file(path) for name, path in paths.items()}
    source_lock = _read_json(paths[SOURCE_LOCK_NAME])
    provenance = _read_json(paths[PROVENANCE_NAME])
    repository_id, revision, locked_paths = _validate_complete_source(
        source_lock, provenance, source_hashes=source_hashes_before
    )

    frequencies: dict[str, Counter[str]] = {field: Counter() for field in FREQUENCY_FIELDS}
    labels = Counter[str]()
    eligible_labels = Counter[str]()
    model_counts = Counter[tuple[str, str]]()
    gate_accepts = Counter[str]()
    gate_rejects = Counter[str]()
    rejection_reasons = Counter[str]()
    rows_by_shard = Counter[str]()
    image_names = Counter[str]()
    image_name_labels: defaultdict[str, set[str]] = defaultdict(set)
    content_groups = Counter[str]()
    content_group_labels: defaultdict[str, set[str]] = defaultdict(set)
    blank_image_names = 0
    blank_content_groups = 0
    total_rows = 0

    with paths[SOURCE_CATALOG_NAME].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_catalog_headers(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            shard_path, _ = _validate_row_provenance(
                row,
                row_number=row_number,
                repository_id=repository_id,
                revision=revision,
                locked_paths=locked_paths,
            )
            total_rows += 1
            rows_by_shard[shard_path] += 1

            geometry, is_exact_512 = _geometry_value(row["source_width"], row["source_height"])
            is_png = _canonical_text(row["format"]) == "png"
            is_rgb = _canonical_text(row["mode"]) == "rgb"
            is_explicit_non_nsfw = _parse_nsfw(row["nsfw_flag"]) is False
            label = _parse_label(row["label"])
            is_valid_label = label != "invalid"
            gates = {
                "exact_512x512": is_exact_512,
                "png": is_png,
                "rgb": is_rgb,
                "explicit_non_nsfw": is_explicit_non_nsfw,
                "valid_label": is_valid_label,
            }
            for gate, passed in gates.items():
                if passed:
                    gate_accepts[gate] += 1
                else:
                    gate_rejects[gate] += 1
                    rejection_reasons[f"not_{gate}"] += 1
            quality_eligible = all(gates.values())

            frequencies["source_geometry"][geometry] += 1
            frequencies["format"][_normalise_text(row["format"]).upper()] += 1
            frequencies["mode"][_normalise_text(row["mode"]).upper()] += 1
            frequencies["nsfw"][_nsfw_frequency_value(row["nsfw_flag"])] += 1
            for field in ("source_split", "architecture", "subset", "real_source"):
                frequencies[field][_normalise_text(row[field])] += 1

            labels[label] += 1
            model_counts[(label, _normalise_text(row["model_name"]))] += 1
            if quality_eligible:
                eligible_labels[label] += 1

            image_name = _normalise_text(row["image_name"])
            if image_name == MISSING_VALUE:
                blank_image_names += 1
            else:
                image_names[image_name] += 1
                image_name_labels[image_name].add(label)
            content_group = _normalise_text(row["content_group_id"])
            if content_group == MISSING_VALUE:
                blank_content_groups += 1
            else:
                content_groups[content_group] += 1
                content_group_labels[content_group].add(label)

    declared_row_count = _require_int(
        provenance.get("catalog_row_count"), field="catalog_row_count"
    )
    if declared_row_count != total_rows:
        raise ValueError(
            "source_catalog.csv row count does not match provenance.json: "
            f"observed {total_rows}, declared {declared_row_count}"
        )
    declared_rows_by_shard = provenance.get("rows_scanned_by_shard")
    if not isinstance(declared_rows_by_shard, dict):
        raise TypeError("provenance.json must contain rows_scanned_by_shard")
    if set(declared_rows_by_shard) != locked_paths:
        raise ValueError("rows_scanned_by_shard does not cover exactly the locked source shards")
    if any(
        not isinstance(count, int) or isinstance(count, bool)
        for count in declared_rows_by_shard.values()
    ):
        raise ValueError("rows_scanned_by_shard counts must be integers")
    if sum(declared_rows_by_shard.values()) != total_rows:
        raise ValueError("rows_scanned_by_shard does not sum to catalog_row_count")
    observed_rows_by_shard = {path: rows_by_shard[path] for path in locked_paths}
    if observed_rows_by_shard != declared_rows_by_shard:
        raise ValueError("source_catalog.csv per-shard row counts do not match provenance.json")

    source_hashes_after = {name: sha256_file(path) for name, path in paths.items()}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError(
            "Source catalogue changed during audit; refusing to write derived results"
        )

    image_name_stats, image_name_records = _duplicate_summary(
        image_names, image_name_labels, blank_rows=blank_image_names, total_rows=total_rows
    )
    content_group_stats, content_group_records = _duplicate_summary(
        content_groups, content_group_labels, blank_rows=blank_content_groups, total_rows=total_rows
    )
    quality_counts = {
        gate: {"accepted": gate_accepts[gate], "rejected": gate_rejects[gate]}
        for gate in QUALITY_GATES
    }
    quality_counts["all_quality_gates"] = {
        "accepted": sum(eligible_labels.values()),
        "rejected": total_rows - sum(eligible_labels.values()),
    }
    timestamp = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    summary: dict[str, object] = {
        "schema_version": "highres_catalog_audit_v1",
        "created_at_utc": timestamp,
        "audit_kind": "offline_catalogue_quality_and_provenance_audit_not_candidate_selection",
        "source_catalog_provenance": {
            "input_directory": str(source_directory),
            "provenance_file": PROVENANCE_NAME,
            "provenance_sha256": source_hashes_before[PROVENANCE_NAME],
            "scanner_schema_version": provenance["schema_version"],
            "repository_id": repository_id,
            "requested_revision": provenance.get("requested_revision"),
            "resolved_revision": revision,
            "source_lock_file": SOURCE_LOCK_NAME,
            "source_lock_sha256": source_hashes_before[SOURCE_LOCK_NAME],
            "source_catalog_file": SOURCE_CATALOG_NAME,
            "source_catalog_sha256": source_hashes_before[SOURCE_CATALOG_NAME],
            "scan_scope": provenance["scan_scope"],
            "catalog_row_count_declared": declared_row_count,
            "catalog_row_count_observed": total_rows,
        },
        "total_rows": total_rows,
        "label_counts": {label: labels[label] for label in ("0", "1", "invalid")},
        "class_balance": {
            "valid_catalog_labels": _class_balance(labels),
            "all_quality_gates": _class_balance(eligible_labels),
        },
        "quality_gates": {
            "criteria": quality_counts,
            "rejection_reason_counts_overlap_allowed": dict(sorted(rejection_reasons.items())),
            "notes": (
                "Each criterion is counted independently, so rejection reasons can overlap. "
                "all_quality_gates requires every criterion to pass."
            ),
        },
        "frequency_table_files": {field: f"{field}_counts.csv" for field in FREQUENCY_FIELDS},
        "model_counts_by_label_file": "model_counts_by_label.csv",
        "duplicates": {
            "image_name": image_name_stats,
            "content_group_id": content_group_stats,
        },
    }

    output = require_fresh_output_dir(destination)
    _write_json(output / "summary.json", summary)
    for field, counter in frequencies.items():
        _write_frequency_table(output / f"{field}_counts.csv", counter)
    _write_frequency_table(output / "label_counts.csv", Counter(labels))
    _write_frequency_table(output / "quality_eligible_label_counts.csv", Counter(eligible_labels))
    with (output / "quality_gate_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("gate", "accepted", "rejected"))
        writer.writeheader()
        for gate in (*QUALITY_GATES, "all_quality_gates"):
            writer.writerow({"gate": gate, **quality_counts[gate]})
    _write_model_counts(output / "model_counts_by_label.csv", model_counts)
    duplicate_statistics = {
        "image_name": image_name_stats,
        "content_group_id": content_group_stats,
    }
    _write_duplicate_statistics(output / "duplicate_statistics.csv", duplicate_statistics)
    _write_duplicate_groups(output / "duplicate_image_name_groups.csv", image_name_records)
    _write_duplicate_groups(output / "duplicate_content_group_id_groups.csv", content_group_records)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a complete HighRes-v1 CommunityForensics metadata catalogue offline; "
            "no image download, selection, or network access occurs."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Complete scanner output directory with source_lock.json, source_catalog.csv, and provenance.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for the immutable derived audit evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_catalog(args.input_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
