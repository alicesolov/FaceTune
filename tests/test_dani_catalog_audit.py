from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_dani_catalog.py"
SPEC = importlib.util.spec_from_file_location("audit_dani_catalog", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

REPOSITORY_ID = "Renyang/DANI"
REVISION = "a" * 40
SHARDS = ("data/a.parquet", "data/b.parquet")


def sample_row(
    *,
    shard_path: str = SHARDS[0],
    row_index: int = 0,
    source_index: str = "source-0",
    reference: bool = False,
    declared_size: int = 1024,
    **overrides: str,
) -> dict[str, str]:
    source_hash = hashlib.sha256(f"dani-source-index:{source_index}".encode()).hexdigest()
    row = {
        "locator": f"{REPOSITORY_ID}@{REVISION}:{shard_path}:{row_index}",
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "shard_path": shard_path,
        "row_index": str(row_index),
        "source_index": source_index,
        "source_index_hash": source_hash,
        "source_index_group_id": f"upstream-index:{source_hash}",
        "declared_size": str(declared_size),
        "category": "outdoor",
        "class_id": "17",
        "model": "COCO" if reference else "SD_XL",
        "gen_type": "reference" if reference else "T2I",
        "reference": str(reference),
        "label": "0" if reference else "1",
    }
    row.update(overrides)
    return row


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _refresh_provenance_hashes(source_dir: Path) -> None:
    provenance_path = source_dir / audit.PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_lock_sha256"] = audit.sha256_file(source_dir / audit.SOURCE_LOCK_NAME)
    provenance["source_catalog_sha256"] = audit.sha256_file(source_dir / audit.SOURCE_CATALOG_NAME)
    _write_json(provenance_path, provenance)


def write_scanner_output(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    partial: bool = False,
    raw_column: str | None = None,
    candidate_selection: bool = False,
) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_lock = {
        "schema_version": audit.SOURCE_LOCK_SCHEMA_VERSION,
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "license": audit.SOURCE_LICENSE,
        "repo_type": "dataset",
        "tree_recursive": True,
        "source_schema_columns": list(audit.SOURCE_SCHEMA_COLUMNS),
        "metadata_columns": list(audit.META_COLUMNS),
        "excluded_binary_columns": ["image"],
        "shard_count": len(SHARDS),
        "shards": [
            {
                "path": path,
                "size": 123,
                "blob_id": None,
                "lfs": {"size": 123, "sha256": "b" * 64, "pointer_size": 135},
                "xet_hash": None,
            }
            for path in SHARDS
        ],
    }
    source_lock_path = source_dir / audit.SOURCE_LOCK_NAME
    _write_json(source_lock_path, source_lock)

    catalog_path = source_dir / audit.SOURCE_CATALOG_NAME
    columns = [*audit.CATALOG_COLUMNS]
    if raw_column is not None:
        columns.append(raw_column)
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            candidate = dict(row)
            if raw_column is not None:
                candidate[raw_column] = "forbidden value"
            writer.writerow(candidate)

    rows_by_shard = Counter(row["shard_path"] for row in rows)
    if partial:
        scope = {
            "kind": "partial_scout",
            "partial": True,
            "complete": False,
            "eligible_for_candidate_selection": False,
            "eligible_for_training": False,
            "eligible_for_external_descriptive_evaluation": False,
            "internal_selection_blocker": audit.INTERNAL_SELECTION_BLOCKER,
            "available_shard_count": len(SHARDS),
            "selected_shard_count": 1,
            "limit_shards": 1,
            "selected_shards": [SHARDS[0]],
        }
    else:
        scope = {
            "kind": audit.COMPLETE_SCAN_KIND,
            "partial": False,
            "complete": True,
            "eligible_for_candidate_selection": candidate_selection,
            "eligible_for_training": False,
            "eligible_for_external_descriptive_evaluation": True,
            "internal_selection_blocker": audit.INTERNAL_SELECTION_BLOCKER,
            "available_shard_count": len(SHARDS),
            "selected_shard_count": len(SHARDS),
            "limit_shards": None,
            "selected_shards": list(SHARDS),
        }
    provenance = {
        "schema_version": audit.SCAN_SCHEMA_VERSION,
        "repository_id": REPOSITORY_ID,
        "requested_revision": REVISION,
        "revision": REVISION,
        "resolved_revision": REVISION,
        "source_lock": audit.SOURCE_LOCK_NAME,
        "source_lock_sha256": audit.sha256_file(source_lock_path),
        "source_catalog": audit.SOURCE_CATALOG_NAME,
        "source_catalog_sha256": audit.sha256_file(catalog_path),
        "catalog_kind": audit.CATALOG_KIND,
        "catalog_row_count": len(rows),
        "metadata_columns": list(audit.META_COLUMNS),
        "excluded_binary_columns": ["image"],
        "image_materialised": False,
        "image_decoded": False,
        "pairing_status": {
            "recoverable_from_catalog": False,
            "documented_parent_group_field": None,
            "source_index_role": "upstream image-level identifier only",
            "internal_selection_blocker": audit.INTERNAL_SELECTION_BLOCKER,
        },
        "rows_scanned_by_shard": {path: rows_by_shard[path] for path in SHARDS},
        "scan_scope": scope,
    }
    _write_json(source_dir / audit.PROVENANCE_NAME, provenance)
    return source_dir


def fixed_now() -> datetime:
    return datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def test_refuses_incomplete_scout_before_writing_output(tmp_path: Path) -> None:
    source = tmp_path / "incomplete-scout"
    source.mkdir()
    (source / audit.SOURCE_LOCK_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)

    assert not (tmp_path / "audit").exists()


def test_refuses_partial_source_scan(tmp_path: Path) -> None:
    source = write_scanner_output(tmp_path, [sample_row()], partial=True)

    with pytest.raises(ValueError, match="partial or invalid"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)

    assert not (tmp_path / "audit").exists()


def test_refuses_candidate_selection_even_for_complete_dani_scan(tmp_path: Path) -> None:
    source = write_scanner_output(tmp_path, [sample_row()], candidate_selection=True)

    with pytest.raises(ValueError, match="eligible_for_candidate_selection"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)

    assert not (tmp_path / "audit").exists()


def test_refuses_catalog_sha_mismatch(tmp_path: Path) -> None:
    source = write_scanner_output(tmp_path, [sample_row()])
    provenance_path = source / audit.PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_catalog_sha256"] = "0" * 64
    _write_json(provenance_path, provenance)

    with pytest.raises(ValueError, match="source_catalog.csv SHA-256 does not match"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)


def test_refuses_existing_output_without_touching_it(tmp_path: Path) -> None:
    source = write_scanner_output(tmp_path, [sample_row()])
    output = tmp_path / "audit"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit.audit_catalog(source, output, now=fixed_now)

    assert marker.read_text(encoding="utf-8") == "do not overwrite"


def test_reports_distribution_duplicate_statistics_and_keeps_input_immutable(
    tmp_path: Path,
) -> None:
    rows = [
        sample_row(source_index="shared", reference=True),
        sample_row(row_index=1, source_index="shared", reference=False),
        sample_row(
            shard_path=SHARDS[1],
            row_index=0,
            source_index="unique",
            reference=False,
            category="",
            class_id="",
            model="",
            gen_type="",
        ),
    ]
    source = write_scanner_output(tmp_path, rows)
    input_hashes_before = {
        name: audit.sha256_file(source / name)
        for name in (audit.SOURCE_LOCK_NAME, audit.SOURCE_CATALOG_NAME, audit.PROVENANCE_NAME)
    }

    report = audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)

    assert report["source_catalog_provenance"]["catalog_row_count_observed"] == 3
    assert report["duplicates"]["source_index"]["duplicate_value_count"] == 1
    assert report["duplicates"]["source_index"]["cross_label_duplicate_value_count"] == 1
    assert report["metadata_completeness"]["category"] == {"present": 2, "missing": 1}
    assert report["eligibility"]["eligible_for_candidate_selection"] is False
    assert report["eligibility"]["eligible_for_training"] is False
    assert report["created_at_utc"] == "2026-08-29T12:00:00+00:00"

    input_hashes_after = {
        name: audit.sha256_file(source / name)
        for name in (audit.SOURCE_LOCK_NAME, audit.SOURCE_CATALOG_NAME, audit.PROVENANCE_NAME)
    }
    assert input_hashes_after == input_hashes_before
    output = tmp_path / "audit"
    assert (output / "summary.json").is_file()
    assert (output / "declared_size_model_gen_type_reference_counts.csv").is_file()
    assert (output / "duplicate_source_index_groups.csv").is_file()
    with (output / "duplicate_source_index_groups.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        duplicate_rows = list(csv.DictReader(handle))
    assert duplicate_rows[0]["source_index_hash"] != "shared"
    assert duplicate_rows[0]["cross_label"] == "True"


@pytest.mark.parametrize("raw_column", ["image", "image_data", "prompt"])
def test_refuses_raw_image_or_prompt_columns(tmp_path: Path, raw_column: str) -> None:
    source = write_scanner_output(tmp_path, [sample_row()], raw_column=raw_column)

    with pytest.raises(ValueError, match="must not contain raw image/prompt columns"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)

    assert not (tmp_path / "audit").exists()


def test_refuses_invalid_source_index_hash(tmp_path: Path) -> None:
    row = sample_row(source_index="incorrect-hash", source_index_hash="0" * 64)
    source = write_scanner_output(tmp_path, [row])

    with pytest.raises(ValueError, match="invalid source_index_hash"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)


def test_refuses_inconsistent_reference_label(tmp_path: Path) -> None:
    row = sample_row(reference=True, label="1")
    source = write_scanner_output(tmp_path, [row])

    with pytest.raises(ValueError, match="inconsistent reference and label"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)


def test_refuses_row_index_gap_even_when_locator_matches(tmp_path: Path) -> None:
    row = sample_row(row_index=1)
    source = write_scanner_output(tmp_path, [row])

    with pytest.raises(ValueError, match="nonsequential row_index"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)


def test_refuses_per_shard_count_mismatch(tmp_path: Path) -> None:
    source = write_scanner_output(tmp_path, [sample_row()])
    provenance_path = source / audit.PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["rows_scanned_by_shard"][SHARDS[0]] = 2
    _write_json(provenance_path, provenance)

    with pytest.raises(ValueError, match="rows_scanned_by_shard does not sum"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)
