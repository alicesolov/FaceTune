from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_highres_catalog.py"
SPEC = importlib.util.spec_from_file_location("audit_highres_catalog", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

REPOSITORY_ID = "OwensLab/CommunityForensics-Small"
REVISION = "a" * 40
SHARDS = ("data/a.parquet", "data/b.parquet")


def sample_row(
    *,
    shard_path: str = SHARDS[0],
    row_index: int = 0,
    label: str = "1",
    image_name: str = "sample.png",
    content_group_id: str = "prompt:unique",
    **overrides: str,
) -> dict[str, str]:
    row = {
        "locator": f"{REPOSITORY_ID}@{REVISION}:{shard_path}:{row_index}",
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "shard_path": shard_path,
        "row_index": str(row_index),
        "image_name": image_name,
        "format": "PNG",
        "source_width": "512",
        "source_height": "512",
        "mode": "RGB",
        "model_name": "model-a",
        "nsfw_flag": "False",
        "prompt_hash": "b" * 64,
        "prompt_present": "True",
        "content_group_id": content_group_id,
        "real_source": "COCO",
        "subset": "Systematic",
        "source_split": "train",
        "label": label,
        "architecture": "LatDiff",
    }
    row.update(overrides)
    return row


def write_scanner_output(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    partial: bool = False,
    raw_column: str | None = None,
) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_lock = {
        "schema_version": audit.SOURCE_LOCK_SCHEMA_VERSION,
        "repository_id": REPOSITORY_ID,
        "revision": REVISION,
        "excluded_binary_columns": ["image_data"],
        "shard_count": len(SHARDS),
        "shards": [{"path": path} for path in SHARDS],
    }
    source_lock_path = source_dir / audit.SOURCE_LOCK_NAME
    source_lock_path.write_text(json.dumps(source_lock), encoding="utf-8")

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
                candidate[raw_column] = "raw source text"
            writer.writerow(candidate)

    rows_by_shard = Counter(row["shard_path"] for row in rows)
    if partial:
        scope = {
            "kind": "partial_scout",
            "partial": True,
            "complete": False,
            "eligible_for_candidate_selection": False,
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
            "eligible_for_candidate_selection": True,
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
        "catalog_row_count": len(rows),
        "excluded_binary_columns": ["image_data"],
        "image_data_materialised": False,
        "image_data_decoded": False,
        "rows_scanned_by_shard": {path: rows_by_shard[path] for path in SHARDS},
        "scan_scope": scope,
    }
    (source_dir / audit.PROVENANCE_NAME).write_text(json.dumps(provenance), encoding="utf-8")
    return source_dir


def fixed_now() -> datetime:
    return datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def test_refuses_partial_source_scan(tmp_path: Path) -> None:
    source = write_scanner_output(tmp_path, [sample_row()], partial=True)

    with pytest.raises(ValueError, match="Refusing partial or noncomplete"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)

    assert not (tmp_path / "audit").exists()


def test_refuses_catalog_sha_mismatch(tmp_path: Path) -> None:
    source = write_scanner_output(tmp_path, [sample_row()])
    provenance_path = source / audit.PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_catalog_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="source_catalog.csv SHA-256 does not match"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)

    assert not (tmp_path / "audit").exists()


def test_refuses_existing_output_without_touching_it(tmp_path: Path) -> None:
    source = write_scanner_output(tmp_path, [sample_row()])
    output = tmp_path / "audit"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit.audit_catalog(source, output, now=fixed_now)

    assert marker.read_text(encoding="utf-8") == "do not overwrite"


def test_reports_quality_gates_duplicate_statistics_and_keeps_input_immutable(
    tmp_path: Path,
) -> None:
    rows = [
        sample_row(label="0", image_name="same.png", content_group_id="prompt:shared"),
        sample_row(
            row_index=1,
            label="1",
            image_name="same.png",
            content_group_id="prompt:shared",
        ),
        sample_row(
            row_index=2,
            label="2",
            image_name="broken.png",
            content_group_id="prompt:broken",
            format="JPEG",
            source_width="256",
            mode="RGBA",
            nsfw_flag="True",
        ),
        sample_row(
            shard_path=SHARDS[1],
            row_index=0,
            label="1",
            image_name="unknown.png",
            content_group_id="prompt:unknown",
            nsfw_flag="",
        ),
    ]
    source = write_scanner_output(tmp_path, rows)
    input_hashes_before = {
        name: audit.sha256_file(source / name)
        for name in (audit.SOURCE_LOCK_NAME, audit.SOURCE_CATALOG_NAME, audit.PROVENANCE_NAME)
    }

    report = audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)

    assert report["total_rows"] == 4
    assert report["label_counts"] == {"0": 1, "1": 2, "invalid": 1}
    assert report["quality_gates"]["criteria"] == {
        "exact_512x512": {"accepted": 3, "rejected": 1},
        "png": {"accepted": 3, "rejected": 1},
        "rgb": {"accepted": 3, "rejected": 1},
        "explicit_non_nsfw": {"accepted": 2, "rejected": 2},
        "valid_label": {"accepted": 3, "rejected": 1},
        "all_quality_gates": {"accepted": 2, "rejected": 2},
    }
    assert report["class_balance"]["all_quality_gates"]["real_label_0"] == 1
    assert report["class_balance"]["all_quality_gates"]["ai_generated_label_1"] == 1
    assert report["duplicates"]["image_name"]["duplicate_value_count"] == 1
    assert report["duplicates"]["image_name"]["cross_label_duplicate_value_count"] == 1
    assert report["duplicates"]["content_group_id"]["cross_label_duplicate_row_count"] == 2
    assert (
        report["source_catalog_provenance"]["source_catalog_sha256"]
        == input_hashes_before[audit.SOURCE_CATALOG_NAME]
    )
    assert report["created_at_utc"] == "2026-08-29T12:00:00+00:00"

    input_hashes_after = {
        name: audit.sha256_file(source / name)
        for name in (audit.SOURCE_LOCK_NAME, audit.SOURCE_CATALOG_NAME, audit.PROVENANCE_NAME)
    }
    assert input_hashes_after == input_hashes_before
    output = tmp_path / "audit"
    assert (output / "summary.json").is_file()
    assert (output / "source_geometry_counts.csv").is_file()
    assert (output / "model_counts_by_label.csv").is_file()
    assert (output / "duplicate_content_group_id_groups.csv").is_file()
    with (output / "quality_gate_counts.csv").open(encoding="utf-8", newline="") as handle:
        gates = {row["gate"]: row for row in csv.DictReader(handle)}
    assert gates["all_quality_gates"] == {
        "gate": "all_quality_gates",
        "accepted": "2",
        "rejected": "2",
    }


@pytest.mark.parametrize("raw_column", ["prompt", "image_data"])
def test_refuses_raw_prompt_or_image_columns(tmp_path: Path, raw_column: str) -> None:
    source = write_scanner_output(tmp_path, [sample_row()], raw_column=raw_column)

    with pytest.raises(ValueError, match="must not contain raw image/prompt columns"):
        audit.audit_catalog(source, tmp_path / "audit", now=fixed_now)

    assert not (tmp_path / "audit").exists()
