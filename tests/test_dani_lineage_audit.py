from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_image_detector import dani, dani_lineage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_dani_lineage.py"
SPEC = importlib.util.spec_from_file_location("audit_dani_lineage", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

REVISION = dani.PINNED_REVISION
MAPPING_REVISION = "b" * 40
MAPPING_URL = (
    "https://raw.githubusercontent.com/ryliu68/DJudge/"
    f"{MAPPING_REVISION}/demo_code/Collect_AIGI_data/data/image_captions_dict_new.json"
)
SHARDS = ("data/a.parquet", "data/b.parquet")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _mapping() -> dict[str, object]:
    return {
        "10": {
            "image_id": 10,
            "image_name": "000000000010.jpg",
            "captions": [{"caption_id": 101, "caption": "First caption."}],
        },
        "20": {
            "image_id": 20,
            "image_name": "000000000020.jpg",
            "captions": [{"caption_id": 202, "caption": "Second caption."}],
        },
    }


def _row(
    *,
    shard_path: str,
    row_index: int,
    parent_id: int,
    caption_id: int,
    reference: bool,
) -> dict[str, object]:
    source_index = str(row_index + (0 if shard_path == SHARDS[0] else 100))
    return {
        "locator": f"{dani.REPOSITORY_ID}@{REVISION}:{shard_path}:{row_index}",
        "repository_id": dani.REPOSITORY_ID,
        "revision": REVISION,
        "shard_path": shard_path,
        "row_index": row_index,
        "source_index": source_index,
        "source_index_hash": dani.source_index_hash(source_index),
        "image_path_basename": f"{parent_id}_{caption_id}.jpg",
        "parent_coco_image_id": parent_id,
        "coco_caption_id": caption_id,
        "declared_size": 1024,
        "category": "outdoor",
        "class_id": "7",
        "model": "COCO" if reference else "SD_XL",
        "gen_type": "reference" if reference else "T2I",
        "reference": reference,
        "label": 0 if reference else 1,
    }


def _write_scan(tmp_path: Path, *, partial: bool = False) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    mapping_path = tmp_path / "mapping.json"
    _write_json(mapping_path, _mapping())
    lock = {
        "schema_version": audit.SOURCE_LOCK_SCHEMA,
        "repository_id": dani.REPOSITORY_ID,
        "revision": REVISION,
        "license": dani.SOURCE_LICENSE,
        "repo_type": "dataset",
        "tree_recursive": True,
        "source_schema_columns": list(dani.SOURCE_SCHEMA_COLUMNS),
        "projection_contract": dani_lineage.lineage_projection_contract(),
        "shard_count": len(SHARDS),
        "shards": [{"path": path} for path in SHARDS],
    }
    lock_path = source / dani_lineage.LINEAGE_SOURCE_LOCK_NAME
    _write_json(lock_path, lock)
    rows = [
        _row(
            shard_path=shard,
            row_index=index,
            parent_id=parent,
            caption_id=caption,
            reference=reference,
        )
        for shard in SHARDS
        for index, (parent, caption, reference) in enumerate(
            [(10, 101, True), (10, 101, False), (20, 202, True), (20, 202, False)]
        )
    ]
    catalog_path = source / dani_lineage.LINEAGE_CATALOG_NAME
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dani_lineage.LINEAGE_CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    scope = dani_lineage.build_lineage_scan_scope(
        all_shards=[{"path": path} for path in SHARDS],
        selected_shards=[{"path": SHARDS[0]}] if partial else [{"path": path} for path in SHARDS],
        limit_shards=1 if partial else None,
    )
    provenance = {
        "schema_version": audit.SCAN_SCHEMA,
        "repository_id": dani.REPOSITORY_ID,
        "requested_revision": REVISION,
        "revision": REVISION,
        "resolved_revision": REVISION,
        "source_lock": dani_lineage.LINEAGE_SOURCE_LOCK_NAME,
        "source_lock_sha256": audit.sha256_file(lock_path),
        "lineage_catalog": dani_lineage.LINEAGE_CATALOG_NAME,
        "lineage_catalog_sha256": audit.sha256_file(catalog_path),
        "catalog_kind": audit.CATALOG_KIND,
        "catalog_row_count": len(rows),
        "projection_contract": dani_lineage.lineage_projection_contract(),
        "projection_observation": {
            "image_path_requested": True,
            "image_path_materialised": True,
            "image_bytes_requested": False,
            "image_bytes_materialised": False,
            "image_bytes_decoded": False,
            "batch_image_struct_children_required": ["path"],
            "http_range_image_byte_disjointness_verified": False,
        },
        "lineage_status": {
            "upstream_mapping_join_performed": False,
            "upstream_mapping_join_verified": False,
        },
        "rows_scanned_by_shard": {path: 4 for path in SHARDS},
        "scan_scope": scope,
    }
    _write_json(source / dani_lineage.LINEAGE_PROVENANCE_NAME, provenance)
    return source, mapping_path


def _audit(source: Path, mapping: Path, output: Path) -> dict[str, object]:
    return audit.audit_lineage(
        source,
        mapping,
        output,
        mapping_url=MAPPING_URL,
        mapping_revision=MAPPING_REVISION,
        now=lambda: datetime(2026, 8, 30, tzinfo=UTC),
    )


def _refresh_catalog_hash(source: Path) -> None:
    provenance_path = source / dani_lineage.LINEAGE_PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["lineage_catalog_sha256"] = audit.sha256_file(
        source / dani_lineage.LINEAGE_CATALOG_NAME
    )
    _write_json(provenance_path, provenance)


def test_exact_mapping_join_proves_candidate_keys_but_keeps_training_blocked(
    tmp_path: Path,
) -> None:
    source, mapping = _write_scan(tmp_path)
    input_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in [*source.iterdir(), mapping]
    }

    report = _audit(source, mapping, tmp_path / "audit")

    assert report["coverage"] == {
        "catalog_rows_exact_pair_joined": 8,
        "catalog_rows_unjoined": 0,
        "observed_parent_count": 2,
        "observed_caption_pair_count": 2,
        "mapping_parent_coverage_fraction": 1.0,
        "mapping_caption_pair_coverage_fraction": 1.0,
        "cross_label_parent_count": 2,
        "cross_label_caption_pair_count": 2,
        "all_verified_parents_cross_labels": True,
        "all_verified_caption_pairs_cross_labels": True,
    }
    assert report["eligibility"]["candidate_parent_group_verified_against_pinned_djudge_mapping"]
    assert report["eligibility"]["eligible_for_training"] is False
    assert report["caption_text_emitted"] is False
    summary_text = (tmp_path / "audit" / "summary.json").read_text(encoding="utf-8")
    assert "First caption" not in summary_text
    assert input_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_hashes
    }


def test_refuses_partial_scan_before_writing_output(tmp_path: Path) -> None:
    source, mapping = _write_scan(tmp_path, partial=True)
    with pytest.raises(ValueError, match="partial or invalid"):
        _audit(source, mapping, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_refuses_catalog_hash_tampering(tmp_path: Path) -> None:
    source, mapping = _write_scan(tmp_path)
    catalog = source / dani_lineage.LINEAGE_CATALOG_NAME
    catalog.write_text(catalog.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        _audit(source, mapping, tmp_path / "audit")


def test_refuses_unverified_catalog_pair(tmp_path: Path) -> None:
    source, mapping = _write_scan(tmp_path)
    catalog = source / dani_lineage.LINEAGE_CATALOG_NAME
    rows = list(csv.DictReader(catalog.open(encoding="utf-8", newline="")))
    rows[0]["image_path_basename"] = "10_999.jpg"
    rows[0]["coco_caption_id"] = "999"
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dani_lineage.LINEAGE_CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_catalog_hash(source)
    with pytest.raises(ValueError, match="no exact pinned mapping pair"):
        _audit(source, mapping, tmp_path / "audit")


@pytest.mark.parametrize("defect", ["parent_mismatch", "duplicate_caption"])
def test_refuses_invalid_mapping_structure(tmp_path: Path, defect: str) -> None:
    source, mapping = _write_scan(tmp_path)
    payload = _mapping()
    if defect == "parent_mismatch":
        payload["10"]["image_id"] = 11
        expected = "disagrees with image_id"
    else:
        payload["20"]["captions"][0]["caption_id"] = 101
        expected = "duplicate caption_id"
    _write_json(mapping, payload)
    with pytest.raises(ValueError, match=expected):
        _audit(source, mapping, tmp_path / "audit")


def test_refuses_existing_output_without_touching_it(tmp_path: Path) -> None:
    source, mapping = _write_scan(tmp_path)
    output = tmp_path / "audit"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _audit(source, mapping, output)
    assert marker.read_text(encoding="utf-8") == "keep"
