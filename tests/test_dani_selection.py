from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_image_detector import dani, dani_lineage, dani_selection

REVISION = dani.PINNED_REVISION
SHARD = "data/sample.parquet"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _lineage_row(
    *,
    parent_id: int,
    caption_id: int,
    cell: str,
    row_index: int,
) -> dict[str, object]:
    label, model, gen_type = dani_selection.CELL_DEFINITIONS[cell]
    source_index = str(row_index)
    reference = label == "0"
    return {
        "locator": f"{dani.REPOSITORY_ID}@{REVISION}:{SHARD}:{row_index}",
        "repository_id": dani.REPOSITORY_ID,
        "revision": REVISION,
        "shard_path": SHARD,
        "row_index": row_index,
        "source_index": source_index,
        "source_index_hash": dani.source_index_hash(source_index),
        "image_path_basename": f"{parent_id}_{caption_id}.jpg",
        "parent_coco_image_id": parent_id,
        "coco_caption_id": caption_id,
        "declared_size": 1024,
        "category": "outdoor",
        "class_id": "7",
        "model": model,
        "gen_type": gen_type,
        "reference": reference,
        "label": int(label),
    }


def _coco_payload(records: list[tuple[int, int]]) -> dict[str, object]:
    licenses = [
        {"id": 1, "name": "NC-SA", "url": "http://license/1"},
        {"id": 2, "name": "NC", "url": "http://license/2"},
        {"id": 4, "name": "BY", "url": "http://license/4"},
    ]
    return {
        "licenses": licenses,
        "images": [
            {
                "id": parent_id,
                "file_name": f"{parent_id:012d}.jpg",
                "license": license_id,
            }
            for parent_id, license_id in records
        ],
        "annotations": [],
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "lineage"
    source.mkdir()
    rows: list[dict[str, object]] = []
    row_index = 0
    for parent_id, complete in [(10, True), (20, True), (30, True), (40, False)]:
        cells = list(dani_selection.CELL_DEFINITIONS)
        if not complete:
            cells.remove("fake_dalle3_t2i")
        for cell in cells:
            rows.append(
                _lineage_row(
                    parent_id=parent_id,
                    caption_id=parent_id * 100 + 1,
                    cell=cell,
                    row_index=row_index,
                )
            )
            row_index += 1
        if parent_id == 10:
            rows.append(
                _lineage_row(
                    parent_id=parent_id,
                    caption_id=parent_id * 100 + 1,
                    cell="real_coco",
                    row_index=row_index,
                )
            )
            row_index += 1
    catalog_path = source / dani_lineage.LINEAGE_CATALOG_NAME
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dani_lineage.LINEAGE_CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lock_path = source / dani_lineage.LINEAGE_SOURCE_LOCK_NAME
    provenance_path = source / dani_lineage.LINEAGE_PROVENANCE_NAME
    _write_json(lock_path, {"locked": True})
    _write_json(provenance_path, {"complete": True})
    lineage_summary_path = tmp_path / "lineage_summary.json"
    _write_json(
        lineage_summary_path,
        {
            "schema_version": dani_selection.LINEAGE_AUDIT_SCHEMA,
            "source": {
                "lineage_catalog_sha256": dani.sha256_file(catalog_path),
                "provenance_sha256": dani.sha256_file(provenance_path),
                "source_lock_sha256": dani.sha256_file(lock_path),
            },
            "eligibility": {
                "candidate_parent_group_verified_against_pinned_djudge_mapping": True,
                "candidate_caption_pair_verified_against_pinned_djudge_mapping": True,
                "eligible_for_training": False,
            },
        },
    )
    archive_path = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "annotations/captions_train2017.json",
            json.dumps(_coco_payload([])),
        )
        archive.writestr(
            "annotations/captions_val2017.json",
            json.dumps(_coco_payload([(10, 2), (20, 4), (30, 1), (40, 2)])),
        )
    coco_summary_path = tmp_path / "coco_summary.json"
    _write_json(
        coco_summary_path,
        {
            "schema_version": dani_selection.COCO_IDENTITY_SCHEMA,
            "inputs": {
                "annotations_archive_sha256": dani.sha256_file(archive_path),
                "lineage_summary_sha256": dani.sha256_file(lineage_summary_path),
            },
            "verified_dani_subset": {
                "parent_count": 5000,
                "parent_split_counts": {"val2017": 5000},
            },
            "eligibility": {
                "official_coco_parent_identity_verified": True,
                "official_coco_caption_identity_verified": True,
                "eligible_for_training": False,
            },
        },
    )
    return source, lineage_summary_path, coco_summary_path, archive_path


def _build(tmp_path: Path, *, seed: int = 17) -> tuple[dict[str, object], Path]:
    source, lineage, coco, archive = _write_fixture(tmp_path)
    output = tmp_path / "selection"
    report = dani_selection.build_selection(
        source,
        lineage,
        coco,
        archive,
        output,
        seed=seed,
        now=lambda: datetime(2026, 8, 30, tzinfo=UTC),
    )
    return report, output


def test_builds_caption_matched_license_filtered_parent_selection(tmp_path: Path) -> None:
    report, output = _build(tmp_path)
    rows = list(
        csv.DictReader(
            (output / dani_selection.SELECTION_CATALOG_NAME).open(encoding="utf-8", newline="")
        )
    )

    assert report["counts"]["license_eligible_parent_count"] == 3
    assert report["counts"]["incomplete_required_cell_parent_count"] == 1
    assert report["counts"]["selected_parent_count"] == 2
    assert report["counts"]["selected_row_count"] == 10
    assert report["counts"]["geometry_candidate_row_count"] == 11
    assert report["eligibility"]["eligible_for_selected_geometry_scan"] is True
    assert report["eligibility"]["eligible_for_selected_byte_materialisation"] is False
    assert report["eligibility"]["eligible_for_training"] is False
    assert (output / dani_selection.GEOMETRY_CANDIDATES_NAME).is_file()
    assert report["geometry_candidates_sha256"] == dani.sha256_file(
        output / dani_selection.GEOMETRY_CANDIDATES_NAME
    )
    assert {row["parent_coco_image_id"] for row in rows} == {"10", "20"}
    assert Counter(row["cell"] for row in rows) == {
        cell: 2 for cell in dani_selection.CELL_DEFINITIONS
    }
    for parent_id in ("10", "20"):
        parent_rows = [row for row in rows if row["parent_coco_image_id"] == parent_id]
        assert len({row["coco_caption_id"] for row in parent_rows}) == 1
        assert len({row["split"] for row in parent_rows}) == 1
        assert len({row["leakage_group"] for row in parent_rows}) == 1


def test_selection_is_byte_reproducible(tmp_path: Path) -> None:
    _, first = _build(tmp_path / "first", seed=29)
    _, second = _build(tmp_path / "second", seed=29)

    first_catalog = first / dani_selection.SELECTION_CATALOG_NAME
    second_catalog = second / dani_selection.SELECTION_CATALOG_NAME
    assert first_catalog.read_bytes() == second_catalog.read_bytes()
    first_geometry = first / dani_selection.GEOMETRY_CANDIDATES_NAME
    second_geometry = second / dani_selection.GEOMETRY_CANDIDATES_NAME
    assert first_geometry.read_bytes() == second_geometry.read_bytes()


def test_parent_split_is_stratified_reproducible_and_disjoint() -> None:
    parent_licenses = {
        **{index: 2 for index in range(20)},
        **{100 + index: 4 for index in range(20)},
    }
    first = dani_selection.assign_parent_splits(parent_licenses, seed=7)
    second = dani_selection.assign_parent_splits(parent_licenses, seed=7)

    assert first == second
    assert Counter(first.values()) == {"train": 28, "val": 6, "test": 6}
    for license_id in (2, 4):
        assert Counter(
            split for parent_id, split in first.items() if parent_licenses[parent_id] == license_id
        ) == {"train": 14, "val": 3, "test": 3}


def test_refuses_tampered_lineage_catalog(tmp_path: Path) -> None:
    source, lineage, coco, archive = _write_fixture(tmp_path)
    catalog = source / dani_lineage.LINEAGE_CATALOG_NAME
    catalog.write_text(catalog.read_text() + "\n")

    with pytest.raises(ValueError, match="SHA-256"):
        dani_selection.build_selection(
            source,
            lineage,
            coco,
            archive,
            tmp_path / "selection",
        )


def test_refuses_existing_output_without_touching_it(tmp_path: Path) -> None:
    source, lineage, coco, archive = _write_fixture(tmp_path)
    output = tmp_path / "selection"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        dani_selection.build_selection(source, lineage, coco, archive, output)

    assert marker.read_text() == "keep"
