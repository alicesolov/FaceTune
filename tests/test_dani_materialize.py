from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from ai_image_detector import dani, dani_core, dani_materialize, dani_selection

BYTE_CAP = 5_000_000
SHARD_PATH = "data/sample.parquet"


def _jpeg(size: int, color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _selection(row_index: int, *, cell: str) -> dict[str, str]:
    label, model, gen_type = dani_selection.CELL_DEFINITIONS[cell]
    parent = 10
    values: dict[str, object] = {
        "selection_id": f"core-{row_index}",
        "geometry_candidate_id": f"candidate-{row_index}",
        "provisional_selection_id": f"provisional-{row_index}",
        "is_provisional_selection": True,
        "split": "train",
        "leakage_group": f"coco-parent:{parent}",
        "parent_coco_image_id": parent,
        "coco_caption_id": 1001,
        "official_coco_split": "val2017",
        "official_coco_license_id": 4,
        "official_coco_license_name": "BY",
        "official_coco_license_url": "https://license/4",
        "cell": cell,
        "label": label,
        "generator": "real" if label == "0" else f"{model}:{gen_type}",
        "model": model,
        "gen_type": gen_type,
        "declared_size": 1024,
        "locator": f"locator-{row_index}",
        "repository_id": dani.REPOSITORY_ID,
        "revision": dani.PINNED_REVISION,
        "shard_path": SHARD_PATH,
        "row_index": row_index,
        "source_index": 100 + row_index,
        "source_index_hash": dani.source_index_hash(str(100 + row_index)),
        "image_path_basename": f"{parent}_1001.jpg",
        "category": "outdoor",
        "class_id": "7",
    }
    return {
        key: str(values[key])
        for key in ("selection_id", *dani_selection.GEOMETRY_CANDIDATE_COLUMNS)
    }


def _fixture(tmp_path: Path, *, second_size: int = 1024) -> tuple[Path, Path]:
    source_shard = tmp_path / "source.parquet"
    selection = [
        _selection(0, cell="real_coco"),
        _selection(1, cell="fake_sdxl_t2i"),
    ]
    records = []
    for row, size, color in zip(
        selection,
        (1024, second_size),
        ((20, 30, 40), (120, 130, 140)),
        strict=True,
    ):
        records.append(
            {
                "index": int(row["source_index"]),
                "image": {
                    "bytes": _jpeg(size, color),
                    "path": row["image_path_basename"],
                },
                "size": 1024,
                "category": row["category"],
                "class_id": row["class_id"],
                "model": row["model"],
                "gen_type": row["gen_type"],
                "reference": row["label"] == "0",
            }
        )
    pq.write_table(pa.Table.from_pylist(records), source_shard, row_group_size=1)

    core = tmp_path / "core"
    core.mkdir()
    spec = core / dani_core.CORE_SPEC_NAME
    selection_path = core / dani_core.CORE_SELECTION_NAME
    shard_plan = core / dani_core.CORE_SHARD_PLAN_NAME
    spec.write_text("{}\n", encoding="utf-8")
    with selection_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("selection_id", *dani_selection.GEOMETRY_CANDIDATE_COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selection)
    with shard_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dani_core.SHARD_PLAN_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "processing_order": 1,
                "shard_path": SHARD_PATH,
                "expected_size_bytes": source_shard.stat().st_size,
                "expected_sha256": dani.sha256_file(source_shard),
                "selected_row_count": 2,
            }
        )
    (core / dani_core.CORE_PROVENANCE_NAME).write_text(
        json.dumps(
            {
                "schema_version": dani_core.CORE_SCHEMA_VERSION,
                "core_spec_sha256": dani.sha256_file(spec),
                "core_selection_sha256": dani.sha256_file(selection_path),
                "shard_plan_sha256": dani.sha256_file(shard_plan),
                "image_bytes_requested": False,
                "image_bytes_read": False,
                "budget": {"hard_cap_bytes": BYTE_CAP},
                "eligibility": {
                    "eligible_for_bounded_shard_materialisation": True,
                    "eligible_for_training": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return core, source_shard


def _downloader(source_shard: Path):
    def download(shard_path: str, staging: Path, expected_size: int, expected_sha256: str) -> Path:
        assert expected_size == source_shard.stat().st_size
        assert expected_sha256 == dani.sha256_file(source_shard)
        target = staging / shard_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_shard, target)
        return target

    return download


def test_materializes_exact_rows_and_removes_verified_staging_shard(tmp_path: Path) -> None:
    core, source_shard = _fixture(tmp_path)
    staging = tmp_path / "staging"
    output = tmp_path / "materialized"

    report = dani_materialize.materialize_core(
        core,
        staging,
        output,
        downloader=_downloader(source_shard),
        byte_cap=BYTE_CAP,
    )

    assert report["counts"]["materialized_row_count"] == 2
    assert report["counts"]["decoded_format_counts"] == {"JPEG": 2}
    assert report["counts"]["decoded_mode_counts"] == {"RGB": 2}
    assert report["eligibility"]["eligible_for_duplicate_and_leakage_audit"] is True
    assert not (staging / SHARD_PATH).exists()
    rows = list(
        csv.DictReader(
            (output / dani_materialize.MATERIALIZED_MANIFEST_NAME).open(encoding="utf-8")
        )
    )
    assert len(rows) == 2
    assert {row["decoded_width"] for row in rows} == {"1024"}
    assert all((output / row["materialized_path"]).is_file() for row in rows)


def test_fails_closed_on_observed_low_resolution(tmp_path: Path) -> None:
    core, source_shard = _fixture(tmp_path, second_size=512)
    staging = tmp_path / "staging"
    output = tmp_path / "materialized"

    with pytest.raises(ValueError, match="geometry mismatch"):
        dani_materialize.materialize_core(
            core,
            staging,
            output,
            downloader=_downloader(source_shard),
            byte_cap=BYTE_CAP,
        )

    assert not (output / dani_materialize.MATERIALIZED_PROVENANCE_NAME).exists()
    assert (staging / SHARD_PATH).is_file()


def test_refuses_core_byte_cap_drift(tmp_path: Path) -> None:
    core, source_shard = _fixture(tmp_path)

    with pytest.raises(ValueError, match="byte cap differs"):
        dani_materialize.materialize_core(
            core,
            tmp_path / "staging",
            tmp_path / "materialized",
            downloader=_downloader(source_shard),
            byte_cap=BYTE_CAP - 1,
        )


def test_resume_peak_does_not_count_existing_range_parts_twice(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    materialized = tmp_path / "materialized"
    target = staging / "data" / "sample.parquet"
    parts = target.with_name(target.name + ".range-parts")
    parts.mkdir(parents=True)
    materialized.mkdir()
    (parts / "000.part").write_bytes(b"x" * 700)
    (staging / "unrelated.lock").write_bytes(b"l" * 11)
    (materialized / "partial.csv").write_bytes(b"m" * 13)

    peak = dani_materialize._projected_range_download_peak(
        staging,
        materialized,
        target,
        expected_size=1_000,
    )

    assert peak == 11 + 13 + 2_000
