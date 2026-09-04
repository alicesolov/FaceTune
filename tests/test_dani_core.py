from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_image_detector import dani, dani_core, dani_geometry, dani_selection

SHARDS = {
    "data/s7.parquet": (700, "7" * 64),
    "data/s8.parquet": (800, "8" * 64),
    "data/s9.parquet": (900, "9" * 64),
    "data/s11.parquet": (1_100, "a" * 64),
}


def _candidate(parent: int, cell: str, source_index: int, shard_path: str) -> dict[str, str]:
    label, model, gen_type = dani_selection.CELL_DEFINITIONS[cell]
    values: dict[str, object] = {
        "geometry_candidate_id": f"candidate-{parent}-{cell}-{source_index}",
        "provisional_selection_id": f"provisional-{parent}-{cell}",
        "is_provisional_selection": "True",
        "split": "train" if parent == 10 else "test",
        "leakage_group": f"coco-parent:{parent}",
        "parent_coco_image_id": parent,
        "coco_caption_id": parent * 100 + 1,
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
        "locator": f"locator-{source_index}",
        "repository_id": dani.REPOSITORY_ID,
        "revision": dani.PINNED_REVISION,
        "shard_path": shard_path,
        "row_index": source_index,
        "source_index": source_index,
        "source_index_hash": dani.source_index_hash(str(source_index)),
        "image_path_basename": f"{parent}_{parent * 100 + 1}.jpg",
        "category": "outdoor",
        "class_id": "7",
    }
    return {key: str(values[key]) for key in dani_selection.GEOMETRY_CANDIDATE_COLUMNS}


def _candidates() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for offset, parent in enumerate((10, 20)):
        real_start = 100_000 + offset
        for resolution_index in range(4):
            rows.append(
                _candidate(
                    parent,
                    "real_coco",
                    real_start + resolution_index * dani_core.REAL_RESOLUTION_STRIDE,
                    "data/s11.parquet",
                )
            )
        synthetic = {
            "fake_dalle3_t2i": "data/s9.parquet",
            "fake_sdxl_i2i": "data/s7.parquet",
            "fake_sdxl_t2i": "data/s8.parquet",
            "fake_sdxl_ti2i": "data/s9.parquet",
        }
        for index, (cell, shard) in enumerate(synthetic.items(), start=1):
            rows.append(_candidate(parent, cell, 300_000 + offset * 10 + index, shard))
    return rows


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    preselection = tmp_path / "preselection"
    lineage = tmp_path / "lineage"
    preselection.mkdir()
    lineage.mkdir()
    lock = lineage / "source_lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": "dani_lineage_source_lock_v1",
                "repository_id": dani.REPOSITORY_ID,
                "revision": dani.PINNED_REVISION,
                "shards": [
                    {
                        "path": path,
                        "lfs": {"size": size, "sha256": sha256},
                    }
                    for path, (size, sha256) in SHARDS.items()
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (preselection / dani_selection.SELECTION_SPEC_NAME).write_text(
        json.dumps(
            {"input_hashes": {"lineage_source_lock_sha256": dani.sha256_file(lock)}},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return preselection, lineage


def test_builds_complete_five_cell_core_under_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = _candidates()
    preselection, lineage = _inputs(tmp_path)
    monkeypatch.setattr(
        dani_geometry,
        "_validate_preselection",
        lambda path: (candidates, {"geometry_candidates_sha256": "frozen"}),
    )

    output = tmp_path / "core"
    report = dani_core.build_core_plan(
        preselection,
        lineage,
        output,
        byte_cap=100_000_000,
        now=lambda: datetime(2026, 8, 30, tzinfo=UTC),
    )

    assert report["counts"]["parent_count"] == 2
    assert report["counts"]["selected_row_count"] == 10
    assert report["counts"]["required_shard_count"] == 4
    assert report["eligibility"]["eligible_for_bounded_shard_materialisation"] is True
    rows = list(csv.DictReader((output / dani_core.CORE_SELECTION_NAME).open(encoding="utf-8")))
    real = [row for row in rows if row["cell"] == "real_coco"]
    assert {int(row["source_index"]) for row in real} == {
        100_000 + 3 * dani_core.REAL_RESOLUTION_STRIDE,
        100_001 + 3 * dani_core.REAL_RESOLUTION_STRIDE,
    }


def test_refuses_inconsistent_real_resolution_stride() -> None:
    candidates = _candidates()
    real = next(row for row in candidates if row["cell"] == "real_coco")
    real["source_index"] = str(int(real["source_index"]) + 1)

    with pytest.raises(ValueError, match="four ordered real resolutions"):
        dani_core._choose_core_rows(candidates)


def test_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preselection, lineage = _inputs(tmp_path)
    output = tmp_path / "core"
    output.mkdir()
    monkeypatch.setattr(
        dani_geometry,
        "_validate_preselection",
        lambda path: (_candidates(), {"geometry_candidates_sha256": "frozen"}),
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        dani_core.build_core_plan(preselection, lineage, output)
