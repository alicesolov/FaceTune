from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_image_detector import dani, dani_integrity, dani_materialize, dani_shortcut


def test_build_source_shortcut_manifest_points_only_path_to_original(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    original_image = source / "images" / "original.jpg"
    original_image.parent.mkdir()
    original_image.write_bytes(b"original")
    source_row = {column: "x" for column in dani_materialize.MATERIALIZED_COLUMNS}
    source_row.update(
        {
            "selection_id": "sample",
            "split": "train",
            "label": "1",
            "cell": "fake_sdxl_t2i",
            "generator": "SD_XL:T2I",
            "parent_coco_image_id": "42",
            "materialized_path": "images/original.jpg",
        }
    )
    source_manifest = source / dani_materialize.MATERIALIZED_MANIFEST_NAME
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dani_materialize.MATERIALIZED_COLUMNS)
        writer.writeheader()
        writer.writerow(source_row)
    source_provenance = source / dani_materialize.MATERIALIZED_PROVENANCE_NAME
    source_provenance.write_text(
        json.dumps({"materialized_manifest_sha256": dani.sha256_file(source_manifest)}),
        encoding="utf-8",
    )
    training = tmp_path / dani_integrity.TRAINING_MANIFEST_NAME
    training_columns = (*dani_materialize.MATERIALIZED_COLUMNS, "path", "group_id")
    training_row = {column: source_row.get(column, "canonical") for column in training_columns}
    training_row["path"] = "canonical.png"
    with training.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=training_columns)
        writer.writeheader()
        writer.writerow(training_row)
    monkeypatch.setattr(
        dani_integrity,
        "validate_training_manifest",
        lambda path: {"training_manifest_sha256": dani.sha256_file(path)},
    )

    summary = dani_shortcut.build_source_shortcut_manifest(
        source, training, tmp_path / "shortcut"
    )

    assert summary["eligibility"]["eligible_for_pixel_training"] is False
    with (tmp_path / "shortcut" / dani_shortcut.MANIFEST_NAME).open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["path"] == str(original_image.resolve())
    assert row["selection_id"] == "sample"
