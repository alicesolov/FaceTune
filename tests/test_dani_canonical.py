from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from ai_image_detector import (
    dani,
    dani_canonical,
    dani_integrity,
    dani_materialize,
    dani_selection,
)


def _source_row(root: Path, selection_id: str, label: str, cell: str, image_format: str) -> dict[str, str]:
    relative = Path("images") / f"{selection_id}.{image_format.lower()}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    color = (31, 63, 127) if label == "0" else (223, 191, 95)
    Image.new("RGB", (1024, 1024), color).save(path, format=image_format)
    encoded = path.read_bytes()
    pixel = Image.open(path).convert("RGB")
    values = {column: "x" for column in dani_materialize.MATERIALIZED_COLUMNS}
    expected_label, model, gen_type = dani_selection.CELL_DEFINITIONS[cell]
    assert expected_label == label
    values.update(
        {
            "selection_id": selection_id,
            "split": "train",
            "leakage_group": "coco-parent:1",
            "parent_coco_image_id": "1",
            "cell": cell,
            "label": label,
            "model": model,
            "gen_type": gen_type,
            "materialized_path": relative.as_posix(),
            "encoded_size_bytes": str(len(encoded)),
            "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
            "decoded_width": "1024",
            "decoded_height": "1024",
            "decoded_mode": "RGB",
            "decoded_format": image_format,
            "decoded_pixel_sha256_rgb": hashlib.sha256(pixel.tobytes()).hexdigest(),
            "decoded_phash_rgb": "0000000000000000" if label == "0" else "ffffffffffffffff",
        }
    )
    return values


def _source(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    root.mkdir()
    rows = [
        _source_row(root, "real", "0", "real_coco", "JPEG"),
        _source_row(root, "fake", "1", "fake_sdxl_t2i", "PNG"),
    ]
    manifest = root / dani_materialize.MATERIALIZED_MANIFEST_NAME
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dani_materialize.MATERIALIZED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (root / dani_materialize.MATERIALIZED_PROVENANCE_NAME).write_text(
        json.dumps(
            {
                "schema_version": dani_materialize.MATERIALIZATION_SCHEMA_VERSION,
                "materialized_manifest_sha256": dani.sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "source-audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": dani_integrity.INTEGRITY_SCHEMA_VERSION,
                "inputs": {"materialized_manifest_sha256": dani.sha256_file(manifest)},
                "counts": {
                    "row_count": 2,
                    "cross_label_exact_duplicate_group_count": 0,
                    "cross_split_integrity_component_count": 0,
                },
                "shortcut_audit": {"format_mode_support_balanced_between_labels": False},
                "eligibility": {"eligible_for_training": False},
            }
        ),
        encoding="utf-8",
    )
    return root, audit


def test_canonicalize_dani_preserves_pixels_and_enables_fresh_audit(tmp_path: Path) -> None:
    source, source_audit = _source(tmp_path)
    output = tmp_path / "canonical"

    provenance = dani_canonical.canonicalize_dani(
        source,
        source_audit,
        output,
        workers=2,
        now=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert provenance["counts"]["decoded_format_counts"] == {"PNG": 2}
    assert provenance["counts"]["decoded_mode_counts"] == {"RGB": 2}
    with (output / dani_materialize.MATERIALIZED_MANIFEST_NAME).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["decoded_format"] for row in rows} == {"PNG"}
    assert {row["decoded_mode"] for row in rows} == {"RGB"}
    for row in rows:
        with Image.open(output / row["materialized_path"]) as image:
            assert image.size == (1024, 1024)
            assert image.mode == "RGB"
            assert image.format == "PNG"

    report = dani_integrity.audit_integrity(output, tmp_path / "canonical-audit")
    assert report["eligibility"]["eligible_for_training"] is True
    assert (tmp_path / "canonical-audit" / dani_integrity.TRAINING_MANIFEST_NAME).is_file()
