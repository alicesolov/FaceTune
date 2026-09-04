from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from ai_image_detector import dani, dani_integrity, dani_materialize, dani_selection
from ai_image_detector.manifest import load_manifest


def _row(
    root: Path,
    *,
    selection_id: str,
    parent: int,
    split: str,
    cell: str,
    phash: str,
    content: bytes,
    decoded_format: str = "JPEG",
) -> dict[str, str]:
    label, model, gen_type = dani_selection.CELL_DEFINITIONS[cell]
    relative = Path("images") / f"{selection_id}.jpg"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    encoded_sha = hashlib.sha256(content).hexdigest()
    pixel_sha = hashlib.sha256(b"pixels:" + content).hexdigest()
    values: dict[str, object] = {
        "selection_id": selection_id,
        "geometry_candidate_id": f"candidate-{selection_id}",
        "provisional_selection_id": f"provisional-{selection_id}",
        "is_provisional_selection": True,
        "split": split,
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
        "locator": f"locator-{selection_id}",
        "repository_id": dani.REPOSITORY_ID,
        "revision": dani.PINNED_REVISION,
        "shard_path": "data/sample.parquet",
        "row_index": parent,
        "source_index": parent,
        "source_index_hash": dani.source_index_hash(str(parent)),
        "image_path_basename": f"{parent}_{parent * 100 + 1}.jpg",
        "category": "outdoor",
        "class_id": "7",
        "materialized_path": relative.as_posix(),
        "encoded_size_bytes": len(content),
        "encoded_sha256": encoded_sha,
        "decoded_width": 1024,
        "decoded_height": 1024,
        "decoded_mode": "RGB",
        "decoded_format": decoded_format,
        "decoded_pixel_sha256_rgb": pixel_sha,
        "decoded_phash_rgb": phash,
    }
    return {key: str(values[key]) for key in dani_materialize.MATERIALIZED_COLUMNS}


def _materialized(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    root = tmp_path / "materialized"
    root.mkdir(exist_ok=True)
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
                "counts": {"materialized_row_count": len(rows)},
                "eligibility": {
                    "all_selected_rows_materialized": True,
                    "all_decoded_geometry_exact_1024": True,
                    "eligible_for_duplicate_and_leakage_audit": True,
                    "eligible_for_training": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def test_integrity_audit_emits_training_manifest_when_gates_pass(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    rows = [
        _row(
            root,
            selection_id="real-train",
            parent=10,
            split="train",
            cell="real_coco",
            phash="0000000000000000",
            content=b"real-train",
        ),
        _row(
            root,
            selection_id="fake-train",
            parent=10,
            split="train",
            cell="fake_sdxl_t2i",
            phash="00ff00ff00ff00ff",
            content=b"fake-train",
        ),
        _row(
            root,
            selection_id="real-test",
            parent=20,
            split="test",
            cell="real_coco",
            phash="ffffffffffffffff",
            content=b"real-test",
        ),
        _row(
            root,
            selection_id="fake-test",
            parent=20,
            split="test",
            cell="fake_sdxl_t2i",
            phash="ff00ff00ff00ff00",
            content=b"fake-test",
        ),
    ]
    materialized = _materialized(tmp_path, rows)

    report = dani_integrity.audit_integrity(materialized, tmp_path / "audit")

    assert report["eligibility"]["eligible_for_training"] is True
    assert report["counts"]["cross_split_integrity_component_count"] == 0
    training_path = tmp_path / "audit" / dani_integrity.TRAINING_MANIFEST_NAME
    assert training_path.is_file()
    training = load_manifest(training_path, check_paths=True)
    assert len(training) == 4
    assert {
        "sha256",
        "phash",
        "source_sha256",
        "source_pixel_sha256",
        "source_phash",
        "parent_group",
        "leakage_group",
        "integrity_component",
    }.issubset(training.columns)
    assert training["source_id"].tolist() == training["selection_id"].tolist()
    assert training["group_id"].tolist() == training["parent_group"].tolist()
    assert training["leakage_group"].tolist() == training["integrity_component"].tolist()
    assert training.groupby("parent_group")["leakage_group"].nunique().eq(1).all()
    validated = dani_integrity.validate_training_manifest(training_path)
    assert validated["training_manifest_sha256"] == dani.sha256_file(training_path)


def test_training_manifest_validation_rejects_changed_bytes(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    rows = [
        _row(
            root,
            selection_id="real-train",
            parent=10,
            split="train",
            cell="real_coco",
            phash="0000000000000000",
            content=b"real-train",
        ),
        _row(
            root,
            selection_id="fake-train",
            parent=10,
            split="train",
            cell="fake_sdxl_t2i",
            phash="ffffffffffffffff",
            content=b"fake-train",
        ),
    ]
    materialized = _materialized(tmp_path, rows)
    dani_integrity.audit_integrity(materialized, tmp_path / "audit")
    manifest = tmp_path / "audit" / dani_integrity.TRAINING_MANIFEST_NAME
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="differs from its integrity summary"):
        dani_integrity.validate_training_manifest(manifest)


def test_integrity_audit_blocks_cross_split_near_duplicate_component(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    rows = [
        _row(
            root,
            selection_id="real-train",
            parent=10,
            split="train",
            cell="real_coco",
            phash="0000000000000000",
            content=b"real-train",
        ),
        _row(
            root,
            selection_id="real-test",
            parent=20,
            split="test",
            cell="real_coco",
            phash="0000000000000001",
            content=b"real-test",
        ),
    ]
    materialized = _materialized(tmp_path, rows)

    report = dani_integrity.audit_integrity(materialized, tmp_path / "audit")

    assert report["eligibility"]["eligible_for_training"] is False
    assert report["counts"]["cross_split_integrity_component_count"] == 1
    assert not (tmp_path / "audit" / dani_integrity.TRAINING_MANIFEST_NAME).exists()


def test_integrity_audit_blocks_format_support_shortcut(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    rows = [
        _row(
            root,
            selection_id="real",
            parent=10,
            split="train",
            cell="real_coco",
            phash="0000000000000000",
            content=b"real",
            decoded_format="JPEG",
        ),
        _row(
            root,
            selection_id="fake",
            parent=10,
            split="train",
            cell="fake_sdxl_t2i",
            phash="ffffffffffffffff",
            content=b"fake",
            decoded_format="PNG",
        ),
    ]
    materialized = _materialized(tmp_path, rows)

    report = dani_integrity.audit_integrity(materialized, tmp_path / "audit")

    assert report["shortcut_audit"]["format_mode_support_balanced_between_labels"] is False
    assert report["eligibility"]["eligible_for_training"] is False
