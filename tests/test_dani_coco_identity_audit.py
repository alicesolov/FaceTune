from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_dani_coco_identity.py"
SPEC = importlib.util.spec_from_file_location("audit_dani_coco_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

MAPPING_REVISION = "a" * 40
TRANSPORT_REVISION = "b" * 40
TRANSPORT_URL = (
    "https://huggingface.co/datasets/example/coco/resolve/"
    f"{TRANSPORT_REVISION}/annotations_trainval2017.zip"
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _mapping() -> dict[str, object]:
    return {
        "10": {
            "image_id": 10,
            "image_name": "000000000010.jpg",
            "captions": [{"caption_id": 101, "caption": "Exact caption ten."}],
        },
        "20": {
            "image_id": 20,
            "image_name": "000000000020.jpg",
            "captions": [{"caption_id": 202, "caption": "Exact caption twenty."}],
        },
    }


def _caption_payload(
    *,
    image_id: int,
    caption_id: int,
    caption: str,
) -> dict[str, object]:
    return {
        "info": {"year": 2017},
        "licenses": [{"id": 4, "name": "Attribution License"}],
        "images": [
            {
                "id": image_id,
                "file_name": f"{image_id:012d}.jpg",
                "license": 4,
            }
        ],
        "annotations": [{"id": caption_id, "image_id": image_id, "caption": caption}],
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    mapping_path = tmp_path / "mapping.json"
    _write_json(mapping_path, _mapping())
    mapping_sha = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    lineage_path = tmp_path / "lineage_summary.json"
    _write_json(
        lineage_path,
        {
            "schema_version": audit.LINEAGE_AUDIT_SCHEMA,
            "network_accessed": False,
            "image_bytes_read": False,
            "mapping": {"sha256": mapping_sha, "revision": MAPPING_REVISION},
            "coverage": {
                "catalog_rows_unjoined": 0,
                "mapping_parent_coverage_fraction": 1.0,
                "mapping_caption_pair_coverage_fraction": 1.0,
                "all_verified_parents_cross_labels": True,
                "all_verified_caption_pairs_cross_labels": True,
            },
            "eligibility": {
                "candidate_parent_group_verified_against_pinned_djudge_mapping": True,
                "candidate_caption_pair_verified_against_pinned_djudge_mapping": True,
                "eligible_for_training": False,
            },
        },
    )
    archive_path = tmp_path / "annotations_trainval2017.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            audit.CAPTION_MEMBERS["train2017"],
            json.dumps(
                _caption_payload(
                    image_id=10,
                    caption_id=101,
                    caption="Exact caption ten.",
                )
            ),
        )
        archive.writestr(
            audit.CAPTION_MEMBERS["val2017"],
            json.dumps(
                _caption_payload(
                    image_id=20,
                    caption_id=202,
                    caption="Exact caption twenty.",
                )
            ),
        )
    return lineage_path, mapping_path, archive_path


def _run(
    monkeypatch: pytest.MonkeyPatch,
    lineage: Path,
    mapping: Path,
    archive: Path,
    output: Path,
) -> dict[str, object]:
    monkeypatch.setattr(audit, "OFFICIAL_ARCHIVE_SIZE", archive.stat().st_size)
    return audit.audit_coco_identity(
        lineage,
        mapping,
        archive,
        output,
        mapping_revision=MAPPING_REVISION,
        transport_url=TRANSPORT_URL,
        transport_revision=TRANSPORT_REVISION,
        expected_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
        now=lambda: datetime(2026, 8, 30, tzinfo=UTC),
    )


def test_exact_official_join_verifies_identity_but_keeps_training_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage, mapping, archive = _write_inputs(tmp_path)

    report = _run(monkeypatch, lineage, mapping, archive, tmp_path / "audit")

    assert report["verified_dani_subset"] == {
        "parent_count": 2,
        "caption_pair_count": 2,
        "exact_parent_filename_matches": 2,
        "exact_caption_id_parent_text_matches": 2,
        "parent_split_counts": {"train2017": 1, "val2017": 1},
        "image_license_id_counts": {"4": 2},
    }
    assert report["eligibility"]["official_coco_parent_identity_verified"] is True
    assert report["eligibility"]["eligible_for_training"] is False
    assert report["caption_text_emitted"] is False
    assert "Exact caption" not in (tmp_path / "audit" / "summary.json").read_text()


def test_refuses_archive_checksum_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage, mapping, archive = _write_inputs(tmp_path)
    monkeypatch.setattr(audit, "OFFICIAL_ARCHIVE_SIZE", archive.stat().st_size)
    with pytest.raises(ValueError, match="MD5"):
        audit.audit_coco_identity(
            lineage,
            mapping,
            archive,
            tmp_path / "audit",
            mapping_revision=MAPPING_REVISION,
            transport_url=TRANSPORT_URL,
            transport_revision=TRANSPORT_REVISION,
            expected_md5="0" * 32,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image_id", 999, "absent from official COCO"),
        ("caption", "Changed caption.", "text differs from official COCO"),
    ],
)
def test_refuses_djudge_identity_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    lineage, mapping, archive = _write_inputs(tmp_path)
    payload = _mapping()
    if field == "image_id":
        payload["10"]["image_id"] = value
        payload["10"]["image_name"] = "000000000999.jpg"
        moved = payload.pop("10")
        payload["999"] = moved
    else:
        payload["10"]["captions"][0]["caption"] = value
    _write_json(mapping, payload)
    lineage_payload = json.loads(lineage.read_text())
    lineage_payload["mapping"]["sha256"] = hashlib.sha256(mapping.read_bytes()).hexdigest()
    _write_json(lineage, lineage_payload)
    with pytest.raises(ValueError, match=message):
        _run(monkeypatch, lineage, mapping, archive, tmp_path / "audit")


def test_refuses_existing_output_without_touching_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage, mapping, archive = _write_inputs(tmp_path)
    output = tmp_path / "audit"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _run(monkeypatch, lineage, mapping, archive, output)
    assert marker.read_text() == "keep"
