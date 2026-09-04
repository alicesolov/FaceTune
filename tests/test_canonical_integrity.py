from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from ai_image_detector import canonical_integrity as integrity
from ai_image_detector import highres_defactify as highres
from ai_image_detector.manifest import load_manifest
from ai_image_detector.reproducibility import sha256_file


def _pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        image.load()
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def _synthetic_source_digest(name: str) -> str:
    return hashlib.sha256(f"synthetic-source:{name}".encode()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_provenance_hashes(corpus: Path) -> None:
    manifest_path = corpus / highres.HIGHRES_MANIFEST_NAME
    source_lock_path = corpus / highres.SOURCE_LOCK_NAME
    provenance_path = corpus / highres.PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(manifest_path)
    source_lock_sha256 = sha256_file(source_lock_path)
    provenance["highres_manifest_sha256"] = manifest_sha256
    provenance["source_lock_sha256"] = source_lock_sha256
    provenance["output_files_sha256"][manifest_path.name] = manifest_sha256
    provenance["output_files_sha256"][source_lock_path.name] = source_lock_sha256
    _write_json(provenance_path, provenance)


def _write_fixture(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    corpus = tmp_path / "corpus"
    images = corpus / highres.CANONICAL_IMAGES_DIRECTORY
    images.mkdir(parents=True)
    entries = (
        ("train", 0, "real"),
        ("train", 1, "sd21"),
        ("val", 0, "real"),
        ("val", 1, "sd3"),
        ("test", 0, "real"),
        ("test", 1, "sdxl"),
    )
    rows: list[dict[str, object]] = []
    for index, (split, label, generator) in enumerate(entries):
        source_id = f"source-{index:02d}"
        image_path = images / f"{hashlib.sha256(source_id.encode()).hexdigest()}.png"
        Image.new(
            "RGB",
            (highres.TARGET_SIZE, highres.TARGET_SIZE),
            color=(index * 19, index * 37, index * 53),
        ).save(image_path, format="PNG", optimize=False, compress_level=9)
        rows.append(
            {
                "path": f"{highres.CANONICAL_IMAGES_DIRECTORY}/{image_path.name}",
                "label": label,
                "split": split,
                "generator": generator,
                "group_id": f"group-{index // 2:02d}",
                "leakage_group": f"component-{index // 2:02d}",
                "source_id": source_id,
                "caption": f"unique caption {index}",
                "official_split": split,
                "source_path": f"raw/{source_id}.png",
                "source_width": 512,
                "source_height": 512,
                "source_format": "PNG",
                "source_file_bytes": 1000 + index,
                "source_sha256": _synthetic_source_digest(f"bytes-{index}"),
                "source_pixel_sha256": _synthetic_source_digest(f"pixels-{index}"),
                "source_phash": f"{index + 100:016x}",
                "crop_left": 0,
                "crop_top": 0,
                "crop_size": highres.TARGET_SIZE,
                "width": highres.TARGET_SIZE,
                "height": highres.TARGET_SIZE,
                "format": "PNG",
                "file_bytes": image_path.stat().st_size,
                "sha256": sha256_file(image_path),
                "pixel_sha256": _pixel_sha256(image_path),
                "phash": f"{index + 200:016x}",
                "source_repository_id": highres.SOURCE_REPOSITORY_ID,
                "source_revision": "a" * 40,
            }
        )

    manifest_path = corpus / highres.HIGHRES_MANIFEST_NAME
    pd.DataFrame(rows, columns=highres.HIGHRES_MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    source_lock_path = corpus / highres.SOURCE_LOCK_NAME
    _write_json(
        source_lock_path,
        {
            "schema_version": highres.SOURCE_LOCK_SCHEMA_VERSION,
            "source_repository_id": highres.SOURCE_REPOSITORY_ID,
            "source_revision": "a" * 40,
            "minimum_short_side": highres.TARGET_SIZE,
            "selected_generators": list(highres.SELECTED_GENERATORS),
            "output_image_policy": {
                "accepted_format": "PNG",
                "accepted_mode": "RGB",
                "target_size": highres.TARGET_SIZE,
            },
            "split_policy": {
                "kind": "preserve_upstream_roles_after_component_exclusion_v1",
                "upstream_column": "split",
                "roles": list(highres.SPLIT_NAMES),
                "cross_role_components_excluded": True,
            },
            "leakage_component_policy": {
                "exact_source_keys": [
                    "source_sha256",
                    "source_pixel_sha256",
                    "source_phash",
                ],
                "exact_output_keys": ["sha256", "pixel_sha256", "phash"],
                "near_phash_hamming_threshold": 8,
                "near_phash_interpretation": (
                    "over-inclusive candidate boundary, not duplicate identity evidence"
                ),
            },
        },
    )
    for filename in (
        highres.NEAR_PHASH_LINKS_NAME,
        highres.COMPONENT_COUNTS_NAME,
        highres.SPLIT_COUNTS_NAME,
        highres.EXCLUDED_COMPONENTS_NAME,
    ):
        (corpus / filename).write_text("fixture\n", encoding="utf-8")
    source_lock_sha256 = sha256_file(source_lock_path)
    manifest_sha256 = sha256_file(manifest_path)
    output_files = {
        filename: sha256_file(corpus / filename) for filename in integrity.REQUIRED_EVIDENCE_FILES
    }
    _write_json(
        corpus / highres.PROVENANCE_NAME,
        {
            "schema_version": highres.MANIFEST_SCHEMA_VERSION,
            "source_lock": highres.SOURCE_LOCK_NAME,
            "source_lock_sha256": source_lock_sha256,
            "highres_manifest": highres.HIGHRES_MANIFEST_NAME,
            "highres_manifest_sha256": manifest_sha256,
            "output_files_sha256": output_files,
            "file_verification": {
                "rows_verified": len(rows),
                "decoded_rgb_png_rows": len(rows),
                "canonical_images_directory": highres.CANONICAL_IMAGES_DIRECTORY,
            },
            "eligibility": {
                "eligible_for_exploratory_sensitivity_training": True,
                "eligible_for_primary_highres_training": False,
                "eligible_for_model_selection": False,
                "eligible_for_external_evaluation": False,
                "scope_limitations": ["Fixture scope limitation."],
            },
        },
    )
    return manifest_path, load_manifest(manifest_path, check_paths=True)


def test_validates_absolute_paths_from_load_manifest_and_returns_json_safe_summary(
    tmp_path: Path,
) -> None:
    manifest_path, frame = _write_fixture(tmp_path)

    assert frame["path"].map(lambda value: Path(value).is_absolute()).all()
    summary = integrity.validate_defactify_exploratory_corpus(manifest_path, frame)

    assert summary["schema_version"] == integrity.INTEGRITY_SCHEMA_VERSION
    assert summary["manifest"]["rows"] == 6
    assert summary["manifest"]["frame_matches_immutable_manifest"] is True
    assert summary["canonical_images"]["encoded_size"] == [384, 384]
    assert summary["canonical_images"]["byte_sha256_matches"] == 6
    assert summary["split_isolation"]["rows_by_split"] == {"test": 2, "train": 2, "val": 2}
    assert summary["eligibility"]["eligible_for_primary_highres_training"] is False
    json.dumps(summary)


def test_rejects_tampered_exploratory_eligibility(tmp_path: Path) -> None:
    manifest_path, _ = _write_fixture(tmp_path)
    provenance_path = manifest_path.parent / highres.PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["eligibility"]["eligible_for_model_selection"] = True
    _write_json(provenance_path, provenance)
    frame = load_manifest(manifest_path, check_paths=True)

    with pytest.raises(ValueError, match="unsupported Defactify exploratory eligibility"):
        integrity.validate_defactify_exploratory_corpus(manifest_path, frame)


def test_rejects_tampered_canonical_image_before_decode(tmp_path: Path) -> None:
    manifest_path, frame = _write_fixture(tmp_path)
    image_path = Path(frame.loc[0, "path"])
    image_path.write_bytes(image_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="byte SHA-256"):
        integrity.validate_defactify_exploratory_corpus(manifest_path, frame)


def test_rejects_tampered_manifest_hash_metadata(tmp_path: Path) -> None:
    manifest_path, _ = _write_fixture(tmp_path)
    raw = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    raw.loc[0, "sha256"] = "0" * 64
    raw.to_csv(manifest_path, index=False)
    _refresh_provenance_hashes(manifest_path.parent)
    frame = load_manifest(manifest_path, check_paths=True)

    with pytest.raises(ValueError, match="byte SHA-256"):
        integrity.validate_defactify_exploratory_corpus(manifest_path, frame)


def test_rejects_non_rgb_png_even_when_hashes_match(tmp_path: Path) -> None:
    manifest_path, _ = _write_fixture(tmp_path)
    raw = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    image_path = manifest_path.parent / raw.loc[0, "path"]
    Image.new("RGBA", (384, 384), color=(10, 20, 30, 40)).save(image_path, format="PNG")
    raw.loc[0, "file_bytes"] = str(image_path.stat().st_size)
    raw.loc[0, "sha256"] = sha256_file(image_path)
    raw.loc[0, "pixel_sha256"] = _pixel_sha256(image_path)
    raw.to_csv(manifest_path, index=False)
    _refresh_provenance_hashes(manifest_path.parent)
    frame = load_manifest(manifest_path, check_paths=True)

    with pytest.raises(ValueError, match="PNG/RGB"):
        integrity.validate_defactify_exploratory_corpus(manifest_path, frame)


def test_rejects_tampered_provenance_source_lock_hash(tmp_path: Path) -> None:
    manifest_path, frame = _write_fixture(tmp_path)
    provenance_path = manifest_path.parent / highres.PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_lock_sha256"] = "0" * 64
    _write_json(provenance_path, provenance)

    with pytest.raises(ValueError, match="source_lock_sha256"):
        integrity.validate_defactify_exploratory_corpus(manifest_path, frame)


def test_rejects_source_hash_crossing_splits(tmp_path: Path) -> None:
    manifest_path, _ = _write_fixture(tmp_path)
    raw = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    raw.loc[2, "source_sha256"] = raw.loc[0, "source_sha256"]
    raw.to_csv(manifest_path, index=False)
    _refresh_provenance_hashes(manifest_path.parent)
    frame = load_manifest(manifest_path, check_paths=True)

    with pytest.raises(ValueError, match="source_sha256.*crosses splits"):
        integrity.validate_defactify_exploratory_corpus(manifest_path, frame)
