"""Verify pinned D-Judge lineage records against official COCO 2017 captions.

The command is offline. It reads a previously verified DANI lineage summary, the pinned D-Judge
mapping, and a checksum-verified COCO annotation archive. It never reads or downloads image bytes
and never emits caption text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LINEAGE_AUDIT_SCHEMA = "dani_lineage_mapping_audit_v1"
COCO_AUDIT_SCHEMA = "dani_coco_identity_audit_v1"
OFFICIAL_ARCHIVE_SIZE = 252_907_541
OFFICIAL_ARCHIVE_MD5 = "f4bbac642086de4f52a3fdda2de5fa2c"
OFFICIAL_SOURCE_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CAPTION_MEMBERS = {
    "train2017": "annotations/captions_train2017.json",
    "val2017": "annotations/captions_val2017.json",
}


def file_digest(path: str | Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _require_revision(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be an immutable lowercase 40-character commit SHA")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_lineage_summary(
    summary: Mapping[str, object],
    *,
    mapping_sha256: str,
    mapping_revision: str,
) -> None:
    if summary.get("schema_version") != LINEAGE_AUDIT_SCHEMA:
        raise ValueError("lineage summary has an unsupported schema_version")
    if summary.get("network_accessed") is not False:
        raise ValueError("lineage summary must prove an offline audit")
    if summary.get("image_bytes_read") is not False:
        raise ValueError("lineage summary must prove that no image bytes were read")
    mapping = summary.get("mapping")
    coverage = summary.get("coverage")
    eligibility = summary.get("eligibility")
    if (
        not isinstance(mapping, dict)
        or not isinstance(coverage, dict)
        or not isinstance(eligibility, dict)
    ):
        raise TypeError("lineage summary mapping, coverage, and eligibility must be objects")
    if mapping.get("sha256") != mapping_sha256:
        raise ValueError("D-Judge mapping SHA-256 differs from the lineage audit")
    if mapping.get("revision") != mapping_revision:
        raise ValueError("D-Judge mapping revision differs from the lineage audit")
    required_coverage = {
        "catalog_rows_unjoined": 0,
        "mapping_parent_coverage_fraction": 1.0,
        "mapping_caption_pair_coverage_fraction": 1.0,
        "all_verified_parents_cross_labels": True,
        "all_verified_caption_pairs_cross_labels": True,
    }
    if any(coverage.get(key) != value for key, value in required_coverage.items()):
        raise ValueError("lineage summary does not prove complete D-Judge mapping coverage")
    if (
        eligibility.get("candidate_parent_group_verified_against_pinned_djudge_mapping") is not True
        or eligibility.get("candidate_caption_pair_verified_against_pinned_djudge_mapping")
        is not True
        or eligibility.get("eligible_for_training") is not False
    ):
        raise ValueError("lineage summary has an invalid fail-closed eligibility state")


def _load_djudge_mapping(path: Path) -> tuple[dict[int, tuple[str, dict[int, str]]], int]:
    raw = _read_object(path)
    result: dict[int, tuple[str, dict[int, str]]] = {}
    caption_ids: set[int] = set()
    pair_count = 0
    for parent_key, record in raw.items():
        if not parent_key.isdecimal() or parent_key.startswith("0"):
            raise ValueError(f"D-Judge parent key {parent_key!r} is not canonical")
        parent_id = int(parent_key)
        if not isinstance(record, dict) or set(record) != {"image_id", "image_name", "captions"}:
            raise ValueError(f"D-Judge parent {parent_key} has an unexpected schema")
        if record.get("image_id") != parent_id:
            raise ValueError(f"D-Judge parent {parent_key} disagrees with image_id")
        image_name = record.get("image_name")
        if image_name != f"{parent_id:012d}.jpg":
            raise ValueError(f"D-Judge parent {parent_key} has an invalid image_name")
        captions = record.get("captions")
        if not isinstance(captions, list) or not captions:
            raise ValueError(f"D-Judge parent {parent_key} has no captions")
        by_id: dict[int, str] = {}
        for caption in captions:
            if not isinstance(caption, dict) or set(caption) != {"caption_id", "caption"}:
                raise ValueError(f"D-Judge parent {parent_key} has an invalid caption record")
            caption_id = caption.get("caption_id")
            text = caption.get("caption")
            if isinstance(caption_id, bool) or not isinstance(caption_id, int) or caption_id <= 0:
                raise ValueError(f"D-Judge parent {parent_key} has an invalid caption_id")
            if caption_id in caption_ids:
                raise ValueError(f"D-Judge mapping duplicates caption_id {caption_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"D-Judge parent {parent_key} has empty caption text")
            caption_ids.add(caption_id)
            by_id[caption_id] = text
            pair_count += 1
        result[parent_id] = (image_name, by_id)
    return result, pair_count


def _load_coco_captions(
    archive_path: Path,
) -> tuple[dict[int, tuple[str, str, int]], dict[int, tuple[int, str]], dict[str, object]]:
    images: dict[int, tuple[str, str, int]] = {}
    captions: dict[int, tuple[int, str]] = {}
    split_image_counts: dict[str, int] = {}
    split_caption_counts: dict[str, int] = {}
    licence_ids = Counter[int]()
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"COCO archive CRC check failed at {corrupt_member}")
        names = set(archive.namelist())
        missing = sorted(set(CAPTION_MEMBERS.values()).difference(names))
        if missing:
            raise ValueError("COCO archive is missing caption members: " + ", ".join(missing))
        for split, member in CAPTION_MEMBERS.items():
            try:
                payload = json.load(archive.open(member))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid COCO JSON in {member}: {error}") from error
            if not isinstance(payload, dict):
                raise TypeError(f"{member} must contain a JSON object")
            raw_images = payload.get("images")
            raw_captions = payload.get("annotations")
            raw_licences = payload.get("licenses")
            if not isinstance(raw_images, list) or not isinstance(raw_captions, list):
                raise TypeError(f"{member} has an invalid COCO caption schema")
            if not isinstance(raw_licences, list) or not raw_licences:
                raise ValueError(f"{member} does not declare image licence records")
            split_image_counts[split] = len(raw_images)
            split_caption_counts[split] = len(raw_captions)
            for position, image in enumerate(raw_images):
                if not isinstance(image, dict):
                    raise TypeError(f"{member} image {position} must be an object")
                image_id = image.get("id")
                file_name = image.get("file_name")
                licence_id = image.get("license")
                if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id <= 0:
                    raise ValueError(f"{member} image {position} has invalid id")
                if file_name != f"{image_id:012d}.jpg":
                    raise ValueError(f"{member} image {image_id} has invalid file_name")
                if isinstance(licence_id, bool) or not isinstance(licence_id, int):
                    raise TypeError(f"{member} image {image_id} has invalid license")
                if image_id in images:
                    raise ValueError(f"COCO image id {image_id} occurs in multiple 2017 splits")
                images[image_id] = (split, file_name, licence_id)
                licence_ids[licence_id] += 1
            for position, caption in enumerate(raw_captions):
                if not isinstance(caption, dict):
                    raise TypeError(f"{member} caption {position} must be an object")
                caption_id = caption.get("id")
                image_id = caption.get("image_id")
                text = caption.get("caption")
                if (
                    isinstance(caption_id, bool)
                    or not isinstance(caption_id, int)
                    or caption_id <= 0
                    or isinstance(image_id, bool)
                    or not isinstance(image_id, int)
                    or image_id <= 0
                    or not isinstance(text, str)
                    or not text.strip()
                ):
                    raise ValueError(f"{member} caption {position} has invalid fields")
                if image_id not in images:
                    raise ValueError(f"{member} caption {caption_id} refers to an unknown image")
                if caption_id in captions:
                    raise ValueError(f"COCO caption id {caption_id} occurs more than once")
                captions[caption_id] = (image_id, text)
    evidence: dict[str, object] = {
        "split_image_counts": split_image_counts,
        "split_caption_counts": split_caption_counts,
        "total_image_count": len(images),
        "total_caption_count": len(captions),
        "image_license_id_counts": {
            str(identifier): count for identifier, count in sorted(licence_ids.items())
        },
    }
    return images, captions, evidence


def audit_coco_identity(
    lineage_summary_path: str | Path,
    mapping_path: str | Path,
    annotations_zip_path: str | Path,
    output_dir: str | Path,
    *,
    mapping_revision: str,
    transport_url: str,
    transport_revision: str,
    expected_md5: str = OFFICIAL_ARCHIVE_MD5,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Run a fail-closed, byte-free DANI-to-COCO identity audit."""
    lineage_path = Path(lineage_summary_path)
    mapping_file = Path(mapping_path)
    archive_path = Path(annotations_zip_path)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing DANI COCO audit: {destination}")
    for path in (lineage_path, mapping_file, archive_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required audit input does not exist: {path}")
    if len(expected_md5) != 32 or any(
        character not in "0123456789abcdef" for character in expected_md5
    ):
        raise ValueError("expected_md5 must be a lowercase MD5 digest")
    if archive_path.stat().st_size != OFFICIAL_ARCHIVE_SIZE:
        raise ValueError("COCO annotation archive size differs from the official object")
    archive_md5 = file_digest(archive_path, "md5")
    if archive_md5 != expected_md5:
        raise ValueError("COCO annotation archive MD5 differs from the official object")
    archive_sha256 = file_digest(archive_path, "sha256")
    mapping_sha256 = file_digest(mapping_file, "sha256")
    lineage_sha256 = file_digest(lineage_path, "sha256")
    immutable_mapping_revision = _require_revision(mapping_revision, field="mapping_revision")
    immutable_transport_revision = _require_revision(transport_revision, field="transport_revision")
    if (
        not transport_url.startswith("https://")
        or immutable_transport_revision not in transport_url
    ):
        raise ValueError("transport_url must be HTTPS and contain its immutable revision")
    lineage_summary = _read_object(lineage_path)
    _validate_lineage_summary(
        lineage_summary,
        mapping_sha256=mapping_sha256,
        mapping_revision=immutable_mapping_revision,
    )
    djudge, djudge_pair_count = _load_djudge_mapping(mapping_file)
    coco_images, coco_captions, coco_evidence = _load_coco_captions(archive_path)

    selected_splits = Counter[str]()
    selected_licences = Counter[int]()
    exact_text_matches = 0
    for parent_id, (image_name, djudge_captions) in djudge.items():
        official_image = coco_images.get(parent_id)
        if official_image is None:
            raise ValueError(
                f"D-Judge parent {parent_id} is absent from official COCO 2017 captions"
            )
        split, official_name, licence_id = official_image
        if official_name != image_name:
            raise ValueError(f"D-Judge parent {parent_id} file name differs from official COCO")
        selected_splits[split] += 1
        selected_licences[licence_id] += 1
        for caption_id, djudge_text in djudge_captions.items():
            official_caption = coco_captions.get(caption_id)
            if official_caption is None:
                raise ValueError(f"D-Judge caption {caption_id} is absent from official COCO")
            official_parent, official_text = official_caption
            if official_parent != parent_id:
                raise ValueError(
                    f"D-Judge caption {caption_id} maps to a different official COCO parent"
                )
            if official_text != djudge_text:
                raise ValueError(f"D-Judge caption {caption_id} text differs from official COCO")
            exact_text_matches += 1

    input_hashes_after = {
        "lineage_summary_sha256": file_digest(lineage_path, "sha256"),
        "mapping_sha256": file_digest(mapping_file, "sha256"),
        "annotations_archive_sha256": file_digest(archive_path, "sha256"),
    }
    input_hashes_before = {
        "lineage_summary_sha256": lineage_sha256,
        "mapping_sha256": mapping_sha256,
        "annotations_archive_sha256": archive_sha256,
    }
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("COCO audit input changed during verification")
    created = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    report: dict[str, object] = {
        "schema_version": COCO_AUDIT_SCHEMA,
        "created_at_utc": created,
        "audit_kind": "offline_dani_djudge_to_official_coco2017_caption_identity",
        "network_accessed": False,
        "image_bytes_read": False,
        "caption_text_emitted": False,
        "inputs": {
            **input_hashes_before,
            "mapping_revision": immutable_mapping_revision,
            "official_source_url": OFFICIAL_SOURCE_URL,
            "transport_url": transport_url,
            "transport_revision": immutable_transport_revision,
            "annotations_archive_size": OFFICIAL_ARCHIVE_SIZE,
            "annotations_archive_md5": archive_md5,
        },
        "official_coco_catalog": coco_evidence,
        "verified_dani_subset": {
            "parent_count": len(djudge),
            "caption_pair_count": djudge_pair_count,
            "exact_parent_filename_matches": len(djudge),
            "exact_caption_id_parent_text_matches": exact_text_matches,
            "parent_split_counts": dict(sorted(selected_splits.items())),
            "image_license_id_counts": {
                str(identifier): count for identifier, count in sorted(selected_licences.items())
            },
        },
        "eligibility": {
            "official_coco_parent_identity_verified": True,
            "official_coco_caption_identity_verified": True,
            "eligible_for_parent_group_definition": True,
            "eligible_for_candidate_selection": False,
            "eligible_for_split_assignment": False,
            "eligible_for_training": False,
            "remaining_blockers": [
                (
                    "Local non-commercial research use and attribution must be documented across "
                    "DANI CC BY-NC 4.0, D-Judge MIT, COCO annotations CC BY 4.0, and the individual "
                    "COCO/Flickr image terms; raw images must not be redistributed."
                ),
                (
                    "A deterministic 1024-source candidate selection and parent-group split must "
                    "be frozen before any image bytes are requested."
                ),
                (
                    "Selected image bytes still require geometry, container, mode, corruption, "
                    "exact/perceptual duplicate, and shortcut audits before training."
                ),
            ],
        },
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify pinned DANI/D-Judge lineage against official COCO 2017 captions."
    )
    parser.add_argument("lineage_summary", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--mapping-revision", required=True)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--transport-url", required=True)
    parser.add_argument("--transport-revision", required=True)
    parser.add_argument("--expected-md5", default=OFFICIAL_ARCHIVE_MD5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_coco_identity(
        args.lineage_summary,
        args.mapping,
        args.annotations_zip,
        args.output_dir,
        mapping_revision=args.mapping_revision,
        transport_url=args.transport_url,
        transport_revision=args.transport_revision,
        expected_md5=args.expected_md5,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
