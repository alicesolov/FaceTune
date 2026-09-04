"""Fail-closed integrity checks for the frozen Defactify-HR exploratory corpus.

The high-resolution manifest records canonical PNG crops, while ``load_manifest`` resolves their
paths to absolute locations.  This module verifies the immutable CSV and its adjacent evidence
files before accepting that resolved DataFrame for an experiment.  It deliberately verifies the
encoded image mode and geometry before any RGB conversion so a later transform cannot hide a
non-canonical raster.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from .highres_defactify import (
    CANONICAL_IMAGES_DIRECTORY,
    COMPONENT_COUNTS_NAME,
    EXCLUDED_COMPONENTS_NAME,
    HIGHRES_MANIFEST_COLUMNS,
    HIGHRES_MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    NEAR_PHASH_LINKS_NAME,
    PROVENANCE_NAME,
    SELECTED_GENERATORS,
    SOURCE_LOCK_NAME,
    SOURCE_LOCK_SCHEMA_VERSION,
    SPLIT_COUNTS_NAME,
    SPLIT_NAMES,
    TARGET_SIZE,
)
from .reproducibility import sha256_file

INTEGRITY_SCHEMA_VERSION = "defactify_exploratory_canonical_integrity_v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PHASH_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

SOURCE_FIELDS = (
    "source_path",
    "source_width",
    "source_height",
    "source_format",
    "source_file_bytes",
    "source_sha256",
    "source_pixel_sha256",
    "source_phash",
    "source_repository_id",
    "source_revision",
)
SPLIT_ISOLATION_KEYS = (
    "leakage_group",
    "group_id",
    "caption",
    "source_path",
    "source_id",
    "source_sha256",
    "source_pixel_sha256",
    "source_phash",
    "sha256",
    "pixel_sha256",
    "phash",
)
REQUIRED_EVIDENCE_FILES = (
    SOURCE_LOCK_NAME,
    HIGHRES_MANIFEST_NAME,
    NEAR_PHASH_LINKS_NAME,
    COMPONENT_COUNTS_NAME,
    SPLIT_COUNTS_NAME,
    EXCLUDED_COMPONENTS_NAME,
)
EXPECTED_SOURCE_KEYS = ("source_sha256", "source_pixel_sha256", "source_phash")
EXPECTED_OUTPUT_KEYS = ("sha256", "pixel_sha256", "phash")
EXPLORATORY_ELIGIBILITY_FIELDS = (
    "eligible_for_exploratory_sensitivity_training",
    "eligible_for_primary_highres_training",
    "eligible_for_model_selection",
    "eligible_for_external_evaluation",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _require_text(value: object, *, field: str) -> str:
    if value is None or pd.isna(value) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = _require_text(value, field=field)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return digest


def _require_phash(value: object, *, field: str) -> str:
    digest = _require_text(value, field=field)
    if not _PHASH_PATTERN.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase 16-character pHash")
    return digest


def _require_revision(value: object, *, field: str) -> str:
    revision = _require_text(value, field=field)
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{field} must be a 40-character lowercase commit SHA")
    return revision


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    text = str(value)
    if not text.isdecimal():
        raise ValueError(f"{field} must be a canonical integer")
    parsed = int(text)
    if parsed < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return parsed


def _validated_exploratory_eligibility(provenance: Mapping[str, object]) -> dict[str, bool]:
    eligibility = provenance.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise TypeError("provenance.json.eligibility must be an object")
    if set(eligibility) != set(EXPLORATORY_ELIGIBILITY_FIELDS) | {"scope_limitations"}:
        raise ValueError("provenance.json has an unsupported exploratory eligibility schema")
    resolved: dict[str, bool] = {}
    for field in EXPLORATORY_ELIGIBILITY_FIELDS:
        value = eligibility.get(field)
        if not isinstance(value, bool):
            raise TypeError(f"provenance.json.eligibility.{field} must be boolean")
        resolved[field] = value
    if resolved != {
        "eligible_for_exploratory_sensitivity_training": True,
        "eligible_for_primary_highres_training": False,
        "eligible_for_model_selection": False,
        "eligible_for_external_evaluation": False,
    }:
        raise ValueError(
            "provenance.json has an unsupported Defactify exploratory eligibility state"
        )
    limitations = eligibility.get("scope_limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise ValueError("provenance.json.eligibility.scope_limitations must be nonempty text")
    return resolved


def _resolve_manifest_image_path(value: object, manifest_path: Path) -> Path:
    candidate = Path(_require_text(value, field="manifest image path"))
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    for root in (Path.cwd(), *manifest_path.parents):
        resolved = root / candidate
        if resolved.exists():
            return resolved.resolve()
    return (manifest_path.parent / candidate).resolve(strict=False)


def _validate_schema(frame: pd.DataFrame, *, context: str) -> None:
    expected = tuple(HIGHRES_MANIFEST_COLUMNS)
    actual = tuple(frame.columns)
    if actual != expected:
        missing = [column for column in expected if column not in frame.columns]
        unexpected = [column for column in frame.columns if column not in expected]
        raise ValueError(
            f"{context} schema is incompatible with HIGHRES_MANIFEST_COLUMNS; "
            f"missing={missing}, unexpected={unexpected}, order_matches={actual == expected}"
        )
    if frame.empty:
        raise ValueError(f"{context} must not be empty")


def _normalised_frame_values(frame: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    normalised = frame.loc[:, list(HIGHRES_MANIFEST_COLUMNS)].copy().reset_index(drop=True)
    normalised["path"] = normalised["path"].map(
        lambda value: str(_resolve_manifest_image_path(value, manifest_path))
    )
    for column in HIGHRES_MANIFEST_COLUMNS:
        if column != "path":
            normalised[column] = normalised[column].map(str)
    # ``load_manifest`` intentionally uses pandas' normal CSV inference. A pHash made entirely
    # of decimal digits may consequently arrive as an integer and lose leading zeroes. Restore
    # its fixed-width hexadecimal representation before comparing it to the immutable CSV.
    for column in ("source_phash", "phash"):
        normalised[column] = normalised[column].map(
            lambda value: value.zfill(16) if value.isdecimal() and len(value) <= 16 else value
        )
    return normalised


def _validate_frame_matches_manifest(frame: pd.DataFrame, manifest_path: Path) -> None:
    raw_frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    _validate_schema(raw_frame, context="Immutable high-resolution manifest")
    _validate_schema(frame, context="Provided high-resolution frame")
    expected = _normalised_frame_values(raw_frame, manifest_path)
    observed = _normalised_frame_values(frame, manifest_path)
    if len(observed) != len(expected):
        raise ValueError(
            "Provided high-resolution frame row count differs from the immutable manifest: "
            f"frame={len(observed)}, manifest={len(expected)}"
        )
    for column in HIGHRES_MANIFEST_COLUMNS:
        mismatches = expected[column].ne(observed[column])
        if mismatches.any():
            row_index = int(mismatches.idxmax())
            raise ValueError(
                "Provided high-resolution frame differs from the immutable manifest at "
                f"row {row_index}, column {column!r}"
            )


def _require_declared_hash(
    payload: Mapping[str, object],
    *,
    field: str,
    actual: str,
    context: str,
) -> None:
    expected = _require_sha256(payload.get(field), field=f"{context}.{field}")
    if expected != actual:
        raise ValueError(f"{context}.{field} does not match the current file SHA-256")


def _validate_evidence(
    manifest_path: Path,
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], str, str, Path]:
    evidence_dir = manifest_path.parent
    provenance_path = evidence_dir / PROVENANCE_NAME
    source_lock_path = evidence_dir / SOURCE_LOCK_NAME
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing canonical corpus provenance: {provenance_path}")
    if not source_lock_path.is_file():
        raise FileNotFoundError(f"Missing canonical corpus source lock: {source_lock_path}")

    provenance = _read_json_object(provenance_path)
    source_lock = _read_json_object(source_lock_path)
    if provenance.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("provenance.json has an unsupported schema_version")
    if source_lock.get("schema_version") != SOURCE_LOCK_SCHEMA_VERSION:
        raise ValueError("source_lock.json has an unsupported schema_version")
    if provenance.get("highres_manifest") != manifest_path.name:
        raise ValueError("provenance.json does not refer to the requested high-resolution manifest")
    if provenance.get("source_lock") != source_lock_path.name:
        raise ValueError(
            "provenance.json does not refer to source_lock.json in the manifest directory"
        )

    manifest_hash = sha256_file(manifest_path)
    source_lock_hash = sha256_file(source_lock_path)
    _require_declared_hash(
        provenance,
        field="highres_manifest_sha256",
        actual=manifest_hash,
        context="provenance.json",
    )
    _require_declared_hash(
        provenance,
        field="source_lock_sha256",
        actual=source_lock_hash,
        context="provenance.json",
    )
    output_hashes = provenance.get("output_files_sha256")
    if not isinstance(output_hashes, Mapping):
        raise TypeError("provenance.json.output_files_sha256 must be an object")
    if set(output_hashes) != set(REQUIRED_EVIDENCE_FILES):
        missing = sorted(set(REQUIRED_EVIDENCE_FILES).difference(output_hashes))
        unexpected = sorted(set(output_hashes).difference(REQUIRED_EVIDENCE_FILES))
        raise ValueError(
            "provenance.json.output_files_sha256 must lock every canonical sidecar; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for filename in REQUIRED_EVIDENCE_FILES:
        sidecar = evidence_dir / filename
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing canonical corpus evidence sidecar: {sidecar}")
        _require_declared_hash(
            output_hashes,
            field=filename,
            actual=sha256_file(sidecar),
            context="provenance.json.output_files_sha256",
        )

    file_verification = provenance.get("file_verification")
    if not isinstance(file_verification, Mapping):
        raise TypeError("provenance.json.file_verification must be an object")
    if _require_int(file_verification.get("rows_verified"), field="rows_verified") != len(frame):
        raise ValueError("provenance.json rows_verified does not match the provided frame")
    if _require_int(
        file_verification.get("decoded_rgb_png_rows"), field="decoded_rgb_png_rows"
    ) != len(frame):
        raise ValueError("provenance.json decoded_rgb_png_rows does not match the provided frame")
    images_directory = _require_text(
        file_verification.get("canonical_images_directory"),
        field="canonical_images_directory",
    )
    if Path(images_directory).is_absolute() or images_directory != CANONICAL_IMAGES_DIRECTORY:
        raise ValueError("provenance.json has an unsupported canonical_images_directory")
    images_root = (evidence_dir / images_directory).resolve(strict=False)

    split_policy = source_lock.get("split_policy")
    if not isinstance(split_policy, Mapping):
        raise TypeError("source_lock.json.split_policy must be an object")
    if split_policy.get("kind") != "preserve_upstream_roles_after_component_exclusion_v1":
        raise ValueError("source_lock.json does not lock the upstream-role preservation policy")
    if split_policy.get("upstream_column") != "split":
        raise ValueError("source_lock.json split_policy has an unexpected upstream column")
    if split_policy.get("roles") != list(SPLIT_NAMES):
        raise ValueError("source_lock.json split_policy does not lock train/val/test roles")
    if split_policy.get("cross_role_components_excluded") is not True:
        raise ValueError("source_lock.json does not prove cross-role components were excluded")
    if not frame["official_split"].astype(str).eq(frame["split"].astype(str)).all():
        raise ValueError(
            "Manifest split roles differ from the locked upstream official_split roles"
        )

    policy = source_lock.get("output_image_policy")
    if not isinstance(policy, Mapping):
        raise TypeError("source_lock.json.output_image_policy must be an object")
    if policy.get("accepted_format") != "PNG" or policy.get("accepted_mode") != "RGB":
        raise ValueError("source_lock.json does not lock the canonical PNG/RGB encoding")
    if _require_int(policy.get("target_size"), field="source_lock target_size") != TARGET_SIZE:
        raise ValueError("source_lock.json target_size differs from the canonical protocol")
    if (
        _require_int(source_lock.get("minimum_short_side"), field="minimum_short_side")
        < TARGET_SIZE
    ):
        raise ValueError("source_lock.json minimum_short_side is below the canonical crop size")
    leakage_policy = source_lock.get("leakage_component_policy")
    if not isinstance(leakage_policy, Mapping):
        raise TypeError("source_lock.json.leakage_component_policy must be an object")
    if leakage_policy.get("exact_source_keys") != list(EXPECTED_SOURCE_KEYS):
        raise ValueError("source_lock.json does not lock the required source exact keys")
    if leakage_policy.get("exact_output_keys") != list(EXPECTED_OUTPUT_KEYS):
        raise ValueError("source_lock.json does not lock the required output exact keys")
    _require_int(
        leakage_policy.get("near_phash_hamming_threshold"),
        field="source_lock near_phash_hamming_threshold",
    )
    if leakage_policy.get("near_phash_interpretation") != (
        "over-inclusive candidate boundary, not duplicate identity evidence"
    ):
        raise ValueError("source_lock.json has an unsupported pHash candidate-link interpretation")
    source_repository_id = _require_text(
        source_lock.get("source_repository_id"), field="source_lock source_repository_id"
    )
    source_revision = _require_revision(
        source_lock.get("source_revision"), field="source_lock source_revision"
    )
    selected_generators = source_lock.get("selected_generators")
    if not isinstance(selected_generators, list) or not selected_generators:
        raise TypeError("source_lock.json selected_generators must be a nonempty list")
    if len(selected_generators) != len(set(selected_generators)) or any(
        not isinstance(value, str) or not value.strip() for value in selected_generators
    ):
        raise ValueError("source_lock.json selected_generators must be unique nonempty strings")

    observed_generators = set(frame.loc[frame["label"].astype(str) == "1", "generator"].astype(str))
    if set(selected_generators) != set(SELECTED_GENERATORS):
        raise ValueError("source_lock.json selected_generators differs from the locked protocol")
    if observed_generators != set(selected_generators):
        raise ValueError(
            "Manifest fake generators do not match source_lock.json selected_generators"
        )
    if not frame["source_repository_id"].map(str).eq(source_repository_id).all():
        raise ValueError("Manifest source_repository_id differs from source_lock.json")
    if not frame["source_revision"].map(str).eq(source_revision).all():
        raise ValueError("Manifest source_revision differs from source_lock.json")

    return provenance, source_lock, manifest_hash, source_lock_hash, images_root


def _validate_row_metadata(row: pd.Series, *, row_index: int) -> None:
    prefix = f"manifest row {row_index}"
    for column in SOURCE_FIELDS:
        _require_text(row[column], field=f"{prefix}.{column}")
    for column in (
        "generator",
        "group_id",
        "leakage_group",
        "source_id",
        "caption",
        "official_split",
    ):
        _require_text(row[column], field=f"{prefix}.{column}")
    if str(row["label"]) not in {"0", "1"}:
        raise ValueError(f"{prefix}.label must be binary 0 or 1")
    if str(row["label"]) == "0" and row["generator"] != "real":
        raise ValueError(f"{prefix} real label must use generator='real'")
    if str(row["label"]) == "1" and row["generator"] not in SELECTED_GENERATORS:
        raise ValueError(f"{prefix} fake label has an unsupported generator")
    if str(row["split"]) not in set(SPLIT_NAMES):
        raise ValueError(f"{prefix}.split must be one of {list(SPLIT_NAMES)}")
    for column in ("source_width", "source_height", "source_file_bytes"):
        _require_int(row[column], field=f"{prefix}.{column}", minimum=1)
    for column in ("crop_left", "crop_top"):
        _require_int(row[column], field=f"{prefix}.{column}")
    for column in ("crop_size", "width", "height"):
        if _require_int(row[column], field=f"{prefix}.{column}", minimum=1) != TARGET_SIZE:
            raise ValueError(f"{prefix}.{column} differs from the canonical {TARGET_SIZE}px size")
    if row["format"] != "PNG":
        raise ValueError(f"{prefix}.format must be PNG")
    _require_int(row["file_bytes"], field=f"{prefix}.file_bytes", minimum=1)
    for column in ("source_sha256", "source_pixel_sha256", "sha256", "pixel_sha256"):
        _require_sha256(row[column], field=f"{prefix}.{column}")
    for column in ("source_phash", "phash"):
        _require_phash(row[column], field=f"{prefix}.{column}")
    _require_revision(row["source_revision"], field=f"{prefix}.source_revision")


def _validate_canonical_image(
    row: pd.Series,
    *,
    row_index: int,
    manifest_path: Path,
    images_root: Path,
) -> None:
    image_path = _resolve_manifest_image_path(row["path"], manifest_path)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Manifest row {row_index} canonical image is missing: {image_path}"
        )
    try:
        image_path.relative_to(images_root)
    except ValueError as error:
        raise ValueError(
            f"Manifest row {row_index} canonical image is outside the locked images directory"
        ) from error
    expected_name = hashlib.sha256(str(row["source_id"]).encode()).hexdigest() + ".png"
    if image_path.name != expected_name:
        raise ValueError(
            f"Manifest row {row_index} canonical filename does not match its source_id contract"
        )
    actual_sha256 = sha256_file(image_path)
    if actual_sha256 != row["sha256"]:
        raise ValueError(f"Manifest row {row_index} canonical image byte SHA-256 does not match")
    if image_path.stat().st_size != _require_int(
        row["file_bytes"], field=f"manifest row {row_index}.file_bytes", minimum=1
    ):
        raise ValueError(f"Manifest row {row_index} canonical image byte count does not match")
    try:
        with Image.open(image_path) as verify_image:
            verify_image.verify()
        with Image.open(image_path) as encoded:
            encoded.load()
            encoded_format = encoded.format
            encoded_mode = encoded.mode
            encoded_size = encoded.size
            encoded_frames = getattr(encoded, "n_frames", 1)
            if (
                encoded_format != "PNG"
                or encoded_mode != "RGB"
                or encoded_size
                != (
                    TARGET_SIZE,
                    TARGET_SIZE,
                )
            ):
                raise ValueError(
                    f"Manifest row {row_index} is not an encoded PNG/RGB "
                    f"{TARGET_SIZE}x{TARGET_SIZE} raster before conversion"
                )
            if encoded_frames != 1:
                raise ValueError(f"Manifest row {row_index} canonical PNG must contain one frame")
            decoded = ImageOps.exif_transpose(encoded).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Manifest row {row_index} canonical image cannot be decoded") from error
    pixel_sha256 = hashlib.sha256(decoded.tobytes()).hexdigest()
    if pixel_sha256 != row["pixel_sha256"]:
        raise ValueError(f"Manifest row {row_index} canonical image pixel SHA-256 does not match")


def _validate_split_isolation(frame: pd.DataFrame) -> None:
    if frame["source_id"].duplicated().any():
        duplicate = str(frame.loc[frame["source_id"].duplicated(keep=False), "source_id"].iloc[0])
        raise ValueError(f"Manifest source_id must be unique; duplicate={duplicate!r}")
    group_sizes = frame.groupby("group_id", dropna=False).size()
    incomplete_groups = group_sizes[group_sizes != 2]
    if not incomplete_groups.empty:
        raise ValueError(
            "Manifest group_id must contain exactly one frozen binary pair; "
            f"first invalid group={incomplete_groups.index[0]!r}"
        )
    group_labels = frame.groupby("group_id", dropna=False)["label"].agg(set)
    invalid_groups = group_labels[group_labels.map(lambda labels: labels != {"0", "1"})]
    if not invalid_groups.empty:
        raise ValueError(
            "Manifest group_id must contain one real and one fake label; "
            f"first invalid group={invalid_groups.index[0]!r}"
        )
    for key in SPLIT_ISOLATION_KEYS:
        values = frame[key]
        blank = values.isna() | values.astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"Manifest split-isolation key {key!r} contains blank values")
        split_counts = frame.groupby(key, dropna=False)["split"].nunique()
        crossing = split_counts[split_counts > 1]
        if not crossing.empty:
            raise ValueError(
                f"Manifest split-isolation key {key!r} crosses splits: {crossing.index[0]!r}"
            )
        if key in {"source_sha256", "source_pixel_sha256", "sha256", "pixel_sha256"}:
            label_counts = frame.groupby(key, dropna=False)["label"].nunique()
            conflicting = label_counts[label_counts > 1]
            if not conflicting.empty:
                raise ValueError(f"Manifest {key!r} has a cross-label duplicate")


def validate_defactify_exploratory_corpus(
    manifest_path: str | Path,
    frame: pd.DataFrame,
) -> dict[str, object]:
    """Verify immutable evidence, canonical rasters, and split isolation before training.

    ``frame`` may come directly from :func:`ai_image_detector.manifest.load_manifest`; absolute
    path resolution is normalised before it is compared with the immutable CSV.
    """
    resolved_manifest = Path(manifest_path).resolve()
    if not resolved_manifest.is_file():
        raise FileNotFoundError(f"Missing high-resolution manifest: {resolved_manifest}")
    if resolved_manifest.name != HIGHRES_MANIFEST_NAME:
        raise ValueError(
            f"Expected canonical manifest name {HIGHRES_MANIFEST_NAME!r}, got "
            f"{resolved_manifest.name!r}"
        )

    _validate_frame_matches_manifest(frame, resolved_manifest)
    verified_frame = _normalised_frame_values(frame, resolved_manifest)
    provenance, source_lock, manifest_hash, source_lock_hash, images_root = _validate_evidence(
        resolved_manifest, verified_frame
    )
    eligibility = _validated_exploratory_eligibility(provenance)
    for row_index, row in verified_frame.iterrows():
        _validate_row_metadata(row, row_index=int(row_index))
        _validate_canonical_image(
            row,
            row_index=int(row_index),
            manifest_path=resolved_manifest,
            images_root=images_root,
        )
    _validate_split_isolation(verified_frame)

    rows_by_split = {
        str(split): int(count)
        for split, count in verified_frame["split"].value_counts(sort=False).sort_index().items()
    }
    return {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "manifest": {
            "path": str(resolved_manifest),
            "sha256": manifest_hash,
            "rows": len(verified_frame),
            "frame_matches_immutable_manifest": True,
        },
        "provenance": {
            "path": str(resolved_manifest.parent / PROVENANCE_NAME),
            "schema_version": provenance["schema_version"],
        },
        "source_lock": {
            "path": str(resolved_manifest.parent / SOURCE_LOCK_NAME),
            "sha256": source_lock_hash,
            "schema_version": source_lock["schema_version"],
            "source_repository_id": source_lock["source_repository_id"],
            "source_revision": source_lock["source_revision"],
        },
        "canonical_images": {
            "root": str(images_root),
            "verified_rows": len(verified_frame),
            "encoded_format": "PNG",
            "encoded_mode": "RGB",
            "encoded_size": [TARGET_SIZE, TARGET_SIZE],
            "byte_sha256_matches": len(verified_frame),
            "pixel_sha256_matches": len(verified_frame),
        },
        "split_isolation": {
            "keys_checked": list(SPLIT_ISOLATION_KEYS),
            "rows_by_split": rows_by_split,
        },
        "eligibility": eligibility,
    }
