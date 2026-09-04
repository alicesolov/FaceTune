"""Losslessly re-encode audited DANI rasters to a common RGB PNG container."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import imagehash
from PIL import Image, ImageOps

from . import dani, dani_materialize

CANONICAL_SCHEMA_VERSION: Final = "dani_rgb1024_canonicalization_v1"
SOURCE_INTEGRITY_SCHEMA_VERSION: Final = "dani_highres_integrity_v1"
PARTIAL_MANIFEST_NAME: Final = "materialized.partial.csv"
LINEAGE_NAME: Final = "source_lineage.csv"
LINEAGE_COLUMNS: Final = (
    "selection_id",
    "source_materialized_path",
    "source_encoded_size_bytes",
    "source_encoded_sha256",
    "source_decoded_mode",
    "source_decoded_format",
    "source_decoded_pixel_sha256_rgb",
    "source_decoded_phash_rgb",
    "canonical_materialized_path",
    "canonical_encoded_size_bytes",
    "canonical_encoded_sha256",
    "canonical_decoded_pixel_sha256_rgb",
    "canonical_decoded_phash_rgb",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_source(source: Path, audit_summary_path: Path) -> list[dict[str, str]]:
    manifest_path = source / dani_materialize.MATERIALIZED_MANIFEST_NAME
    provenance_path = source / dani_materialize.MATERIALIZED_PROVENANCE_NAME
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Completed source DANI materialisation is missing")
    provenance = _read_json(provenance_path)
    if provenance.get("schema_version") != dani_materialize.MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("Canonicalisation requires the original DANI materialisation")
    if provenance.get("materialized_manifest_sha256") != dani.sha256_file(manifest_path):
        raise ValueError("Source DANI manifest differs from provenance")

    audit = _read_json(audit_summary_path)
    if audit.get("schema_version") != SOURCE_INTEGRITY_SCHEMA_VERSION:
        raise ValueError("Source DANI integrity summary has an unsupported schema")
    inputs = audit.get("inputs")
    counts = audit.get("counts")
    shortcut = audit.get("shortcut_audit")
    eligibility = audit.get("eligibility")
    if not isinstance(inputs, dict) or inputs.get("materialized_manifest_sha256") != dani.sha256_file(
        manifest_path
    ):
        raise ValueError("Source integrity audit does not identify this DANI manifest")
    if (
        not isinstance(counts, dict)
        or counts.get("cross_label_exact_duplicate_group_count") != 0
        or counts.get("cross_split_integrity_component_count") != 0
    ):
        raise ValueError("Source DANI has duplicate or cross-split leakage blockers")
    if (
        not isinstance(shortcut, dict)
        or shortcut.get("format_mode_support_balanced_between_labels") is not False
        or not isinstance(eligibility, dict)
        or eligibility.get("eligible_for_training") is not False
    ):
        raise ValueError("Canonicalisation is allowed only for the isolated format/mode blocker")

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != dani_materialize.MATERIALIZED_COLUMNS:
            raise ValueError("Source DANI manifest schema differs from the locked schema")
        rows = [dict(row) for row in reader]
    if counts.get("row_count") != len(rows):
        raise ValueError("Source DANI audit row count differs from the manifest")
    return rows


def _canonicalize_one(source: Path, destination: Path, row: Mapping[str, str]) -> dict[str, str]:
    source_path = source / row["materialized_path"]
    relative = Path(dani_materialize.IMAGES_DIRECTORY) / f"{row['selection_id']}.png"
    output_path = destination / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".png.tmp")
    with Image.open(source_path) as image:
        canonical = ImageOps.exif_transpose(image).convert("RGB")
        if canonical.size != (1024, 1024):
            raise ValueError(f"{row['selection_id']} is not an exact 1024 x 1024 raster")
        canonical.save(temporary_path, format="PNG", compress_level=6, optimize=False)
    os.replace(temporary_path, output_path)
    encoded = output_path.read_bytes()
    pixel_bytes = canonical.tobytes()
    values = dict(row)
    values.update(
        {
            "materialized_path": relative.as_posix(),
            "encoded_size_bytes": str(len(encoded)),
            "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
            "decoded_width": "1024",
            "decoded_height": "1024",
            "decoded_mode": "RGB",
            "decoded_format": "PNG",
            "decoded_pixel_sha256_rgb": hashlib.sha256(pixel_bytes).hexdigest(),
            "decoded_phash_rgb": str(imagehash.phash(canonical)),
        }
    )
    return {column: values[column] for column in dani_materialize.MATERIALIZED_COLUMNS}


def canonicalize_dani(
    source_dir: str | Path,
    source_audit_summary: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 8,
    now: datetime | None = None,
) -> dict[str, object]:
    """Create a fresh RGB/PNG derivative while retaining a byte-level lineage record."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    source = Path(source_dir)
    audit_path = Path(source_audit_summary)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite canonical DANI corpus: {destination}")
    rows = _load_source(source, audit_path)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            canonical_rows = list(
                executor.map(lambda row: _canonicalize_one(source, destination, row), rows)
            )
        manifest_path = destination / dani_materialize.MATERIALIZED_MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=dani_materialize.MATERIALIZED_COLUMNS)
            writer.writeheader()
            writer.writerows(canonical_rows)
        lineage_path = destination / LINEAGE_NAME
        with lineage_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LINEAGE_COLUMNS)
            writer.writeheader()
            for source_row, canonical_row in zip(rows, canonical_rows, strict=True):
                writer.writerow(
                    {
                        "selection_id": source_row["selection_id"],
                        "source_materialized_path": source_row["materialized_path"],
                        "source_encoded_size_bytes": source_row["encoded_size_bytes"],
                        "source_encoded_sha256": source_row["encoded_sha256"],
                        "source_decoded_mode": source_row["decoded_mode"],
                        "source_decoded_format": source_row["decoded_format"],
                        "source_decoded_pixel_sha256_rgb": source_row[
                            "decoded_pixel_sha256_rgb"
                        ],
                        "source_decoded_phash_rgb": source_row["decoded_phash_rgb"],
                        "canonical_materialized_path": canonical_row["materialized_path"],
                        "canonical_encoded_size_bytes": canonical_row["encoded_size_bytes"],
                        "canonical_encoded_sha256": canonical_row["encoded_sha256"],
                        "canonical_decoded_pixel_sha256_rgb": canonical_row[
                            "decoded_pixel_sha256_rgb"
                        ],
                        "canonical_decoded_phash_rgb": canonical_row["decoded_phash_rgb"],
                    }
                )
        created_at = (datetime.now(UTC) if now is None else now).astimezone(UTC).isoformat()
        provenance: dict[str, object] = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "source": {
                "materialized_manifest_sha256": dani.sha256_file(
                    source / dani_materialize.MATERIALIZED_MANIFEST_NAME
                ),
                "materialized_provenance_sha256": dani.sha256_file(
                    source / dani_materialize.MATERIALIZED_PROVENANCE_NAME
                ),
                "integrity_summary_sha256": dani.sha256_file(audit_path),
            },
            "canonicalization": {
                "geometry": "exact_1024_square_preserved",
                "pixel_operation": "exif_transpose_then_rgb_conversion_only",
                "container": "PNG",
                "mode": "RGB",
                "png_compress_level": 6,
                "png_optimize": False,
            },
            "materialized_manifest": manifest_path.name,
            "materialized_manifest_sha256": dani.sha256_file(manifest_path),
            "source_lineage": lineage_path.name,
            "source_lineage_sha256": dani.sha256_file(lineage_path),
            "counts": {
                "materialized_row_count": len(canonical_rows),
                "decoded_format_counts": dict(
                    Counter(row["decoded_format"] for row in canonical_rows)
                ),
                "decoded_mode_counts": dict(Counter(row["decoded_mode"] for row in canonical_rows)),
            },
            "eligibility": {
                "all_selected_rows_materialized": True,
                "all_decoded_geometry_exact_1024": True,
                "source_lineage_preserved": True,
                "eligible_for_duplicate_and_leakage_audit": True,
                "eligible_for_training": False,
                "remaining_blocker": "Run a fresh integrity audit on canonical encoded and decoded bytes.",
            },
        }
        _write_json(destination / dani_materialize.MATERIALIZED_PROVENANCE_NAME, provenance)
        return provenance
    except BaseException:  # noqa: TRY203 - leave an explicit incomplete-corpus boundary
        # Keep the original source immutable. The incomplete derivative is intentionally obvious
        # (no provenance.json) and must be removed or investigated before a fresh run.
        raise


def validate_canonical_provenance(provenance: Mapping[str, object]) -> None:
    """Validate the eligibility fields consumed by the common integrity auditor."""
    if provenance.get("schema_version") != CANONICAL_SCHEMA_VERSION:
        raise ValueError("DANI canonical corpus has an unsupported schema_version")
    eligibility = provenance.get("eligibility")
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("all_selected_rows_materialized") is not True
        or eligibility.get("all_decoded_geometry_exact_1024") is not True
        or eligibility.get("source_lineage_preserved") is not True
        or eligibility.get("eligible_for_duplicate_and_leakage_audit") is not True
        or eligibility.get("eligible_for_training") is not False
    ):
        raise ValueError("DANI canonical corpus is not eligible for integrity audit")
