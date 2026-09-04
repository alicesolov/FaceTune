"""Build a non-training manifest for DANI acquisition-pipeline shortcut diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final

from . import dani, dani_integrity, dani_materialize

SHORTCUT_SCHEMA_VERSION: Final = "dani_source_shortcut_manifest_v1"
MANIFEST_NAME: Final = "source_shortcut_manifest.csv"
SUMMARY_NAME: Final = "summary.json"


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), [dict(row) for row in reader]


def build_source_shortcut_manifest(
    source_dir: str | Path,
    canonical_training_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Point an audited split back at original files for metadata-only diagnostics."""
    source = Path(source_dir)
    training_path = Path(canonical_training_manifest)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite DANI shortcut artifact: {destination}")
    integrity = dani_integrity.validate_training_manifest(training_path)
    source_manifest_path = source / dani_materialize.MATERIALIZED_MANIFEST_NAME
    source_provenance_path = source / dani_materialize.MATERIALIZED_PROVENANCE_NAME
    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    if source_provenance.get("materialized_manifest_sha256") != dani.sha256_file(
        source_manifest_path
    ):
        raise ValueError("Original DANI manifest differs from its provenance")
    source_columns, source_rows = _read_csv(source_manifest_path)
    if source_columns != dani_materialize.MATERIALIZED_COLUMNS:
        raise ValueError("Original DANI manifest schema differs from the locked schema")
    training_columns, training_rows = _read_csv(training_path)
    source_by_id = {row["selection_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows) or len(training_rows) != len(source_rows):
        raise ValueError("Source and canonical training manifests do not have a 1:1 selection")

    output_rows: list[dict[str, str]] = []
    for row in training_rows:
        original = source_by_id.get(row["selection_id"])
        if original is None:
            raise ValueError(f"Missing original DANI row {row['selection_id']}")
        for key in ("split", "label", "cell", "generator", "parent_coco_image_id"):
            if row[key] != original[key]:
                raise ValueError(f"Original and canonical DANI differ in {key}")
        output = dict(row)
        original_path = (source / original["materialized_path"]).resolve()
        output["path"] = str(original_path)
        output_rows.append(output)

    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = destination / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=training_columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(output_rows)
    summary: dict[str, object] = {
        "schema_version": SHORTCUT_SCHEMA_VERSION,
        "role": "acquisition_pipeline_shortcut_diagnostic_only",
        "inputs": {
            "source_materialized_manifest_sha256": dani.sha256_file(source_manifest_path),
            "source_materialized_provenance_sha256": dani.sha256_file(source_provenance_path),
            "canonical_training_manifest_sha256": integrity["training_manifest_sha256"],
        },
        "manifest": manifest_path.name,
        "manifest_sha256": dani.sha256_file(manifest_path),
        "row_count": len(output_rows),
        "eligibility": {
            "eligible_for_metadata_shortcut_diagnostic": True,
            "eligible_for_pixel_training": False,
            "eligible_for_model_selection": False,
            "eligible_for_external_evaluation": False,
        },
    }
    (destination / SUMMARY_NAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
