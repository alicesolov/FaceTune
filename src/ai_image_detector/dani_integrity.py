"""Offline duplicate, leakage, and shortcut audit for materialised DANI images."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from . import dani, dani_canonical, dani_materialize, dani_selection

INTEGRITY_SCHEMA_VERSION: Final = "dani_highres_integrity_v1"
SUMMARY_NAME: Final = "summary.json"
NEAR_LINKS_NAME: Final = "near_phash_links.csv"
DUPLICATE_GROUPS_NAME: Final = "exact_duplicate_groups.csv"
TRAINING_MANIFEST_NAME: Final = "training_manifest.csv"
DEFAULT_PHASH_THRESHOLD: Final = 4
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
PHASH_PATTERN: Final = re.compile(r"^[0-9a-f]{16}$")
NEAR_LINK_COLUMNS: Final = (
    "left_selection_id",
    "right_selection_id",
    "left_parent_coco_image_id",
    "right_parent_coco_image_id",
    "left_split",
    "right_split",
    "left_label",
    "right_label",
    "phash_distance",
)
DUPLICATE_COLUMNS: Final = (
    "key",
    "value",
    "row_count",
    "labels",
    "splits",
    "parent_coco_image_ids",
    "selection_ids",
)
TRAINING_COLUMNS: Final = (
    *(column for column in dani_materialize.MATERIALIZED_COLUMNS if column != "leakage_group"),
    "path",
    "group_id",
    "source_id",
    "sha256",
    "phash",
    "source_sha256",
    "source_pixel_sha256",
    "source_phash",
    "parent_group",
    "leakage_group",
    "integrity_component",
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _portable_image_path(materialized_dir: Path, relative_path: str) -> str:
    """Prefer a repository-relative path while retaining support for external audit roots."""
    absolute = (materialized_dir / relative_path).resolve()
    try:
        return absolute.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(absolute)


def _training_row(
    row: Mapping[str, str],
    *,
    materialized_dir: Path,
    integrity_component: str,
) -> dict[str, str]:
    """Adapt frozen DANI evidence to the generic training-manifest contract."""
    parent_group = row["leakage_group"]
    encoded_sha = row["encoded_sha256"]
    pixel_sha = row["decoded_pixel_sha256_rgb"]
    phash = row["decoded_phash_rgb"]
    return {
        **row,
        "path": _portable_image_path(materialized_dir, row["materialized_path"]),
        "group_id": parent_group,
        "source_id": row["selection_id"],
        "sha256": encoded_sha,
        "phash": phash,
        "source_sha256": encoded_sha,
        "source_pixel_sha256": pixel_sha,
        "source_phash": phash,
        "parent_group": parent_group,
        # The connected component, rather than only the COCO parent, is the split-isolation key.
        "leakage_group": integrity_component,
        "integrity_component": integrity_component,
    }


def validate_training_manifest(manifest_path: str | Path) -> dict[str, object]:
    """Require an adjacent successful DANI integrity record for the exact manifest bytes."""
    manifest = Path(manifest_path).resolve()
    if manifest.name != TRAINING_MANIFEST_NAME:
        raise ValueError(f"DANI training requires the audited {TRAINING_MANIFEST_NAME}")
    summary_path = manifest.parent / SUMMARY_NAME
    if not manifest.is_file() or not summary_path.is_file():
        raise FileNotFoundError("DANI training manifest or adjacent integrity summary is missing")
    summary = _read_json(summary_path)
    if summary.get("schema_version") != INTEGRITY_SCHEMA_VERSION:
        raise ValueError("DANI integrity summary has an unsupported schema_version")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get("training_manifest") != manifest.name:
        raise ValueError("DANI integrity summary does not identify this training manifest")
    eligibility = summary.get("eligibility")
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("eligible_for_training") is not True
        or eligibility.get("eligible_for_model_selection") is not False
        or eligibility.get("eligible_for_external_evaluation") is not False
    ):
        raise ValueError("DANI integrity summary is not eligible for controlled training")
    actual_hash = dani.sha256_file(manifest)
    if summary.get("training_manifest_sha256") != actual_hash:
        raise ValueError("DANI training manifest differs from its integrity summary")
    counts = summary.get("counts")
    if not isinstance(counts, dict) or not isinstance(counts.get("row_count"), int):
        raise TypeError("DANI integrity summary has no validated row count")
    return summary


def _load_materialized(materialized_dir: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    manifest_path = materialized_dir / dani_materialize.MATERIALIZED_MANIFEST_NAME
    provenance_path = materialized_dir / dani_materialize.MATERIALIZED_PROVENANCE_NAME
    if not manifest_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Completed DANI materialisation manifest/provenance is missing")
    provenance = _read_json(provenance_path)
    schema_version = provenance.get("schema_version")
    if schema_version == dani_canonical.CANONICAL_SCHEMA_VERSION:
        dani_canonical.validate_canonical_provenance(provenance)
    elif schema_version != dani_materialize.MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("DANI materialisation has an unsupported schema_version")
    if provenance.get("materialized_manifest_sha256") != dani.sha256_file(manifest_path):
        raise ValueError("DANI materialised manifest differs from provenance")
    eligibility = provenance.get("eligibility")
    if schema_version == dani_materialize.MATERIALIZATION_SCHEMA_VERSION and (
        not isinstance(eligibility, dict)
        or eligibility.get("all_selected_rows_materialized") is not True
        or eligibility.get("all_decoded_geometry_exact_1024") is not True
        or eligibility.get("eligible_for_duplicate_and_leakage_audit") is not True
        or eligibility.get("eligible_for_training") is not False
    ):
        raise ValueError("DANI materialisation is not eligible for integrity audit")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != dani_materialize.MATERIALIZED_COLUMNS:
            raise ValueError("DANI materialised manifest schema differs from the locked schema")
        rows = [dict(row) for row in reader]
    counts = provenance.get("counts")
    if not isinstance(counts, dict) or counts.get("materialized_row_count") != len(rows):
        raise ValueError("DANI materialised row count differs from provenance")
    if not rows or len({row["selection_id"] for row in rows}) != len(rows):
        raise ValueError("DANI materialised manifest is empty or has duplicate IDs")
    return rows, provenance


def _validate_rows(rows: Sequence[Mapping[str, str]], materialized_dir: Path) -> None:
    paths: set[Path] = set()
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row_number, row in enumerate(rows, start=2):
        if row["cell"] not in dani_selection.CELL_DEFINITIONS:
            raise ValueError(f"DANI materialised row {row_number} has an unknown cell")
        expected_label, expected_model, expected_gen_type = dani_selection.CELL_DEFINITIONS[
            row["cell"]
        ]
        if (row["label"], row["model"], row["gen_type"]) != (
            expected_label,
            expected_model,
            expected_gen_type,
        ):
            raise ValueError(f"DANI materialised row {row_number} changes cell semantics")
        if row["split"] not in dani_selection.SPLIT_ORDER:
            raise ValueError(f"DANI materialised row {row_number} has an invalid split")
        if row["decoded_width"] != "1024" or row["decoded_height"] != "1024":
            raise ValueError(f"DANI materialised row {row_number} changes decoded geometry")
        for field in (
            "encoded_sha256",
            "decoded_pixel_sha256_rgb",
        ):
            if not SHA256_PATTERN.fullmatch(row[field]):
                raise ValueError(f"DANI materialised row {row_number} has invalid {field}")
        if not PHASH_PATTERN.fullmatch(row["decoded_phash_rgb"]):
            raise ValueError(f"DANI materialised row {row_number} has invalid pHash")
        relative = Path(row["materialized_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"DANI materialised row {row_number} has an unsafe path")
        image_path = materialized_dir / relative
        if image_path in paths or not image_path.is_file():
            raise ValueError(f"DANI materialised row {row_number} path is missing or duplicated")
        paths.add(image_path)
        if image_path.stat().st_size != int(row["encoded_size_bytes"]):
            raise ValueError(f"DANI materialised row {row_number} encoded size changed")
        if dani.sha256_file(image_path) != row["encoded_sha256"]:
            raise ValueError(f"DANI materialised row {row_number} encoded bytes changed")
        group_splits[row["leakage_group"]].add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A frozen DANI parent group crosses splits")


def _exact_groups(
    rows: Sequence[Mapping[str, str]], union_find: UnionFind
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key in ("encoded_sha256", "decoded_pixel_sha256_rgb"):
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row[key]].append(index)
        for value, members in groups.items():
            if len(members) < 2:
                continue
            first = members[0]
            for member in members[1:]:
                union_find.union(first, member)
            output.append(
                {
                    "key": key,
                    "value": value,
                    "row_count": len(members),
                    "labels": "+".join(sorted({rows[index]["label"] for index in members})),
                    "splits": "+".join(sorted({rows[index]["split"] for index in members})),
                    "parent_coco_image_ids": "+".join(
                        sorted({rows[index]["parent_coco_image_id"] for index in members}, key=int)
                    ),
                    "selection_ids": "+".join(
                        sorted(rows[index]["selection_id"] for index in members)
                    ),
                }
            )
    return output


def _near_links(
    rows: Sequence[Mapping[str, str]],
    union_find: UnionFind,
    *,
    threshold: int,
) -> list[dict[str, object]]:
    values = [int(row["decoded_phash_rgb"], 16) for row in rows]
    parents = [row["parent_coco_image_id"] for row in rows]
    links: list[dict[str, object]] = []
    for left in range(len(rows)):
        for right in range(left):
            if parents[left] == parents[right]:
                continue
            distance = (values[left] ^ values[right]).bit_count()
            if distance > threshold:
                continue
            union_find.union(left, right)
            links.append(
                {
                    "left_selection_id": rows[left]["selection_id"],
                    "right_selection_id": rows[right]["selection_id"],
                    "left_parent_coco_image_id": parents[left],
                    "right_parent_coco_image_id": parents[right],
                    "left_split": rows[left]["split"],
                    "right_split": rows[right]["split"],
                    "left_label": rows[left]["label"],
                    "right_label": rows[right]["label"],
                    "phash_distance": distance,
                }
            )
    return links


def audit_integrity(
    materialized_dir: str | Path,
    output_dir: str | Path,
    *,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
    now: datetime | None = None,
) -> dict[str, object]:
    """Audit all frozen bytes and emit a training manifest only when every gate passes."""
    if phash_threshold < 0 or phash_threshold > 64:
        raise ValueError("pHash threshold must be in 0..64")
    source = Path(materialized_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing DANI integrity audit: {destination}")
    rows, provenance = _load_materialized(source)
    _validate_rows(rows, source)
    union_find = UnionFind(len(rows))
    by_parent: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_parent[row["parent_coco_image_id"]].append(index)
    for members in by_parent.values():
        for member in members[1:]:
            union_find.union(members[0], member)
    exact_groups = _exact_groups(rows, union_find)
    near_links = _near_links(rows, union_find, threshold=phash_threshold)

    component_splits: defaultdict[int, set[str]] = defaultdict(set)
    component_members: defaultdict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        root = union_find.find(index)
        component_splits[root].add(row["split"])
        component_members[root].append(index)
    cross_split_components = [root for root, splits in component_splits.items() if len(splits) > 1]
    cross_label_exact = sum(
        "0" in group["labels"] and "1" in group["labels"] for group in exact_groups
    )
    format_by_label = {
        label: sorted({row["decoded_format"] for row in rows if row["label"] == label})
        for label in ("0", "1")
    }
    mode_by_label = {
        label: sorted({row["decoded_mode"] for row in rows if row["label"] == label})
        for label in ("0", "1")
    }
    format_mode_balanced = (
        format_by_label["0"] == format_by_label["1"] and mode_by_label["0"] == mode_by_label["1"]
    )
    passed = cross_label_exact == 0 and not cross_split_components and format_mode_balanced
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / DUPLICATE_GROUPS_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DUPLICATE_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(exact_groups)
    with (destination / NEAR_LINKS_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NEAR_LINK_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(near_links)
    if passed:
        component_ids: dict[int, str] = {}
        for root, members in component_members.items():
            stable = "\n".join(sorted(rows[index]["selection_id"] for index in members))
            component_ids[root] = (
                "dani-integrity:" + hashlib.sha256(stable.encode()).hexdigest()[:24]
            )
        with (destination / TRAINING_MANIFEST_NAME).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=TRAINING_COLUMNS, extrasaction="raise")
            writer.writeheader()
            for index, row in enumerate(rows):
                component = component_ids[union_find.find(index)]
                writer.writerow(
                    _training_row(
                        row,
                        materialized_dir=source,
                        integrity_component=component,
                    )
                )
    created_at = (datetime.now(UTC) if now is None else now).astimezone(UTC).isoformat()
    summary: dict[str, object] = {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "inputs": {
            "materialized_provenance_sha256": dani.sha256_file(
                source / dani_materialize.MATERIALIZED_PROVENANCE_NAME
            ),
            "materialized_manifest_sha256": provenance["materialized_manifest_sha256"],
        },
        "counts": {
            "row_count": len(rows),
            "parent_count": len(by_parent),
            "integrity_component_count": len(component_members),
            "exact_duplicate_group_count": len(exact_groups),
            "cross_label_exact_duplicate_group_count": cross_label_exact,
            "cross_parent_near_phash_link_count": len(near_links),
            "cross_split_integrity_component_count": len(cross_split_components),
            "row_count_by_split_cell": {
                f"{split}:{cell}": count
                for (split, cell), count in sorted(
                    Counter((row["split"], row["cell"]) for row in rows).items()
                )
            },
        },
        "shortcut_audit": {
            "decoded_formats_by_label": format_by_label,
            "decoded_modes_by_label": mode_by_label,
            "format_mode_support_balanced_between_labels": format_mode_balanced,
        },
        "phash": {
            "distance_threshold": phash_threshold,
            "interpretation": "candidate leakage boundary, not duplicate identity proof",
        },
        "artifacts": {
            "exact_duplicate_groups": DUPLICATE_GROUPS_NAME,
            "near_phash_links": NEAR_LINKS_NAME,
            "training_manifest": TRAINING_MANIFEST_NAME if passed else None,
        },
        "eligibility": {
            "eligible_for_training": passed,
            "eligible_for_model_selection": False,
            "eligible_for_external_evaluation": False,
            "remaining_blocker": (
                "Run declared MPS training hypotheses using validation only; internal test and "
                "external evaluation remain untouched for model selection."
                if passed
                else "Resolve cross-label duplicates, cross-split components, or format/mode support imbalance."
            ),
        },
    }
    if passed:
        summary["training_manifest_sha256"] = dani.sha256_file(destination / TRAINING_MANIFEST_NAME)
    _write_json(destination / SUMMARY_NAME, summary)
    return summary
