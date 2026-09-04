"""Freeze the complete five-cell DANI research corpus under a strict local byte cap."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from . import dani, dani_geometry, dani_selection

CORE_SCHEMA_VERSION: Final = "dani_highres_core_v1"
CORE_SELECTION_NAME: Final = "core_selection.csv"
CORE_SPEC_NAME: Final = "core_spec.json"
CORE_SHARD_PLAN_NAME: Final = "shard_plan.csv"
CORE_PROVENANCE_NAME: Final = "provenance.json"
CORE_CELLS: Final = tuple(dani_selection.CELL_DEFINITIONS)
REAL_RESOLUTION_STRIDE: Final = 25_014
SHARD_PLAN_COLUMNS: Final = (
    "processing_order",
    "shard_path",
    "expected_size_bytes",
    "expected_sha256",
    "selected_row_count",
)


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


def _load_source_lock(
    lineage_scan_dir: Path, expected_sha256: str
) -> tuple[Path, dict[str, dict[str, object]]]:
    lock_path = lineage_scan_dir / "source_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"DANI source lock does not exist: {lock_path}")
    if dani.sha256_file(lock_path) != expected_sha256:
        raise ValueError("DANI source lock differs from the frozen preselection")
    payload = _read_json(lock_path)
    if (
        payload.get("schema_version") != "dani_lineage_source_lock_v1"
        or payload.get("repository_id") != dani.REPOSITORY_ID
        or payload.get("revision") != dani.PINNED_REVISION
    ):
        raise ValueError("DANI source lock identity differs from the pinned source")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list):
        raise TypeError("DANI source lock shards must be a list")
    shards: dict[str, dict[str, object]] = {}
    for raw in raw_shards:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise TypeError("DANI source lock contains an invalid shard")
        path = raw["path"]
        lfs = raw.get("lfs")
        if not isinstance(lfs, dict):
            raise TypeError(f"DANI source lock shard {path} has no LFS identity")
        size = lfs.get("size")
        sha256 = lfs.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ValueError(f"DANI source lock shard {path} has invalid LFS metadata")
        shards[path] = {"size": size, "sha256": sha256}
    return lock_path, shards


def _choose_core_rows(candidates: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: defaultdict[tuple[str, str], defaultdict[str, list[Mapping[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in candidates:
        grouped[(row["parent_coco_image_id"], row["coco_caption_id"])][row["cell"]].append(row)
    selected: list[dict[str, str]] = []
    for pair, cells in sorted(grouped.items(), key=lambda item: int(item[0][0])):
        if any(cell not in cells for cell in CORE_CELLS):
            raise ValueError(f"DANI core pair {pair} is missing a required cell")
        real_rows = sorted(cells["real_coco"], key=lambda row: int(row["source_index"]))
        real_indexes = [int(row["source_index"]) for row in real_rows]
        if len(real_rows) != 4 or any(
            right - left != REAL_RESOLUTION_STRIDE for left, right in pairwise(real_indexes)
        ):
            raise ValueError(f"DANI core pair {pair} lacks the four ordered real resolutions")
        chosen = [real_rows[-1]]
        for cell in CORE_CELLS[1:]:
            if len(cells[cell]) != 1:
                raise ValueError(f"DANI core pair {pair} has {len(cells[cell])} rows for {cell}")
            chosen.append(cells[cell][0])
        if (
            len({row["split"] for row in chosen}) != 1
            or len({row["leakage_group"] for row in chosen}) != 1
        ):
            raise ValueError(f"DANI core pair {pair} changes its frozen group split")
        for row in chosen:
            selected.append(
                {
                    **row,
                    "selection_id": "dani-core:"
                    + dani_selection.stable_hash(
                        CORE_SCHEMA_VERSION,
                        row["parent_coco_image_id"],
                        row["coco_caption_id"],
                        row["cell"],
                        row["locator"],
                    ),
                }
            )
    selected.sort(
        key=lambda row: (
            dani_selection.SPLIT_ORDER[row["split"]],
            int(row["parent_coco_image_id"]),
            CORE_CELLS.index(row["cell"]),
        )
    )
    return selected


def build_core_plan(
    preselection_dir: str | Path,
    lineage_scan_dir: str | Path,
    output_dir: str | Path,
    *,
    byte_cap: int = dani_selection.MATERIALISATION_BYTE_BUDGET,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Freeze the complete five-cell DANI core and its bounded shard processing plan."""
    if byte_cap <= 0:
        raise ValueError("DANI core byte cap must be positive")
    source = Path(preselection_dir)
    lineage = Path(lineage_scan_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing DANI core plan: {destination}")
    candidates, preselection_hashes = dani_geometry._validate_preselection(source)
    spec = _read_json(source / dani_selection.SELECTION_SPEC_NAME)
    input_hashes = spec.get("input_hashes")
    if not isinstance(input_hashes, dict) or not isinstance(
        input_hashes.get("lineage_source_lock_sha256"), str
    ):
        raise TypeError("DANI preselection does not pin its lineage source lock")
    lock_path, locked_shards = _load_source_lock(
        lineage, input_hashes["lineage_source_lock_sha256"]
    )
    selected = _choose_core_rows(candidates)
    parent_count = len({row["parent_coco_image_id"] for row in selected})
    if len(selected) != parent_count * len(CORE_CELLS):
        raise AssertionError("DANI core does not have exactly five rows per parent")
    theoretical_rgb_bytes = len(selected) * 1024 * 1024 * 3
    if theoretical_rgb_bytes > byte_cap:
        raise ValueError("DANI core decoded RGB upper bound exceeds the byte cap")

    shard_counts = Counter(row["shard_path"] for row in selected)
    shard_rows: list[dict[str, object]] = []
    for shard_path, count in shard_counts.items():
        identity = locked_shards.get(shard_path)
        if identity is None:
            raise ValueError(f"DANI core shard is absent from source lock: {shard_path}")
        shard_rows.append(
            {
                "shard_path": shard_path,
                "expected_size_bytes": identity["size"],
                "expected_sha256": identity["sha256"],
                "selected_row_count": count,
            }
        )
    shard_rows.sort(key=lambda row: (-int(row["expected_size_bytes"]), str(row["shard_path"])))
    for order, row in enumerate(shard_rows, start=1):
        row["processing_order"] = order
    shard_bytes = sum(int(row["expected_size_bytes"]) for row in shard_rows)
    if shard_bytes > byte_cap:
        raise ValueError("Required DANI core shards exceed the byte cap")
    processed_upper = 0
    peak_staged_upper = 0
    for row in shard_rows:
        size = int(row["expected_size_bytes"])
        peak_staged_upper = max(peak_staged_upper, processed_upper + 2 * size)
        processed_upper += size
    if peak_staged_upper > byte_cap:
        raise ValueError("Safe download-extract staging upper bound exceeds the byte cap")

    created_at = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    core_spec: dict[str, object] = {
        "schema_version": CORE_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "byte_cap": byte_cap,
        "included_cells": list(CORE_CELLS),
        "excluded_cells": [
            cell for cell in dani_selection.CELL_DEFINITIONS if cell not in CORE_CELLS
        ],
        "real_row_rule": (
            "maximum source_index among four caption-matched real rows separated by 25014; "
            "decoded geometry must still be verified from bytes"
        ),
        "synthetic_row_rule": "frozen preselection row for SDXL I2I and SDXL T2I",
        "preselection_hashes": preselection_hashes,
        "lineage_source_lock_sha256": dani.sha256_file(lock_path),
    }
    provenance: dict[str, object] = {
        "schema_version": CORE_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "image_bytes_requested": False,
        "image_bytes_read": False,
        "counts": {
            "parent_count": parent_count,
            "selected_row_count": len(selected),
            "selected_row_count_by_cell": dict(
                sorted(Counter(row["cell"] for row in selected).items())
            ),
            "selected_row_count_by_split": dict(
                sorted(Counter(row["split"] for row in selected).items())
            ),
            "required_shard_count": len(shard_rows),
        },
        "budget": {
            "hard_cap_bytes": byte_cap,
            "theoretical_decoded_rgb_bytes": theoretical_rgb_bytes,
            "required_pinned_shard_bytes": shard_bytes,
            "worst_case_staged_download_plus_extraction_bytes": peak_staged_upper,
        },
        "eligibility": {
            "eligible_for_bounded_shard_materialisation": True,
            "eligible_for_training": False,
            "remaining_blocker": (
                "Download each pinned shard in processing order, verify its LFS SHA-256, extract "
                "only frozen rows, then pass decoded geometry/container/hash/duplicate/leakage audits."
            ),
        },
    }
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / CORE_SPEC_NAME, core_spec)
    with (destination / CORE_SELECTION_NAME).open("w", encoding="utf-8", newline="") as handle:
        fields = ("selection_id", *dani_selection.GEOMETRY_CANDIDATE_COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(selected)
    with (destination / CORE_SHARD_PLAN_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHARD_PLAN_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(shard_rows)
    provenance["core_spec_sha256"] = dani.sha256_file(destination / CORE_SPEC_NAME)
    provenance["core_selection_sha256"] = dani.sha256_file(destination / CORE_SELECTION_NAME)
    provenance["shard_plan_sha256"] = dani.sha256_file(destination / CORE_SHARD_PLAN_NAME)
    _write_json(destination / CORE_PROVENANCE_NAME, provenance)
    return provenance
