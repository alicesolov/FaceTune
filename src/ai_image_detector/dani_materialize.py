"""Bounded, resumable materialisation of a frozen DANI core from pinned Parquet shards."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import imagehash
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError

from . import dani, dani_core, dani_selection

MATERIALIZATION_SCHEMA_VERSION: Final = "dani_highres_materialization_v1"
PARTIAL_MANIFEST_NAME: Final = "materialized.partial.csv"
MATERIALIZED_MANIFEST_NAME: Final = "materialized_manifest.csv"
MATERIALIZED_PROVENANCE_NAME: Final = "provenance.json"
IMAGES_DIRECTORY: Final = "images"
ALLOWED_FORMATS: Final = {"JPEG", "PNG", "WEBP"}
MATERIALIZED_EXTRA_COLUMNS: Final = (
    "materialized_path",
    "encoded_size_bytes",
    "encoded_sha256",
    "decoded_width",
    "decoded_height",
    "decoded_mode",
    "decoded_format",
    "decoded_pixel_sha256_rgb",
    "decoded_phash_rgb",
)
MATERIALIZED_COLUMNS: Final = (
    "selection_id",
    *dani_selection.GEOMETRY_CANDIDATE_COLUMNS,
    *MATERIALIZED_EXTRA_COLUMNS,
)

Downloader = Callable[[str, Path, int, str], Path]
ProgressCallback = Callable[[str], None]


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


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _projected_range_download_peak(
    staging_dir: Path,
    materialized_dir: Path,
    local_path: Path,
    expected_size: int,
) -> int:
    """Bound peak bytes without counting already downloaded range parts twice.

    A complete range download temporarily holds both all parts and the assembled shard. Existing
    resumable parts are already included in ``staging_dir`` and replace, rather than add to, the
    eventual full set of parts.
    """
    parts_dir = local_path.with_name(local_path.name + ".range-parts")
    unrelated_staging = _directory_bytes(staging_dir) - _directory_bytes(parts_dir)
    return unrelated_staging + _directory_bytes(materialized_dir) + 2 * expected_size


def _load_core_plan(
    core_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    paths = {
        "spec": core_dir / dani_core.CORE_SPEC_NAME,
        "selection": core_dir / dani_core.CORE_SELECTION_NAME,
        "shards": core_dir / dani_core.CORE_SHARD_PLAN_NAME,
        "provenance": core_dir / dani_core.CORE_PROVENANCE_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("DANI core plan is incomplete: " + ", ".join(missing))
    provenance = _read_json(paths["provenance"])
    if provenance.get("schema_version") != dani_core.CORE_SCHEMA_VERSION:
        raise ValueError("DANI core plan has an unsupported schema_version")
    expected_hashes = {
        "core_spec_sha256": dani.sha256_file(paths["spec"]),
        "core_selection_sha256": dani.sha256_file(paths["selection"]),
        "shard_plan_sha256": dani.sha256_file(paths["shards"]),
    }
    for field, digest in expected_hashes.items():
        if provenance.get(field) != digest:
            raise ValueError(f"DANI core {field} differs from provenance")
    eligibility = provenance.get("eligibility")
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("eligible_for_bounded_shard_materialisation") is not True
        or eligibility.get("eligible_for_training") is not False
        or provenance.get("image_bytes_requested") is not False
        or provenance.get("image_bytes_read") is not False
    ):
        raise ValueError("DANI core plan is not in the required materialisation state")
    with paths["selection"].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = ("selection_id", *dani_selection.GEOMETRY_CANDIDATE_COLUMNS)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError("DANI core selection schema differs from the locked schema")
        selection = [dict(row) for row in reader]
    if not selection or len({row["selection_id"] for row in selection}) != len(selection):
        raise ValueError("DANI core selection is empty or has duplicate IDs")
    with paths["shards"].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != dani_core.SHARD_PLAN_COLUMNS:
            raise ValueError("DANI core shard plan schema differs from the locked schema")
        shards = [dict(row) for row in reader]
    if not shards or [int(row["processing_order"]) for row in shards] != list(
        range(1, len(shards) + 1)
    ):
        raise ValueError("DANI core shard processing order is invalid")
    expected_counts = Counter(row["shard_path"] for row in selection)
    if expected_counts != Counter(
        {row["shard_path"]: int(row["selected_row_count"]) for row in shards}
    ):
        raise ValueError("DANI core shard counts differ from selection")
    return selection, shards, provenance


def _default_downloader(
    shard_path: str, staging_dir: Path, expected_size: int, expected_sha256: str
) -> Path:
    downloaded = hf_hub_download(
        repo_id=dani.REPOSITORY_ID,
        filename=shard_path,
        repo_type="dataset",
        revision=dani.PINNED_REVISION,
        local_dir=staging_dir,
    )
    return Path(downloaded)


def make_range_downloader(*, workers: int = 8) -> Downloader:
    """Create a direct pinned-HTTPS downloader with resumable parallel byte ranges."""
    if workers <= 0:
        raise ValueError("range download workers must be positive")

    def download(
        shard_path: str,
        staging_dir: Path,
        expected_size: int,
        expected_sha256: str,
    ) -> Path:
        from .range_download import download_range_file

        url = (
            f"https://huggingface.co/datasets/{dani.REPOSITORY_ID}/resolve/"
            f"{dani.PINNED_REVISION}/{shard_path}"
        )
        return download_range_file(
            url,
            staging_dir / shard_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            workers=workers,
        )

    return download


def _verify_shard(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Downloaded DANI shard does not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"DANI shard size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = dani.sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"DANI shard SHA-256 mismatch for {path}")


def _validate_source_record(
    selection: Mapping[str, str], record: Mapping[str, object]
) -> tuple[bytes, str]:
    source_index = int(selection["source_index"])
    expected: dict[str, object] = {
        "index": source_index,
        "size": int(selection["declared_size"]),
        "category": selection["category"] or None,
        "class_id": selection["class_id"] or None,
        "model": selection["model"],
        "gen_type": selection["gen_type"],
        "reference": selection["label"] == "0",
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise ValueError(
                f"DANI Parquet {field} mismatch for source_index {source_index}: "
                f"expected {expected_value!r}, got {record.get(field)!r}"
            )
    image = record.get("image")
    if not isinstance(image, dict):
        raise TypeError(f"DANI Parquet image is not an object for source_index {source_index}")
    image_bytes = image.get("bytes")
    image_path = image.get("path")
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError(f"DANI Parquet image bytes are empty for source_index {source_index}")
    if not isinstance(image_path, str) or Path(image_path).name != selection["image_path_basename"]:
        raise ValueError(f"DANI Parquet image path mismatch for source_index {source_index}")
    return image_bytes, image_path


def _inspect_image(image_bytes: bytes, *, source_index: str) -> dict[str, object]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            decoded_format = source.format
            decoded_mode = source.mode
            source.verify()
        with Image.open(io.BytesIO(image_bytes)) as source:
            normalized = ImageOps.exif_transpose(source).convert("RGB")
            normalized.load()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(
            f"DANI image is corrupt for source_index {source_index}: {error}"
        ) from error
    if decoded_format not in ALLOWED_FORMATS:
        raise ValueError(
            f"DANI image has unsupported format {decoded_format!r} for source_index {source_index}"
        )
    if normalized.size != (1024, 1024):
        raise ValueError(
            f"DANI image geometry mismatch for source_index {source_index}: {normalized.size}"
        )
    return {
        "encoded_size_bytes": len(image_bytes),
        "encoded_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "decoded_width": normalized.width,
        "decoded_height": normalized.height,
        "decoded_mode": decoded_mode,
        "decoded_format": decoded_format,
        "decoded_pixel_sha256_rgb": hashlib.sha256(normalized.tobytes()).hexdigest(),
        "decoded_phash_rgb": str(imagehash.phash(normalized)),
    }


def _extension(decoded_format: str) -> str:
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[decoded_format]


def _read_partial(
    path: Path, selection_by_id: Mapping[str, Mapping[str, str]], output_dir: Path
) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MATERIALIZED_COLUMNS:
            raise ValueError("Existing DANI partial manifest has an invalid schema")
        rows = [dict(row) for row in reader]
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        selection_id = row["selection_id"]
        source = selection_by_id.get(selection_id)
        if source is None or selection_id in seen:
            raise ValueError(f"DANI partial row {row_number} is unknown or duplicated")
        seen.add(selection_id)
        for field in ("selection_id", *dani_selection.GEOMETRY_CANDIDATE_COLUMNS):
            if row[field] != source[field]:
                raise ValueError(f"DANI partial row {row_number} differs from frozen selection")
        image_path = output_dir / row["materialized_path"]
        if (
            not image_path.is_file()
            or image_path.stat().st_size != int(row["encoded_size_bytes"])
            or dani.sha256_file(image_path) != row["encoded_sha256"]
        ):
            raise ValueError(f"DANI partial row {row_number} image bytes changed")
    return rows


def _selected_row_groups(parquet_file: Any, selected_indexes: set[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    offset = 0
    for group_index in range(parquet_file.num_row_groups):
        count = parquet_file.metadata.row_group(group_index).num_rows
        if any(offset <= index < offset + count for index in selected_indexes):
            groups.append((group_index, offset))
        offset += count
    if selected_indexes and max(selected_indexes) >= offset:
        raise ValueError("DANI selection row_index exceeds its pinned Parquet shard")
    return groups


def materialize_core(
    core_dir: str | Path,
    staging_dir: str | Path,
    output_dir: str | Path,
    *,
    downloader: Downloader | None = None,
    byte_cap: int = dani_selection.MATERIALISATION_BYTE_BUDGET,
    progress: ProgressCallback | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Download, verify, extract, and resume exact selected DANI rows under the byte cap."""
    if byte_cap <= 0:
        raise ValueError("DANI materialisation byte cap must be positive")
    core = Path(core_dir)
    staging = Path(staging_dir)
    destination = Path(output_dir)
    selection, shard_plan, core_provenance = _load_core_plan(core)
    budget = core_provenance.get("budget")
    if not isinstance(budget, dict) or budget.get("hard_cap_bytes") != byte_cap:
        raise ValueError("DANI materialisation byte cap differs from the frozen core plan")
    destination.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    final_provenance = destination / MATERIALIZED_PROVENANCE_NAME
    if final_provenance.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed DANI materialisation: {destination}"
        )
    partial_path = destination / PARTIAL_MANIFEST_NAME
    final_manifest = destination / MATERIALIZED_MANIFEST_NAME
    if final_manifest.exists():
        raise ValueError("DANI final manifest exists without final provenance")
    selection_by_id = {row["selection_id"]: row for row in selection}
    completed = _read_partial(partial_path, selection_by_id, destination)
    completed_ids = {row["selection_id"] for row in completed}
    fetch = _default_downloader if downloader is None else downloader

    mode = "a" if partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATERIALIZED_COLUMNS, extrasaction="raise")
        if mode == "w":
            writer.writeheader()
        for shard in shard_plan:
            shard_path = shard["shard_path"]
            shard_selection = [
                row
                for row in selection
                if row["shard_path"] == shard_path and row["selection_id"] not in completed_ids
            ]
            local_path = staging / shard_path
            expected_size = int(shard["expected_size_bytes"])
            expected_sha256 = shard["expected_sha256"]
            if not shard_selection:
                if local_path.is_file():
                    local_path.unlink()
                continue
            if not local_path.is_file():
                if progress is not None:
                    progress(f"download {shard_path}")
                projected_download_peak = _projected_range_download_peak(
                    staging,
                    destination,
                    local_path,
                    expected_size,
                )
                if projected_download_peak > byte_cap:
                    raise ValueError("Parallel range staging would exceed the hard byte cap")
                local_path = fetch(shard_path, staging, expected_size, expected_sha256)
            if _directory_bytes(staging) + _directory_bytes(destination) > byte_cap:
                raise ValueError("DANI staging plus materialised bytes exceed the hard cap")
            if progress is not None:
                progress(f"verify {shard_path}")
            _verify_shard(local_path, expected_size, expected_sha256)
            current_total_bytes = _directory_bytes(staging) + _directory_bytes(destination)

            import pyarrow.parquet as pq

            by_row_index = {int(row["row_index"]): row for row in shard_selection}
            with pq.ParquetFile(local_path) as parquet_file:
                groups = _selected_row_groups(parquet_file, set(by_row_index))
                found: set[int] = set()
                for group_index, group_offset in groups:
                    table = parquet_file.read_row_group(
                        group_index,
                        columns=(
                            "index",
                            "image",
                            "size",
                            "category",
                            "class_id",
                            "model",
                            "gen_type",
                            "reference",
                        ),
                        use_threads=False,
                    )
                    for row_index, selected in by_row_index.items():
                        local_index = row_index - group_offset
                        if not 0 <= local_index < table.num_rows:
                            continue
                        record = table.slice(local_index, 1).to_pylist()[0]
                        image_bytes, _ = _validate_source_record(selected, record)
                        inspected = _inspect_image(
                            image_bytes, source_index=selected["source_index"]
                        )
                        suffix = hashlib.sha256(selected["selection_id"].encode()).hexdigest()
                        relative = (
                            Path(IMAGES_DIRECTORY)
                            / selected["split"]
                            / selected["cell"]
                            / f"{suffix}{_extension(str(inspected['decoded_format']))}"
                        )
                        target = destination / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        projected_bytes = current_total_bytes + len(image_bytes)
                        if projected_bytes > byte_cap:
                            raise ValueError("Next DANI image would exceed the hard byte cap")
                        temporary = target.with_suffix(target.suffix + ".part")
                        temporary.write_bytes(image_bytes)
                        temporary.replace(target)
                        current_total_bytes = projected_bytes
                        materialized = {
                            **selected,
                            "materialized_path": relative.as_posix(),
                            **inspected,
                        }
                        writer.writerow(materialized)
                        handle.flush()
                        completed.append({key: str(value) for key, value in materialized.items()})
                        completed_ids.add(selected["selection_id"])
                        found.add(row_index)
                    del table
            if found != set(by_row_index):
                missing = sorted(set(by_row_index) - found)
                raise ValueError(f"DANI shard omitted {len(missing)} selected row(s)")
            local_path.unlink()
            if progress is not None:
                progress(
                    f"extracted {len(shard_selection)} rows from {shard_path}; staging removed"
                )

    final_rows = _read_partial(partial_path, selection_by_id, destination)
    by_id = {row["selection_id"]: row for row in final_rows}
    if len(by_id) != len(selection):
        raise AssertionError("DANI materialisation did not consume the complete frozen selection")
    with final_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATERIALIZED_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(by_id[row["selection_id"]] for row in selection)
    partial_path.unlink()
    created_at = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    formats = Counter(row["decoded_format"] for row in final_rows)
    modes = Counter(row["decoded_mode"] for row in final_rows)
    encoded_bytes = sum(int(row["encoded_size_bytes"]) for row in final_rows)
    provenance: dict[str, object] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "core_provenance_sha256": dani.sha256_file(core / dani_core.CORE_PROVENANCE_NAME),
        "materialized_manifest": MATERIALIZED_MANIFEST_NAME,
        "materialized_manifest_sha256": dani.sha256_file(final_manifest),
        "counts": {
            "materialized_row_count": len(final_rows),
            "row_count_by_cell": dict(sorted(Counter(row["cell"] for row in final_rows).items())),
            "decoded_format_counts": dict(sorted(formats.items())),
            "decoded_mode_counts": dict(sorted(modes.items())),
        },
        "budget": {
            "hard_cap_bytes": byte_cap,
            "materialized_encoded_bytes": encoded_bytes,
            "final_output_directory_bytes": _directory_bytes(destination),
            "remaining_bytes_at_completion": byte_cap - _directory_bytes(destination),
        },
        "eligibility": {
            "all_selected_rows_materialized": True,
            "all_decoded_geometry_exact_1024": True,
            "eligible_for_duplicate_and_leakage_audit": True,
            "eligible_for_training": False,
            "remaining_blocker": (
                "Audit exact encoded/pixel hashes, perceptual duplicate components, cross-split "
                "leakage and container/mode shortcuts; then freeze the canonical training manifest."
            ),
        },
    }
    _write_json(final_provenance, provenance)
    return provenance
