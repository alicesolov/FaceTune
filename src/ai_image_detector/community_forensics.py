"""Metadata-only acquisition helpers for the CommunityForensics HighRes-v1 reservoir.

This module deliberately keeps the binary ``image_data`` column out of the first acquisition
stage.  It creates an auditable source catalogue, not a trainable image manifest: no image has
been materialised, decoded, hashed, or assigned a HighRes-v1 split at this point.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

REPOSITORY_ID = "OwensLab/CommunityForensics-Small"
PINNED_REVISION = "6c539a534c07917307c381f5af4053c6091b5278"
SOURCE_LICENSE = "CC-BY-NC-SA-4.0"
IMAGE_DATA_COLUMN = "image_data"
IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# The upstream Parquet schema is intentionally specified in source order.  Keeping the binary
# field visible here makes the exclusion below reviewable instead of relying on a vague wildcard.
SOURCE_SCHEMA_COLUMNS = (
    "image_name",
    "format",
    "resolution",
    "mode",
    IMAGE_DATA_COLUMN,
    "model_name",
    "nsfw_flag",
    "prompt",
    "real_source",
    "subset",
    "split",
    "label",
    "architecture",
)
META_COLUMNS = [column for column in SOURCE_SCHEMA_COLUMNS if column != IMAGE_DATA_COLUMN]

CANDIDATE_COLUMNS = (
    "locator",
    "repository_id",
    "revision",
    "shard_path",
    "row_index",
    "image_name",
    "format",
    "source_width",
    "source_height",
    "mode",
    "model_name",
    "nsfw_flag",
    "prompt_hash",
    "prompt_present",
    "content_group_id",
    "real_source",
    "subset",
    "source_split",
    "label",
    "architecture",
)


def get_hf_api() -> HfApi:
    """Create the Hub client late so tests never need network credentials or state."""
    return HfApi()


def get_dataset_loader() -> Callable[..., Iterable[Mapping[str, object]]]:
    """Return the dataset loader late so callers can inject a local fake in tests."""
    from datasets import load_dataset

    return load_dataset


def ensure_metadata_schema() -> None:
    """Fail closed if a future edit accidentally sends image bytes into this scan."""
    if len(SOURCE_SCHEMA_COLUMNS) != 13:
        raise RuntimeError("CommunityForensics source schema must contain exactly 13 fields")
    if IMAGE_DATA_COLUMN not in SOURCE_SCHEMA_COLUMNS:
        raise RuntimeError("CommunityForensics source schema must declare image_data explicitly")
    if IMAGE_DATA_COLUMN in META_COLUMNS:
        raise RuntimeError("Metadata scan must never request the image_data column")
    if len(META_COLUMNS) != 12:
        raise RuntimeError("Metadata projection must contain the 12 non-binary source fields")


def _value_from(entry: object, name: str) -> object | None:
    if isinstance(entry, Mapping):
        return entry.get(name)
    return getattr(entry, name, None)


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object | None, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer when present, got {value!r}")
    return int(value)


def canonical_shards(tree_entries: Iterable[object]) -> list[dict[str, object]]:
    """Project Hub tree entries into a sorted, JSON-stable list of Parquet identities."""
    shards: list[dict[str, object]] = []
    for entry in tree_entries:
        raw_path = _value_from(entry, "path")
        if not isinstance(raw_path, str) or not raw_path.endswith(".parquet"):
            continue
        lfs = _value_from(entry, "lfs")
        shards.append(
            {
                "path": raw_path,
                "size": _optional_int(_value_from(entry, "size"), field="size"),
                "blob_id": _optional_text(_value_from(entry, "blob_id")),
                "lfs": {
                    "size": _optional_int(_value_from(lfs, "size"), field="lfs.size"),
                    "sha256": _optional_text(_value_from(lfs, "sha256")),
                    "pointer_size": _optional_int(
                        _value_from(lfs, "pointer_size"), field="lfs.pointer_size"
                    ),
                },
                "xet_hash": _optional_text(_value_from(entry, "xet_hash")),
            }
        )
    ordered = sorted(shards, key=lambda shard: str(shard["path"]))
    paths = [str(shard["path"]) for shard in ordered]
    if not ordered:
        raise ValueError("Source lock found no .parquet shards")
    if len(paths) != len(set(paths)):
        raise ValueError("Source lock found duplicate .parquet shard paths")
    return ordered


def require_immutable_revision(revision: str) -> str:
    """Reject mutable branch/tag names at every source-lock and locator boundary."""
    if not IMMUTABLE_REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"revision must be a 40-character lowercase commit SHA, got {revision!r}")
    return revision


def resolve_revision(
    *,
    repository_id: str,
    requested_revision: str,
    api: Any | None = None,
) -> str:
    """Resolve a user-facing ref to the immutable commit used everywhere downstream."""
    hub_api = get_hf_api() if api is None else api
    info = hub_api.dataset_info(repository_id, revision=requested_revision)
    resolved_revision = _value_from(info, "sha")
    if not isinstance(resolved_revision, str):
        raise TypeError("Hugging Face dataset_info did not return a commit SHA")
    return require_immutable_revision(resolved_revision)


def list_source_shards(
    *,
    repository_id: str = REPOSITORY_ID,
    revision: str = PINNED_REVISION,
    api: Any | None = None,
) -> list[dict[str, object]]:
    """List every revision-pinned Parquet shard through the Hub tree API."""
    require_immutable_revision(revision)
    hub_api = get_hf_api() if api is None else api
    tree_entries = hub_api.list_repo_tree(
        repository_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
    )
    return canonical_shards(tree_entries)


def build_source_lock(
    *,
    repository_id: str,
    revision: str,
    shards: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the canonical, full source lock before narrowing the scan scope."""
    immutable_revision = require_immutable_revision(revision)
    locked_shards = sorted((dict(shard) for shard in shards), key=lambda shard: str(shard["path"]))
    paths = [str(shard["path"]) for shard in locked_shards]
    if len(paths) != len(set(paths)):
        raise ValueError("Source lock cannot contain duplicate .parquet shard paths")
    return {
        "schema_version": "community_forensics_source_lock_v1",
        "repository_id": repository_id,
        "revision": immutable_revision,
        "license": SOURCE_LICENSE,
        "repo_type": "dataset",
        "tree_recursive": True,
        "source_schema_columns": list(SOURCE_SCHEMA_COLUMNS),
        "metadata_columns": list(META_COLUMNS),
        "excluded_binary_columns": [IMAGE_DATA_COLUMN],
        "shard_count": len(locked_shards),
        "shards": locked_shards,
    }


def stable_locator(
    repository_id: str,
    revision: str,
    shard_path: str,
    row_index: int,
) -> str:
    """Return a source-stable locator that never depends on mutable streaming order."""
    if not repository_id or not revision or not shard_path:
        raise ValueError("repository_id, revision, and shard_path must be nonempty")
    if isinstance(row_index, bool) or not isinstance(row_index, Integral) or row_index < 0:
        raise ValueError(f"row_index must be a nonnegative integer, got {row_index!r}")
    return f"{repository_id}@{require_immutable_revision(revision)}:{shard_path}:{int(row_index)}"


def validate_resolution(value: object) -> tuple[int, int]:
    """Validate the source ``[width, height]`` metadata without opening an image."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("resolution must be a two-item [width, height] sequence")
    width, height = value
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or not isinstance(height, Integral)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"resolution dimensions must be positive integers, got {value!r}")
    return int(width), int(height)


def validate_label(value: object) -> int:
    """Accept only the declared binary CommunityForensics task labels."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in {0, 1}:
        raise ValueError(f"label must be 0 (real) or 1 (AI-generated), got {value!r}")
    return int(value)


def normalized_prompt_hash(value: object, *, locator: str) -> tuple[str, bool]:
    """Hash normalised prompt text, giving each absent prompt its own source group.

    The second return value indicates whether a nonempty prompt was present.  Missing and blank
    prompts deliberately hash a locator-qualified marker instead of becoming one giant component.
    """
    text = "" if value is None else str(value)
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    if normalized:
        payload = f"prompt:{normalized}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), True
    payload = f"blank-prompt-locator:{locator}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), False


def project_metadata_record(
    record: Mapping[str, object],
    *,
    repository_id: str,
    revision: str,
    shard_path: str,
    row_index: int,
) -> dict[str, object]:
    """Project one non-binary Parquet row into the source candidate catalogue."""
    missing = [column for column in META_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Metadata row is missing required columns: {missing}")
    locator = stable_locator(repository_id, revision, shard_path, row_index)
    width, height = validate_resolution(record["resolution"])
    label = validate_label(record["label"])
    prompt_hash, prompt_present = normalized_prompt_hash(record["prompt"], locator=locator)
    content_group_id = f"prompt:{prompt_hash}" if prompt_present else f"blank:{locator}"
    return {
        "locator": locator,
        "repository_id": repository_id,
        "revision": revision,
        "shard_path": shard_path,
        "row_index": int(row_index),
        "image_name": record["image_name"],
        "format": record["format"],
        "source_width": width,
        "source_height": height,
        "mode": record["mode"],
        "model_name": record["model_name"],
        "nsfw_flag": record["nsfw_flag"],
        "prompt_hash": prompt_hash,
        "prompt_present": prompt_present,
        "content_group_id": content_group_id,
        "real_source": record["real_source"],
        "subset": record["subset"],
        "source_split": record["split"],
        "label": label,
        "architecture": record["architecture"],
    }


def shard_url(*, repository_id: str, revision: str, shard_path: str) -> str:
    """Build a revision-pinned Hugging Face filesystem URL for exactly one Parquet shard."""
    if not shard_path or shard_path.startswith("/"):
        raise ValueError(f"shard_path must be a repository-relative path, got {shard_path!r}")
    return f"hf://datasets/{repository_id}@{require_immutable_revision(revision)}/{shard_path}"


def load_metadata_shard(
    *,
    repository_id: str,
    revision: str,
    shard_path: str,
    cache_dir: str | Path | None = None,
    dataset_loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> Iterable[Mapping[str, object]]:
    """Open one Parquet shard in streaming mode while requesting metadata columns only."""
    ensure_metadata_schema()
    loader = get_dataset_loader() if dataset_loader is None else dataset_loader
    return loader(
        "parquet",
        data_files=shard_url(
            repository_id=repository_id,
            revision=revision,
            shard_path=shard_path,
        ),
        split="train",
        streaming=True,
        columns=META_COLUMNS,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )


def require_fresh_output_dir(output_dir: str | Path) -> Path:
    """Create a new output directory and refuse to mix a scan with prior evidence."""
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing metadata scan output: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_scan_scope(
    *,
    all_shards: Sequence[Mapping[str, object]],
    selected_shards: Sequence[Mapping[str, object]],
    limit_shards: int | None,
) -> dict[str, object]:
    """Describe whether a scan can represent the locked source in its entirety."""
    partial = limit_shards is not None
    return {
        "kind": "partial_scout" if partial else "complete_source_scan",
        "partial": partial,
        "complete": not partial and len(selected_shards) == len(all_shards),
        "eligible_for_candidate_selection": not partial and len(selected_shards) == len(all_shards),
        "eligible_for_training": False,
        "available_shard_count": len(all_shards),
        "selected_shard_count": len(selected_shards),
        "limit_shards": limit_shards,
        "selected_shards": [str(shard["path"]) for shard in selected_shards],
    }


def _select_shards(
    shards: Sequence[Mapping[str, object]], limit_shards: int | None
) -> list[Mapping[str, object]]:
    if limit_shards is not None and limit_shards <= 0:
        raise ValueError("limit_shards must be a positive integer when supplied")
    if limit_shards is None:
        return list(shards)
    return list(shards[:limit_shards])


def scan_metadata(
    output_dir: str | Path,
    *,
    repository_id: str = REPOSITORY_ID,
    revision: str = PINNED_REVISION,
    limit_shards: int | None = None,
    cache_dir: str | Path | None = None,
    api: Any | None = None,
    dataset_loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Write a revision-pinned, metadata-only candidate catalogue for HighRes-v1.

    Passing ``limit_shards`` is intentionally a scout-only mode.  Its report remains partial even
    if that numerical limit happens to include every currently visible shard.
    """
    ensure_metadata_schema()
    output = require_fresh_output_dir(output_dir)
    hub_api = get_hf_api() if api is None else api
    resolved_revision = resolve_revision(
        repository_id=repository_id,
        requested_revision=revision,
        api=hub_api,
    )
    all_shards = list_source_shards(
        repository_id=repository_id,
        revision=resolved_revision,
        api=hub_api,
    )
    selected_shards = _select_shards(all_shards, limit_shards)
    source_lock = build_source_lock(
        repository_id=repository_id,
        revision=resolved_revision,
        shards=all_shards,
    )
    source_lock_path = output / "source_lock.json"
    write_json(source_lock_path, source_lock)

    catalog_path = output / "source_catalog.csv"
    scanned_rows_by_shard: dict[str, int] = {}
    candidate_count = 0
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for shard in selected_shards:
            shard_path = str(shard["path"])
            row_count = 0
            records = load_metadata_shard(
                repository_id=repository_id,
                revision=resolved_revision,
                shard_path=shard_path,
                cache_dir=cache_dir,
                dataset_loader=dataset_loader,
            )
            for row_index, record in enumerate(records):
                writer.writerow(
                    project_metadata_record(
                        record,
                        repository_id=repository_id,
                        revision=resolved_revision,
                        shard_path=shard_path,
                        row_index=row_index,
                    )
                )
                row_count += 1
                candidate_count += 1
            scanned_rows_by_shard[shard_path] = row_count

    scope = build_scan_scope(
        all_shards=all_shards,
        selected_shards=selected_shards,
        limit_shards=limit_shards,
    )
    timestamp = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    report: dict[str, object] = {
        "schema_version": "community_forensics_metadata_scan_v1",
        "repository_id": repository_id,
        "requested_revision": revision,
        "revision": resolved_revision,
        "resolved_revision": resolved_revision,
        "created_at_utc": timestamp,
        "source_lock": source_lock_path.name,
        "source_lock_sha256": sha256_file(source_lock_path),
        "source_catalog": catalog_path.name,
        "source_catalog_sha256": sha256_file(catalog_path),
        "catalog_kind": "metadata_only_candidate_catalog_not_trainable_manifest",
        "catalog_row_count": candidate_count,
        "metadata_columns": list(META_COLUMNS),
        "excluded_binary_columns": [IMAGE_DATA_COLUMN],
        "image_data_materialised": False,
        "image_data_decoded": False,
        "rows_scanned_by_shard": scanned_rows_by_shard,
        "scan_scope": scope,
    }
    write_json(output / "provenance.json", report)
    return report
