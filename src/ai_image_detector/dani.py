"""Metadata-only acquisition helpers for the DANI high-resolution source audit.

The initial DANI pass records a revision-pinned source catalogue only. It deliberately excludes
the binary image field at the Parquet projection boundary: no image is materialised, decoded,
hashed, or assigned to a training split by this module.

The reference field is the upstream declaration of natural versus generated origin. The documented
index field is retained as provenance and a conservative grouping candidate, but this scanner does
not claim that equal or related indexes prove one-to-one semantic pairing.

The public non-binary schema does not expose a documented COCO parent/caption group. A complete
catalogue is therefore useful for source auditing but remains blocked from internal corpus
selection until a separate, revision-pinned mapping audit proves a safe group key.
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

REPOSITORY_ID = "Renyang/DANI"
PINNED_REVISION = "870e29fcdc13c405fae35442899e9ba1da11691d"
SOURCE_LICENSE = "CC-BY-NC-4.0"
IMAGE_COLUMN = "image"
IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
INTERNAL_SELECTION_BLOCKER = (
    "The public DANI metadata schema lacks a documented parent COCO/caption group; "
    "source_index is an image-level upstream identifier, not proven pair provenance."
)

# The order follows the upstream README and must remain explicit. Keeping image visible here makes
# the exclusion in META_COLUMNS reviewable instead of relying on an implicit wildcard.
SOURCE_SCHEMA_COLUMNS = (
    "index",
    IMAGE_COLUMN,
    "size",
    "category",
    "class_id",
    "model",
    "gen_type",
    "reference",
)
META_COLUMNS = [column for column in SOURCE_SCHEMA_COLUMNS if column != IMAGE_COLUMN]

CANDIDATE_COLUMNS = (
    "locator",
    "repository_id",
    "revision",
    "shard_path",
    "row_index",
    "source_index",
    "source_index_hash",
    "source_index_group_id",
    "declared_size",
    "category",
    "class_id",
    "model",
    "gen_type",
    "reference",
    "label",
)


def get_hf_api() -> HfApi:
    """Create the Hub client late so tests do not need network credentials or state."""
    return HfApi()


def get_dataset_loader() -> Callable[..., Iterable[Mapping[str, object]]]:
    """Return the datasets loader late so tests can inject a non-network fake."""
    from datasets import load_dataset

    return load_dataset


def ensure_metadata_schema() -> None:
    """Fail closed if a future edit sends DANI image bytes into the scan."""
    if len(SOURCE_SCHEMA_COLUMNS) != 8:
        raise RuntimeError("DANI source schema must contain exactly 8 fields")
    if IMAGE_COLUMN not in SOURCE_SCHEMA_COLUMNS:
        raise RuntimeError("DANI source schema must declare the image field explicitly")
    if IMAGE_COLUMN in META_COLUMNS:
        raise RuntimeError("DANI metadata scan must never request the image column")
    if len(META_COLUMNS) != 7:
        raise RuntimeError("DANI metadata projection must contain the 7 non-binary source fields")


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
    """List every revision-pinned DANI Parquet shard through the Hub tree API."""
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
    """Build a canonical full source lock before narrowing the scanning scope."""
    immutable_revision = require_immutable_revision(revision)
    locked_shards = sorted((dict(shard) for shard in shards), key=lambda shard: str(shard["path"]))
    paths = [str(shard["path"]) for shard in locked_shards]
    if len(paths) != len(set(paths)):
        raise ValueError("Source lock cannot contain duplicate .parquet shard paths")
    return {
        "schema_version": "dani_source_lock_v1",
        "repository_id": repository_id,
        "revision": immutable_revision,
        "license": SOURCE_LICENSE,
        "repo_type": "dataset",
        "tree_recursive": True,
        "source_schema_columns": list(SOURCE_SCHEMA_COLUMNS),
        "metadata_columns": list(META_COLUMNS),
        "excluded_binary_columns": [IMAGE_COLUMN],
        "shard_count": len(locked_shards),
        "shards": locked_shards,
    }


def stable_locator(
    repository_id: str,
    revision: str,
    shard_path: str,
    row_index: int,
) -> str:
    """Return a source-stable locator independent of mutable streaming order."""
    if not repository_id or not revision or not shard_path:
        raise ValueError("repository_id, revision, and shard_path must be nonempty")
    if isinstance(row_index, bool) or not isinstance(row_index, Integral) or row_index < 0:
        raise ValueError(f"row_index must be a nonnegative integer, got {row_index!r}")
    return f"{repository_id}@{require_immutable_revision(revision)}:{shard_path}:{int(row_index)}"


def validate_size(value: object) -> int:
    """Validate the declared upstream raster size without opening an image."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"size must be a positive integer, got {value!r}")
    return int(value)


def validate_reference(value: object) -> bool:
    """Accept only the documented boolean DANI natural-image declaration."""
    if not isinstance(value, bool):
        raise TypeError(f"reference must be a boolean, got {value!r}")
    return value


def normalise_source_index(value: object) -> str:
    """Return the nonempty upstream index as a provenance identifier."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"index must be a nonempty identifier, got {value!r}")
    index = str(value).strip()
    if not index:
        raise ValueError("index must be a nonempty identifier")
    return index


def source_index_hash(value: str) -> str:
    """Hash one upstream index for grouping without claiming it is an image pair."""
    payload = f"dani-source-index:{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def project_metadata_record(
    record: Mapping[str, object],
    *,
    repository_id: str,
    revision: str,
    shard_path: str,
    row_index: int,
) -> dict[str, object]:
    """Project one non-binary DANI row into the metadata-only candidate catalogue."""
    missing = [column for column in META_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"Metadata row is missing required columns: {missing}")
    locator = stable_locator(repository_id, revision, shard_path, row_index)
    upstream_index = normalise_source_index(record["index"])
    upstream_index_hash = source_index_hash(upstream_index)
    reference = validate_reference(record["reference"])
    return {
        "locator": locator,
        "repository_id": repository_id,
        "revision": revision,
        "shard_path": shard_path,
        "row_index": int(row_index),
        "source_index": upstream_index,
        "source_index_hash": upstream_index_hash,
        # This group is an upstream-index boundary only. It is not evidence of paired scenes.
        "source_index_group_id": f"upstream-index:{upstream_index_hash}",
        "declared_size": validate_size(record["size"]),
        "category": _optional_text(record["category"]),
        "class_id": _optional_text(record["class_id"]),
        "model": _optional_text(record["model"]),
        "gen_type": _optional_text(record["gen_type"]),
        "reference": reference,
        "label": 0 if reference else 1,
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
    """Stream one Parquet shard while requesting DANI metadata columns only."""
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
        raise FileExistsError(f"Refusing to overwrite existing DANI metadata scan output: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256_file(path: str | Path) -> str:
    """Return a byte-exact SHA-256 digest without parsing the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Write canonical human-readable JSON used in the source evidence lock."""
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
    """Describe whether a scan represents every shard in the immutable source lock."""
    partial = limit_shards is not None
    complete = not partial and len(selected_shards) == len(all_shards)
    return {
        "kind": "partial_scout" if partial else "complete_source_scan",
        "partial": partial,
        "complete": complete,
        # Complete metadata makes descriptive auditing possible, but cannot recover the whole
        # parent-image component needed for an honest internal train/validation/test split.
        "eligible_for_candidate_selection": False,
        "eligible_for_training": False,
        "eligible_for_external_descriptive_evaluation": complete,
        "internal_selection_blocker": INTERNAL_SELECTION_BLOCKER,
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
    """Write a revision-pinned, metadata-only DANI candidate catalogue.

    A limit-shards run is deliberately a scout, even if the numerical limit happens to cover every
    currently visible shard. Such output is never eligible for candidate selection.
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
        "schema_version": "dani_metadata_scan_v1",
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
        "excluded_binary_columns": [IMAGE_COLUMN],
        "image_materialised": False,
        "image_decoded": False,
        "pairing_status": {
            "recoverable_from_catalog": False,
            "documented_parent_group_field": None,
            "source_index_role": "upstream image-level identifier only",
            "internal_selection_blocker": INTERNAL_SELECTION_BLOCKER,
        },
        "rows_scanned_by_shard": scanned_rows_by_shard,
        "scan_scope": scope,
    }
    write_json(output / "provenance.json", report)
    return report
