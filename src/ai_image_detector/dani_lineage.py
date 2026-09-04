"""Safe metadata projection for reconstructing candidate DANI COCO lineage.

The DANI Parquet image field is a struct with two physical leaves: image.bytes and image.path.
This module permits the path leaf only. It never requests the parent image field or image.bytes,
and rejects a projected Arrow batch if the binary child is present.

A parsed basename is evidence of a candidate COCO image/caption lineage, not image identity, byte
format, a duplicate key, or an approved train/validation/test split key. The later mapping audit
must join these identifiers against version-pinned upstream metadata before any selection decision.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import dani

IMAGE_PATH_COLUMN = "image.path"
LINEAGE_FSSPEC_CACHE_TYPE = "none"
LINEAGE_FSSPEC_BLOCK_SIZE = 65_536
LINEAGE_ARROW_USE_THREADS = False
MAPPING_PROJECTION_COLUMNS = (
    "index",
    IMAGE_PATH_COLUMN,
    "size",
    "category",
    "class_id",
    "model",
    "gen_type",
    "reference",
)
LINEAGE_PATH_PATTERN = re.compile(
    r"^(?P<coco_image_id>[1-9][0-9]*)_(?P<coco_caption_id>[1-9][0-9]*)\.jpg$"
)
LINEAGE_CANDIDATE_COLUMNS = (
    "locator",
    "repository_id",
    "revision",
    "shard_path",
    "row_index",
    "source_index",
    "source_index_hash",
    "image_path_basename",
    "parent_coco_image_id",
    "coco_caption_id",
    "declared_size",
    "category",
    "class_id",
    "model",
    "gen_type",
    "reference",
    "label",
)
LINEAGE_SELECTION_BLOCKER = (
    "Path syntax yields only reconstructed candidate lineage; an immutable upstream mapping join, "
    "a COCO metadata join, and later byte/pixel audits are required before any split or training."
)
LINEAGE_SOURCE_LOCK_NAME = "source_lock.json"
LINEAGE_CATALOG_NAME = "lineage_catalog.csv"
LINEAGE_PROVENANCE_NAME = "provenance.json"


def ensure_mapping_projection() -> None:
    """Fail closed if the lineage phase would select the parent image or binary image leaf."""
    if dani.IMAGE_COLUMN in MAPPING_PROJECTION_COLUMNS:
        raise RuntimeError("DANI lineage projection must not request the parent image field")
    if "image.bytes" in MAPPING_PROJECTION_COLUMNS:
        raise RuntimeError("DANI lineage projection must never request image.bytes")
    if IMAGE_PATH_COLUMN not in MAPPING_PROJECTION_COLUMNS:
        raise RuntimeError("DANI lineage projection must request image.path explicitly")
    if len(MAPPING_PROJECTION_COLUMNS) != len(set(MAPPING_PROJECTION_COLUMNS)):
        raise RuntimeError("DANI lineage projection contains duplicate columns")


def lineage_url(*, repository_id: str, revision: str, shard_path: str) -> str:
    """Build an immutable direct Parquet URL for range-backed metadata projection."""
    if not repository_id:
        raise ValueError("repository_id must be nonempty")
    if not shard_path or shard_path.startswith("/") or not shard_path.endswith(".parquet"):
        raise ValueError(
            f"shard_path must be a repository-relative Parquet path, got {shard_path!r}"
        )
    immutable_revision = dani.require_immutable_revision(revision)
    return (
        f"https://huggingface.co/datasets/{repository_id}/resolve/{immutable_revision}/{shard_path}"
    )


@contextmanager
def open_lineage_parquet(
    *,
    repository_id: str,
    revision: str,
    shard_path: str,
    block_size: int = LINEAGE_FSSPEC_BLOCK_SIZE,
) -> Iterator[Any]:
    """Open one pinned Parquet shard with HTTP range caching disabled.

    The caller reads only the exact nested projection through iter_projected_metadata_rows. Disabling
    fsspec cache/read-ahead makes an optional HTTP-range audit meaningful: it prevents a local cache
    from expanding a requested metadata interval into an image-byte interval.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    ensure_mapping_projection()
    import fsspec
    import pyarrow.parquet as pq

    url = lineage_url(
        repository_id=repository_id,
        revision=revision,
        shard_path=shard_path,
    )
    with fsspec.open(
        url,
        "rb",
        cache_type=LINEAGE_FSSPEC_CACHE_TYPE,
        block_size=block_size,
    ).open() as handle:
        yield pq.ParquetFile(handle)


def lineage_projection_contract(
    *, block_size: int = LINEAGE_FSSPEC_BLOCK_SIZE
) -> dict[str, object]:
    """Describe the exact nested read contract without claiming HTTP-range proof not yet run."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return {
        "arrow_columns": list(MAPPING_PROJECTION_COLUMNS),
        "parent_image_column_requested": False,
        "permitted_image_child_columns": [IMAGE_PATH_COLUMN],
        "excluded_binary_image_child_columns": ["image.bytes"],
        "required_physical_leaf_paths": [IMAGE_PATH_COLUMN, "image.bytes"],
        "pyarrow_use_threads": LINEAGE_ARROW_USE_THREADS,
        "fsspec_cache_type": LINEAGE_FSSPEC_CACHE_TYPE,
        "fsspec_block_size": block_size,
    }


def parquet_leaf_paths(parquet_file: Any) -> tuple[str, ...]:
    """Return the physical leaf paths listed in the Parquet footer."""
    schema = parquet_file.metadata.schema
    return tuple(schema.column(index).path for index in range(len(schema)))


def ensure_safe_lineage_leaves(parquet_file: Any) -> None:
    """Require separate image.path and image.bytes leaves before a nested projection."""
    leaves = set(parquet_leaf_paths(parquet_file))
    if IMAGE_PATH_COLUMN not in leaves:
        raise ValueError("DANI Parquet footer does not expose image.path as a physical leaf")
    if "image.bytes" not in leaves:
        raise ValueError("DANI Parquet footer does not expose image.bytes as a physical leaf")


def ensure_path_only_batch(batch: Any) -> None:
    """Reject an Arrow batch unless its image struct contains exactly the non-binary path child."""
    import pyarrow as pa

    try:
        image_field = batch.schema.field(dani.IMAGE_COLUMN)
    except KeyError as error:
        raise ValueError("Projected DANI batch has no image path struct") from error
    if not pa.types.is_struct(image_field.type):
        raise ValueError("Projected DANI image field must be an Arrow struct")
    child_names = tuple(
        image_field.type[index].name for index in range(image_field.type.num_fields)
    )
    if child_names != ("path",):
        raise ValueError(
            f"Projected DANI image struct must contain only path, but has children {child_names!r}"
        )


def iter_projected_metadata_rows(
    parquet_file: Any,
    *,
    batch_size: int = 1024,
) -> Iterator[Mapping[str, object]]:
    """Yield DANI metadata rows after verifying a nested image.path-only Arrow projection."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ensure_mapping_projection()
    ensure_safe_lineage_leaves(parquet_file)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=list(MAPPING_PROJECTION_COLUMNS),
        use_threads=LINEAGE_ARROW_USE_THREADS,
    ):
        ensure_path_only_batch(batch)
        for record in batch.to_pylist():
            if not isinstance(record, Mapping):
                raise TypeError("Projected DANI batch contains a non-mapping record")
            yield record


def parse_lineage_basename(value: object) -> tuple[int, int, str]:
    """Parse one DANI image.path basename into candidate COCO image and caption identifiers."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"image.path must be a nonempty basename, got {value!r}")
    if value != value.strip() or "/" in value or "\\" in value:
        raise ValueError(f"image.path must be a plain basename, got {value!r}")
    match = LINEAGE_PATH_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            "image.path does not match the documented candidate image_id_caption_id.jpg syntax"
        )
    return int(match["coco_image_id"]), int(match["coco_caption_id"]), value


def _path_from_projected_image(value: object) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("Projected image metadata must be a mapping containing only path")
    keys = set(value)
    if keys != {"path"}:
        raise ValueError(f"Projected image metadata must contain only path, got {sorted(keys)!r}")
    return parse_lineage_basename(value["path"])[2]


def project_lineage_record(
    record: Mapping[str, object],
    *,
    repository_id: str,
    revision: str,
    shard_path: str,
    row_index: int,
) -> dict[str, object]:
    """Project a path-only DANI record into a blocked candidate-lineage catalogue row."""
    required = {
        "index",
        dani.IMAGE_COLUMN,
        "size",
        "category",
        "class_id",
        "model",
        "gen_type",
        "reference",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Projected DANI metadata row is missing required columns: {missing}")
    locator = dani.stable_locator(repository_id, revision, shard_path, row_index)
    image_path = _path_from_projected_image(record[dani.IMAGE_COLUMN])
    parent_coco_image_id, coco_caption_id, _ = parse_lineage_basename(image_path)
    source_index = dani.normalise_source_index(record["index"])
    reference = dani.validate_reference(record["reference"])
    return {
        "locator": locator,
        "repository_id": repository_id,
        "revision": revision,
        "shard_path": shard_path,
        "row_index": int(row_index),
        "source_index": source_index,
        "source_index_hash": dani.source_index_hash(source_index),
        "image_path_basename": image_path,
        "parent_coco_image_id": parent_coco_image_id,
        "coco_caption_id": coco_caption_id,
        "declared_size": dani.validate_size(record["size"]),
        "category": dani._optional_text(record["category"]),
        "class_id": dani._optional_text(record["class_id"]),
        "model": dani._optional_text(record["model"]),
        "gen_type": dani._optional_text(record["gen_type"]),
        "reference": reference,
        "label": 0 if reference else 1,
    }


def build_lineage_source_lock(
    *,
    repository_id: str,
    revision: str,
    shards: Sequence[Mapping[str, object]],
    block_size: int = LINEAGE_FSSPEC_BLOCK_SIZE,
) -> dict[str, object]:
    """Lock the full source revision plus the exact non-binary lineage projection contract."""
    base_lock = dani.build_source_lock(
        repository_id=repository_id,
        revision=revision,
        shards=shards,
    )
    return {
        "schema_version": "dani_lineage_source_lock_v1",
        "repository_id": base_lock["repository_id"],
        "revision": base_lock["revision"],
        "license": base_lock["license"],
        "repo_type": base_lock["repo_type"],
        "tree_recursive": base_lock["tree_recursive"],
        "source_schema_columns": list(dani.SOURCE_SCHEMA_COLUMNS),
        "projection_contract": lineage_projection_contract(block_size=block_size),
        "shard_count": base_lock["shard_count"],
        "shards": base_lock["shards"],
    }


def select_lineage_shards(
    shards: Sequence[Mapping[str, object]], limit_shards: int | None
) -> list[Mapping[str, object]]:
    """Choose a bounded scout without allowing a numerical limit to masquerade as full coverage."""
    if limit_shards is not None and limit_shards <= 0:
        raise ValueError("limit_shards must be a positive integer when supplied")
    return list(shards if limit_shards is None else shards[:limit_shards])


def build_lineage_scan_scope(
    *,
    all_shards: Sequence[Mapping[str, object]],
    selected_shards: Sequence[Mapping[str, object]],
    limit_shards: int | None,
) -> dict[str, object]:
    """Describe a blocked lineage candidate scan without calling it a selected corpus."""
    partial = limit_shards is not None
    complete = not partial and len(selected_shards) == len(all_shards)
    return {
        "kind": "partial_lineage_scout" if partial else "complete_lineage_candidate_scan",
        "partial": partial,
        "complete": complete,
        "eligible_for_lineage_audit": complete,
        "eligible_for_candidate_selection": False,
        "eligible_for_training": False,
        "eligible_for_external_descriptive_evaluation": False,
        "selection_blocker": LINEAGE_SELECTION_BLOCKER,
        "available_shard_count": len(all_shards),
        "selected_shard_count": len(selected_shards),
        "limit_shards": limit_shards,
        "selected_shards": [str(shard["path"]) for shard in selected_shards],
    }


def load_lineage_metadata_shard(
    *,
    repository_id: str,
    revision: str,
    shard_path: str,
    batch_size: int = 1024,
    block_size: int = LINEAGE_FSSPEC_BLOCK_SIZE,
) -> Iterable[Mapping[str, object]]:
    """Stream one source shard through the Arrow path-only projection contract."""
    with open_lineage_parquet(
        repository_id=repository_id,
        revision=revision,
        shard_path=shard_path,
        block_size=block_size,
    ) as parquet_file:
        yield from iter_projected_metadata_rows(parquet_file, batch_size=batch_size)


def scan_lineage_metadata(
    output_dir: str | Path,
    *,
    repository_id: str = dani.REPOSITORY_ID,
    revision: str = dani.PINNED_REVISION,
    limit_shards: int | None = None,
    batch_size: int = 1024,
    block_size: int = LINEAGE_FSSPEC_BLOCK_SIZE,
    api: Any | None = None,
    shard_loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Create a revision-pinned candidate-lineage catalogue without materialising images.

    The output is intentionally a *blocked* reconstruction aid. Parsing image.path establishes a
    candidate COCO parent/caption key only; it is not proof of file identity or a split manifest.
    """
    if batch_size <= 0 or block_size <= 0:
        raise ValueError("batch_size and block_size must be positive")
    ensure_mapping_projection()
    output = dani.require_fresh_output_dir(output_dir)
    hub_api = dani.get_hf_api() if api is None else api
    resolved_revision = dani.resolve_revision(
        repository_id=repository_id,
        requested_revision=revision,
        api=hub_api,
    )
    all_shards = dani.list_source_shards(
        repository_id=repository_id,
        revision=resolved_revision,
        api=hub_api,
    )
    selected_shards = select_lineage_shards(all_shards, limit_shards)
    source_lock_path = output / LINEAGE_SOURCE_LOCK_NAME
    dani.write_json(
        source_lock_path,
        build_lineage_source_lock(
            repository_id=repository_id,
            revision=resolved_revision,
            shards=all_shards,
            block_size=block_size,
        ),
    )

    loader = load_lineage_metadata_shard if shard_loader is None else shard_loader
    catalog_path = output / LINEAGE_CATALOG_NAME
    rows_scanned_by_shard: dict[str, int] = {}
    candidate_count = 0
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LINEAGE_CANDIDATE_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for shard in selected_shards:
            shard_path = str(shard["path"])
            row_count = 0
            records = loader(
                repository_id=repository_id,
                revision=resolved_revision,
                shard_path=shard_path,
                batch_size=batch_size,
                block_size=block_size,
            )
            for row_index, record in enumerate(records):
                writer.writerow(
                    project_lineage_record(
                        record,
                        repository_id=repository_id,
                        revision=resolved_revision,
                        shard_path=shard_path,
                        row_index=row_index,
                    )
                )
                row_count += 1
                candidate_count += 1
            rows_scanned_by_shard[shard_path] = row_count

    scope = build_lineage_scan_scope(
        all_shards=all_shards,
        selected_shards=selected_shards,
        limit_shards=limit_shards,
    )
    timestamp = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    report: dict[str, object] = {
        "schema_version": "dani_lineage_metadata_scan_v1",
        "repository_id": repository_id,
        "requested_revision": revision,
        "revision": resolved_revision,
        "resolved_revision": resolved_revision,
        "created_at_utc": timestamp,
        "source_lock": source_lock_path.name,
        "source_lock_sha256": dani.sha256_file(source_lock_path),
        "lineage_catalog": catalog_path.name,
        "lineage_catalog_sha256": dani.sha256_file(catalog_path),
        "catalog_kind": "path_derived_lineage_candidate_catalog_not_trainable_manifest",
        "catalog_row_count": candidate_count,
        "projection_contract": lineage_projection_contract(block_size=block_size),
        "projection_observation": {
            "image_path_requested": True,
            "image_path_materialised": True,
            "image_bytes_requested": False,
            "image_bytes_materialised": False,
            "image_bytes_decoded": False,
            "batch_image_struct_children_required": ["path"],
            "http_range_image_byte_disjointness_verified": False,
        },
        "lineage_status": {
            "path_syntax_parsed": True,
            "candidate_parent_group_key": "parent_coco_image_id",
            "candidate_caption_key": "coco_caption_id",
            "upstream_mapping_join_performed": False,
            "upstream_mapping_join_verified": False,
            "selection_blocker": LINEAGE_SELECTION_BLOCKER,
        },
        "rows_scanned_by_shard": rows_scanned_by_shard,
        "scan_scope": scope,
    }
    dani.write_json(output / LINEAGE_PROVENANCE_NAME, report)
    return report
