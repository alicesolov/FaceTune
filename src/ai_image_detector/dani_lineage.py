"""Safe metadata projection for reconstructing candidate DANI COCO lineage.

The DANI Parquet image field is a struct with two physical leaves: image.bytes and image.path.
This module permits the path leaf only. It never requests the parent image field or image.bytes,
and rejects a projected Arrow batch if the binary child is present.

A parsed basename is evidence of a candidate COCO image/caption lineage, not image identity, byte
format, a duplicate key, or an approved train/validation/test split key. The later mapping audit
must join these identifiers against version-pinned upstream metadata before any selection decision.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from . import dani

IMAGE_PATH_COLUMN = "image.path"
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
    block_size: int = 65_536,
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
    with fsspec.open(url, "rb", cache_type="none", block_size=block_size).open() as handle:
        yield pq.ParquetFile(handle)


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
        use_threads=False,
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
