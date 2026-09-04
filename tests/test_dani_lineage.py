from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ai_image_detector import dani, dani_lineage


def write_parquet_with_image_struct(
    tmp_path: Path,
    *,
    paths: list[str] | None = None,
    include_path: bool = True,
) -> Path:
    image_paths = paths or ["235057_22583.jpg", "325483_135530.jpg"]
    children = [pa.array([b"first-image-bytes", b"second-image-bytes"])]
    names = ["bytes"]
    if include_path:
        children.append(pa.array(image_paths))
        names.append("path")
    images = pa.StructArray.from_arrays(children, names=names)
    table = pa.table(
        {
            "index": pa.array([7, 8]),
            "image": images,
            "size": pa.array([1024, 1024]),
            "category": pa.array(["outdoor", "outdoor"]),
            "class_id": pa.array(["17", "17"]),
            "model": pa.array(["SD_XL", "COCO"]),
            "gen_type": pa.array(["T2I", "reference"]),
            "reference": pa.array([False, True]),
        }
    )
    path = tmp_path / "sample.parquet"
    pq.write_table(table, path)
    return path


class SpyParquet:
    def __init__(self, parquet_file: pq.ParquetFile) -> None:
        self.parquet_file = parquet_file
        self.metadata = parquet_file.metadata
        self.calls: list[dict[str, object]] = []

    def iter_batches(self, **kwargs: object):
        self.calls.append(dict(kwargs))
        return self.parquet_file.iter_batches(**kwargs)


def sample_projected_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "index": 7,
        "image": {"path": "235057_22583.jpg"},
        "size": 1024,
        "category": "outdoor",
        "class_id": "17",
        "model": "SD_XL",
        "gen_type": "T2I",
        "reference": False,
    }
    record.update(overrides)
    return record


def test_mapping_projection_forbids_parent_image_and_binary_leaf() -> None:
    dani_lineage.ensure_mapping_projection()

    assert dani.IMAGE_COLUMN not in dani_lineage.MAPPING_PROJECTION_COLUMNS
    assert "image.bytes" not in dani_lineage.MAPPING_PROJECTION_COLUMNS
    assert "image.path" in dani_lineage.MAPPING_PROJECTION_COLUMNS


def test_nested_projection_reads_path_child_without_bytes(tmp_path: Path) -> None:
    path = write_parquet_with_image_struct(tmp_path)
    with pq.ParquetFile(path) as parquet_file:
        spy = SpyParquet(parquet_file)
        rows = list(dani_lineage.iter_projected_metadata_rows(spy, batch_size=1))

    assert spy.calls == [
        {
            "batch_size": 1,
            "columns": list(dani_lineage.MAPPING_PROJECTION_COLUMNS),
            "use_threads": False,
        }
    ]
    assert rows == [
        {
            "index": 7,
            "image": {"path": "235057_22583.jpg"},
            "size": 1024,
            "category": "outdoor",
            "class_id": "17",
            "model": "SD_XL",
            "gen_type": "T2I",
            "reference": False,
        },
        {
            "index": 8,
            "image": {"path": "325483_135530.jpg"},
            "size": 1024,
            "category": "outdoor",
            "class_id": "17",
            "model": "COCO",
            "gen_type": "reference",
            "reference": True,
        },
    ]
    assert all(set(row["image"]) == {"path"} for row in rows)


def test_projection_refuses_parquet_without_path_leaf(tmp_path: Path) -> None:
    path = write_parquet_with_image_struct(tmp_path, include_path=False)

    with pq.ParquetFile(path) as parquet_file, pytest.raises(ValueError, match="image.path"):
        list(dani_lineage.iter_projected_metadata_rows(parquet_file))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("235057_22583.jpg", (235057, 22583, "235057_22583.jpg")),
        ("1_1.jpg", (1, 1, "1_1.jpg")),
    ],
)
def test_parse_lineage_basename(value: str, expected: tuple[int, int, str]) -> None:
    assert dani_lineage.parse_lineage_basename(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0_1.jpg",
        "1_0.jpg",
        "1_2.png",
        "folder/1_2.jpg",
        "1_2.jpg ",
        "captioned.jpg",
    ],
)
def test_parse_lineage_basename_rejects_ambiguous_syntax(value: str) -> None:
    with pytest.raises(ValueError, match="image.path"):
        dani_lineage.parse_lineage_basename(value)


def test_project_lineage_record_keeps_row_identity_and_blocks_binary_image_struct() -> None:
    projected = dani_lineage.project_lineage_record(
        sample_projected_record(),
        repository_id=dani.REPOSITORY_ID,
        revision=dani.PINNED_REVISION,
        shard_path="data/train-00000-of-00012.parquet",
        row_index=5,
    )

    assert projected["locator"].endswith(":data/train-00000-of-00012.parquet:5")
    assert projected["parent_coco_image_id"] == 235057
    assert projected["coco_caption_id"] == 22583
    assert projected["label"] == 1
    assert projected["source_index_hash"] == dani.source_index_hash("7")

    with pytest.raises(ValueError, match="only path"):
        dani_lineage.project_lineage_record(
            sample_projected_record(image={"path": "235057_22583.jpg", "bytes": b"forbidden"}),
            repository_id=dani.REPOSITORY_ID,
            revision=dani.PINNED_REVISION,
            shard_path="data/train-00000-of-00012.parquet",
            row_index=5,
        )


def test_lineage_url_requires_immutable_revision() -> None:
    assert (
        dani_lineage.lineage_url(
            repository_id=dani.REPOSITORY_ID,
            revision=dani.PINNED_REVISION,
            shard_path="data/train-00000-of-00012.parquet",
        )
        == "https://huggingface.co/datasets/Renyang/DANI/resolve/"
        "870e29fcdc13c405fae35442899e9ba1da11691d/data/train-00000-of-00012.parquet"
    )
    with pytest.raises(ValueError, match="commit SHA"):
        dani_lineage.lineage_url(
            repository_id=dani.REPOSITORY_ID,
            revision="main",
            shard_path="data/train-00000-of-00012.parquet",
        )
