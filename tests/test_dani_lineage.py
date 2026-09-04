from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

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


class FakeHubApi:
    def __init__(self, entries: list[object]) -> None:
        self.entries = entries

    def dataset_info(self, *_: object, **__: object) -> SimpleNamespace:
        return SimpleNamespace(sha=dani.PINNED_REVISION)

    def list_repo_tree(self, *_: object, **__: object) -> list[object]:
        return self.entries


def fake_shard(path: str, *, size: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        size=size,
        blob_id=f"blob-{path}",
        lfs=SimpleNamespace(size=size, sha256=f"lfs-{path}", pointer_size=135),
        xet_hash=f"xet-{path}",
    )


def test_lineage_source_lock_records_path_only_projection_contract() -> None:
    lock = dani_lineage.build_lineage_source_lock(
        repository_id=dani.REPOSITORY_ID,
        revision=dani.PINNED_REVISION,
        shards=dani.canonical_shards([fake_shard("data/a.parquet")]),
    )

    assert lock["schema_version"] == "dani_lineage_source_lock_v1"
    assert lock["projection_contract"] == {
        "arrow_columns": list(dani_lineage.MAPPING_PROJECTION_COLUMNS),
        "parent_image_column_requested": False,
        "permitted_image_child_columns": ["image.path"],
        "excluded_binary_image_child_columns": ["image.bytes"],
        "required_physical_leaf_paths": ["image.path", "image.bytes"],
        "pyarrow_use_threads": False,
        "fsspec_cache_type": "none",
        "fsspec_block_size": 65_536,
    }


def test_partial_lineage_scan_writes_only_blocked_path_catalogue(tmp_path: Path) -> None:
    api = FakeHubApi([fake_shard("data/b.parquet"), fake_shard("data/a.parquet")])
    loader_calls: list[dict[str, object]] = []

    def fake_loader(**kwargs: object) -> list[dict[str, object]]:
        loader_calls.append(kwargs)
        assert kwargs["shard_path"] == "data/a.parquet"
        assert kwargs["batch_size"] == 3
        assert kwargs["block_size"] == 4096
        return [sample_projected_record()]

    output = tmp_path / "lineage-scout"
    report = dani_lineage.scan_lineage_metadata(
        output,
        revision="main",
        limit_shards=1,
        batch_size=3,
        block_size=4096,
        api=api,
        shard_loader=fake_loader,
        now=lambda: datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert len(loader_calls) == 1
    assert report["scan_scope"] == {
        "kind": "partial_lineage_scout",
        "partial": True,
        "complete": False,
        "eligible_for_lineage_audit": False,
        "eligible_for_candidate_selection": False,
        "eligible_for_training": False,
        "eligible_for_external_descriptive_evaluation": False,
        "selection_blocker": dani_lineage.LINEAGE_SELECTION_BLOCKER,
        "available_shard_count": 2,
        "selected_shard_count": 1,
        "limit_shards": 1,
        "selected_shards": ["data/a.parquet"],
    }
    assert report["projection_contract"]["fsspec_block_size"] == 4096
    assert report["projection_observation"] == {
        "image_path_requested": True,
        "image_path_materialised": True,
        "image_bytes_requested": False,
        "image_bytes_materialised": False,
        "image_bytes_decoded": False,
        "batch_image_struct_children_required": ["path"],
        "http_range_image_byte_disjointness_verified": False,
    }
    assert report["lineage_status"]["candidate_parent_group_key"] == "parent_coco_image_id"
    assert report["lineage_status"]["upstream_mapping_join_verified"] is False

    with (output / "lineage_catalog.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "locator": "Renyang/DANI@870e29fcdc13c405fae35442899e9ba1da11691d:data/a.parquet:0",
            "repository_id": "Renyang/DANI",
            "revision": dani.PINNED_REVISION,
            "shard_path": "data/a.parquet",
            "row_index": "0",
            "source_index": "7",
            "source_index_hash": dani.source_index_hash("7"),
            "image_path_basename": "235057_22583.jpg",
            "parent_coco_image_id": "235057",
            "coco_caption_id": "22583",
            "declared_size": "1024",
            "category": "outdoor",
            "class_id": "17",
            "model": "SD_XL",
            "gen_type": "T2I",
            "reference": "False",
            "label": "1",
        }
    ]
    assert "bytes" not in rows[0]
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["lineage_catalog_sha256"] == report["lineage_catalog_sha256"]


def test_complete_lineage_scan_is_audit_eligible_but_still_not_trainable(tmp_path: Path) -> None:
    api = FakeHubApi([fake_shard("data/a.parquet"), fake_shard("data/b.parquet")])

    def fake_loader(**kwargs: object) -> list[dict[str, object]]:
        return [sample_projected_record(index=kwargs["shard_path"])]

    report = dani_lineage.scan_lineage_metadata(
        tmp_path / "full-lineage",
        api=api,
        shard_loader=fake_loader,
        now=lambda: datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert report["catalog_row_count"] == 2
    assert report["scan_scope"]["complete"] is True
    assert report["scan_scope"]["eligible_for_lineage_audit"] is True
    assert report["scan_scope"]["eligible_for_candidate_selection"] is False
    assert report["scan_scope"]["eligible_for_training"] is False
    source_lock = json.loads((tmp_path / "full-lineage" / "source_lock.json").read_text())
    assert [shard["path"] for shard in source_lock["shards"]] == [
        "data/a.parquet",
        "data/b.parquet",
    ]


def test_limit_covering_all_current_shards_remains_a_partial_lineage_scout() -> None:
    shards = dani.canonical_shards([fake_shard("data/a.parquet"), fake_shard("data/b.parquet")])

    scope = dani_lineage.build_lineage_scan_scope(
        all_shards=shards,
        selected_shards=shards,
        limit_shards=2,
    )

    assert scope["partial"] is True
    assert scope["complete"] is False
    assert scope["eligible_for_lineage_audit"] is False


def test_invalid_path_fails_before_writing_final_provenance(tmp_path: Path) -> None:
    api = FakeHubApi([fake_shard("data/a.parquet")])

    def fake_loader(**_: object) -> list[dict[str, object]]:
        return [sample_projected_record(image={"path": "not-a-dani-lineage.png"})]

    output = tmp_path / "invalid-lineage"
    with pytest.raises(ValueError, match="image.path"):
        dani_lineage.scan_lineage_metadata(output, api=api, shard_loader=fake_loader)

    assert (output / "source_lock.json").is_file()
    assert not (output / "provenance.json").exists()


def test_lineage_scan_refuses_existing_output_without_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "lineage-output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        dani_lineage.scan_lineage_metadata(output, api=FakeHubApi([fake_shard("data/a.parquet")]))

    assert marker.read_text(encoding="utf-8") == "do not overwrite"
