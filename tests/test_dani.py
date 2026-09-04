from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_image_detector import dani


def sample_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "index": "coco-000001",
        "image": b"must-not-be-read",
        "size": 1024,
        "category": "outdoor",
        "class_id": "17",
        "model": "COCO",
        "gen_type": "reference",
        "reference": True,
    }
    record.update(overrides)
    return record


class ImagePoisonedRecord(dict[str, object]):
    def __getitem__(self, key: str) -> object:
        if key == dani.IMAGE_COLUMN:
            raise AssertionError("metadata projection attempted to read DANI image bytes")
        return super().__getitem__(key)


class FakeHubApi:
    def __init__(
        self, entries: list[object], *, resolved_revision: str = dani.PINNED_REVISION
    ) -> None:
        self.entries = entries
        self.resolved_revision = resolved_revision
        self.info_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def dataset_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
        self.info_calls.append((args, kwargs))
        return SimpleNamespace(sha=self.resolved_revision)

    def list_repo_tree(self, *args: object, **kwargs: object) -> list[object]:
        self.calls.append((args, kwargs))
        return self.entries


def fake_shard(path: str, *, size: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        size=size,
        blob_id=f"blob-{path}",
        lfs=SimpleNamespace(size=size, sha256=f"lfs-{path}", pointer_size=135),
        xet_hash=f"xet-{path}",
    )


def test_schema_excludes_binary_image_column() -> None:
    dani.ensure_metadata_schema()

    assert len(dani.SOURCE_SCHEMA_COLUMNS) == 8
    assert dani.IMAGE_COLUMN in dani.SOURCE_SCHEMA_COLUMNS
    assert dani.IMAGE_COLUMN not in dani.META_COLUMNS
    assert len(dani.META_COLUMNS) == 7


def test_source_lock_lists_sorted_parquet_shards_with_identifiers() -> None:
    api = FakeHubApi(
        [
            fake_shard("data/z.parquet", size=20),
            SimpleNamespace(path="README.md"),
            fake_shard("data/a.parquet", size=10),
        ]
    )

    shards = dani.list_source_shards(api=api)
    lock = dani.build_source_lock(
        repository_id=dani.REPOSITORY_ID,
        revision=dani.PINNED_REVISION,
        shards=shards,
    )

    assert [shard["path"] for shard in shards] == ["data/a.parquet", "data/z.parquet"]
    assert shards[0] == {
        "path": "data/a.parquet",
        "size": 10,
        "blob_id": "blob-data/a.parquet",
        "lfs": {"size": 10, "sha256": "lfs-data/a.parquet", "pointer_size": 135},
        "xet_hash": "xet-data/a.parquet",
    }
    assert lock["excluded_binary_columns"] == ["image"]
    assert api.calls == [
        (
            (dani.REPOSITORY_ID,),
            {"repo_type": "dataset", "revision": dani.PINNED_REVISION, "recursive": True},
        )
    ]


def test_projected_record_uses_reference_label_and_never_reads_image() -> None:
    real = dani.project_metadata_record(
        ImagePoisonedRecord(sample_record()),
        repository_id=dani.REPOSITORY_ID,
        revision=dani.PINNED_REVISION,
        shard_path="data/a.parquet",
        row_index=0,
    )
    generated = dani.project_metadata_record(
        ImagePoisonedRecord(sample_record(index="coco-000002", reference=False, model="SD_XL")),
        repository_id=dani.REPOSITORY_ID,
        revision=dani.PINNED_REVISION,
        shard_path="data/a.parquet",
        row_index=1,
    )

    assert real["label"] == 0
    assert real["reference"] is True
    assert generated["label"] == 1
    assert generated["reference"] is False
    assert real["source_index_group_id"].startswith("upstream-index:")
    assert real["source_index_group_id"] != generated["source_index_group_id"]


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("index", "", ValueError, "index"),
        ("index", True, ValueError, "index"),
        ("size", 0, ValueError, "size"),
        ("size", True, ValueError, "size"),
        ("reference", "False", TypeError, "reference"),
        ("reference", 0, TypeError, "reference"),
    ],
)
def test_invalid_key_metadata_is_rejected(
    field: str, value: object, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        dani.project_metadata_record(
            sample_record(**{field: value}),
            repository_id=dani.REPOSITORY_ID,
            revision=dani.PINNED_REVISION,
            shard_path="data/a.parquet",
            row_index=0,
        )


def test_partial_scan_is_explicitly_ineligible_and_never_reads_image(tmp_path: Path) -> None:
    api = FakeHubApi([fake_shard("data/b.parquet"), fake_shard("data/a.parquet")])
    loader_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_loader(*args: object, **kwargs: object) -> list[ImagePoisonedRecord]:
        loader_calls.append((args, kwargs))
        assert kwargs["columns"] == dani.META_COLUMNS
        assert dani.IMAGE_COLUMN not in kwargs["columns"]
        assert kwargs["streaming"] is True
        return [ImagePoisonedRecord(sample_record())]

    output = tmp_path / "scout"
    report = dani.scan_metadata(
        output,
        revision="main",
        limit_shards=1,
        cache_dir=tmp_path / "cache",
        api=api,
        dataset_loader=fake_loader,
        now=lambda: datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert len(loader_calls) == 1
    assert loader_calls[0][0] == ("parquet",)
    assert f"@{dani.PINNED_REVISION}/" in loader_calls[0][1]["data_files"]
    assert loader_calls[0][1]["data_files"].endswith("/data/a.parquet")
    assert report["scan_scope"] == {
        "kind": "partial_scout",
        "partial": True,
        "complete": False,
        "eligible_for_candidate_selection": False,
        "eligible_for_training": False,
        "eligible_for_external_descriptive_evaluation": False,
        "internal_selection_blocker": dani.INTERNAL_SELECTION_BLOCKER,
        "available_shard_count": 2,
        "selected_shard_count": 1,
        "limit_shards": 1,
        "selected_shards": ["data/a.parquet"],
    }
    assert report["image_materialised"] is False
    assert report["image_decoded"] is False
    assert report["pairing_status"]["recoverable_from_catalog"] is False

    with (output / "source_catalog.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert candidates[0]["locator"].endswith(":data/a.parquet:0")
    assert "image" not in candidates[0]
    source_lock = json.loads((output / "source_lock.json").read_text(encoding="utf-8"))
    assert source_lock["revision"] == dani.PINNED_REVISION
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["scan_scope"]["partial"] is True
    assert provenance["source_catalog_sha256"] == report["source_catalog_sha256"]


def test_limit_shards_stays_partial_even_when_it_covers_every_shard() -> None:
    shards = dani.canonical_shards([fake_shard("data/a.parquet"), fake_shard("data/b.parquet")])

    scope = dani.build_scan_scope(
        all_shards=shards,
        selected_shards=shards,
        limit_shards=2,
    )

    assert scope["partial"] is True
    assert scope["complete"] is False
    assert scope["eligible_for_candidate_selection"] is False
    assert scope["eligible_for_external_descriptive_evaluation"] is False


def test_complete_dani_catalog_is_still_blocked_from_internal_selection() -> None:
    shards = dani.canonical_shards([fake_shard("data/a.parquet"), fake_shard("data/b.parquet")])

    scope = dani.build_scan_scope(
        all_shards=shards,
        selected_shards=shards,
        limit_shards=None,
    )

    assert scope["complete"] is True
    assert scope["eligible_for_candidate_selection"] is False
    assert scope["eligible_for_external_descriptive_evaluation"] is True
    assert scope["internal_selection_blocker"] == dani.INTERNAL_SELECTION_BLOCKER


def test_fresh_output_directory_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "new-output"
    assert dani.require_fresh_output_dir(output) == output

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        dani.require_fresh_output_dir(output)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        dani.scan_metadata(output, api=FakeHubApi([fake_shard("data/a.parquet")]))
