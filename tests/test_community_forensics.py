from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_image_detector import community_forensics as community


def sample_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "image_name": "sample.png",
        "format": "PNG",
        "resolution": [512, 512],
        "mode": "RGB",
        "image_data": b"must-not-be-read",
        "model_name": "model-a",
        "nsfw_flag": False,
        "prompt": "A bright red apple",
        "real_source": "COCO",
        "subset": "train",
        "split": "train",
        "label": 1,
        "architecture": "diffusion",
    }
    record.update(overrides)
    return record


class ImagePoisonedRecord(dict[str, object]):
    def __getitem__(self, key: str) -> object:
        if key == community.IMAGE_DATA_COLUMN:
            raise AssertionError("metadata projection attempted to read image_data")
        return super().__getitem__(key)


class FakeHubApi:
    def __init__(
        self, entries: list[object], *, resolved_revision: str = community.PINNED_REVISION
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


def test_schema_excludes_binary_image_data_column() -> None:
    community.ensure_metadata_schema()

    assert len(community.SOURCE_SCHEMA_COLUMNS) == 13
    assert community.IMAGE_DATA_COLUMN in community.SOURCE_SCHEMA_COLUMNS
    assert community.IMAGE_DATA_COLUMN not in community.META_COLUMNS
    assert len(community.META_COLUMNS) == 12


def test_source_lock_lists_sorted_parquet_shards_with_identifiers() -> None:
    api = FakeHubApi(
        [
            fake_shard("data/z.parquet", size=20),
            SimpleNamespace(path="README.md"),
            fake_shard("data/a.parquet", size=10),
        ]
    )

    shards = community.list_source_shards(api=api)
    lock = community.build_source_lock(
        repository_id=community.REPOSITORY_ID,
        revision=community.PINNED_REVISION,
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
    assert [shard["path"] for shard in lock["shards"]] == ["data/a.parquet", "data/z.parquet"]
    assert lock["excluded_binary_columns"] == ["image_data"]
    assert api.calls == [
        (
            (community.REPOSITORY_ID,),
            {"repo_type": "dataset", "revision": community.PINNED_REVISION, "recursive": True},
        )
    ]


def test_locator_and_prompt_hash_are_stable_without_collapsing_blank_prompts() -> None:
    first = community.project_metadata_record(
        sample_record(prompt="   "),
        repository_id=community.REPOSITORY_ID,
        revision=community.PINNED_REVISION,
        shard_path="data/a.parquet",
        row_index=0,
    )
    second = community.project_metadata_record(
        sample_record(prompt=None),
        repository_id=community.REPOSITORY_ID,
        revision=community.PINNED_REVISION,
        shard_path="data/a.parquet",
        row_index=1,
    )
    normalized_one = community.project_metadata_record(
        sample_record(prompt="  A\nbright red APPLE  "),
        repository_id=community.REPOSITORY_ID,
        revision=community.PINNED_REVISION,
        shard_path="data/a.parquet",
        row_index=2,
    )
    normalized_two = community.project_metadata_record(
        sample_record(prompt="a bright red apple"),
        repository_id=community.REPOSITORY_ID,
        revision=community.PINNED_REVISION,
        shard_path="data/a.parquet",
        row_index=3,
    )

    expected = (
        "OwensLab/CommunityForensics-Small@6c539a534c07917307c381f5af4053c6091b5278:"
        "data/a.parquet:0"
    )
    assert first["locator"] == expected
    assert first["prompt_present"] is False
    assert second["prompt_present"] is False
    assert first["content_group_id"] != second["content_group_id"]
    assert first["prompt_hash"] != second["prompt_hash"]
    assert normalized_one["prompt_present"] is True
    assert normalized_one["prompt_hash"] == normalized_two["prompt_hash"]
    assert normalized_one["content_group_id"] == normalized_two["content_group_id"]


def test_mutable_requested_revision_is_resolved_before_source_locking() -> None:
    api = FakeHubApi([fake_shard("data/a.parquet")])

    resolved = community.resolve_revision(
        repository_id=community.REPOSITORY_ID,
        requested_revision="main",
        api=api,
    )

    assert resolved == community.PINNED_REVISION
    assert api.info_calls == [
        ((community.REPOSITORY_ID,), {"revision": "main"}),
    ]
    with pytest.raises(ValueError, match="commit SHA"):
        community.list_source_shards(revision="main", api=api)
    with pytest.raises(ValueError, match="commit SHA"):
        community.stable_locator(community.REPOSITORY_ID, "main", "data/a.parquet", 0)
    with pytest.raises(ValueError, match="commit SHA"):
        community.shard_url(
            repository_id=community.REPOSITORY_ID,
            revision="main",
            shard_path="data/a.parquet",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("resolution", [512], "resolution"),
        ("resolution", [0, 512], "resolution"),
        ("resolution", [512, True], "resolution"),
        ("label", 2, "label"),
        ("label", True, "label"),
    ],
)
def test_invalid_resolution_or_label_rows_are_rejected(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        community.project_metadata_record(
            sample_record(**{field: value}),
            repository_id=community.REPOSITORY_ID,
            revision=community.PINNED_REVISION,
            shard_path="data/a.parquet",
            row_index=0,
        )


def test_partial_scan_is_explicitly_ineligible_and_never_reads_image_data(tmp_path: Path) -> None:
    api = FakeHubApi([fake_shard("data/b.parquet"), fake_shard("data/a.parquet")])
    loader_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_loader(*args: object, **kwargs: object) -> list[ImagePoisonedRecord]:
        loader_calls.append((args, kwargs))
        assert kwargs["columns"] == community.META_COLUMNS
        assert community.IMAGE_DATA_COLUMN not in kwargs["columns"]
        assert kwargs["streaming"] is True
        return [ImagePoisonedRecord(sample_record())]

    output = tmp_path / "scout"
    report = community.scan_metadata(
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
    assert f"@{community.PINNED_REVISION}/" in loader_calls[0][1]["data_files"]
    assert loader_calls[0][1]["data_files"].endswith("/data/a.parquet")
    assert loader_calls[0][1]["cache_dir"] == str(tmp_path / "cache")
    assert report["scan_scope"] == {
        "kind": "partial_scout",
        "partial": True,
        "complete": False,
        "eligible_for_candidate_selection": False,
        "eligible_for_training": False,
        "available_shard_count": 2,
        "selected_shard_count": 1,
        "limit_shards": 1,
        "selected_shards": ["data/a.parquet"],
    }
    assert report["requested_revision"] == "main"
    assert report["resolved_revision"] == community.PINNED_REVISION
    assert api.info_calls == [((community.REPOSITORY_ID,), {"revision": "main"})]
    assert api.calls == [
        (
            (community.REPOSITORY_ID,),
            {"repo_type": "dataset", "revision": community.PINNED_REVISION, "recursive": True},
        )
    ]
    assert report["image_data_materialised"] is False
    assert report["image_data_decoded"] is False
    assert len(report["source_catalog_sha256"]) == 64

    with (output / "source_catalog.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert candidates[0]["locator"].endswith(":data/a.parquet:0")
    assert "@main:" not in candidates[0]["locator"]
    assert "image_data" not in candidates[0]
    assert "prompt" not in candidates[0]
    source_lock = json.loads((output / "source_lock.json").read_text(encoding="utf-8"))
    assert source_lock["revision"] == community.PINNED_REVISION
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["scan_scope"]["partial"] is True
    assert provenance["source_catalog_sha256"] == report["source_catalog_sha256"]


def test_limit_shards_stays_partial_even_when_it_covers_every_shard() -> None:
    shards = community.canonical_shards(
        [fake_shard("data/a.parquet"), fake_shard("data/b.parquet")]
    )

    scope = community.build_scan_scope(
        all_shards=shards,
        selected_shards=shards,
        limit_shards=2,
    )

    assert scope["partial"] is True
    assert scope["complete"] is False
    assert scope["eligible_for_candidate_selection"] is False


def test_fresh_output_directory_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "new-output"
    assert community.require_fresh_output_dir(output) == output

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        community.require_fresh_output_dir(output)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        community.scan_metadata(output, api=FakeHubApi([fake_shard("data/a.parquet")]))
