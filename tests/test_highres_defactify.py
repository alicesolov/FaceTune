from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import imagehash
import pandas as pd
import pytest
from PIL import Image

from ai_image_detector import highres_defactify as highres


def _write_rgb_png(path: Path, *, seed: int, size: tuple[int, int] = (384, 384)) -> None:
    generator = random.Random(seed)
    pixels = generator.randbytes(size[0] * size[1] * 3)
    image = Image.frombytes("RGB", size, pixels)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _source_row(
    path: Path,
    *,
    group_id: str,
    source_id: str,
    label: int,
    generator: str,
    official_split: str,
) -> dict[str, object]:
    with Image.open(path) as image:
        image.load()
        digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        phash = str(imagehash.phash(image.convert("RGB")))
        width, height = image.size
        image_format = image.format
        mode = image.mode
    return {
        "path": str(path),
        "label": label,
        "split": official_split,
        "generator": generator,
        "group_id": group_id,
        "source_id": source_id,
        "caption": f"caption for {group_id}",
        "label_b_consistent": True,
        "width": width,
        "height": height,
        "format": image_format,
        "file_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "phash": phash,
        "pixel_sha256": digest,
        "mode": mode,
    }


def write_source_fixture(tmp_path: Path, *, groups: int = 48) -> tuple[Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows: list[dict[str, object]] = []
    for group_number in range(groups):
        group_id = f"group-{group_number:03d}"
        if group_number < round(groups * 0.70):
            official_split = "train"
        elif group_number < round(groups * 0.85):
            official_split = "val"
        else:
            official_split = "test"
        for label, generator in ((0, "real"), (1, "sd21"), (1, "sd3"), (1, "sdxl")):
            path = raw_dir / f"{group_id}-{generator}.png"
            _write_rgb_png(path, seed=group_number * 100 + label * 10 + len(generator))
            rows.append(
                _source_row(
                    path,
                    group_id=group_id,
                    source_id=f"source-{group_id}-{generator}",
                    label=label,
                    generator=generator,
                    official_split=official_split,
                )
            )
    manifest = raw_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    provenance = raw_dir / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "repository_id": highres.SOURCE_REPOSITORY_ID,
                "revision": "a" * 40,
                "streaming": False,
                "limit_per_split": None,
                "records": len(rows),
                "manifest": str(manifest),
                "manifest_sha256": highres.sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    return manifest, provenance


def test_selects_balanced_deterministic_pairs_from_caption_groups(tmp_path: Path) -> None:
    manifest, provenance = write_source_fixture(tmp_path, groups=12)
    frame, _, _ = highres.validate_source_inputs(manifest, provenance)

    selected, report = highres.select_caption_matched_rows(frame, selection_seed=42)

    assert report["candidate_group_count"] == 12
    assert report["candidate_row_count"] == 24
    assert report["assigned_generator_group_counts"] == {"sd21": 4, "sd3": 4, "sdxl": 4}
    assert selected.groupby("group_id").size().eq(2).all()
    assert (
        selected.groupby("group_id")["label"]
        .agg(set)
        .map(lambda values: values == {"0", "1"})
        .all()
    )


def test_rejects_duplicate_source_id_before_output_paths_are_created(tmp_path: Path) -> None:
    manifest, provenance = write_source_fixture(tmp_path, groups=6)
    frame, _, _ = highres.validate_source_inputs(manifest, provenance)
    # These otherwise valid observations would hash to one canonical output filename. A source
    # identifier must be an immutable observation key, not merely a display label.
    first_real = frame.index[(frame["group_id"] == "group-000") & (frame["generator"] == "real")][0]
    second_real = frame.index[(frame["group_id"] == "group-001") & (frame["generator"] == "real")][
        0
    ]
    frame.loc[second_real, "source_id"] = frame.loc[first_real, "source_id"]

    with pytest.raises(ValueError, match="unique 'source_id'"):
        highres.select_caption_matched_rows(frame, selection_seed=42)


def test_source_hashes_join_leakage_components_before_split_allocation() -> None:
    frame = pd.DataFrame(
        {
            "group_id": ["caption-a", "caption-b"],
            "source_id": ["source-a", "source-b"],
            "label": ["0", "0"],
            "generator": ["real", "real"],
            # The group-seeded rendered output is deliberately distinct, but the two raw-source
            # observations are byte-identical. They must become one component before splitting.
            "source_sha256": ["a" * 64, "a" * 64],
            "source_pixel_sha256": ["b" * 64, "b" * 64],
            "source_phash": ["0" * 16, "0" * 16],
            "sha256": ["c" * 64, "d" * 64],
            "pixel_sha256": ["e" * 64, "f" * 64],
            "phash": ["0" * 15 + "1", "f" * 16],
        }
    )

    components, links = highres.build_leakage_components(frame, phash_distance_threshold=0)

    assert links.empty
    assert components["leakage_group"].nunique() == 1


def test_builds_canonical_native_crops_and_preserves_upstream_group_disjoint_splits(
    tmp_path: Path,
) -> None:
    manifest, provenance = write_source_fixture(tmp_path)
    output = tmp_path / "highres"

    report = highres.build_highres_manifest(
        output,
        source_manifest=manifest,
        source_provenance=provenance,
        phash_distance_threshold=0,
        crop_seed=17,
        selection_seed=17,
        now=datetime(2026, 8, 29, tzinfo=UTC),
    )

    frozen = pd.read_csv(output / highres.HIGHRES_MANIFEST_NAME, dtype=str)
    assert len(frozen) == 96
    assert report["selection"]["candidate_group_count"] == 48
    assert set(frozen["width"]) == {"384"}
    assert set(frozen["height"]) == {"384"}
    assert set(frozen["format"]) == {"PNG"}
    assert frozen["split"].equals(frozen["official_split"])
    assert frozen.groupby("leakage_group")["split"].nunique().eq(1).all()
    assert frozen.groupby("group_id")["split"].nunique().eq(1).all()
    assert set(frozen["split"]) == {"train", "val", "test"}
    counts = frozen.groupby(["split", "label", "generator"]).size()
    for split in ("train", "val", "test"):
        assert counts[split, "0", "real"] > 0
        for generator in highres.SELECTED_GENERATORS:
            assert counts[split, "1", generator] > 0
    for path in frozen["path"]:
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert image.size == (384, 384)
            assert image.format == "PNG"
    saved_report = json.loads((output / highres.PROVENANCE_NAME).read_text(encoding="utf-8"))
    assert saved_report["created_at_utc"] == "2026-08-29T00:00:00+00:00"
    assert saved_report["file_verification"]["rows_verified"] == 96
    assert (
        saved_report["splits"]["policy"]
        == "preserved_upstream_train_val_test_after_component_exclusion"
    )


def test_preserves_upstream_roles_by_excluding_cross_role_components() -> None:
    rows: list[dict[str, object]] = []
    for split in highres.SPLIT_NAMES:
        for generator in highres.SELECTED_GENERATORS:
            component = f"{split}-{generator}"
            rows.extend(
                (
                    {
                        "leakage_group": component,
                        "official_split": split,
                        "group_id": component,
                        "label": "0",
                        "generator": "real",
                    },
                    {
                        "leakage_group": component,
                        "official_split": split,
                        "group_id": component,
                        "label": "1",
                        "generator": generator,
                    },
                )
            )
    rows.extend(
        (
            {
                "leakage_group": "cross-role",
                "official_split": "train",
                "group_id": "cross-a",
                "label": "0",
                "generator": "real",
            },
            {
                "leakage_group": "cross-role",
                "official_split": "val",
                "group_id": "cross-b",
                "label": "1",
                "generator": "sd21",
            },
        )
    )

    retained, excluded = highres.preserve_upstream_component_splits(pd.DataFrame(rows))

    assert "cross-role" not in set(retained["leakage_group"])
    assert excluded.to_dict("records") == [
        {
            "leakage_group": "cross-role",
            "rows": 2,
            "caption_groups": 2,
            "upstream_splits": "train+val",
        }
    ]
    assert retained["split"].equals(retained["official_split"])


def test_crop_coordinates_depend_on_group_and_not_on_label() -> None:
    first = highres.deterministic_crop_coordinates(
        group_id="same-group", width=640, height=480, crop_seed=9
    )
    second = highres.deterministic_crop_coordinates(
        group_id="same-group", width=640, height=480, crop_seed=9
    )
    changed = highres.deterministic_crop_coordinates(
        group_id="other-group", width=640, height=480, crop_seed=9
    )

    assert first == second
    assert 0 <= first[0] <= 256
    assert 0 <= first[1] <= 96
    assert first != changed


def test_rejects_source_manifest_hash_mismatch_before_output(tmp_path: Path) -> None:
    manifest, provenance = write_source_fixture(tmp_path, groups=3)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    provenance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        highres.build_highres_manifest(
            tmp_path / "output", source_manifest=manifest, source_provenance=provenance
        )

    assert not (tmp_path / "output").exists()


def test_rejects_existing_output_without_overwriting(tmp_path: Path) -> None:
    manifest, provenance = write_source_fixture(tmp_path, groups=3)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        highres.build_highres_manifest(
            output, source_manifest=manifest, source_provenance=provenance
        )

    assert marker.read_text(encoding="utf-8") == "do not overwrite"
