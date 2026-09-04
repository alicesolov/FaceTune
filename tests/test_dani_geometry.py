from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from ai_image_detector import dani, dani_geometry, dani_selection


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _candidate(parent: int, cell: str, source_index: int) -> dict[str, object]:
    label, model, gen_type = dani_selection.CELL_DEFINITIONS[cell]
    base = {
        "split": "train",
        "leakage_group": f"coco-parent:{parent}",
        "parent_coco_image_id": parent,
        "coco_caption_id": parent * 100 + 1,
        "official_coco_split": "val2017",
        "official_coco_license_id": 4,
        "official_coco_license_name": "BY",
        "official_coco_license_url": "https://license/4",
        "cell": cell,
        "label": label,
        "generator": "real" if label == "0" else f"{model}:{gen_type}",
        "model": model,
        "gen_type": gen_type,
        "declared_size": 1024,
        "locator": f"{dani.REPOSITORY_ID}@{dani.PINNED_REVISION}:data/sample:{source_index}",
        "repository_id": dani.REPOSITORY_ID,
        "revision": dani.PINNED_REVISION,
        "shard_path": "data/sample.parquet",
        "row_index": source_index,
        "source_index": source_index,
        "source_index_hash": dani.source_index_hash(str(source_index)),
        "image_path_basename": f"{parent}_{parent * 100 + 1}.jpg",
        "category": "outdoor",
        "class_id": "7",
    }
    provisional_id = f"selected-{parent}-{cell}"
    return {
        "geometry_candidate_id": f"candidate-{source_index}",
        "provisional_selection_id": provisional_id,
        "is_provisional_selection": cell != "real_coco" or source_index % 2 == 0,
        **base,
    }


def _preselection(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    source = tmp_path / "preselection"
    source.mkdir(parents=True)
    candidates: list[dict[str, object]] = []
    source_index = 0
    for parent in (10, 20):
        candidates.append(_candidate(parent, "real_coco", source_index))
        source_index += 1
        candidates.append(_candidate(parent, "real_coco", source_index))
        source_index += 1
        for cell in list(dani_selection.CELL_DEFINITIONS)[1:]:
            candidates.append(_candidate(parent, cell, source_index))
            source_index += 1
    candidates[0]["category"] = ""
    candidates[0]["class_id"] = ""
    spec = source / dani_selection.SELECTION_SPEC_NAME
    catalog = source / dani_selection.SELECTION_CATALOG_NAME
    candidate_path = source / dani_selection.GEOMETRY_CANDIDATES_NAME
    _write_json(spec, {"selection": "fixture"})
    catalog.write_text("selection_id\n", encoding="utf-8")
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=dani_selection.GEOMETRY_CANDIDATE_COLUMNS,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(candidates)
    _write_json(
        source / dani_selection.SELECTION_PROVENANCE_NAME,
        {
            "schema_version": dani_selection.SELECTION_SCHEMA_VERSION,
            "selection_spec_sha256": dani.sha256_file(spec),
            "selection_catalog_sha256": dani.sha256_file(catalog),
            "geometry_candidates_sha256": dani.sha256_file(candidate_path),
            "image_bytes_requested": False,
            "image_bytes_read": False,
            "counts": {"geometry_candidate_row_count": len(candidates)},
            "eligibility": {
                "eligible_for_selected_geometry_scan": True,
                "eligible_for_selected_byte_materialisation": False,
                "eligible_for_training": False,
            },
        },
    )
    return source, candidates


def _payload(candidate: dict[str, object], *, width: int) -> dict[str, object]:
    source_index = int(candidate["source_index"])
    return {
        "rows": [
            {
                "row_idx": source_index,
                "row": {
                    "index": source_index,
                    "image": {
                        "src": (
                            f"https://datasets-server.huggingface.co/cached-assets/"
                            f"Renyang/DANI/--/{dani.PINNED_REVISION}/--/default/train/"
                            f"{source_index}/image/image.jpg?temporary=secret"
                        ),
                        "height": width,
                        "width": width,
                    },
                    "size": int(candidate["declared_size"]),
                    "category": candidate["category"] or None,
                    "class_id": candidate["class_id"] or None,
                    "model": candidate["model"],
                    "gen_type": candidate["gen_type"],
                    "reference": candidate["label"] == "0",
                },
                "truncated_cells": [],
            }
        ],
        "partial": False,
    }


def _requester(candidates: list[dict[str, object]], widths: dict[int, int]):
    by_index = {int(row["source_index"]): row for row in candidates}

    def request(url: str, timeout: float) -> dict[str, object]:
        assert timeout > 0
        query = parse_qs(urlparse(url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        rows = []
        for index in range(offset, offset + length):
            if index in by_index:
                rows.append(_payload(by_index[index], width=widths[index])["rows"][0])
            else:
                rows.append({"row_idx": index, "row": {}, "truncated_cells": []})
        return {"rows": rows, "partial": False}

    return request


def test_exact_geometry_scan_passes_without_persisting_asset_urls(tmp_path: Path) -> None:
    source, candidates = _preselection(tmp_path)
    widths = {int(row["source_index"]): 1024 for row in candidates}
    for parent in (10, 20):
        real = [
            row
            for row in candidates
            if row["parent_coco_image_id"] == parent and row["cell"] == "real_coco"
        ]
        widths[int(real[0]["source_index"])] = 224
    report = dani_geometry.scan_geometry(
        source,
        tmp_path / "audit",
        workers=2,
        chunk_size=3,
        requester=_requester(candidates, widths),
        min_request_interval=0,
        now=lambda: datetime(2026, 8, 30, tzinfo=UTC),
    )

    assert report["summary"]["resolution_counts"] == {"1024x1024": 10, "224x224": 2}
    assert report["eligibility"]["eligible_for_selection_finalisation"] is True
    assert report["eligibility"]["eligible_for_selected_byte_materialisation"] is False
    catalog = (tmp_path / "audit" / dani_geometry.GEOMETRY_CATALOG_NAME).read_text()
    assert "temporary=secret" not in catalog
    assert "https://" in catalog  # only the frozen COCO licence URL remains


def test_geometry_gate_fails_when_a_synthetic_cell_is_low_resolution(tmp_path: Path) -> None:
    source, candidates = _preselection(tmp_path)
    widths = {int(row["source_index"]): 1024 for row in candidates}
    for parent in (10, 20):
        first_real = next(
            row
            for row in candidates
            if row["parent_coco_image_id"] == parent and row["cell"] == "real_coco"
        )
        widths[int(first_real["source_index"])] = 224
    synthetic = next(row for row in candidates if row["cell"] != "real_coco")
    widths[int(synthetic["source_index"])] = 512

    report = dani_geometry.scan_geometry(
        source,
        tmp_path / "audit",
        requester=_requester(candidates, widths),
        min_request_interval=0,
    )

    assert report["summary"]["parent_count_failing_exact_geometry_gate"] == 1
    assert report["eligibility"]["eligible_for_selection_finalisation"] is False


def test_geometry_scan_resumes_from_completed_observations(tmp_path: Path) -> None:
    source, candidates = _preselection(tmp_path)
    widths = {int(row["source_index"]): 1024 for row in candidates}
    calls: Counter[int] = Counter()

    def flaky(url: str, timeout: float) -> dict[str, object]:
        index = int(parse_qs(urlparse(url).query)["offset"][0])
        calls[index] += 1
        if index == 1:
            raise OSError("temporary test failure")
        return _payload(candidates[index], width=widths[index])

    output = tmp_path / "audit"
    with pytest.raises(RuntimeError, match=r"range 1\+1"):
        dani_geometry.scan_geometry(
            source,
            output,
            workers=1,
            chunk_size=1,
            max_attempts=1,
            min_request_interval=0,
            max_viewer_length=1,
            requester=flaky,
        )
    assert calls[0] == 1

    dani_geometry.scan_geometry(
        source,
        output,
        workers=1,
        chunk_size=1,
        requester=_requester(candidates, widths),
        min_request_interval=0,
        max_viewer_length=1,
    )
    assert calls[0] == 1


def test_geometry_scan_rejects_viewer_identity_mismatch(tmp_path: Path) -> None:
    source, candidates = _preselection(tmp_path)

    def wrong_index(url: str, timeout: float) -> dict[str, object]:
        payload = _payload(candidates[0], width=1024)
        payload["rows"][0]["row_idx"] = 999  # type: ignore[index]
        return payload

    with pytest.raises(ValueError, match="row_idx mismatch"):
        dani_geometry.scan_geometry(
            source,
            tmp_path / "audit",
            workers=1,
            chunk_size=1,
            max_attempts=1,
            min_request_interval=0,
            max_viewer_length=1,
            requester=wrong_index,
        )
