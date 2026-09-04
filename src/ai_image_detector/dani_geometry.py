"""Exact-row DANI geometry audit through the Hugging Face Dataset Viewer.

The audit requests one metadata row at a time and validates the response against the frozen
preselection. It records decoded dimensions but never requests the returned image asset URL.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from . import dani, dani_selection

GEOMETRY_SCHEMA_VERSION: Final = "dani_exact_row_geometry_audit_v1"
GEOMETRY_CATALOG_NAME: Final = "geometry_catalog.csv"
GEOMETRY_PARTIAL_NAME: Final = "geometry_catalog.partial.csv"
GEOMETRY_PROVENANCE_NAME: Final = "provenance.json"
DEFAULT_VIEWER_ROWS_ENDPOINT: Final = "https://datasets-server.huggingface.co/rows"
DEFAULT_VIEWER_CONFIG: Final = "default"
DEFAULT_VIEWER_SPLIT: Final = "train"
MAX_RESPONSE_BYTES: Final = 1_000_000
OBSERVED_COLUMNS: Final = (
    *dani_selection.GEOMETRY_CANDIDATE_COLUMNS,
    "observed_width",
    "observed_height",
    "observed_square",
    "observed_target_1024",
    "viewer_revision_verified",
)

JsonRequester = Callable[[str, float], Mapping[str, object]]
ProgressCallback = Callable[[int, int], None]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_request_json(url: str, timeout: float) -> Mapping[str, object]:
    request = Request(url, headers={"User-Agent": "ai-image-detector-research/0.1"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("Dataset Viewer response exceeds the metadata-only size limit")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise TypeError("Dataset Viewer response must be a JSON object")
    return decoded


def _canonical_source_index(raw: str, *, row_number: int) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"geometry candidate row {row_number} has non-integer source_index"
        ) from error
    if value < 0 or str(value) != raw:
        raise ValueError(f"geometry candidate row {row_number} has non-canonical source_index")
    return value


def _validate_preselection(preselection_dir: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    paths = {
        "spec": preselection_dir / dani_selection.SELECTION_SPEC_NAME,
        "catalog": preselection_dir / dani_selection.SELECTION_CATALOG_NAME,
        "candidates": preselection_dir / dani_selection.GEOMETRY_CANDIDATES_NAME,
        "provenance": preselection_dir / dani_selection.SELECTION_PROVENANCE_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("DANI preselection is incomplete: " + ", ".join(missing))
    provenance = _read_json(paths["provenance"])
    if provenance.get("schema_version") != dani_selection.SELECTION_SCHEMA_VERSION:
        raise ValueError("DANI preselection has an unsupported schema_version")
    eligibility = provenance.get("eligibility")
    if not isinstance(eligibility, dict):
        raise TypeError("DANI preselection eligibility must be an object")
    if (
        eligibility.get("eligible_for_selected_geometry_scan") is not True
        or eligibility.get("eligible_for_selected_byte_materialisation") is not False
        or eligibility.get("eligible_for_training") is not False
        or provenance.get("image_bytes_requested") is not False
        or provenance.get("image_bytes_read") is not False
    ):
        raise ValueError("DANI preselection is not in the required geometry-only state")
    hashes = {
        "selection_spec_sha256": dani.sha256_file(paths["spec"]),
        "selection_catalog_sha256": dani.sha256_file(paths["catalog"]),
        "geometry_candidates_sha256": dani.sha256_file(paths["candidates"]),
        "selection_provenance_sha256": dani.sha256_file(paths["provenance"]),
    }
    for key in ("selection_spec_sha256", "selection_catalog_sha256", "geometry_candidates_sha256"):
        if provenance.get(key) != hashes[key]:
            raise ValueError(f"DANI preselection {key} differs from provenance")

    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_source_indexes: set[int] = set()
    with paths["candidates"].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != dani_selection.GEOMETRY_CANDIDATE_COLUMNS:
            raise ValueError("DANI geometry candidate schema differs from the locked schema")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"geometry candidate row {row_number} differs from its schema")
            source_index = _canonical_source_index(row["source_index"], row_number=row_number)
            candidate_id = row["geometry_candidate_id"]
            if not candidate_id or candidate_id in seen_ids:
                raise ValueError(f"geometry candidate row {row_number} has a duplicate/empty ID")
            if source_index in seen_source_indexes:
                raise ValueError(f"geometry candidate row {row_number} duplicates source_index")
            if row["repository_id"] != dani.REPOSITORY_ID:
                raise ValueError(f"geometry candidate row {row_number} changes repository_id")
            if row["revision"] != dani.PINNED_REVISION:
                raise ValueError(f"geometry candidate row {row_number} changes revision")
            if row["source_index_hash"] != dani.source_index_hash(row["source_index"]):
                raise ValueError(f"geometry candidate row {row_number} has an invalid index hash")
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"geometry candidate row {row_number} has an invalid label")
            seen_ids.add(candidate_id)
            seen_source_indexes.add(source_index)
            candidates.append(dict(row))
    if not candidates:
        raise ValueError("DANI geometry candidate catalogue is empty")
    counts = provenance.get("counts")
    if not isinstance(counts, dict) or counts.get("geometry_candidate_row_count") != len(
        candidates
    ):
        raise ValueError("DANI geometry candidate count differs from provenance")
    return candidates, hashes


def viewer_row_url(endpoint: str, source_index: int) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Dataset Viewer endpoint must be a query-free HTTPS URL")
    query = urlencode(
        {
            "dataset": dani.REPOSITORY_ID,
            "config": DEFAULT_VIEWER_CONFIG,
            "split": DEFAULT_VIEWER_SPLIT,
            "offset": source_index,
            "length": 1,
        }
    )
    return f"{endpoint}?{query}"


def validate_viewer_payload(
    candidate: Mapping[str, str], payload: Mapping[str, object]
) -> dict[str, object]:
    """Validate one exact Viewer response and discard its temporary image asset URL."""
    source_index = int(candidate["source_index"])
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(
            f"Viewer did not return exactly one object for source_index {source_index}"
        )
    envelope = rows[0]
    if envelope.get("row_idx") != source_index:
        raise ValueError(f"Viewer row_idx mismatch for source_index {source_index}")
    row = envelope.get("row")
    if not isinstance(row, dict):
        raise TypeError(f"Viewer row is not an object for source_index {source_index}")
    expected: dict[str, object] = {
        "index": source_index,
        "size": int(candidate["declared_size"]),
        "category": candidate["category"],
        "class_id": candidate["class_id"],
        "model": candidate["model"],
        "gen_type": candidate["gen_type"],
        "reference": candidate["label"] == "0",
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise ValueError(
                f"Viewer {field} mismatch for source_index {source_index}: "
                f"expected {expected_value!r}, got {row.get(field)!r}"
            )
    image = row.get("image")
    if not isinstance(image, dict):
        raise TypeError(f"Viewer image metadata is not an object for source_index {source_index}")
    width = image.get("width")
    height = image.get("height")
    src = image.get("src")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or not isinstance(src, str)
        or not src
    ):
        raise ValueError(f"Viewer returned invalid image metadata for source_index {source_index}")
    revision_segment = f"/--/{candidate['revision']}/--/"
    row_segment = f"/{DEFAULT_VIEWER_SPLIT}/{source_index}/image/"
    if revision_segment not in src or row_segment not in src:
        raise ValueError(f"Viewer asset provenance mismatch for source_index {source_index}")
    if envelope.get("truncated_cells") not in ([], None):
        raise ValueError(f"Viewer truncated the exact row for source_index {source_index}")
    if payload.get("partial") is not False:
        raise ValueError(f"Viewer response is partial for source_index {source_index}")
    return {
        **candidate,
        "observed_width": width,
        "observed_height": height,
        "observed_square": width == height,
        "observed_target_1024": width == height == dani_selection.DECLARED_SOURCE_SIZE,
        "viewer_revision_verified": True,
    }


def _fetch_one(
    candidate: Mapping[str, str],
    *,
    endpoint: str,
    timeout: float,
    max_attempts: int,
    requester: JsonRequester,
    sleep: Callable[[float], None],
) -> dict[str, object]:
    url = viewer_row_url(endpoint, int(candidate["source_index"]))
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return validate_viewer_payload(candidate, requester(url, timeout))
        except (OSError, TypeError, ValueError) as error:
            last_error = error
            if attempt + 1 < max_attempts:
                sleep(min(4.0, 0.5 * (2**attempt)))
    assert last_error is not None
    raise RuntimeError(
        f"Dataset Viewer geometry failed for source_index {candidate['source_index']} "
        f"after {max_attempts} attempt(s): {last_error}"
    ) from last_error


def _read_completed_prefix(
    path: Path, candidates: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OBSERVED_COLUMNS:
            raise ValueError(f"Existing geometry catalogue has an invalid schema: {path}")
        rows = [dict(row) for row in reader]
    if len(rows) > len(candidates):
        raise ValueError("Existing geometry catalogue exceeds the frozen candidate plan")
    for index, row in enumerate(rows):
        candidate = candidates[index]
        for field in dani_selection.GEOMETRY_CANDIDATE_COLUMNS:
            if row[field] != candidate[field]:
                raise ValueError(f"Existing geometry row {index + 2} is not a candidate prefix")
        width = int(row["observed_width"])
        height = int(row["observed_height"])
        expected_flags = {
            "observed_square": str(width == height),
            "observed_target_1024": str(width == height == dani_selection.DECLARED_SOURCE_SIZE),
            "viewer_revision_verified": "True",
        }
        if (
            width <= 0
            or height <= 0
            or any(row[key] != value for key, value in expected_flags.items())
        ):
            raise ValueError(f"Existing geometry row {index + 2} has invalid observations")
    return rows


def _summarise(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    resolution_counts = Counter(f"{row['observed_width']}x{row['observed_height']}" for row in rows)
    cell_resolution_counts = Counter(
        (str(row["cell"]), f"{row['observed_width']}x{row['observed_height']}") for row in rows
    )
    by_parent: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_coco_image_id"])].append(row)
    bad_parent_count = 0
    for parent_rows in by_parent.values():
        real_1024 = sum(
            row["cell"] == "real_coco" and str(row["observed_target_1024"]) == "True"
            for row in parent_rows
        )
        synthetic = [row for row in parent_rows if row["cell"] != "real_coco"]
        synthetic_cells = {str(row["cell"]) for row in synthetic}
        if (
            real_1024 != 1
            or synthetic_cells != set(dani_selection.CELL_DEFINITIONS) - {"real_coco"}
            or len(synthetic) != 4
            or any(str(row["observed_target_1024"]) != "True" for row in synthetic)
        ):
            bad_parent_count += 1
    return {
        "candidate_row_count": len(rows),
        "parent_count": len(by_parent),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "cell_resolution_counts": {
            f"{cell}:{resolution}": count
            for (cell, resolution), count in sorted(cell_resolution_counts.items())
        },
        "parent_count_failing_exact_geometry_gate": bad_parent_count,
        "all_parent_groups_pass_exact_geometry_gate": bad_parent_count == 0,
    }


def scan_geometry(
    preselection_dir: str | Path,
    output_dir: str | Path,
    *,
    endpoint: str = DEFAULT_VIEWER_ROWS_ENDPOINT,
    workers: int = 16,
    chunk_size: int = 256,
    timeout: float = 60.0,
    max_attempts: int = 3,
    requester: JsonRequester | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: ProgressCallback | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Run or resume the exact-row audit, writing final provenance only after completion."""
    if workers <= 0 or chunk_size <= 0 or timeout <= 0 or max_attempts <= 0:
        raise ValueError("workers, chunk_size, timeout, and max_attempts must be positive")
    source = Path(preselection_dir)
    destination = Path(output_dir)
    candidates, input_hashes = _validate_preselection(source)
    destination.mkdir(parents=True, exist_ok=True)
    provenance_path = destination / GEOMETRY_PROVENANCE_NAME
    partial_path = destination / GEOMETRY_PARTIAL_NAME
    final_path = destination / GEOMETRY_CATALOG_NAME
    if provenance_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed DANI geometry audit: {destination}")
    if partial_path.exists() and final_path.exists():
        raise ValueError("Geometry audit cannot contain both partial and final catalogues")
    working_path = final_path if final_path.exists() else partial_path
    completed = _read_completed_prefix(working_path, candidates)
    if final_path.exists() and len(completed) != len(candidates):
        raise ValueError("Final geometry catalogue is incomplete")

    request_json = _default_request_json if requester is None else requester
    if not final_path.exists():
        mode = "a" if partial_path.exists() else "w"
        with partial_path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OBSERVED_COLUMNS, extrasaction="raise")
            if mode == "w":
                writer.writeheader()
            start = len(completed)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for chunk_start in range(start, len(candidates), chunk_size):
                    chunk = candidates[chunk_start : chunk_start + chunk_size]
                    futures = [
                        executor.submit(
                            _fetch_one,
                            candidate,
                            endpoint=endpoint,
                            timeout=timeout,
                            max_attempts=max_attempts,
                            requester=request_json,
                            sleep=sleep,
                        )
                        for candidate in chunk
                    ]
                    results = [future.result() for future in futures]
                    writer.writerows(results)
                    handle.flush()
                    completed.extend(results)  # type: ignore[arg-type]
                    if progress is not None:
                        progress(len(completed), len(candidates))
        partial_path.replace(final_path)

    hashes_after = {
        "selection_spec_sha256": dani.sha256_file(source / dani_selection.SELECTION_SPEC_NAME),
        "selection_catalog_sha256": dani.sha256_file(
            source / dani_selection.SELECTION_CATALOG_NAME
        ),
        "geometry_candidates_sha256": dani.sha256_file(
            source / dani_selection.GEOMETRY_CANDIDATES_NAME
        ),
        "selection_provenance_sha256": dani.sha256_file(
            source / dani_selection.SELECTION_PROVENANCE_NAME
        ),
    }
    if hashes_after != input_hashes:
        raise RuntimeError("DANI preselection changed while geometry was being scanned")
    final_rows = _read_completed_prefix(final_path, candidates)
    if len(final_rows) != len(candidates):
        raise AssertionError("DANI geometry audit did not consume the complete candidate plan")
    summary = _summarise(final_rows)
    created = (datetime.now(UTC) if now is None else now()).astimezone(UTC).isoformat()
    passed = summary["all_parent_groups_pass_exact_geometry_gate"] is True
    provenance: dict[str, object] = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "created_at_utc": created,
        "viewer_request_contract": {
            "endpoint": endpoint,
            "dataset": dani.REPOSITORY_ID,
            "revision_verified_in_each_asset_locator": dani.PINNED_REVISION,
            "config": DEFAULT_VIEWER_CONFIG,
            "split": DEFAULT_VIEWER_SPLIT,
            "length": 1,
        },
        "input_hashes": input_hashes,
        "geometry_catalog": GEOMETRY_CATALOG_NAME,
        "geometry_catalog_sha256": dani.sha256_file(final_path),
        "image_asset_urls_returned_but_not_persisted": True,
        "image_asset_urls_requested": False,
        "image_bytes_requested": False,
        "image_bytes_read": False,
        "summary": summary,
        "eligibility": {
            "exact_geometry_scan_complete": True,
            "eligible_for_selection_finalisation": passed,
            "eligible_for_selected_byte_materialisation": False,
            "eligible_for_training": False,
            "remaining_blocker": (
                "Finalize one observed 1024 real row plus the four observed 1024 synthetic rows "
                "per frozen parent/caption group, then audit selected image bytes."
                if passed
                else "One or more frozen parent/caption groups lack the required exact 1024 geometry."
            ),
        },
    }
    _write_json(provenance_path, provenance)
    return provenance
