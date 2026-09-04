"""Build an auditable native-384, caption-matched Defactify exploratory manifest.

This module starts from the already pinned local Defactify RGB PNG corpus.  It deliberately keeps
only caption groups with one verified real image and one verified image from each selected
high-resolution generator. The result is a controlled within-Defactify sensitivity corpus, not a
geometry-neutral primary training set or a general AI-image detector benchmark.

Every selected file is reopened and checked against its source manifest before a new split is
assigned.  It is then rendered under one fixed native-crop PNG policy.  The split operates on
connected leakage components formed by caption groups, source and output exact hashes, and
an over-inclusive near-pHash candidate boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image, ImageOps

SOURCE_REPOSITORY_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
SOURCE_MANIFEST_NAME = "manifest.csv"
SOURCE_PROVENANCE_NAME = "provenance.json"
SOURCE_LOCK_NAME = "source_lock.json"
HIGHRES_MANIFEST_NAME = "highres_manifest.csv"
PROVENANCE_NAME = "provenance.json"
NEAR_PHASH_LINKS_NAME = "near_phash_links.csv"
COMPONENT_COUNTS_NAME = "component_counts.csv"
SPLIT_COUNTS_NAME = "split_counts.csv"
EXCLUDED_COMPONENTS_NAME = "excluded_components.csv"
CANONICAL_IMAGES_DIRECTORY = "images"

TARGET_SIZE = 384
MIN_SHORT_SIDE = TARGET_SIZE
PHASH_DISTANCE_THRESHOLD = 8
SPLIT_NAMES = ("train", "val", "test")
SELECTED_GENERATORS = ("sd21", "sd3", "sdxl")
SOURCE_LOCK_SCHEMA_VERSION = "defactify_exploratory_source_lock_v2"
MANIFEST_SCHEMA_VERSION = "defactify_exploratory_native384_manifest_v2"

SOURCE_REQUIRED_COLUMNS = (
    "path",
    "label",
    "split",
    "generator",
    "group_id",
    "source_id",
    "caption",
    "label_b_consistent",
    "width",
    "height",
    "format",
    "file_bytes",
    "sha256",
    "phash",
)
HIGHRES_MANIFEST_COLUMNS = (
    "path",
    "label",
    "split",
    "generator",
    "group_id",
    "leakage_group",
    "source_id",
    "caption",
    "official_split",
    "source_path",
    "source_width",
    "source_height",
    "source_format",
    "source_file_bytes",
    "source_sha256",
    "source_pixel_sha256",
    "source_phash",
    "crop_left",
    "crop_top",
    "crop_size",
    "width",
    "height",
    "format",
    "file_bytes",
    "sha256",
    "pixel_sha256",
    "phash",
    "source_repository_id",
    "source_revision",
)
FORBIDDEN_OUTPUT_COLUMNS = frozenset({"image", "image_data", "image_bytes", "raw_image"})


class UnionFind:
    """Small deterministic union-find implementation for leakage-component construction."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class FileVerification:
    """Observed content facts used in a selected high-resolution manifest row."""

    pixel_sha256: str
    phash: str
    width: int
    height: int
    mode: str
    image_format: str
    file_bytes: int


def sha256_file(path: str | Path) -> str:
    """Return a byte-exact SHA-256 digest without transforming the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_fresh_output_dir(output_dir: str | Path) -> Path:
    """Create a new evidence directory and refuse to mix a rerun with prior output."""
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Defactify exploratory output: {path}"
        )
    path.mkdir(parents=True, exist_ok=False)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = _require_text(value, field=field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return digest


def _require_revision(value: object, *, field: str) -> str:
    revision = _require_text(value, field=field)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(f"{field} must be a 40-character lowercase commit SHA")
    return revision


def _canonical_nonnegative_int(value: object, *, field: str) -> int:
    text = str(value)
    if not text.isdecimal():
        raise ValueError(f"{field} must be a canonical nonnegative decimal integer")
    return int(text)


def _canonical_positive_int(value: object, *, field: str) -> int:
    parsed = _canonical_nonnegative_int(value, field=field)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _parse_bool(value: object, *, field: str) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"{field} must be the canonical boolean True or False")


def _resolve_image_path(value: object, *, source_manifest: Path) -> Path:
    candidate = Path(_require_text(value, field="path"))
    if candidate.is_absolute():
        return candidate
    search_roots = (Path.cwd(), *source_manifest.parents)
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return candidate


def _validate_source_manifest(frame: pd.DataFrame) -> None:
    missing = [column for column in SOURCE_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Source manifest is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Source manifest is empty")
    forbidden = sorted(set(frame.columns).intersection(FORBIDDEN_OUTPUT_COLUMNS))
    if forbidden:
        raise ValueError(f"Source manifest contains unsupported raw image columns: {forbidden}")
    labels = set(frame["label"].astype(str))
    if not labels.issubset({"0", "1"}):
        raise ValueError("Source manifest labels must be exactly binary 0/1 values")
    blank_keys = [
        column
        for column in ("path", "generator", "group_id", "source_id", "caption", "sha256", "phash")
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any()
    ]
    if blank_keys:
        raise ValueError(f"Source manifest has blank required provenance keys: {blank_keys}")


def _require_unique_values(frame: pd.DataFrame, column: str, *, context: str) -> None:
    """Refuse ambiguous identifiers before they can alias output files or split components."""
    duplicates = frame.loc[frame[column].duplicated(keep=False), column].astype(str)
    if not duplicates.empty:
        raise ValueError(
            f"{context} requires unique {column!r} values; first duplicate: {duplicates.iloc[0]!r}"
        )


def validate_source_inputs(
    source_manifest: str | Path,
    source_provenance: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    """Verify the local raw corpus lock before inspecting any selected image bytes."""
    manifest_path = Path(source_manifest)
    provenance_path = Path(source_provenance)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing source provenance: {provenance_path}")
    manifest_hash = sha256_file(manifest_path)
    provenance_hash = sha256_file(provenance_path)
    provenance = _read_json(provenance_path)
    if provenance.get("repository_id") != SOURCE_REPOSITORY_ID:
        raise ValueError("Source provenance has an unexpected repository_id")
    revision = _require_revision(provenance.get("revision"), field="source provenance revision")
    if provenance.get("streaming") is not False or provenance.get("limit_per_split") is not None:
        raise ValueError(
            "High-resolution selection requires the complete non-streaming Defactify corpus"
        )
    if provenance.get("manifest_sha256") != manifest_hash:
        raise ValueError("Source manifest SHA-256 does not match source provenance")
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    _validate_source_manifest(frame)
    if provenance.get("records") != len(frame):
        raise ValueError("Source provenance record count does not match source manifest")
    return (
        frame,
        provenance,
        {
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": manifest_hash,
            "source_provenance": str(provenance_path),
            "source_provenance_sha256": provenance_hash,
            "source_repository_id": SOURCE_REPOSITORY_ID,
            "source_revision": revision,
        },
    )


def select_caption_matched_rows(
    frame: pd.DataFrame,
    *,
    min_short_side: int = MIN_SHORT_SIDE,
    generators: Sequence[str] = SELECTED_GENERATORS,
    selection_seed: int = 20260829,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select one deterministic real/fake pair per caption group with balanced fake generators.

    A group may have several rows with a repeated normalized caption.  That is neither treated as
    an extra independent sample nor resolved by looking at pixels: one row per required role is
    selected by a stable hash rank.  Every eligible group receives exactly one fake generator via
    a balanced hash-rank round robin, so the final binary corpus does not repeat a real image three
    times merely to include three synthetic generators.
    """
    if min_short_side <= 0:
        raise ValueError("min_short_side must be positive")
    selected_generators = tuple(generators)
    if tuple(sorted(set(selected_generators))) != tuple(sorted(selected_generators)):
        raise ValueError("generators must be unique")
    if not selected_generators:
        raise ValueError("At least one generated source must be selected")
    working = frame.copy()
    working["_width"] = working["width"].map(
        lambda value: _canonical_positive_int(value, field="source width")
    )
    working["_height"] = working["height"].map(
        lambda value: _canonical_positive_int(value, field="source height")
    )
    working["_label"] = working["label"].map(
        lambda value: _canonical_nonnegative_int(value, field="source label")
    )
    valid_generators = {"real", *selected_generators}
    quality_mask = working[["_width", "_height"]].min(axis=1) >= min_short_side
    relevant = working.loc[quality_mask & working["generator"].isin(valid_generators),].copy()
    if (
        not relevant["label_b_consistent"]
        .map(lambda value: _parse_bool(value, field="label_b_consistent"))
        .all()
    ):
        raise ValueError(
            "A high-resolution candidate contradicts Defactify's binary/source label mapping"
        )

    eligible_groups: list[tuple[str, pd.DataFrame, dict[str, pd.DataFrame]]] = []
    incomplete_group_count = 0
    for group_id, group in relevant.groupby("group_id", sort=True):
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("Source manifest contains an empty group_id")
        real_rows = group.loc[(group["_label"] == 0) & (group["generator"] == "real")]
        generated_rows = {
            generator: group.loc[(group["_label"] == 1) & (group["generator"] == generator)]
            for generator in selected_generators
        }
        if not real_rows.empty and all(not rows.empty for rows in generated_rows.values()):
            eligible_groups.append((str(group_id), real_rows, generated_rows))
        elif not real_rows.empty or any(not rows.empty for rows in generated_rows.values()):
            incomplete_group_count += 1
    if not eligible_groups:
        raise ValueError("No caption groups jointly support real and every selected generator")

    def group_rank(group_id: str) -> tuple[int, str]:
        digest = hashlib.sha256(
            f"{selection_seed}:generator-allocation:{group_id}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "big"), group_id

    def choose_role_row(rows: pd.DataFrame, *, group_id: str, role: str) -> int:
        ranked = []
        for row in rows.itertuples():
            source_hash = _require_sha256(row.sha256, field="source sha256")
            payload = f"{selection_seed}:row-selection:{group_id}:{role}:{source_hash}"
            ranked.append((hashlib.sha256(payload.encode()).hexdigest(), int(row.Index)))
        return min(ranked)[1]

    assigned_generator_counts = Counter[str]()
    chosen_indices: list[int] = []
    for position, (group_id, real_rows, generated_rows) in enumerate(
        sorted(eligible_groups, key=lambda item: group_rank(item[0]))
    ):
        generator = selected_generators[position % len(selected_generators)]
        assigned_generator_counts[generator] += 1
        chosen_indices.append(choose_role_row(real_rows, group_id=group_id, role="real"))
        chosen_indices.append(
            choose_role_row(generated_rows[generator], group_id=group_id, role=generator)
        )
    selected = working.loc[sorted(chosen_indices)].copy()
    selected.drop(columns=["_width", "_height", "_label"], inplace=True)
    if selected.empty:
        raise ValueError(
            "No unambiguous high-resolution caption-matched Defactify groups were found"
        )
    candidate_group_count = len(eligible_groups)
    expected_rows = candidate_group_count * 2
    if len(selected) != expected_rows:
        raise RuntimeError(
            "Selected high-resolution row count is inconsistent with its group contract"
        )
    # `source_id` names a source observation and is used to derive the canonical output filename.
    # Allowing a repeated value would either overwrite an output raster or make the manifest path
    # ambiguous, even if both rows happened to carry the same binary label.
    _require_unique_values(selected, "source_id", context="High-resolution selection")
    return selected, {
        "minimum_short_side": min_short_side,
        "selected_generators": list(selected_generators),
        "selection_seed": selection_seed,
        "candidate_group_count": candidate_group_count,
        "candidate_row_count": len(selected),
        "assigned_generator_group_counts": dict(sorted(assigned_generator_counts.items())),
        "incomplete_relevant_group_count": incomplete_group_count,
        "source_rows_at_or_above_minimum_short_side": int(quality_mask.sum()),
    }


def verify_selected_files(
    selected: pd.DataFrame,
    *,
    source_manifest: Path,
    min_short_side: int,
) -> pd.DataFrame:
    """Reopen every source PNG and record verified immutable source facts before crop rendering."""
    verified_rows: list[dict[str, object]] = []
    for _, row in selected.sort_values(["group_id", "generator", "source_id"]).iterrows():
        path = _resolve_image_path(row["path"], source_manifest=source_manifest)
        if not path.is_file():
            raise FileNotFoundError(f"Selected source image is missing: {path}")
        file_bytes = path.stat().st_size
        expected_bytes = _canonical_positive_int(row["file_bytes"], field="source file_bytes")
        if file_bytes != expected_bytes:
            raise ValueError(f"Selected source image has unexpected byte size: {path}")
        byte_digest = sha256_file(path)
        if byte_digest != _require_sha256(row["sha256"], field="source sha256"):
            raise ValueError(f"Selected source image SHA-256 differs from source manifest: {path}")
        with Image.open(path) as image:
            image.load()
            encoded_mode = image.mode
            image_format = str(image.format)
            decoded = ImageOps.exif_transpose(image).convert("RGB")
            verification = FileVerification(
                pixel_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
                phash=str(imagehash.phash(decoded)),
                width=decoded.width,
                height=decoded.height,
                mode=encoded_mode,
                image_format=image_format,
                file_bytes=file_bytes,
            )
        expected_width = _canonical_positive_int(row["width"], field="source width")
        expected_height = _canonical_positive_int(row["height"], field="source height")
        if (verification.width, verification.height) != (expected_width, expected_height):
            raise ValueError(
                f"Selected source image dimensions differ from source manifest: {path}"
            )
        if min(verification.width, verification.height) < min_short_side:
            raise ValueError(f"Selected source image falls below minimum short side: {path}")
        if verification.mode != "RGB" or verification.image_format != "PNG":
            raise ValueError(f"Selected source image is not an RGB PNG: {path}")
        if verification.phash != _require_text(row["phash"], field="source phash"):
            raise ValueError(f"Selected source image pHash differs from source manifest: {path}")
        output_row = row.to_dict()
        output_row.update(
            {
                "source_path": str(row["path"]),
                "source_width": verification.width,
                "source_height": verification.height,
                "source_format": verification.image_format,
                "source_file_bytes": verification.file_bytes,
                "source_sha256": byte_digest,
                "source_pixel_sha256": verification.pixel_sha256,
                "source_phash": verification.phash,
            }
        )
        verified_rows.append(output_row)
    return pd.DataFrame(verified_rows)


def deterministic_crop_coordinates(
    *,
    group_id: str,
    width: int,
    height: int,
    crop_size: int = TARGET_SIZE,
    crop_seed: int = 20260829,
) -> tuple[int, int]:
    """Return group-shared, label-independent native crop coordinates for one source extent."""
    if width < crop_size or height < crop_size:
        raise ValueError("Source image is smaller than the requested native crop")
    digest = hashlib.sha256(
        f"defactify-highres-native-crop-v1:{crop_seed}:{group_id}".encode()
    ).digest()
    horizontal_fraction = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    vertical_fraction = int.from_bytes(digest[8:16], "big") / (2**64 - 1)
    left = round(horizontal_fraction * (width - crop_size))
    top = round(vertical_fraction * (height - crop_size))
    return int(left), int(top)


def materialize_canonical_crops(
    verified: pd.DataFrame,
    *,
    source_manifest: Path,
    images_dir: Path,
    crop_size: int = TARGET_SIZE,
    crop_seed: int = 20260829,
) -> pd.DataFrame:
    """Write one RGB PNG native crop per selected row under a common encoder policy.

    The crop coordinates depend only on the shared caption-group ID, source geometry and frozen
    seed.  They do not depend on the label, generator, byte size, aspect ratio category, or visual
    inspection.  A source is never stretched, padded, upsampled, or written into a model input in
    its original container.
    """
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    images_dir.mkdir(parents=True, exist_ok=False)
    _require_unique_values(verified, "source_id", context="Canonical crop materialisation")
    output_rows: list[dict[str, object]] = []
    for _, row in verified.sort_values(["group_id", "generator", "source_id"]).iterrows():
        source_path = _resolve_image_path(row["source_path"], source_manifest=source_manifest)
        if sha256_file(source_path) != row["source_sha256"]:
            raise RuntimeError(
                f"Selected source image changed before canonical crop: {source_path}"
            )
        with Image.open(source_path) as image:
            image.load()
            decoded = ImageOps.exif_transpose(image).convert("RGB")
        left, top = deterministic_crop_coordinates(
            group_id=_require_text(row["group_id"], field="group_id"),
            width=decoded.width,
            height=decoded.height,
            crop_size=crop_size,
            crop_seed=crop_seed,
        )
        cropped = decoded.crop((left, top, left + crop_size, top + crop_size))
        if cropped.size != (crop_size, crop_size) or cropped.mode != "RGB":
            raise RuntimeError("Canonical crop did not retain the requested RGB extent")
        name = hashlib.sha256(str(row["source_id"]).encode()).hexdigest() + ".png"
        output_path = images_dir / name
        if output_path.exists():
            raise RuntimeError(f"Canonical output path collision: {output_path}")
        cropped.save(output_path, format="PNG", optimize=False, compress_level=9)
        with Image.open(output_path) as encoded:
            encoded.load()
            output_mode = encoded.mode
            output_format = str(encoded.format)
            decoded_output = ImageOps.exif_transpose(encoded).convert("RGB")
        if (
            decoded_output.size != (crop_size, crop_size)
            or output_format != "PNG"
            or output_mode != "RGB"
        ):
            raise RuntimeError(f"Canonical output failed RGB PNG verification: {output_path}")
        output_row = row.to_dict()
        output_row.update(
            {
                "path": str(output_path),
                "crop_left": left,
                "crop_top": top,
                "crop_size": crop_size,
                "width": crop_size,
                "height": crop_size,
                "format": output_format,
                "file_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "pixel_sha256": hashlib.sha256(decoded_output.tobytes()).hexdigest(),
                "phash": str(imagehash.phash(decoded_output)),
            }
        )
        output_rows.append(output_row)
    return pd.DataFrame(output_rows)


def _hamming_distance(left: str, right: str) -> int:
    if len(left) != 16 or len(right) != 16:
        raise ValueError("pHash values must be 16 hexadecimal characters")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as error:
        raise ValueError("pHash values must be hexadecimal") from error


def build_leakage_components(
    verified: pd.DataFrame,
    *,
    phash_distance_threshold: int = PHASH_DISTANCE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Connect captions, exact hashes, and over-inclusive cross-caption pHash candidates."""
    if phash_distance_threshold < 0:
        raise ValueError("phash_distance_threshold must be nonnegative")
    frame = verified.reset_index(drop=True).copy()
    union_find = UnionFind(len(frame))
    # Source-level keys must participate as well as canonical-output keys.  A group-seeded crop
    # can make two repeated raw files look different at the output level when their source
    # geometries differ; splitting them apart would still leak the underlying source content.
    for column in (
        "group_id",
        "source_sha256",
        "source_pixel_sha256",
        "source_phash",
        "sha256",
        "pixel_sha256",
        "phash",
    ):
        seen: dict[str, int] = {}
        for index, value in enumerate(frame[column].astype(str)):
            if value in seen:
                union_find.union(index, seen[value])
            else:
                seen[value] = index

    links: list[dict[str, object]] = []
    values = frame["phash"].astype(str).tolist()
    groups = frame["group_id"].astype(str).tolist()
    for left in range(len(frame)):
        for right in range(left):
            if groups[left] == groups[right]:
                continue
            distance = _hamming_distance(values[left], values[right])
            if distance <= phash_distance_threshold:
                union_find.union(left, right)
                links.append(
                    {
                        "left_source_id": frame.at[left, "source_id"],
                        "right_source_id": frame.at[right, "source_id"],
                        "left_group_id": groups[left],
                        "right_group_id": groups[right],
                        "left_label": int(frame.at[left, "label"]),
                        "right_label": int(frame.at[right, "label"]),
                        "left_generator": frame.at[left, "generator"],
                        "right_generator": frame.at[right, "generator"],
                        "phash_distance": distance,
                    }
                )

    component_members: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(frame)):
        component_members[union_find.find(index)].append(index)
    for members in component_members.values():
        stable_members = "\n".join(sorted(frame.at[index, "source_id"] for index in members))
        component_id = hashlib.sha256(
            f"defactify-highres-leakage-v1:{stable_members}".encode()
        ).hexdigest()[:24]
        for index in members:
            frame.at[index, "leakage_group"] = f"defactify-highres-v1:{component_id}"
    return (
        frame,
        pd.DataFrame(
            links,
            columns=(
                "left_source_id",
                "right_source_id",
                "left_group_id",
                "right_group_id",
                "left_label",
                "right_label",
                "left_generator",
                "right_generator",
                "phash_distance",
            ),
        ),
    )


def _component_vectors(
    frame: pd.DataFrame,
) -> tuple[dict[str, Counter[tuple[str, str]]], list[tuple[str, str]]]:
    categories = [("0", "real"), *(("1", generator) for generator in SELECTED_GENERATORS)]
    vectors: dict[str, Counter[tuple[str, str]]] = {}
    for component, group in frame.groupby("leakage_group", sort=True):
        vector = Counter((str(row.label), str(row.generator)) for row in group.itertuples())
        vectors[str(component)] = vector
    return vectors, categories


def preserve_upstream_component_splits(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep source-provided roles and drop components that would cross them.

    The Defactify source already supplies train/validation/test roles.  Re-randomising them after
    seeing the high-resolution eligibility gate would make a held-out role available for design
    choices.  Instead, a component is retained only when every member came from one supported
    upstream role; a component spanning roles is excluded from all three outputs.
    """
    if "official_split" not in frame:
        raise ValueError("Cannot preserve upstream roles without an official_split column")
    official = frame["official_split"]
    invalid = official.isna() | ~official.astype(str).isin(SPLIT_NAMES)
    if invalid.any():
        invalid_values = sorted(official.loc[invalid].astype(str).unique())
        raise ValueError(
            "Defactify exploratory corpus requires upstream train/val/test roles; "
            f"found invalid values: {invalid_values}"
        )

    retained_components: list[str] = []
    excluded_rows: list[dict[str, object]] = []
    for leakage_group, component in frame.groupby("leakage_group", sort=True):
        roles = tuple(sorted(component["official_split"].astype(str).unique()))
        if len(roles) == 1:
            retained_components.append(str(leakage_group))
            continue
        excluded_rows.append(
            {
                "leakage_group": str(leakage_group),
                "rows": len(component),
                "caption_groups": int(component["group_id"].nunique()),
                "upstream_splits": "+".join(roles),
            }
        )

    output = frame.loc[frame["leakage_group"].isin(retained_components)].copy()
    if output.empty:
        raise RuntimeError("Every leakage component crossed an upstream split")
    output["split"] = output["official_split"].astype(str)
    vectors, categories = _component_vectors(output)
    counts = output.groupby(["split", "label", "generator"]).size()
    missing = [
        (split, label, generator)
        for split in SPLIT_NAMES
        for label, generator in categories
        if counts.get((split, label, generator), 0) == 0
    ]
    if missing:
        raise RuntimeError(
            f"Upstream split preservation left required label/generator strata empty: {missing}"
        )
    if not vectors:
        raise RuntimeError("No retained components remain after upstream split preservation")
    return (
        output,
        pd.DataFrame(
            excluded_rows,
            columns=("leakage_group", "rows", "caption_groups", "upstream_splits"),
        ),
    )


def _ensure_no_conflicting_duplicates(frame: pd.DataFrame, column: str) -> None:
    conflicting = frame.groupby(column)["label"].nunique()
    bad = conflicting[conflicting > 1]
    if not bad.empty:
        raise ValueError(f"{column} contains cross-label duplicate(s): {bad.index[0]}")


def _validate_output_manifest(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != HIGHRES_MANIFEST_COLUMNS:
        raise RuntimeError("High-resolution output manifest columns differ from the locked schema")
    for column in (
        "source_id",
        "source_sha256",
        "source_pixel_sha256",
        "sha256",
        "pixel_sha256",
    ):
        _ensure_no_conflicting_duplicates(frame, column)
    cross_split = frame.groupby("leakage_group")["split"].nunique()
    if (cross_split > 1).any():
        raise RuntimeError("A leakage component crosses a Defactify exploratory split")
    for column in (
        "source_id",
        "source_sha256",
        "source_pixel_sha256",
        "source_phash",
        "sha256",
        "pixel_sha256",
        "phash",
        "group_id",
        "caption",
    ):
        split_counts = frame.groupby(column)["split"].nunique()
        if (split_counts > 1).any():
            raise RuntimeError(f"Leakage key {column} crosses a Defactify exploratory split")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame, columns: Iterable[str] | None = None) -> None:
    output = frame if columns is None else frame.loc[:, list(columns)]
    output.to_csv(path, index=False)


def build_highres_manifest(
    output_dir: str | Path,
    *,
    source_manifest: str | Path = Path("data/raw/defactify/manifest.csv"),
    source_provenance: str | Path = Path("data/raw/defactify/provenance.json"),
    min_short_side: int = MIN_SHORT_SIDE,
    phash_distance_threshold: int = PHASH_DISTANCE_THRESHOLD,
    crop_size: int = TARGET_SIZE,
    crop_seed: int = 20260829,
    selection_seed: int = 20260829,
    now: datetime | None = None,
) -> dict[str, object]:
    """Build the Defactify exploratory corpus and preserve its upstream held-out roles."""
    if crop_size != TARGET_SIZE:
        raise ValueError(
            f"This protocol fixes its common native crop at {TARGET_SIZE} x {TARGET_SIZE}"
        )
    if min_short_side < crop_size:
        raise ValueError("min_short_side must be at least the native crop size to avoid upsampling")
    source_manifest_path = Path(source_manifest)
    frame, source_provenance_payload, source_identity = validate_source_inputs(
        source_manifest_path,
        source_provenance,
    )
    output = require_fresh_output_dir(output_dir)
    selected, selection_summary = select_caption_matched_rows(
        frame,
        min_short_side=min_short_side,
        selection_seed=selection_seed,
    )
    verified = verify_selected_files(
        selected,
        source_manifest=source_manifest_path,
        min_short_side=min_short_side,
    )
    canonical = materialize_canonical_crops(
        verified,
        source_manifest=source_manifest_path,
        images_dir=output / CANONICAL_IMAGES_DIRECTORY,
        crop_size=crop_size,
        crop_seed=crop_seed,
    )
    canonical["official_split"] = canonical["split"]
    components, near_links = build_leakage_components(
        canonical,
        phash_distance_threshold=phash_distance_threshold,
    )
    allocated, excluded_components = preserve_upstream_component_splits(components)
    allocated["source_repository_id"] = source_identity["source_repository_id"]
    allocated["source_revision"] = source_identity["source_revision"]
    output_frame = allocated.loc[:, list(HIGHRES_MANIFEST_COLUMNS)].copy()
    _validate_output_manifest(output_frame)

    source_lock = {
        "schema_version": SOURCE_LOCK_SCHEMA_VERSION,
        **source_identity,
        "source_record_count": len(frame),
        "source_provenance_records": source_provenance_payload["records"],
        "minimum_short_side": min_short_side,
        "selected_generators": list(SELECTED_GENERATORS),
        "output_image_policy": {
            "copies_or_reencodes_images": True,
            "accepted_format": "PNG",
            "accepted_mode": "RGB",
            "source_short_side_minimum": min_short_side,
            "target_size": crop_size,
            "operation": "group_seeded_native_crop_without_padding_stretch_or_upsampling",
            "crop_seed": crop_seed,
            "png_encoder": "Pillow PNG optimize=False compress_level=9",
        },
        "leakage_component_policy": {
            "caption_group_key": "group_id",
            "exact_source_keys": ["source_sha256", "source_pixel_sha256", "source_phash"],
            "exact_output_keys": ["sha256", "pixel_sha256", "phash"],
            "near_phash_hamming_threshold": phash_distance_threshold,
            "near_phash_interpretation": (
                "over-inclusive candidate boundary, not duplicate identity evidence"
            ),
        },
        "split_policy": {
            "kind": "preserve_upstream_roles_after_component_exclusion_v1",
            "upstream_column": "split",
            "roles": list(SPLIT_NAMES),
            "cross_role_components_excluded": True,
        },
    }
    source_lock_path = output / SOURCE_LOCK_NAME
    _write_json(source_lock_path, source_lock)
    manifest_path = output / HIGHRES_MANIFEST_NAME
    _write_frame(manifest_path, output_frame)
    near_links_path = output / NEAR_PHASH_LINKS_NAME
    _write_frame(
        near_links_path,
        near_links,
        (
            "left_source_id",
            "right_source_id",
            "left_group_id",
            "right_group_id",
            "left_label",
            "right_label",
            "left_generator",
            "right_generator",
            "phash_distance",
        ),
    )
    component_counts = (
        output_frame.groupby(["leakage_group", "split", "label", "generator"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["leakage_group", "label", "generator"])
    )
    _write_frame(output / COMPONENT_COUNTS_NAME, component_counts)
    split_counts = (
        output_frame.groupby(["split", "label", "generator"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "label", "generator"])
    )
    _write_frame(output / SPLIT_COUNTS_NAME, split_counts)
    _write_frame(output / EXCLUDED_COMPONENTS_NAME, excluded_components)

    output_hashes = {
        SOURCE_LOCK_NAME: sha256_file(source_lock_path),
        HIGHRES_MANIFEST_NAME: sha256_file(manifest_path),
        NEAR_PHASH_LINKS_NAME: sha256_file(near_links_path),
        COMPONENT_COUNTS_NAME: sha256_file(output / COMPONENT_COUNTS_NAME),
        SPLIT_COUNTS_NAME: sha256_file(output / SPLIT_COUNTS_NAME),
        EXCLUDED_COMPONENTS_NAME: sha256_file(output / EXCLUDED_COMPONENTS_NAME),
    }
    # Verify the pinned raw inputs did not change while their selected files were opened.
    if sha256_file(source_manifest_path) != source_identity["source_manifest_sha256"]:
        raise RuntimeError("Source manifest changed while building the high-resolution output")
    if sha256_file(source_provenance) != source_identity["source_provenance_sha256"]:
        raise RuntimeError("Source provenance changed while building the high-resolution output")
    timestamp = (datetime.now(UTC) if now is None else now).astimezone(UTC).isoformat()
    report: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": timestamp,
        "source_lock": SOURCE_LOCK_NAME,
        "source_lock_sha256": output_hashes[SOURCE_LOCK_NAME],
        "highres_manifest": HIGHRES_MANIFEST_NAME,
        "highres_manifest_sha256": output_hashes[HIGHRES_MANIFEST_NAME],
        "output_files_sha256": output_hashes,
        "selection": selection_summary,
        "file_verification": {
            "selected_rows_materialized": len(canonical),
            "excluded_rows_materialized": len(canonical) - len(output_frame),
            "rows_verified": len(output_frame),
            "byte_hash_matches_source_manifest": len(output_frame),
            "decoded_pixel_hashes_recorded": len(output_frame),
            "decoded_rgb_png_rows": len(output_frame),
            "exact_byte_duplicate_values": int(output_frame["sha256"].duplicated().sum()),
            "exact_pixel_duplicate_values": int(output_frame["pixel_sha256"].duplicated().sum()),
            "exact_phash_duplicate_values": int(output_frame["phash"].duplicated().sum()),
            "cross_caption_near_phash_link_count": len(near_links),
            "canonical_images_directory": CANONICAL_IMAGES_DIRECTORY,
        },
        "components": {
            "component_count": int(output_frame["leakage_group"].nunique()),
            "largest_component_rows": int(output_frame.groupby("leakage_group").size().max()),
            "near_phash_hamming_threshold": phash_distance_threshold,
            "near_phash_interpretation": "candidate leakage boundary, not duplicate evidence",
            "component_counts": COMPONENT_COUNTS_NAME,
            "excluded_cross_upstream_component_count": len(excluded_components),
            "excluded_cross_upstream_rows": int(excluded_components["rows"].sum()),
            "excluded_components": EXCLUDED_COMPONENTS_NAME,
        },
        "splits": {
            "policy": "preserved_upstream_train_val_test_after_component_exclusion",
            "upstream_column": "official_split",
            "counts": SPLIT_COUNTS_NAME,
        },
        "eligibility": {
            "eligible_for_exploratory_sensitivity_training": True,
            "eligible_for_primary_highres_training": False,
            "eligible_for_model_selection": False,
            "eligible_for_external_evaluation": False,
            "scope_limitations": [
                (
                    "Caption-matched Defactify subset only; not a proof of performance on arbitrary "
                    "images or a primary high-resolution corpus."
                ),
                (
                    "Fixed native 384px crops equalise output dimensions but retain label-correlated "
                    "source-scale and generation-pipeline differences."
                ),
                (
                    "Only SD 2.1, SD 3, and SDXL are selected because the DALL-E 3 and Midjourney "
                    "v6 rows do not meet this source-resolution gate."
                ),
                (
                    "The upstream test role remains locked; no model, threshold, or architecture "
                    "decision may use it."
                ),
            ],
        },
    }
    _write_json(output / PROVENANCE_NAME, report)
    return report
