#!/usr/bin/env python3
"""Create a frozen local Synthbuster + RAISE-1k external-test manifest.

This program deliberately has no downloader and never trains a model.  The data owner must first
obtain the archives under their respective licences, verify them, and extract them locally.  This
keeps the external benchmark locked until the internal model, threshold, and preprocessing have
been frozen.

The canonical synthetic archive is Zenodo record 10066460 (v1, CC BY-NC-SA 4.0).  The companion
RAISE-1k data are separately licensed for scientific, non-commercial use.  See --help for the
local preparation contract.  Original image files are not rewritten: the manifest records raw-file
and decoded-pixel hashes so any later evaluation can state exactly what it used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from ai_image_detector.reproducibility import sha256_file

SYNTHBUSTER_ZENODO_RECORD = "10066460"
SYNTHBUSTER_ZENODO_VERSION = "v1"
SYNTHBUSTER_ZENODO_DOI = "10.5281/zenodo.10066460"
SYNTHBUSTER_ARCHIVE_URL = (
    "https://zenodo.org/api/records/10066460/files/synthbuster.zip/content"
)
SYNTHBUSTER_ARCHIVE_MD5 = "0695bd328e16ea21c5c9cc2ae1d994ff"
RAISE_PORTAL_URL = "https://loki.disi.unitn.it/RAISE/download.html"
RAISE_LICENSE_URL = "https://loki.disi.unitn.it/RAISE/data/RAISE_License.pdf"
RAISE_GRIP_MIRROR_URL = "https://www.grip.unina.it/download/prog/DMimageDetection/real_RAISE_1k.zip"
RAISE_GRIP_MIRROR_MD5 = "a6aad7728226218f22a28b9c9aacaa2c"
GRIP_CHECKSUM_SOURCE_URL = (
    "https://raw.githubusercontent.com/grip-unina/ClipBased-SyntheticImageDetection/"
    "main/data/synthbuster_checksums.md5"
)

CANONICAL_GENERATORS = (
    "dalle2",
    "dalle3",
    "firefly",
    "glide",
    "midjourney_v5",
    "sd13",
    "sd14",
    "sd2",
    "sdxl",
)
GENERATOR_ALIASES = {
    "dalle2": ("dalle2", "dall-e2", "dalle-2", "dall_e_2"),
    "dalle3": ("dalle3", "dall-e3", "dalle-3", "dall_e_3"),
    "firefly": ("firefly", "adobe-firefly", "adobe_firefly"),
    "glide": ("glide",),
    "midjourney_v5": ("midjourney-v5", "midjourney_v5", "midjourney5"),
    "sd13": ("stable-diffusion-1-3", "stable_diffusion_1_3", "sd13", "sd-1-3"),
    "sd14": ("stable-diffusion-1-4", "stable_diffusion_1_4", "sd14", "sd-1-4"),
    "sd2": ("stable-diffusion-2", "stable_diffusion_2", "sd2", "sd-2"),
    "sdxl": ("stable-diffusion-xl", "stable_diffusion_xl", "sdxl"),
}
GENERATOR_CONTEXT = {
    "real": ("camera", "new_real_domain"),
    "dalle2": ("dalle", "same_family_different_version"),
    "dalle3": ("dalle", "same_named_generator"),
    "firefly": ("adobe_firefly", "unseen_family"),
    "glide": ("glide", "unseen_family"),
    "midjourney_v5": ("midjourney", "same_family_different_version"),
    "sd13": ("stable_diffusion", "same_family_different_version"),
    "sd14": ("stable_diffusion", "same_family_different_version"),
    # The archive's public label is simply "Stable Diffusion 2"; it is not safe to equate that
    # with Defactify's explicitly labelled SD 2.1 without inspecting the archive metadata.
    "sd2": ("stable_diffusion", "same_family_version_unspecified"),
    "sdxl": ("stable_diffusion", "same_named_generator"),
}
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
MANIFEST_COLUMNS = (
    "path",
    "label",
    "split",
    "generator",
    "group_id",
    "source_id",
    "source_dataset",
    "source_relative_path",
    "generator_family",
    "defactify_train_relation",
    "width",
    "height",
    "format",
    "file_bytes",
    "sha256",
    "file_sha256",
    "pixel_sha256",
    "phash",
)
EXACT_MATCH_COLUMNS = (
    "external_source_id",
    "external_path",
    "external_label",
    "external_generator",
    "reference_source_id",
    "reference_path",
    "reference_label",
    "reference_generator",
    "sha256",
)
PHASH_MATCH_COLUMNS = EXACT_MATCH_COLUMNS[:-1] + ("phash_distance",)


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, aliases in GENERATOR_ALIASES.items():
        for alias in aliases:
            key = _normalise_name(alias)
            previous = index.setdefault(key, canonical)
            if previous != canonical:
                raise RuntimeError(f"Ambiguous Synthbuster alias {alias!r}")
    return index


ALIAS_INDEX = _alias_index()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file digest without loading a potentially large archive into memory."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(rgb: Image.Image) -> str:
    """Hash an already oriented RGB pixel array and dimensions, independent of container bytes."""
    if rgb.mode != "RGB":
        raise ValueError("pixel_sha256 expects an RGB image")
    digest = hashlib.sha256()
    digest.update(struct.pack(">II", rgb.width, rgb.height))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def image_record(
    image_path: Path,
    *,
    label: int,
    generator: str,
    source_dataset: str,
    source_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Read only enough of an image to make an auditable manifest row."""
    try:
        with Image.open(image_path) as opened:
            original_format = opened.format or image_path.suffix.lstrip(".").upper()
            oriented = ImageOps.exif_transpose(opened)
            rgb = oriented.convert("RGB")
            width, height = rgb.size
            perceptual_hash = str(imagehash.phash(rgb))
            decoded_hash = pixel_sha256(rgb)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Cannot decode image {image_path}: {error}") from error

    absolute_path = image_path.resolve()
    try:
        stored_path = absolute_path.relative_to(repository_root).as_posix()
    except ValueError:
        stored_path = str(absolute_path)
    relative_path = image_path.resolve().relative_to(source_root.resolve()).as_posix()
    source_id = f"{source_dataset}:{generator}:{relative_path}"
    file_hash = sha256_file(absolute_path)
    generator_family, train_relation = GENERATOR_CONTEXT[generator]
    return {
        "path": stored_path,
        "label": label,
        "split": "external",
        "generator": generator,
        # No pairing between a generated image and a particular RAISE file is inferred from a
        # filename.  A source-specific ID prevents accidental grouping claims in later analyses.
        "group_id": source_id,
        "source_id": source_id,
        "source_dataset": source_dataset,
        "source_relative_path": relative_path,
        "generator_family": generator_family,
        "defactify_train_relation": train_relation,
        "width": width,
        "height": height,
        "format": str(original_format),
        "file_bytes": absolute_path.stat().st_size,
        "sha256": file_hash,
        "file_sha256": file_hash,
        "pixel_sha256": decoded_hash,
        "phash": perceptual_hash,
    }


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Expected an extracted image directory, got {directory}")
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_generator_directory(value: str) -> tuple[str, Path]:
    try:
        generator, raw_path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use --generator-dir canonical_name=/path/to/images") from error
    generator = generator.strip()
    if generator not in CANONICAL_GENERATORS:
        raise argparse.ArgumentTypeError(
            f"Unknown generator {generator!r}; expected one of {list(CANONICAL_GENERATORS)}"
        )
    directory = Path(raw_path).expanduser()
    if not directory.is_dir():
        raise argparse.ArgumentTypeError(f"Generator directory does not exist: {directory}")
    return generator, directory


def discover_generator_directories(synthetic_root: Path) -> dict[str, Path]:
    """Find documented generator directory names without assuming a hidden archive layout."""
    if not synthetic_root.is_dir():
        raise ValueError(f"Synthetic root does not exist: {synthetic_root}")
    roots = [synthetic_root]
    nested = synthetic_root / "synthbuster"
    if nested.is_dir():
        roots.append(nested)
    discovered: dict[str, Path] = {}
    for root in roots:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            canonical = ALIAS_INDEX.get(_normalise_name(child.name))
            if canonical is None:
                continue
            previous = discovered.get(canonical)
            if previous is not None and previous.resolve() != child.resolve():
                raise ValueError(
                    f"Found two directories for {canonical}: {previous} and {child}; "
                    "use --generator-dir to make the mapping explicit."
                )
            discovered[canonical] = child
    return discovered


def archive_provenance(
    path: Path | None,
    *,
    expected_md5: str | None,
    verify_known_md5: bool,
    archive_kind: str,
) -> dict[str, Any]:
    if path is None:
        if verify_known_md5:
            raise ValueError("--verify-known-md5 requires the corresponding local archive path")
        return {
            "archive_kind": archive_kind,
            "provided": False,
            "expected_md5": expected_md5,
        }
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Archive does not exist: {resolved}")
    actual_md5 = md5_file(resolved)
    if verify_known_md5 and expected_md5 is None:
        raise ValueError(
            "No publisher checksum is configured for this RAISE archive kind; "
            "do not claim a known-MD5 verification."
        )
    if verify_known_md5 and actual_md5 != expected_md5:
        raise ValueError(
            f"Archive checksum mismatch for {resolved.name}: expected {expected_md5}, got {actual_md5}"
        )
    return {
        "archive_kind": archive_kind,
        "provided": True,
        "path": str(resolved),
        "file_bytes": resolved.stat().st_size,
        "md5": actual_md5,
        "sha256": sha256_file(resolved),
        "expected_md5": expected_md5,
        "expected_md5_matches": actual_md5 == expected_md5 if expected_md5 else None,
    }


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def exact_file_hash_matches(external: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Return byte-for-byte file duplicates; non-matches do not prove visual independence."""
    reference_hash_column = "file_sha256" if "file_sha256" in reference else "sha256"
    if reference_hash_column not in reference or "file_sha256" not in external:
        return _empty_frame(EXACT_MATCH_COLUMNS)
    left = external.loc[
        :, ["source_id", "path", "label", "generator", "file_sha256"]
    ].rename(
        columns={
            "source_id": "external_source_id",
            "path": "external_path",
            "label": "external_label",
            "generator": "external_generator",
            "file_sha256": "sha256",
        }
    )
    right = reference.loc[
        :, ["source_id", "path", "label", "generator", reference_hash_column]
    ].rename(
        columns={
            "source_id": "reference_source_id",
            "path": "reference_path",
            "label": "reference_label",
            "generator": "reference_generator",
            reference_hash_column: "sha256",
        }
    )
    return left.merge(right, on="sha256", how="inner").loc[:, list(EXACT_MATCH_COLUMNS)]


def _phash_value(value: object) -> int:
    text = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{16}", text):
        raise ValueError(f"Expected a 64-bit hexadecimal pHash, got {value!r}")
    return int(text, 16)


def _phash_chunks(value: int, threshold: int) -> list[tuple[int, int]]:
    """Partition 64 bits into threshold+1 chunks for an exact-pigeonhole candidate index."""
    parts = threshold + 1
    quotient, remainder = divmod(64, parts)
    shift = 0
    chunks: list[tuple[int, int]] = []
    for position in range(parts):
        width = quotient + (1 if position < remainder else 0)
        mask = (1 << width) - 1
        chunks.append((position, (value >> shift) & mask))
        shift += width
    return chunks


def near_duplicate_phash_matches(
    external: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    threshold: int,
    max_pairs: int,
) -> pd.DataFrame:
    """Find pHash candidates at Hamming distance <= threshold without an all-pairs scan.

    The chunk index is exact for this threshold: among threshold+1 chunks, at least one chunk is
    identical when at most threshold bits differ.  Candidates are still reported as *candidates*,
    because a pHash collision is not proof of image identity.
    """
    if not 0 <= threshold <= 8:
        raise ValueError("pHash threshold must be in [0, 8] to keep the audit tractable")
    for required in ("source_id", "path", "label", "generator", "phash"):
        if required not in external or required not in reference:
            raise ValueError(f"Both manifests need {required!r} for a pHash audit")

    buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for reference_index, value in enumerate(reference["phash"]):
        parsed = _phash_value(value)
        for key in _phash_chunks(parsed, threshold):
            buckets[key].append((reference_index, parsed))

    matches: list[dict[str, Any]] = []
    for external_index, value in enumerate(external["phash"]):
        parsed = _phash_value(value)
        candidates: dict[int, int] = {}
        for key in _phash_chunks(parsed, threshold):
            for reference_index, reference_value in buckets.get(key, []):
                candidates.setdefault(reference_index, reference_value)
        external_row = external.iloc[external_index]
        for reference_index, reference_value in candidates.items():
            distance = (parsed ^ reference_value).bit_count()
            if distance > threshold:
                continue
            reference_row = reference.iloc[reference_index]
            matches.append(
                {
                    "external_source_id": external_row["source_id"],
                    "external_path": external_row["path"],
                    "external_label": int(external_row["label"]),
                    "external_generator": external_row["generator"],
                    "reference_source_id": reference_row["source_id"],
                    "reference_path": reference_row["path"],
                    "reference_label": int(reference_row["label"]),
                    "reference_generator": reference_row["generator"],
                    "phash_distance": distance,
                }
            )
            if len(matches) > max_pairs:
                raise ValueError(
                    f"More than {max_pairs} pHash candidates; refine the threshold or inspect inputs."
                )
    return pd.DataFrame(matches, columns=list(PHASH_MATCH_COLUMNS))


def read_reference_manifest(path: Path) -> pd.DataFrame:
    # CSV readers otherwise turn an all-numeric pHash such as 0000000000000001 into an integer
    # and silently discard the leading zero required by the 64-bit representation.
    reference = pd.read_csv(
        path,
        dtype={
            "path": str,
            "source_id": str,
            "generator": str,
            "sha256": str,
            "file_sha256": str,
            "phash": str,
        },
    )
    required = {"path", "label", "generator", "source_id", "phash"}
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"Reference manifest is missing {sorted(missing)}")
    return reference


def validate_expected_counts(frame: pd.DataFrame, allow_partial: bool) -> dict[str, Any]:
    expected = {"real": 1000, **{generator: 1000 for generator in CANONICAL_GENERATORS}}
    observed = frame["generator"].value_counts().sort_index().to_dict()
    valid = observed == expected
    if not valid and not allow_partial:
        raise ValueError(
            "Expected exactly 1,000 RAISE images and 1,000 Synthbuster images per generator; "
            f"observed {observed}. Use --allow-partial only for a non-reportable dry run."
        )
    return {"expected": expected, "observed": observed, "complete": valid, "allow_partial": allow_partial}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local external-only Synthbuster + RAISE-1k manifest. This command never "
            "downloads data. Obtain Synthbuster record 10066460 and RAISE-1k separately, accept "
            "their non-commercial research licences, verify archives, extract them, then supply "
            "the local directories."
        )
    )
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--raise-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--generator-dir",
        type=parse_generator_directory,
        action="append",
        default=[],
        metavar="GENERATOR=PATH",
        help="Explicit mapping, repeatable; overrides automatic directory-name discovery.",
    )
    parser.add_argument(
        "--synthbuster-archive",
        type=Path,
        help="Optional locally retained synthbuster.zip; its MD5/SHA-256 are recorded.",
    )
    parser.add_argument(
        "--raise-archive",
        type=Path,
        help="Optional locally retained RAISE archive; its MD5/SHA-256 are recorded.",
    )
    parser.add_argument(
        "--raise-archive-kind",
        choices=("raise_portal", "grip_mirror", "other"),
        default="other",
        help="Only grip_mirror has a known checksum configured by this script.",
    )
    parser.add_argument(
        "--verify-known-md5",
        action="store_true",
        help="Fail unless locally supplied archives match fixed public MD5 values.",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help="Optional locked Defactify manifest to audit exact-file and pHash candidate overlap.",
    )
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--max-near-duplicate-pairs", type=int, default=100_000)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit incomplete counts for a local dry run only; provenance marks it non-complete.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_root}; use --overwrite for this exact target")

    explicit_directories = dict(args.generator_dir)
    if len(explicit_directories) != len(args.generator_dir):
        raise SystemExit("Each --generator-dir generator may be supplied at most once")
    generator_directories = explicit_directories or discover_generator_directories(
        args.synthetic_root.expanduser()
    )
    unknown = set(generator_directories).difference(CANONICAL_GENERATORS)
    if unknown:
        raise SystemExit(f"Unknown generator mappings: {sorted(unknown)}")
    missing = set(CANONICAL_GENERATORS).difference(generator_directories)
    if missing and not args.allow_partial:
        raise SystemExit(
            f"Missing Synthbuster generator directories: {sorted(missing)}; use --generator-dir or "
            "--allow-partial only for a dry run."
        )

    repository_root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    raise_root = args.raise_root.expanduser()
    for path in image_files(raise_root):
        rows.append(
            image_record(
                path,
                label=0,
                generator="real",
                source_dataset="raise1k",
                source_root=raise_root,
                repository_root=repository_root,
            )
        )
    for generator in CANONICAL_GENERATORS:
        directory = generator_directories.get(generator)
        if directory is None:
            continue
        files = image_files(directory)
        if not files:
            raise SystemExit(f"No supported image files found for {generator} in {directory}")
        for path in files:
            rows.append(
                image_record(
                    path,
                    label=1,
                    generator=generator,
                    source_dataset="synthbuster",
                    source_root=directory,
                    repository_root=repository_root,
                )
            )

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)).sort_values(
        ["label", "generator", "source_relative_path"]
    )
    count_report = validate_expected_counts(manifest, args.allow_partial)
    synthbuster_archive = archive_provenance(
        args.synthbuster_archive,
        expected_md5=SYNTHBUSTER_ARCHIVE_MD5,
        verify_known_md5=args.verify_known_md5,
        archive_kind="zenodo_record_10066460_v1",
    )
    raise_expected_md5 = RAISE_GRIP_MIRROR_MD5 if args.raise_archive_kind == "grip_mirror" else None
    raise_archive = archive_provenance(
        args.raise_archive,
        expected_md5=raise_expected_md5,
        verify_known_md5=args.verify_known_md5,
        archive_kind=args.raise_archive_kind,
    )

    reference: pd.DataFrame | None = None
    if args.reference_manifest:
        reference = read_reference_manifest(args.reference_manifest)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    provenance: dict[str, Any] = {
        "benchmark": "Synthbuster + RAISE-1k external locked test",
        "created_utc": datetime.now(UTC).isoformat(),
        "records": len(manifest),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "integrity": count_report,
        "synthbuster": {
            "zenodo_record": SYNTHBUSTER_ZENODO_RECORD,
            "version": SYNTHBUSTER_ZENODO_VERSION,
            "doi": SYNTHBUSTER_ZENODO_DOI,
            "archive_url": SYNTHBUSTER_ARCHIVE_URL,
            "licence": "CC-BY-NC-SA-4.0",
            "archive": synthbuster_archive,
        },
        "raise1k": {
            "portal_url": RAISE_PORTAL_URL,
            "licence_url": RAISE_LICENSE_URL,
            "grip_mirror_url": RAISE_GRIP_MIRROR_URL,
            "grip_mirror_checksum_source_url": GRIP_CHECKSUM_SOURCE_URL,
            "usage_restriction": "scientific, educational, and other non-commercial uses only",
            "archive": raise_archive,
        },
        "image_decoding": {
            "decoder": "Pillow",
            "pillow_version": Image.__version__,
            "imagehash_version": getattr(imagehash, "__version__", None),
            "orientation": "ImageOps.exif_transpose before RGB conversion",
            "pixel_sha256": "SHA-256 over big-endian width,height then RGB pixel bytes",
            "phash": "imagehash.phash default 8x8 DCT hash",
        },
        "external_protocol": {
            "split": "external only",
            "prohibited_uses": ["training", "early stopping", "threshold selection", "augmentation selection"],
            "grouping_note": "No generated-to-RAISE pairing is inferred without an explicit source map.",
        },
    }
    if reference is not None:
        exact_matches = exact_file_hash_matches(manifest, reference)
        near_matches = near_duplicate_phash_matches(
            manifest,
            reference,
            threshold=args.phash_distance,
            max_pairs=args.max_near_duplicate_pairs,
        )
        exact_path = output_root / "exact_file_hash_matches.csv"
        near_path = output_root / "phash_near_duplicate_candidates.csv"
        exact_matches.to_csv(exact_path, index=False)
        near_matches.to_csv(near_path, index=False)
        provenance["defactify_overlap_audit"] = {
            "reference_manifest": str(args.reference_manifest.resolve()),
            "reference_manifest_sha256": sha256_file(args.reference_manifest),
            "exact_file_hash_matches": len(exact_matches),
            "exact_file_hash_report": str(exact_path),
            "phash_distance_threshold": args.phash_distance,
            "phash_candidate_pairs": len(near_matches),
            "phash_candidate_report": str(near_path),
            "interpretation": "pHash rows are candidates requiring visual/source review, not proof of identity.",
        }
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
