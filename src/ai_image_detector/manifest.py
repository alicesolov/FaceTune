"""Manifest validation and leak-aware audit helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"path", "label", "split", "generator", "group_id", "source_id"}
VALID_SPLITS = {"train", "val", "test", "external"}


def load_manifest(path: str | Path, check_paths: bool = False) -> pd.DataFrame:
    manifest_path = Path(path).resolve()
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if not set(frame["label"].dropna().astype(int).unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0 (real) and 1 (AI-generated)")
    unknown_splits = set(frame["split"].dropna().unique()).difference(VALID_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unknown split values: {sorted(unknown_splits)}")
    # Manifests deliberately store repository-relative paths so artifacts stay portable. Resolve
    # them against the first ancestor containing the file; this lets a notebook run from
    # `notebooks/` while command-line scripts run from the repository root.
    search_roots = (Path.cwd(), *manifest_path.parents)

    def resolve_image_path(value: object) -> str:
        image_path = Path(str(value))
        if image_path.is_absolute():
            return str(image_path)
        for root in search_roots:
            candidate = root / image_path
            if candidate.exists():
                return str(candidate)
        return str(image_path)

    frame["path"] = frame["path"].map(resolve_image_path)
    if check_paths:
        absent = frame.loc[~frame["path"].map(lambda value: Path(value).exists()), "path"]
        if not absent.empty:
            raise FileNotFoundError(
                f"{len(absent)} manifest paths do not exist; first: {absent.iloc[0]}"
            )
    return frame


def split_overlap_report(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Find keys that occur in more than one split; these invalidate split isolation."""
    usable = frame.dropna(subset=[key]).copy()
    split_count = usable.groupby(key)["split"].nunique().rename("n_splits")
    leaked = split_count[split_count > 1].index
    return usable[usable[key].isin(leaked)].sort_values([key, "split"])


def audit_summary(frame: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    summary: dict[str, pd.DataFrame | int] = {
        "records_by_split_label": pd.crosstab(frame["split"], frame["label"]),
        "records_by_split_generator": pd.crosstab(frame["split"], frame["generator"]),
        "source_id_overlap_rows": len(split_overlap_report(frame, "source_id")),
        "group_id_overlap_rows": len(split_overlap_report(frame, "group_id")),
    }
    if "caption" in frame.columns:
        summary["caption_overlap_rows"] = len(split_overlap_report(frame, "caption"))
    if "phash" in frame.columns:
        summary["phash_overlap_rows"] = len(split_overlap_report(frame, "phash"))
    if "label_b_consistent" in frame.columns:
        summary["label_b_consistency"] = pd.crosstab(frame["split"], frame["label_b_consistent"])
    return summary
