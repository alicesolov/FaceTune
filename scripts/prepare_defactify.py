#!/usr/bin/env python3
"""Create an auditable local Defactify manifest from its public Hugging Face dataset.

The script intentionally records the source revision and does not guess label meanings. It inspects
the dataset's ClassLabel metadata when available. Use --limit-per-split for a small pilot; omit it
for the complete, documented benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from PIL import Image
from tqdm.auto import tqdm

REPOSITORY_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
LABEL_B_SOURCES = {
    "0": "real",
    "1": "sd21",
    "2": "sdxl",
    "3": "sd3",
    "4": "dalle3",
    "5": "midjourney_v6",
}


def normalized_caption(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else fallback


def class_value(dataset: Dataset | IterableDataset, column: str, value: Any) -> str:
    feature = dataset.features[column]
    names = getattr(feature, "names", None)
    if names is not None and isinstance(value, int):
        return str(names[value])
    return str(value)


def detect_column(
    columns: list[str], alternatives: tuple[str, ...], required: bool = True
) -> str | None:
    normalized = {column.lower(): column for column in columns}
    for candidate in alternatives:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    if required:
        raise KeyError(f"Expected one of {alternatives}; dataset columns are {columns}")
    return None


def map_label(raw: str) -> int:
    value = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"0", "real", "photo", "authentic"}:
        return 0
    if value in {"1", "fake", "ai", "ai_generated", "synthetic"}:
        return 1
    raise ValueError(
        f"Cannot safely map Label_A={raw!r}; inspect dataset metadata and add an explicit mapping."
    )


def save_record(
    record: dict[str, Any],
    dataset: Dataset | IterableDataset,
    split: str,
    index: int,
    output_root: Path,
    image_column: str,
    label_column: str,
    generator_column: str | None,
    caption_column: str | None,
) -> dict[str, Any]:
    image = record[image_column]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a decoded PIL image in {image_column}, got {type(image)!r}")
    raw_label = class_value(dataset, label_column, record[label_column])
    label = map_label(raw_label)
    raw_generator = (
        class_value(dataset, generator_column, record[generator_column])
        if generator_column
        else "unknown"
    )
    declared_source = LABEL_B_SOURCES.get(raw_generator, f"unknown_label_b_{raw_generator}")
    # Label_A is the binary task label. Preserve Label_B separately and audit contradictions instead
    # of allowing an inconsistent source field to overwrite the target label.
    generator = "real" if label == 0 else declared_source
    filename = f"{index:07d}.png"
    path = output_root / "images" / split / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG")
    contents = path.read_bytes()
    caption = record[caption_column] if caption_column else None
    source_id = str(record.get("id", record.get("ID", f"{split}:{index}")))
    return {
        "path": str(path),
        "label": label,
        "split": {"validation": "val", "validation_split": "val"}.get(split, split),
        "generator": generator,
        "group_id": normalized_caption(caption, source_id),
        "source_id": source_id,
        "caption": caption,
        "raw_label": raw_label,
        "raw_generator": raw_generator,
        "label_b_consistent": (label == 0 and declared_source == "real")
        or (label == 1 and declared_source != "real"),
        "width": image.width,
        "height": image.height,
        "format": "PNG",
        "file_bytes": len(contents),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "phash": str(imagehash.phash(image.convert("RGB"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/defactify"))
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming; recommended for a pilot subset."
    )
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.streaming and args.limit_per_split is None:
        raise SystemExit(
            "Streaming a full benchmark is deliberately disabled: choose --limit-per-split or omit --streaming."
        )
    datasets = load_dataset(REPOSITORY_ID, revision=args.revision, streaming=args.streaming)
    if not isinstance(datasets, DatasetDict) and not hasattr(datasets, "keys"):
        raise TypeError(f"Expected a DatasetDict, got {type(datasets)!r}")
    output_root = args.output_root
    all_rows: list[dict[str, Any]] = []
    for split, dataset in datasets.items():
        columns = list(dataset.column_names)
        image_column = detect_column(columns, ("image", "Image"))
        label_column = detect_column(columns, ("Label_A", "label", "Label"))
        generator_column = detect_column(
            columns, ("Label_B", "generator", "source"), required=False
        )
        caption_column = detect_column(
            columns, ("caption", "Caption", "text", "prompt"), required=False
        )
        iterator = iter(dataset)
        limit = args.limit_per_split
        for index, record in enumerate(tqdm(iterator, desc=f"write {split}")):
            if limit is not None and index >= limit:
                break
            all_rows.append(
                save_record(
                    record,
                    dataset,
                    split,
                    index,
                    output_root,
                    image_column,
                    label_column,
                    generator_column,
                    caption_column,
                )
            )
    frame = pd.DataFrame(all_rows)
    manifest = output_root / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest, index=False)
    provenance = {
        "repository_id": REPOSITORY_ID,
        "revision": args.revision,
        "streaming": args.streaming,
        "limit_per_split": args.limit_per_split,
        "records": len(frame),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
