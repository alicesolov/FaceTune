"""Build a leakage-resistant Defactify split from caption, exact-hash and pHash components.

The source's official split is preserved in `official_split`, but it cannot be the primary test
when captions or perceptual hashes cross partitions.  This script assigns each connected component
of those identifiers to exactly one new partition and records the achieved generator counts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ai_image_detector.reproducibility import sha256_file


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def component_ids(frame: pd.DataFrame) -> np.ndarray:
    union_find = UnionFind(len(frame))
    for column in ("group_id", "sha256", "phash"):
        seen: dict[str, int] = {}
        for index, value in enumerate(frame[column].astype(str)):
            if value in seen:
                union_find.union(index, seen[value])
            else:
                seen[value] = index
    roots = np.array([union_find.find(index) for index in range(len(frame))])
    _, encoded = np.unique(roots, return_inverse=True)
    return encoded


def allocate_components(
    frame: pd.DataFrame, components: np.ndarray, seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Greedily match 70/15/15 targets for every declared generator category."""
    categories = sorted(frame["generator"].unique())
    category_index = {category: index for index, category in enumerate(categories)}
    vectors: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(len(categories), dtype=int))
    for row_index, component in enumerate(components):
        vectors[int(component)][category_index[frame.iloc[row_index]["generator"]]] += 1
    targets = np.outer(
        np.array((0.70, 0.15, 0.15)),
        np.array([sum(vector[i] for vector in vectors.values()) for i in range(len(categories))]),
    )
    rng = np.random.default_rng(seed)
    component_order = list(vectors)
    rng.shuffle(component_order)
    component_order.sort(key=lambda component: int(vectors[component].sum()), reverse=True)
    assigned = np.zeros_like(targets, dtype=float)
    component_split: dict[int, int] = {}
    for component in component_order:
        vector = vectors[component]
        costs = []
        for split_index in range(3):
            candidate = assigned.copy()
            candidate[split_index] += vector
            costs.append(float(np.square((candidate - targets) / np.maximum(targets, 1.0)).sum()))
        split_index = int(np.argmin(costs))
        assigned[split_index] += vector
        component_split[component] = split_index
    split_names = np.array(("train", "val", "test"), dtype=object)
    splits = np.array(
        [split_names[component_split[int(component)]] for component in components], dtype=object
    )
    achieved = pd.DataFrame(assigned, index=split_names, columns=categories).astype(int)
    return splits, achieved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    required = {"path", "label", "split", "generator", "group_id", "sha256", "phash"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Manifest lacks columns required for grouped split: {sorted(missing)}")
    components = component_ids(frame)
    grouped_split, achieved = allocate_components(frame, components, args.seed)
    output = frame.copy()
    output["official_split"] = output["split"]
    output["leakage_group"] = components
    output["split"] = grouped_split
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    output.to_csv(manifest_path, index=False)
    report = {
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_file(args.manifest),
        "seed": args.seed,
        "records": len(output),
        "components": int(output["leakage_group"].nunique()),
        "largest_component": int(output.groupby("leakage_group").size().max()),
        "manifest_sha256": sha256_file(manifest_path),
        "achieved_generator_counts": achieved.to_dict(),
    }
    (args.output_root / "provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    achieved.to_csv(args.output_root / "records_by_split_generator.csv")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
