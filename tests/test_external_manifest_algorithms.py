import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_synthbuster_external.py"
SPEC = importlib.util.spec_from_file_location("prepare_synthbuster_external", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def manifest(prefix: str, values: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [f"{prefix}-{index}" for index in range(len(values))],
            "path": [f"{prefix}-{index}.png" for index in range(len(values))],
            "label": [1] * len(values),
            "generator": ["dalle2"] * len(values),
            "phash": [f"{value:016x}" for value in values],
        }
    )


def test_phash_candidate_index_matches_brute_force_boundary_cases() -> None:
    reference_values = [0, 0xF0F0F0F0F0F0F0F0, 0x0123456789ABCDEF]
    external_values = [0b1111, 0xF0F0F0F0F0F0F0FF, 0x1123456789ABCDEF]
    reference = manifest("reference", reference_values)
    external = manifest("external", external_values)

    report = MODULE.near_duplicate_phash_matches(
        external,
        reference,
        threshold=4,
        max_pairs=100,
    )
    observed = {
        (row.external_source_id, row.reference_source_id, row.phash_distance)
        for row in report.itertuples(index=False)
    }
    expected = {
        (f"external-{external_index}", f"reference-{reference_index}", distance)
        for external_index, external_value in enumerate(external_values)
        for reference_index, reference_value in enumerate(reference_values)
        if (distance := (external_value ^ reference_value).bit_count()) <= 4
    }
    assert observed == expected
