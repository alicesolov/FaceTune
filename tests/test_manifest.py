import pandas as pd
import pytest

from ai_image_detector.manifest import audit_summary, load_manifest, split_overlap_report


def sample_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": ["one.png", "two.png", "three.png"],
            "label": [0, 1, 1],
            "split": ["train", "test", "test"],
            "generator": ["real", "sdxl", "sdxl"],
            "group_id": ["a", "b", "b"],
            "source_id": ["one", "two", "three"],
            "caption": ["cat", "dog", "dog"],
        }
    )


def test_overlap_report_finds_cross_split_source() -> None:
    frame = sample_manifest()
    frame.loc[1, "source_id"] = "one"
    report = split_overlap_report(frame, "source_id")
    assert set(report["source_id"]) == {"one"}
    assert audit_summary(frame)["source_id_overlap_rows"] == 2


def test_manifest_validation_rejects_unknown_label(tmp_path) -> None:
    frame = sample_manifest()
    frame.loc[0, "label"] = 2
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="label"):
        load_manifest(path)
