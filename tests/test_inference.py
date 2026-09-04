from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from ai_image_detector.features import CONTROLLED_PREPROCESSING_PROTOCOL
from ai_image_detector.inference import (
    ExperimentLoadError,
    ModelBundle,
    build_evaluation_transform,
    load_selection_record,
    preprocessing_from_run,
)


def test_model_loader_refuses_an_incomplete_or_arbitrary_checkpoint(tmp_path) -> None:
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ExperimentLoadError, match="completed experiment"):
        ModelBundle.load(tmp_path, device_name="cpu")


def test_inference_restores_the_checkpoint_preprocessing_contract() -> None:
    controlled = preprocessing_from_run(
        {
            "preprocessing": {
                "protocol": CONTROLLED_PREPROCESSING_PROTOCOL,
                "version": "1.0",
                "image_size": 128,
            }
        }
    )
    transform = build_evaluation_transform("rgb", controlled)

    assert controlled["protocol"] == CONTROLLED_PREPROCESSING_PROTOCOL
    assert transform(Image.new("RGB", (240, 120))).shape == (3, 128, 128)
    assert preprocessing_from_run({})["protocol"] == "legacy_resize_v1"
    with pytest.raises(ExperimentLoadError, match="version"):
        preprocessing_from_run(
            {
                "preprocessing": {
                    "protocol": CONTROLLED_PREPROCESSING_PROTOCOL,
                    "version": "unknown",
                    "image_size": 128,
                }
            }
        )


def test_selection_record_requires_external_validation_status(tmp_path) -> None:
    record = tmp_path / "selection.json"
    record.write_text(
        """{
  "schema_version": "ai_image_detector_model_selection_v1",
  "selection_status": "internal_only",
  "experiment_dir": "experiment",
  "checkpoint_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
}""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentLoadError, match="frozen_external_validated"):
        load_selection_record(record)


def test_selected_model_is_hash_pinned_and_rejects_legacy_runs(tmp_path, monkeypatch) -> None:
    record = tmp_path / "selection.json"
    checksum = "b" * 64
    record.write_text(
        (
            "{\n"
            '  "schema_version": "ai_image_detector_model_selection_v1",\n'
            '  "selection_status": "frozen_external_validated",\n'
            '  "experiment_dir": "experiment",\n'
            f'  "checkpoint_sha256": "{checksum}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    selected = SimpleNamespace(
        checkpoint_sha256=checksum,
        preprocessing={"protocol": CONTROLLED_PREPROCESSING_PROTOCOL},
        metrics={"roc_auc": 0.7},
    )
    monkeypatch.setattr(ModelBundle, "load", classmethod(lambda cls, *_args, **_kwargs: selected))

    assert ModelBundle.load_selected(record, device_name="cpu") is selected

    selected.preprocessing = {"protocol": "legacy_resize_v1"}
    with pytest.raises(ExperimentLoadError, match="H1-N"):
        ModelBundle.load_selected(record, device_name="cpu")
