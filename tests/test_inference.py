from __future__ import annotations

import pytest
from PIL import Image

from ai_image_detector.features import CONTROLLED_PREPROCESSING_PROTOCOL
from ai_image_detector.inference import (
    ExperimentLoadError,
    ModelBundle,
    build_evaluation_transform,
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
