import pandas as pd
from PIL import Image

from ai_image_detector.baselines import (
    file_metadata_predict,
    fit_file_metadata_logistic,
    image_file_features,
    radial_features,
    radial_preprocessing_metadata,
)
from ai_image_detector.features import CONTROLLED_PREPROCESSING_PROTOCOL


def test_file_metadata_control_has_expected_shape_and_scores(tmp_path) -> None:
    paths = []
    for index, size in enumerate(((20, 10), (22, 11), (80, 40), (82, 41))):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", size, color=(index * 40, 0, 0)).save(path)
        paths.append(str(path))
    frame = pd.DataFrame({"path": paths, "label": [0, 0, 1, 1]})
    assert image_file_features(frame).shape == (4, 4)
    model = fit_file_metadata_logistic(frame, seed=3)
    scores = file_metadata_predict(model, frame)
    assert scores.shape == (4,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_controlled_radial_features_use_a_common_square_raster(tmp_path) -> None:
    paths = []
    for index, size in enumerate(((40, 20), (20, 40))):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", size, color=(index * 100, 10, 20)).save(path)
        paths.append(str(path))

    features = radial_features(
        pd.DataFrame({"path": paths}),
        bins=8,
        preprocessing_protocol=CONTROLLED_PREPROCESSING_PROTOCOL,
    )

    assert features.shape == (2, 8)


def test_controlled_radial_metadata_records_deterministic_crop_for_all_splits() -> None:
    metadata = radial_preprocessing_metadata(CONTROLLED_PREPROCESSING_PROTOCOL)

    assert metadata["train_crop"] == "center_square_crop"
    assert metadata["eval_crop"] == "center_square_crop"
    assert metadata["crop_policy"] == "deterministic_center_square_all_splits"
    assert metadata["augmentation"] == "none"
    assert "neural_train_augmentation" not in metadata
