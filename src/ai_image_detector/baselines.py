"""Small, interpretable baselines that prevent overclaiming neural results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .features import (
    CONTROLLED_PREPROCESSING_PROTOCOL,
    HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL,
    LEGACY_PREPROCESSING_PROTOCOL,
    preprocessing_metadata,
    radial_power_spectrum,
    require_canonical_highres_raster,
    source_normalized_rasterize,
)


def radial_preprocessing_metadata(protocol: str) -> dict[str, object]:
    """Describe the raster actually used by the deterministic radial baseline.

    H1-N neural training uses an epoch-specific random crop, whereas this linear baseline extracts
    one fixed feature vector per image before fitting.  It therefore uses the centre crop on every
    split and must not inherit the neural train-crop metadata unchanged.
    """
    metadata = preprocessing_metadata(protocol)
    if protocol in {CONTROLLED_PREPROCESSING_PROTOCOL, HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL}:
        if protocol == CONTROLLED_PREPROCESSING_PROTOCOL:
            metadata["train_crop"] = "center_square_crop"
            metadata["eval_crop"] = "center_square_crop"
            metadata["crop_policy"] = "deterministic_center_square_all_splits"
        else:
            metadata["crop_policy"] = "precanonicalized_raster_all_splits"
        # The neural-only random flip belongs to the matched RGB/FFT training protocol. This
        # fixed-feature baseline uses no augmentation on any split, so retaining that nested
        # metadata here would falsely describe its fitted input.
        metadata.pop("neural_train_augmentation", None)
        metadata["augmentation"] = "none"
    return metadata


def radial_features(
    frame: pd.DataFrame,
    bins: int = 64,
    preprocessing_protocol: str = LEGACY_PREPROCESSING_PROTOCOL,
) -> np.ndarray:
    """Compute one deterministic radial FFT feature vector per image.

    Controlled H1-N runs centre-crop all three splits before feature extraction; stochastic crops
    would make a fixed linear-feature fit non-reproducible unless they were separately recorded.
    """
    preprocessing = radial_preprocessing_metadata(preprocessing_protocol)
    image_size = int(preprocessing["image_size"])
    rows: list[np.ndarray] = []
    for path in tqdm(frame["path"], desc="radial FFT features"):
        with Image.open(Path(path)) as image:
            if preprocessing_protocol == CONTROLLED_PREPROCESSING_PROTOCOL:
                image = source_normalized_rasterize(image, size=image_size, train=False)
            elif preprocessing_protocol == HIGHRES_CANONICAL_PREPROCESSING_PROTOCOL:
                image = require_canonical_highres_raster(image, size=image_size)
            else:
                image = image.convert("RGB")
            rows.append(radial_power_spectrum(image, size=image_size, bins=bins))
    return np.stack(rows)


def fit_radial_logistic(
    train: pd.DataFrame,
    bins: int = 64,
    seed: int = 0,
    preprocessing_protocol: str = LEGACY_PREPROCESSING_PROTOCOL,
) -> Pipeline:
    features = radial_features(train, bins=bins, preprocessing_protocol=preprocessing_protocol)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
        ]
    )
    return model.fit(features, train["label"].astype(int))


def radial_predict(
    model: Pipeline,
    frame: pd.DataFrame,
    bins: int = 64,
    preprocessing_protocol: str = LEGACY_PREPROCESSING_PROTOCOL,
) -> np.ndarray:
    return model.predict_proba(
        radial_features(frame, bins=bins, preprocessing_protocol=preprocessing_protocol)
    )[:, 1]


def image_file_features(frame: pd.DataFrame) -> np.ndarray:
    """Return deliberately non-semantic file controls, never inputs to an image model.

    Geometry, encoded size, container and colour mode can expose acquisition-pipeline shortcuts.
    A strong score here is a dataset-bias warning, not detector evidence. Fixed one-hot columns
    keep the feature schema stable even when a split does not contain every observed category.
    """
    rows: list[tuple[float, ...]] = []
    for path in frame["path"]:
        image_path = Path(path)
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = (image.format or "OTHER").upper()
            image_mode = image.mode.upper()
        rows.append(
            (
                np.log1p(width),
                np.log1p(height),
                width / max(height, 1),
                np.log1p(image_path.stat().st_size),
                float(image_format == "JPEG"),
                float(image_format == "PNG"),
                float(image_format == "WEBP"),
                float(image_format not in {"JPEG", "PNG", "WEBP"}),
                float(image_mode == "RGB"),
                float(image_mode == "L"),
                float(image_mode == "RGBA"),
                float(image_mode not in {"RGB", "L", "RGBA"}),
            )
        )
    return np.asarray(rows, dtype=np.float32)


def fit_file_metadata_logistic(train: pd.DataFrame, seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
        ]
    ).fit(image_file_features(train), train["label"].astype(int))


def file_metadata_predict(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(image_file_features(frame))[:, 1]
