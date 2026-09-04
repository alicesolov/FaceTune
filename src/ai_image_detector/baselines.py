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
    LEGACY_PREPROCESSING_PROTOCOL,
    preprocessing_metadata,
    radial_power_spectrum,
    source_normalized_rasterize,
)


def radial_features(
    frame: pd.DataFrame,
    bins: int = 64,
    preprocessing_protocol: str = LEGACY_PREPROCESSING_PROTOCOL,
) -> np.ndarray:
    """Compute radial FFT features under an explicitly declared rasterisation contract."""
    preprocessing = preprocessing_metadata(preprocessing_protocol)
    image_size = int(preprocessing["image_size"])
    rows: list[np.ndarray] = []
    for path in tqdm(frame["path"], desc="radial FFT features"):
        with Image.open(Path(path)) as image:
            if preprocessing_protocol == CONTROLLED_PREPROCESSING_PROTOCOL:
                image = source_normalized_rasterize(image, size=image_size, train=False)
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

    Files are canonical RGB PNGs. Consequently `png_bytes` is not original file metadata, but it
    can reveal compression/content correlations introduced by a source corpus. A strong score here
    is a dataset-bias warning, not a useful detector result.
    """
    rows: list[tuple[float, float, float, float]] = []
    for path in frame["path"]:
        image_path = Path(path)
        with Image.open(image_path) as image:
            width, height = image.size
        rows.append(
            (
                np.log1p(width),
                np.log1p(height),
                width / max(height, 1),
                np.log1p(image_path.stat().st_size),
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
