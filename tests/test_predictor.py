"""Tests for the inference predictor: placeholder fallback and verdict mapping."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from model.predict import (
    Predictor,
    PredictionResult,
    map_binary_probability_to_verdict,
)


def _random_image(width: int = 64, height: int = 64) -> Image.Image:
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── map_binary_probability_to_verdict ─────────────────────────────────────────

class TestVerdictMapping:
    def test_low_probability_is_real(self):
        assert map_binary_probability_to_verdict(0.1) == "real"

    def test_high_probability_is_ai_generated(self):
        assert map_binary_probability_to_verdict(0.9) == "ai_generated"

    def test_mid_range_is_uncertain(self):
        assert map_binary_probability_to_verdict(0.5) == "uncertain"

    def test_lower_bound_is_uncertain(self):
        assert map_binary_probability_to_verdict(0.45) == "uncertain"

    def test_upper_bound_is_uncertain(self):
        assert map_binary_probability_to_verdict(0.55) == "uncertain"

    def test_just_below_lower_is_real(self):
        assert map_binary_probability_to_verdict(0.44) == "real"

    def test_just_above_upper_is_ai_generated(self):
        assert map_binary_probability_to_verdict(0.56) == "ai_generated"

    def test_zero_is_real(self):
        assert map_binary_probability_to_verdict(0.0) == "real"

    def test_one_is_ai_generated(self):
        assert map_binary_probability_to_verdict(1.0) == "ai_generated"

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            map_binary_probability_to_verdict(1.5)

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError):
            map_binary_probability_to_verdict(-0.1)

    def test_inverted_thresholds_raise(self):
        with pytest.raises(ValueError):
            map_binary_probability_to_verdict(0.5, lower=0.7, upper=0.3)

    def test_custom_thresholds(self):
        assert map_binary_probability_to_verdict(0.6, lower=0.4, upper=0.7) == "uncertain"
        assert map_binary_probability_to_verdict(0.3, lower=0.4, upper=0.7) == "real"
        assert map_binary_probability_to_verdict(0.8, lower=0.4, upper=0.7) == "ai_generated"


# ── Predictor — no weights (placeholder mode) ─────────────────────────────────

class TestPredictorNoWeights:
    def setup_method(self):
        self.predictor = Predictor(weights_path="nonexistent_model_weights.pth")

    def test_uses_placeholder(self):
        assert self.predictor.using_placeholder is True

    def test_model_is_none(self):
        assert self.predictor.model is None

    def test_predict_returns_uncertain(self):
        result = self.predictor.predict(_random_image())
        assert result.verdict == "uncertain"

    def test_predict_returns_zero_confidence(self):
        result = self.predictor.predict(_random_image())
        assert result.confidence == 0.0

    def test_predict_used_placeholder_flag(self):
        result = self.predictor.predict(_random_image())
        assert result.used_placeholder is True

    def test_predict_returns_prediction_result_type(self):
        result = self.predictor.predict(_random_image())
        assert isinstance(result, PredictionResult)

    def test_predict_probabilities_are_zero(self):
        result = self.predictor.predict(_random_image())
        assert result.probabilities["real"] == 0.0
        assert result.probabilities["ai_generated"] == 0.0

    def test_note_mentions_checkpoint(self):
        result = self.predictor.predict(_random_image())
        assert isinstance(result.note, str)
        assert len(result.note) > 0

    def test_preprocess_returns_correct_shape(self):
        import torch
        tensor = self.predictor.preprocess(_random_image())
        assert tensor.shape == (1, 3, 256, 256)
        assert tensor.dtype == torch.float32
