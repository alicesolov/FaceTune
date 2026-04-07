"""Inference logic for editorial image authenticity screening.

This module supports real architecture construction and local checkpoint loading,
while still falling back safely when trained weights are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from model.resnet import create_resnet50_binary_classifier, load_binary_classifier_weights
from preprocessing.fft_transform import fft_preprocess


DEFAULT_WEIGHTS_PATH = Path("model/model.pth")
DEFAULT_UNCERTAIN_LOWER = 0.45
DEFAULT_UNCERTAIN_UPPER = 0.55


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction output for the Streamlit app.

    Purpose:
    Keep verdicts, confidence values, and explanatory notes in one object.

    Inputs:
    - verdict: Project verdict label.
    - confidence: Confidence score for the returned verdict.
    - probabilities: Mapping of class labels to probabilities when available.
    - note: Editor-facing explanation or limitation note.
    - used_placeholder: Whether the output came from placeholder logic.

    Outputs:
    - Immutable prediction result.

    Key assumptions:
    - Confidence values are in the range `[0, 1]`.

    Failure modes:
    - None in normal construction.
    """

    verdict: str
    confidence: float
    probabilities: dict[str, float]
    note: str
    used_placeholder: bool


class Predictor:
    """Predictor wrapper with real architecture wiring and safe fallback behavior.

    Purpose:
    Centralize preprocessing, model loading, and verdict mapping while the
    project is still missing trained weights.

    Inputs:
    - weights_path: Expected path to serialized model weights.
    - device: Torch device string.

    Outputs:
    - Predictor instance that can preprocess and produce a result object.

    Key assumptions:
    - The project may not have a trained checkpoint yet.

    Failure modes:
    - Weight loading failures are captured and surfaced through safe placeholder
      notes instead of pretending inference is complete.
    """

    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS_PATH, device: str = "cpu"):
        self.weights_path = Path(weights_path)
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None
        self.model_status_note = "Model weights are not available yet."
        self.using_placeholder = True
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Build the real architecture and load local weights if available.

        Purpose:
        Keep inference wiring honest: use the real architecture when possible,
        and use a documented placeholder path when no usable weights exist.

        Inputs:
        - None. Uses instance configuration.

        Outputs:
        - None. Updates predictor state in place.

        Key assumptions:
        - Missing or incompatible weights should not be misrepresented as a
          working trained model.

        Failure modes:
        - Captures import and checkpoint loading failures into
          `model_status_note` and leaves the predictor in placeholder mode.
        """

        try:
            model = create_resnet50_binary_classifier(pretrained=False)
        except Exception as error:
            self.model = None
            self.model_status_note = (
                "Model architecture could not be constructed. "
                f"Reason: {error}"
            )
            self.using_placeholder = True
            return

        if not self.weights_path.exists():
            self.model = None
            self.model_status_note = (
                "Model architecture is defined, but no checkpoint was found at "
                f"`{self.weights_path}`. Returning a safe placeholder result."
            )
            self.using_placeholder = True
            return

        try:
            model = load_binary_classifier_weights(model, str(self.weights_path))
            model.to(self.device)
            model.eval()
        except Exception as error:
            self.model = None
            self.model_status_note = (
                "A checkpoint exists, but it could not be loaded into the baseline "
                f"ResNet-50 architecture. Reason: {error}"
            )
            self.using_placeholder = True
            return

        self.model = model
        self.model_status_note = (
            f"Loaded checkpoint from `{self.weights_path}` into the baseline ResNet-50 architecture."
        )
        self.using_placeholder = False

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Run the documented FFT preprocessing pipeline.

        Purpose:
        Expose preprocessing separately so the app can inspect intermediate data.

        Inputs:
        - image: Uploaded PIL image.

        Outputs:
        - Batched tensor of shape `(1, 3, 256, 256)` on the configured device.

        Key assumptions:
        - The input image has already been validated as a readable image.

        Failure modes:
        - Propagates preprocessing errors for malformed inputs.
        """

        tensor = fft_preprocess(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Image.Image) -> PredictionResult:
        """Generate a safe prediction result for one uploaded image.

        Purpose:
        Run preprocessing and either execute real inference with a compatible
        checkpoint or return a safe placeholder result when no checkpoint is
        available.

        Inputs:
        - image: Uploaded PIL image.

        Outputs:
        - `PredictionResult` for display in the app.

        Key assumptions:
        - Placeholder mode should prefer a cautious `uncertain` result.
        - Real confidence values are only emitted when logits come from a loaded
          checkpoint.

        Failure modes:
        - Propagates preprocessing errors if the upload is not a valid image.
        """

        model_input = self.preprocess(image)

        if self.model is None:
            return PredictionResult(
                verdict="uncertain",
                confidence=0.0,
                probabilities={"real": 0.0, "ai_generated": 0.0},
                note=self.model_status_note,
                used_placeholder=True,
            )

        with torch.inference_mode():
            logits = self.model(model_input)
            probabilities_tensor = F.softmax(logits, dim=1)[0].detach().cpu()

        real_probability = float(probabilities_tensor[0].item())
        ai_generated_probability = float(probabilities_tensor[1].item())
        verdict = map_binary_probability_to_verdict(ai_generated_probability)
        confidence = max(real_probability, ai_generated_probability)

        return PredictionResult(
            verdict=verdict,
            confidence=confidence,
            probabilities={
                "real": real_probability,
                "ai_generated": ai_generated_probability,
            },
            note=self.model_status_note,
            used_placeholder=False,
        )


def map_binary_probability_to_verdict(
    ai_generated_probability: float,
    lower: float = DEFAULT_UNCERTAIN_LOWER,
    upper: float = DEFAULT_UNCERTAIN_UPPER,
) -> str:
    """Map a binary probability into `real`, `ai_generated`, or `uncertain`.

    Purpose:
    Centralize threshold mapping for later real inference logic.

    Inputs:
    - ai_generated_probability: Probability assigned to the AI-generated class.
    - lower: Lower bound of the uncertain band.
    - upper: Upper bound of the uncertain band.

    Outputs:
    - Project verdict string.

    Key assumptions:
    - Values below `lower` indicate `real`.
    - Values above `upper` indicate `ai_generated`.
    - Values inside the band indicate `uncertain`.

    Failure modes:
    - Raises `ValueError` for invalid thresholds or probabilities.
    """

    if not 0.0 <= ai_generated_probability <= 1.0:
        raise ValueError("ai_generated_probability must be between 0 and 1")
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= lower <= upper <= 1")

    if ai_generated_probability < lower:
        return "real"
    if ai_generated_probability > upper:
        return "ai_generated"
    return "uncertain"
