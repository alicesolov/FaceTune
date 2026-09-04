"""Strict, local-only loading and inference for a frozen research checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .features import (
    LEGACY_PREPROCESSING_PROTOCOL,
    FFTTransform,
    RGBTransform,
    preprocessing_metadata,
)
from .models import build_resnet50
from .reproducibility import get_device, sha256_file


class ExperimentLoadError(ValueError):
    """Raised when an experiment directory is incomplete or not eligible for serving."""


def preprocessing_from_run(run: dict[str, Any]) -> dict[str, object]:
    """Validate the exact eval rasterisation declared by a research run.

    The absence of this block identifies a checkpoint produced before H1-N and intentionally
    selects the legacy transform.  A partially specified block is rejected rather than silently
    changing the model's image domain at serving time.
    """
    declared = run.get("preprocessing")
    if declared is None:
        return preprocessing_metadata(LEGACY_PREPROCESSING_PROTOCOL)
    if not isinstance(declared, dict):
        raise ExperimentLoadError("run.json preprocessing metadata must be an object")
    protocol = declared.get("protocol")
    image_size = declared.get("image_size")
    version = declared.get("version")
    if not isinstance(protocol, str) or not isinstance(image_size, int) or not isinstance(version, str):
        raise ExperimentLoadError(
            "run.json preprocessing metadata must include string protocol/version and integer image_size"
        )
    try:
        expected = preprocessing_metadata(protocol, image_size)
    except ValueError as error:
        raise ExperimentLoadError(f"Unsupported checkpoint preprocessing: {error}") from error
    if version != expected["version"]:
        raise ExperimentLoadError(
            f"Checkpoint declares preprocessing version {version!r}, but this runtime supports "
            f"{expected['version']!r} for {protocol!r}"
        )
    return expected


def build_evaluation_transform(
    representation: str, preprocessing: dict[str, object]
) -> RGBTransform | FFTTransform:
    """Build the evaluation transform that exactly matches a validated run artifact."""
    protocol = preprocessing["protocol"]
    image_size = preprocessing["image_size"]
    if not isinstance(protocol, str) or not isinstance(image_size, int):
        raise ExperimentLoadError("Invalid validated preprocessing metadata")
    if representation == "rgb":
        return RGBTransform(size=image_size, train=False, preprocessing_protocol=protocol)
    if representation == "fft":
        return FFTTransform(size=image_size, train=False, preprocessing_protocol=protocol)
    raise ExperimentLoadError(f"Unsupported representation {representation!r}")


@dataclass
class ModelBundle:
    """One explicitly chosen checkpoint and its exact research preprocessing."""

    experiment_dir: Path
    representation: str
    threshold: float
    device: torch.device
    model: torch.nn.Module
    checkpoint_sha256: str
    metrics: dict[str, object]
    preprocessing: dict[str, object]
    transform: RGBTransform | FFTTransform

    @classmethod
    def load(cls, experiment_dir: str | Path, device_name: str = "auto") -> ModelBundle:
        directory = Path(experiment_dir).resolve()
        run_path = directory / "run.json"
        checkpoint_path = directory / "best_model.pt"
        metrics_path = directory / "internal_test_metrics.json"
        if not run_path.is_file() or not checkpoint_path.is_file():
            raise ExperimentLoadError(
                "Expected a completed experiment with run.json and best_model.pt. "
                "Do not serve a partial or arbitrary checkpoint."
            )
        run = json.loads(run_path.read_text(encoding="utf-8"))
        representation = run.get("config", {}).get("representation")
        threshold = run.get("threshold")
        if representation not in {"rgb", "fft"} or not isinstance(threshold, (int, float)):
            raise ExperimentLoadError("run.json lacks a supported representation or validation threshold")
        preprocessing = preprocessing_from_run(run)
        transform = build_evaluation_transform(representation, preprocessing)
        device = get_device(device_name)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ExperimentLoadError("Checkpoint payload has no state_dict")
        model = build_resnet50(pretrained=False)
        try:
            model.load_state_dict(payload["state_dict"])
        except RuntimeError as error:
            raise ExperimentLoadError("Checkpoint state_dict is incompatible with the declared model") from error
        model.to(device).eval()
        metrics: dict[str, object] = {}
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return cls(
            experiment_dir=directory,
            representation=representation,
            threshold=float(threshold),
            device=device,
            model=model,
            checkpoint_sha256=sha256_file(checkpoint_path),
            metrics=metrics,
            preprocessing=preprocessing,
            transform=transform,
        )

    def _transform(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)

    def predict(self, image: Image.Image) -> dict[str, object]:
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            score = float(torch.softmax(self.model(tensor), dim=1)[0, 1].detach().cpu())
        decision = "ai_like" if score >= self.threshold else "not_ai_like"
        return {
            "model_score": score,
            "threshold": self.threshold,
            "model_decision": decision,
            "representation": self.representation,
            "preprocessing": self.preprocessing,
            "experiment": self.experiment_dir.name,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
