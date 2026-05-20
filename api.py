"""FastAPI inference server for the AI image detector.

Endpoints:
  POST /predict      — Upload an image, get verdict + confidence
  GET  /health       — Liveness check
  GET  /model-info   — Checkpoint metadata

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from preprocessing.fft_transform import fft_preprocess

DEFAULT_WEIGHTS = Path("model/model.pth")
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.55

app = FastAPI(
    title="FaceTune AI Image Detector",
    description="Detects AI-generated images using FFT spectral analysis.",
    version="0.2.0",
)

# ── Model state (loaded once at startup) ─────────────────────────────────────

_model: torch.nn.Module | None = None
_model_meta: dict = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def _load_model() -> None:
    global _model, _model_meta
    if not DEFAULT_WEIGHTS.exists():
        return  # model not trained yet — /predict will return 503
    try:
        from training.model_factory import load_checkpoint_any
        _model, _model_meta = load_checkpoint_any(DEFAULT_WEIGHTS, device=_device)
    except Exception as exc:
        print(f"[WARN] Could not load model: {exc}")


# ── Response schemas ──────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    verdict: Literal["real", "ai_generated", "uncertain"]
    confidence: float           # P(ai_generated) after Bayes correction
    p_real: float
    p_ai: float
    model_type: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    model_type: str | None
    epoch: int | None
    val_f1: float | None
    val_fpr: float | None
    weights_path: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_hybrid() -> bool:
    return _model_meta.get("model_type") == "hybrid"


def _predict_image(image: Image.Image) -> tuple[float, float]:
    """Run inference. Returns (p_real, p_ai)."""
    fft_tensor = fft_preprocess(image).unsqueeze(0).to(_device)

    with torch.inference_mode():
        if _is_hybrid():
            try:
                from transformers import CLIPImageProcessor
                proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
                clip_pv = proc(images=image, return_tensors="pt")["pixel_values"].to(_device)
                logits = _model(fft_tensor, clip_pv)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Hybrid inference error: {exc}")
        else:
            logits = _model(fft_tensor)

    probs = F.softmax(logits, dim=1)[0]
    return float(probs[0].item()), float(probs[1].item())


def _verdict(p_ai: float) -> str:
    if p_ai < UNCERTAIN_LOW:
        return "real"
    if p_ai > UNCERTAIN_HIGH:
        return "ai_generated"
    return "uncertain"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Classify an uploaded image as real, ai_generated, or uncertain."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first: python -m training.train",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}")

    t0 = time.perf_counter()
    p_real, p_ai = _predict_image(image)
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        verdict=_verdict(p_ai),
        confidence=round(p_ai, 4),
        p_real=round(p_real, 4),
        p_ai=round(p_ai, 4),
        model_type=_model_meta.get("model_type", "unknown"),
        latency_ms=round(latency_ms, 1),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        device=str(_device),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_type=_model_meta.get("model_type"),
        epoch=_model_meta.get("epoch"),
        val_f1=_model_meta.get("val_f1"),
        val_fpr=_model_meta.get("val_fpr"),
        weights_path=str(DEFAULT_WEIGHTS),
    )
