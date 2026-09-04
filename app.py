"""Small local research interface for an explicitly selected frozen model."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from ai_image_detector.inference import ExperimentLoadError, ModelBundle
from ai_image_detector.uploads import MAX_UPLOAD_BYTES, load_image

app = FastAPI(title="AI Image Detector - research interface", docs_url=None, redoc_url=None)
bundle: ModelBundle | None = None
startup_error: str | None = None


@app.on_event("startup")
def load_selected_experiment() -> None:
    global bundle, startup_error
    directory = os.environ.get("AI_IMAGE_DETECTOR_EXPERIMENT_DIR")
    if not directory:
        startup_error = "Set AI_IMAGE_DETECTOR_EXPERIMENT_DIR to a completed, selected experiment."
        return
    try:
        bundle = ModelBundle.load(directory)
    except ExperimentLoadError as error:
        startup_error = str(error)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ready": bundle is not None, "error": startup_error}


@app.get("/api/model-card")
def model_card() -> dict[str, object]:
    if bundle is None:
        raise HTTPException(status_code=503, detail=startup_error or "No selected model")
    return {
        "experiment": bundle.experiment_dir.name,
        "representation": bundle.representation,
        "preprocessing": bundle.preprocessing,
        "threshold": bundle.threshold,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "internal_metrics": bundle.metrics,
        "limitations": [
            "The score is not a calibrated probability or proof of image origin.",
            "The Defactify benchmark has observed technical dataset bias.",
            "JPEG compression, resizing, blur, unseen generators and distribution shift can change results.",
        ],
    }


@app.post("/api/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    if bundle is None:
        raise HTTPException(status_code=503, detail=startup_error or "No selected model")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a raster image")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        image = load_image(contents)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return bundle.predict(image)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    state = "Model ready" if bundle else "Model is not selected yet"
    detail = "" if bundle else (startup_error or "")
    return f"""<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>AI Image Detector Research</title><style>body{{margin:0;background:#f5f6f8;color:#17212b;font:16px system-ui,sans-serif}}main{{max-width:760px;margin:8vh auto;padding:32px}}section{{background:#fff;border:1px solid #d9dee5;border-radius:18px;padding:28px;box-shadow:0 8px 28px #17212b0d}}h1{{margin-top:0}}button{{background:#123c69;border:0;border-radius:9px;color:white;padding:11px 16px;font-weight:650}}input{{margin:18px 0;display:block}}#result{{white-space:pre-wrap;background:#f1f4f7;border-radius:10px;padding:16px;min-height:22px}}small{{color:#52616f;line-height:1.5}}</style><main><section><p><strong>Research interface</strong></p><h1>AI image model score</h1><p id=\"state\">{state}</p><form id=\"form\"><input id=\"file\" type=\"file\" accept=\"image/*\" required><button>Evaluate image</button></form><pre id=\"result\"></pre><small>This is an experimental model score, not a probability or proof of origin. The decision uses a validation-selected threshold. Compression, resizing, new generators and dataset differences can change the result. Uploaded files are processed in memory and are not retained.</small><p><small>{detail}</small></p></section></main><script>const f=document.querySelector('#form'),r=document.querySelector('#result');f.addEventListener('submit',async e=>{{e.preventDefault();r.textContent='Evaluating…';const d=new FormData();d.append('file',document.querySelector('#file').files[0]);const x=await fetch('/api/predict',{{method:'POST',body:d}});const j=await x.json();r.textContent=x.ok?`Model signal: ${{j.model_decision}}\nScore: ${{j.model_score.toFixed(4)}}\nThreshold: ${{j.threshold.toFixed(4)}}\nRepresentation: ${{j.representation}}`:(j.detail||'Request failed');}});</script></html>"""
