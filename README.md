# AI Image Detector - reproducible research and local evaluation interface

This repository contains the **research and training stage** of a coursework project about
distinguishing real photographs from AI-generated images. It also contains a local interface for a
**separately selected, frozen research checkpoint**, but the research result remains primary. The
point is a reproducible answer to narrowly stated hypotheses, including cases in which an FFT-based
method fails.

## Research question

After source-normalised square-crop rasterisation, does an FFT-magnitude representation improve a
binary real-versus-synthetic classifier relative to an RGB model of equal capacity and budget?

The original Defactify control discovered a serious geometry/source shortcut: all AI images are
square while real photographs vary in aspect ratio. The amendment therefore replaces direct
rectangular-to-square resizing with **source square crop -> common 128x128 LANCZOS raster**, then
calculates FFT only from that shared raster. Read
[the amendment](docs/PROTOCOL_AMENDMENT_GEOMETRY_CONTROL.md) before interpreting any result.

The planned evidence is split into three result families and must never be merged into one score:

| Result family | Data | Permitted conclusion |
| --- | --- | --- |
| Internal | Defactify grouped test | Exploratory stress test after D0; never a provenance claim. |
| External | Synthbuster + RAISE-1k | Confirmatory transfer evidence once the model is frozen. |
| Robustness | Deterministic JPEG, resize, blur transforms | Sensitivity to specified file transformations. |

An internal test accuracy is not evidence that a model can authenticate arbitrary images.

## Reproducible setup

The project deliberately pins Python to 3.12 because the host's default Python may be newer than
the currently supported PyTorch wheel. On macOS, PyTorch will use MPS when available and fall back
to CPU otherwise.

```bash
uv venv --python 3.12
uv sync --extra dev
uv run python -m ipykernel install --user --name ai-image-detector-research --display-name "AI Image Detector Research"
uv run jupyter lab
```

Run notebooks in numerical order. Each notebook either creates a versioned artifact or reads one
created by an earlier notebook. No notebook contains invented metrics; empty result tables remain
empty until a run produces them.

## H1-N experiment commands

The full commands are intentionally explicit. They use the controlled protocol by default; the
legacy resize exists only to reproduce the D0 diagnostic and must not be called a detector result.

```bash
# Pixel-only baseline under the amended rasterisation.
uv run python scripts/run_baselines.py \
  --manifest data/processed/defactify_grouped/manifest.csv \
  --only radial_fft_logistic --output-root artifacts/h1n_controls --seed 7

# Run the equal-budget RGB and FFT comparison for seeds 7, 17, and 42.
uv run python scripts/run_experiment.py \
  --manifest data/processed/defactify_grouped/manifest.csv \
  --representation rgb --output-dir artifacts/h1n_rgb_resnet50_seed7 \
  --seed 7 --epochs 15 --batch-size 32 --learning-rate 1e-4 --patience 4 --from-scratch

uv run python scripts/analyze_predictions.py \
  --experiment-dir artifacts/h1n_rgb_resnet50_seed7 --bootstrap-repeats 2000
```

Run the matching FFT command with `--representation fft` and a distinct output directory. Only
after all three predeclared seeds finish, aggregate them with `scripts/aggregate_experiments.py`;
that script rejects a single run and never picks a best test-set seed.

## Data provenance and acquisition

The initial train/validation/internal-test source is
[Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset):
42k / 9k / 45k records. Its Hugging Face card did not state an aggregate license at the time this
protocol was written; record the source revision and do not redistribute raw images from this
repository. `Synthbuster + RAISE-1k` is a separately held-out external benchmark; it must not enter
training, model selection, threshold selection or augmentation selection. The preparation script
does not download it automatically because RAISE-1k requires acceptance of its research licence.
See [data/README.md](data/README.md) and [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md).

## Layout

```text
notebooks/                  # Executable research narrative
src/ai_image_detector/      # Reusable data, feature, training and evaluation code
scripts/                    # Explicit command-line entry points
tests/                      # Unit and smoke tests for claims made by the pipeline
data/manifests/             # Small provenance and split CSV files only
artifacts/                  # Ignored runs, checkpoints, predictions and figures
reports/generated/          # Ignored automatically generated results tables
```

## Non-claims

The system does not detect faces or video deepfakes, does not prove image provenance, and must not
label softmax output as a calibrated probability before calibration has been evaluated. See the
protocol for exact hypotheses, metrics and threats to validity.

## Local research interface

After a completed experiment has been explicitly selected, serve it locally with the exact
preprocessing and validation-selected threshold used in research:

```bash
AI_IMAGE_DETECTOR_EXPERIMENT_DIR=artifacts/<selected-experiment> .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

The interface never stores uploaded images and calls the output a **model score**, not a
probability or proof of provenance. It deliberately refuses to start without a completed selected
experiment, so the smoke-test checkpoint cannot accidentally be presented as a working detector.
It is intentionally local: serving the PyTorch model needs the same runtime as the research
environment; this repository does not deploy it to a public web host.
