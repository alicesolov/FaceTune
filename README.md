# AI Image Detector - reproducible research and local evaluation interface

This repository contains the **research and training stage** of a coursework project about
distinguishing real photographs from AI-generated images. It also contains a local interface for a
**separately selected, frozen research checkpoint**, but the research result remains primary. The
point is a reproducible answer to narrowly stated hypotheses, including cases in which an FFT-based
method fails.

There are two deliberately separate research tracks:

- **H1-N / Defactify 128 x 128** — a historical low-resolution pilot that established geometry
  controls. Its planned multi-seed series was stopped before completion and is not eligible for
  model selection, external validation or the interface.
- **HighRes-v1 / 384 x 384** — the primary study now being assembled from a revision-pinned,
  audited high-quality source. It begins with a metadata-only source catalog; no high-resolution result
  is claimed until the source, split and materialised files have passed the protocol.

## Research questions and status

The historical H1-N question was whether an FFT-magnitude representation improved a binary
real-versus-synthetic classifier relative to RGB under source-normalised rasterisation. Its scope
and limitations remain in [the preserved Defactify protocol](docs/RESEARCH_PROTOCOL.md).

The primary question is now recorded in [HighRes-v1](docs/HIGHRES_V1_PROTOCOL.md): under a common
384 x 384 raster made from source images of at least 512 pixels, what evidence holds internally and
on locked external tests? This is a different study, not an enlarged Defactify command.

The eventual evidence is split into distinct result families and must never be merged into one
score:

| Result family | Data | Permitted conclusion |
| --- | --- | --- |
| Historical pilot | Defactify grouped 128 x 128 | Exploratory control only; never a provenance claim or model-selection result. |
| HighRes internal | Frozen HighRes-v1 split | Evidence for the named corpus after data and leakage audit. |
| External | Synthbuster + RAISE-1k | Confirmatory transfer evidence only once a HighRes-v1 model is frozen. |
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

There are deliberately only two notebooks, to keep the research path easy to follow:

1. `notebooks/01_internal_training.ipynb` — environment, data audit, controlled preprocessing,
   baseline, neural training, and exploratory internal analysis. It will narrate HighRes-v1 after
   its manifest is frozen; the H1-N cells remain an explicitly labelled historical pilot.
2. `notebooks/02_external_validation_and_results.ipynb` — locked external validation, robustness,
   aggregation, and final limitations. Its commands remain gated until HighRes-v1 is frozen.

Run them in that order. Each notebook either creates a versioned artifact or reads one created by
an earlier stage. No notebook contains invented metrics; empty result tables remain empty until a
run produces them.

The current [evidence-bound coursework draft](reports/draft/ai_image_detector_coursework_draft.md)
is the companion narrative: it labels observed results, diagnostic controls, and pending work
separately rather than filling future tables in advance.

## Historical H1-N commands

These commands are retained to reproduce existing pilot artifacts only. Do **not** launch the
remaining H1-N seed/representation runs; HighRes-v1 has its own data and preprocessing contract.
The legacy resize exists only to reproduce the D0 diagnostic and must not be called a detector
result.

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

The baseline and training launchers treat their output directories as archival records and refuse
to overwrite them. For a deliberate reproduction, choose a new `--output-root` or `--output-dir`
rather than replacing the canonical artifact.

Run the matching FFT command with `--representation fft` and a distinct output directory. Only
after all three predeclared seeds finish, aggregate them with `scripts/aggregate_experiments.py`;
that script rejects a single run and never picks a best test-set seed.

## Data provenance and acquisition

The historical source is [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset).
It remains preserved for the 128 x 128 pilot only; it is not enlarged by upsampling.

The HighRes-v1 train/validation candidate reservoir is
[CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small), pinned
to revision `6c539a534c07917307c381f5af4053c6091b5278`. The new
`scripts/scan_community_forensics_metadata.py` stage reads only Parquet metadata first and creates
an ignored local source lock/source catalog. It deliberately excludes binary image data and
does not make a partial scout eligible for training. See [HighRes-v1](docs/HIGHRES_V1_PROTOCOL.md)
and [data/README.md](data/README.md) before materialising any image.

```bash
# Metadata only: full source catalog, source lock, and provenance. It does not download image bytes.
.venv/bin/python scripts/scan_community_forensics_metadata.py \
  --output-dir data/processed/community_forensics_small_metadata_v1
```

The scanner stores only temporary Hugging Face locks/cache under ignored
`artifacts/cache/huggingface` by default. `--limit-shards N` is only a transport smoke test: its
catalogue is automatically marked partial and cannot be used for selection or training.

`Synthbuster + RAISE-1k` is a separately held-out external benchmark; it must not enter training,
model selection, threshold selection or augmentation selection. The preparation script does not
download it automatically so the benchmark stays locked; RAISE states that its images are for
non-commercial research and educational use. See the [official RAISE terms]
(https://loki.disi.unitn.it/RAISE/download.html).

## Apple GPU readiness

The MPS readiness gate is a reproducible hardware measurement, separate from model evaluation. It
compares fixed synthetic ResNet-50 training steps on CPU and MPS, records actual parameter device,
allocator snapshots and relevant environment settings, and refuses ambiguous MPS fallback:

```bash
.venv/bin/python scripts/benchmark_device.py \
  --image-size 384 --batch-size 64 --warmup-steps 10 --timed-steps 30 --trials 3 \
  --output artifacts/benchmarks/resnet50_384_cpu_vs_mps_b64_v1.json
```

Do not treat its throughput as end-to-end data-loader performance or as evidence of detector
quality.

## Repository map

The repository keeps command-line entry points flat because each one is a separate reproducible
stage, not a throwaway helper:

| Stage | Entry points |
| --- | --- |
| Acquire and audit internal data | `prepare_defactify.py`, `audit_manifest.py`, `make_grouped_split.py` |
| Audit HighRes-v1 source metadata | `scan_community_forensics_metadata.py` |
| Train and analyse internal models | `run_baselines.py`, `run_experiment.py`, `analyze_predictions.py`, `aggregate_experiments.py` |
| Frozen validation only | `prepare_synthbuster_external.py`, `evaluate_external.py`, `evaluate_robustness.py` |

An earlier standalone annotation helper was removed because `prepare_defactify.py` now creates the
same declared generator field and consistency check itself.

```text
notebooks/                  # Two runnable research narratives: internal, then external/results
src/ai_image_detector/      # Reusable data, feature, training and evaluation code
scripts/                    # Explicit pipeline entry points listed above
tests/                      # Unit and smoke tests for claims made by the pipeline
data/                       # Policy and small provenance/manifests only; raw data is ignored
artifacts/                  # Ignored runs, checkpoints, predictions and figures
reports/draft/              # Evidence-bound coursework draft in source form
reports/generated/          # Ignored rendered report deliverables
```

## Why the stages are separate

`prepare_*`, `run_*`, and `evaluate_*` scripts intentionally do not call one another implicitly.
This lets a student inspect a manifest before training, preserve a failed/negative result, and lock
external data before model selection. The README and the two notebooks are the normal entry points;
scripts are there for repeatable execution or automation.

## Non-claims

The system does not detect faces or video deepfakes, does not prove image provenance, and must not
label softmax output as a calibrated probability before calibration has been evaluated. See the
protocol for exact hypotheses, metrics and threats to validity.

## Local research interface

After a completed experiment has been explicitly selected and externally validated, serve it
locally with the exact preprocessing and validation-selected threshold used in research:

```bash
AI_IMAGE_DETECTOR_SELECTION_RECORD=artifacts/model_selections/<selected-model>.json .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

The interface never stores uploaded images and calls the output a **model score**, not a
probability or proof of provenance. It deliberately refuses to start without a hash-pinned,
`frozen_external_validated` selection record. The JSON record must use schema
`ai_image_detector_model_selection_v1` and contain `experiment_dir`, `checkpoint_sha256`, and the
required `selection_status`; the hash must match the selected experiment checkpoint. This prevents the
smoke-test or a legacy/partial checkpoint from being accidentally presented as a working detector.
The historical H1-N selection rule was never activated because its multi-seed series was stopped.
A future HighRes-v1 selection record will require its own hash-pinned preprocessing, manifest and
external-validation evidence; neither an internal test nor external scores may choose a more
flattering checkpoint.
It is intentionally local: serving the PyTorch model needs the same runtime as the research
environment; this repository does not deploy it to a public web host.
