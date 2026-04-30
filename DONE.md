# DONE.md

## Initial project setup

### Completed
- Defined project scope for editorial AI-image detection
- Defined non-goals around face-specific analysis, identity verification, and facial analysis
- Selected a working candidate dataset direction: `Rajarshi-Roy-research/Defactify_Image_Dataset`
- Defined baseline stack: Python + PyTorch + Streamlit
- Defined baseline preprocessing: FFT-based pipeline
- Defined baseline model direction: ResNet-50
- Created project management files: `AGENTS.md`, `TASKS.md`, `DONE.md`

### Files changed
- `AGENTS.md`
- `TASKS.md`
- `DONE.md`

### Major decisions
- The tool is positioned as a newsroom screening assistant, not a forensic truth machine
- The system should support three outcomes: `real`, `ai_generated`, `uncertain`
- Explainability and editorial caution are core requirements
- The project should prefer `uncertain` over overconfident wrong predictions

### Limitations
- Dataset had not yet been inspected at this initial stage
- No training code has been implemented yet
- No metrics have been measured yet
- Threshold policy is still undecided
- Robustness to recompression, screenshots, and reposting is not yet validated

---

## Phase 1 dataset review

### Completed
- Reviewed `AGENTS.md`, `TASKS.md`, and `DONE.md` before making changes
- Inspected the planned dataset choice `Rajarshi-Roy-research/Defactify_Image_Dataset`
- Confirmed the published splits: `train` 42,000, `validation` 9,000, `test` 45,000
- Confirmed the published fields: `Caption`, `Image`, `Label_A`, `Label_B`
- Confirmed `Label_A` is the binary authenticity label and `Label_B` is the generator-source label
- Created a root `README.md` with a short dataset section

### Files changed
- `README.md`
- `TASKS.md`
- `DONE.md`

### Major decisions
- Keep `Rajarshi-Roy-research/Defactify_Image_Dataset` as the current candidate primary dataset
- Keep a broad newsroom scope: images with people and faces remain in scope
- Treat the dataset as suitable for the initial binary task

### Limitations
- Dataset license not yet explicitly recorded
- Dataset composition not yet audited
- Portrait and close-up face prevalence not measured
- No model code, preprocessing code, or tests implemented

---

## Documentation scope update

### Completed
- Revised `README.md` to reflect the broad newsroom scope
- Added portrait/face bias audit tasks
- Aligned `AGENTS.md` with the updated scope

### Files changed
- `AGENTS.md`, `README.md`, `TASKS.md`, `DONE.md`

---

## Foundation layer implementation

### Completed
- Implemented `training/dataset.py` loader skeleton
- Implemented deterministic FFT preprocessing in `preprocessing/fft_transform.py`
- Implemented `model/predict.py` with placeholder-safe model loading
- Implemented minimal `app.py` Streamlit UI

### Files changed
- `training/dataset.py`, `preprocessing/fft_transform.py`, `model/predict.py`, `app.py`

---

## Baseline model wiring

### Completed
- Added `model/resnet.py` with the baseline ResNet-50 binary classifier constructor
- Wired `model/predict.py` to build the real architecture and load local checkpoints
- Updated `app.py` to reflect real architecture wiring

### Files changed
- `model/resnet.py`, `model/predict.py`, `app.py`, `README.md`, `TASKS.md`, `DONE.md`

### Major decisions
- Only run real inference when a local checkpoint exists and can be loaded
- Fall back to explicit placeholder `uncertain` result when weights are missing

---

## Production training pipeline and full MVP

### Completed
- Added `DefactifyTorchDataset` PyTorch Dataset wrapper to `training/dataset.py`
  — applies FFT preprocessing per item, exposes `get_all_labels()` for sampler construction
- Created `training/train.py` — full supervised training loop with:
  - `CrossEntropyLoss` (equivalent to BCEWithLogitsLoss for 2-class heads)
  - `Adam` optimizer + `StepLR` scheduler (halve LR every 3 epochs)
  - Class-balanced `WeightedRandomSampler` to handle label skew
  - Per-epoch accuracy, precision, recall, F1 logging for both splits
  - Best-checkpoint saving to `model/model.pth`
  - GPU/CPU auto-detection
  - CLI: `python -m training.train --epochs 10 --batch-size 32 --device auto`
- Created `training/audit.py` — pre-training dataset audit covering:
  - Class balance (real vs ai_generated) with imbalance ratio warning
  - Duplicate caption detection with top-5 report
  - Generator-family distribution (Label_B mapping to SD2.1/SDXL/SD3/DALL-E 3/Midjourney)
  - CLI: `python -m training.audit --split train`
- Created `training/evaluate.py` — post-training evaluation covering:
  - Accuracy, precision, recall, F1 on any split
  - Confusion matrix with labelled rows/columns
  - Per-generator-family recall slice for ai_generated images
  - CLI: `python -m training.evaluate --split test`
- Created `preprocessing/visualizations.py`:
  - `render_fft_colormap` — viridis-colorized FFT spectrum as uint8 RGB
  - `render_fft_grayscale` — plain grayscale fallback
  - `compute_power_spectrum_stats` — mean, std, centre energy, edge energy
- Created `tests/` with three test modules:
  - `tests/test_fft.py` — shape, dtype, value range, determinism, channel identity, input types
  - `tests/test_dataset.py` — DatasetConfig, normalize_label_a, validate_dataset_columns, DefactifyTorchDataset
  - `tests/test_predictor.py` — verdict mapping (12 cases), placeholder predictor behaviour
- Created `requirements.txt`
- Rewrote `app.py` with three-tab layout:
  - **Prediction** — verdict badge (green/red/orange), confidence metric, editor copy, placeholder notice
  - **FFT Visualization** — colorized spectrum + grayscale, spectrum statistics expander
  - **Diagnostics** — class probability bar chart, confidence progress bar, model status + training instructions
  - Sidebar: model status indicator, upload widget, editorial disclaimer

### Files changed
- `training/dataset.py`
- `training/train.py` (new)
- `training/audit.py` (new)
- `training/evaluate.py` (new)
- `preprocessing/visualizations.py` (new)
- `tests/__init__.py` (new)
- `tests/test_fft.py` (new)
- `tests/test_dataset.py` (new)
- `tests/test_predictor.py` (new)
- `requirements.txt` (new)
- `app.py`
- `TASKS.md`
- `DONE.md`

### Major decisions
- `CrossEntropyLoss` is used instead of `BCEWithLogitsLoss`: the model outputs 2 logits per
  sample, so CrossEntropyLoss is the mathematically correct choice. For binary problems
  with 2 output nodes these losses are equivalent; switching to BCEWithLogitsLoss would
  require changing the model head to 1 output and all dependent inference code.
- `WeightedRandomSampler` is applied on the train split by default; if the audit confirms
  near-perfect balance, the sampler is harmless.
- `num_workers=0` is hard-coded in DataLoader calls because Windows does not support
  fork-based multiprocessing for PyTorch + HuggingFace datasets.
- ImageNet-pretrained backbone is used by default (`pretrained=True`) as the FFT
  channel structure (3 identical grayscale repeats) is still close enough to RGB for
  pretrained low-level filters to provide a useful initialisation.

### Limitations
- The dataset has not yet been downloaded and audited locally; `training/audit.py` must
  be run before training to confirm class balance and duplicate risk.
- No trained `model/model.pth` checkpoint exists yet — inference still falls back to the
  safe `uncertain` placeholder until `training/train.py` is run.
- `num_workers=0` makes dataloader I/O single-threaded; on Linux this can be increased
  for significant speed gains.
- Robustness to recompression, screenshots, and social-media degradation is not yet
  evaluated.
- No `.streamlit/config.toml` theming has been added yet.

### Audit results (all-splits, confirmed)

| Metric | Train | Validation | Test |
|---|---|---|---|
| Rows | 42,000 | 9,000 | 45,000 |
| Unique captions | 6,948 | 1,495 | 1,500 |
| Versions per scene | ~6 | ~6 | **30** |
| Real % | 16.7% | 16.7% | 16.7% |
| AI % | 83.3% | 83.3% | 83.3% |
| Imbalance ratio | 5.0× | 5.0× | 5.0× |
| Cross-split leakage | train↔val: 25, train↔test: 31, val↔test: 7 | — | — |

Key conclusions:
- Dataset is N unique scenes × 6 versions (real + SD2.1 + SDXL + SD3 + DALL-E 3 + Midjourney)
- 5:1 class imbalance — WeightedRandomSampler is non-negotiable
- 55 leaked captions total (~0.4% of unique scenes) — acceptable for baseline
- Test has 30× versions per scene (likely 5 seeds × generators) — inflates size, does not affect eval fairness
- Generator distribution perfectly balanced at 7,000/split (train), 1,500 (val/test)

### Next follow-up
- ✅ `python -m training.audit --all-splits` — completed, findings above
- Run `python -m training.train` to produce the first `model/model.pth` — **in progress**
- Run `python -m training.evaluate --split test` and record metrics in README
- Add robustness evaluation slices (recompressed JPEG, cropped, resized)
- Record the dataset license explicitly

---

## Iteration 2 — training pipeline improvements + app UX overhaul

### Completed
- Extended `training/audit.py`:
  - Added `audit_cross_split_leakage()` — detects caption overlap across all split pairs
  - Added `run_pipeline_audit()` — loads all 3 splits and runs full audit in one command
  - Added `--all-splits` CLI flag: `python -m training.audit --all-splits`
- Improved `training/train.py`:
  - Added `EarlyStopping` class (patience=3, min_delta=1e-4) — halts when val F1 stagnates
  - Best checkpoint now selected by **val F1** instead of val accuracy (more robust under imbalance)
  - Mixed precision training (torch.cuda.amp) enabled automatically on CUDA
  - Added `--patience` CLI flag
- Rewrote `app.py` with major UX improvements:
  - Probability gauge (matplotlib): visual bar with REAL / uncertain / AI zones + live marker
  - Session history in sidebar: tracks last 10 images analyzed in the session
  - Editorial guidance per verdict: plain-language explanation + recommended action list
  - Checkpoint quality panel in Diagnostics: epoch, val F1, accuracy, precision, recall
  - Verdict guide on landing page; three-step workflow explanation
  - Better model status copy in sidebar with checkpoint F1 shown
- Added `model/efficientnet.py`:
  - EfficientNet-B4 drop-in alternative to ResNet-50
  - Same checkpoint format — use after baseline to compare val F1 head-to-head

### Files changed
- `training/audit.py`
- `training/train.py`
- `app.py`
- `model/efficientnet.py` (new)
- `TASKS.md`
- `DONE.md`

### Major decisions
- Val F1 is now the checkpoint selection criterion: for binary classification under
  potential imbalance, F1 is more informative than accuracy (accuracy can be high even
  when the minority class is ignored).
- Early stopping patience=3 is conservative enough to allow the scheduler to take effect
  (LR halves at epoch 3, 6, 9) without letting training continue indefinitely.
- Mixed precision is opt-in at the device level (auto-enabled on CUDA, silent on CPU).
- EfficientNet-B4 is provided as a separate file rather than replacing ResNet-50, so
  the baseline result remains reproducible.

### Limitations
- No augmentation pipeline yet — all images are resized to 256×256 and FFT-transformed
  with no stochastic transforms (flip, crop, colour jitter). This may hurt generalisation.
- EfficientNet-B4 has not been trained or benchmarked yet; it is ready to run.
- The app session history is not persisted across browser reloads.
