# TASKS.md

## Current objective
Build an MVP for detecting whether an editorial image is likely real or AI-generated for newsroom screening.

## Active priorities

### 1. Dataset follow-up validation
- Confirm the dataset license and record the exact license text or identifier in project docs
- Run a dataset composition audit across major content types relevant to newsroom use
- Run a portrait / close-up face bias audit to measure how much of the dataset is dominated by portrait-like imagery
- Inspect class balance for `Label_A` across `train`, `validation`, and `test`
- Check whether captions or near-duplicate images could create train/test leakage risk
- Write down strategy options if the dataset is too portrait-heavy for broad newsroom use

### 2. Strategy options if portrait-heavy bias is confirmed
- Keep the dataset as-is, but document the bias and treat portrait-heavy performance as a known limitation
- Rebalance the training data with additional non-portrait editorial imagery
- Create evaluation slices for portraits, close-up faces, and non-portrait editorial scenes
- Consider a fallback or supplemental dataset if coverage of general editorial scenes is too weak

### 3. First implementation tasks
- Extend `training/dataset.py` from loader skeleton to full dataset inspection utilities
- Add dataset sanity checks for missing images, invalid labels, and empty splits during real split loading
- Add a small inspection script or utility that prints split sizes and label counts
- Add tests for dataset config validation and `Label_A` normalization

### 4. Preprocessing tasks
- Add smoke tests for FFT output shape, dtype, value range, and deterministic behavior
- Decide whether inference and training should share any additional tensor normalization beyond the current FFT pipeline

### 5. Baseline model and inference follow-up
- Verify checkpoint format assumptions for `model/model.pth` and document them
- Add tests for `model/resnet.py` construction and state-dict loading
- Add tests for `model/predict.py` threshold mapping and placeholder fallback behavior
- Decide whether the baseline should use torchvision ImageNet initialization during training

### 6. Documentation tasks
- Keep `README.md` aligned with any new dataset findings
- Keep `DONE.md` updated after each completed phase
- Record any dataset mismatch with the broad newsroom/editorial scope instead of silently working around it

### 7. App follow-up
- Improve the Streamlit app with supported-format validation, clearer result copy, and diagnostics tabs
- Add smoke tests for uploaded-image handling and end-to-end placeholder inference
- Show clearer editor-facing wording for `real`, `ai_generated`, and `uncertain`

## Open questions
- If the dataset license is restrictive for newsroom/demo use, what fallback dataset should replace it?
- Do we optimize more for precision or recall in newsroom screening once training starts?
- What threshold policy should define `uncertain` later in inference?

## Definition of next milestone
The next milestone is reached when dataset follow-up checks are complete, the dataset composition and portrait-bias audits are documented, the loader and baseline inference layer have smoke-test coverage, and the FFT preprocessing pipeline is validated by tests.
