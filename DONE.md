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

### Next follow-up
- Inspect the candidate dataset and document its structure
- Implement the first reproducible dataset loader
- Build FFT preprocessing and a minimal inference-ready pipeline stub

## Phase 1 dataset review

### Completed
- Reviewed `AGENTS.md`, `TASKS.md`, and `DONE.md` before making changes
- Inspected the planned dataset choice `Rajarshi-Roy-research/Defactify_Image_Dataset` from the Hugging Face dataset card and raw dataset `README.md`
- Confirmed the published splits: `train` 42,000, `validation` 9,000, `test` 45,000
- Confirmed the published fields: `Caption`, `Image`, `Label_A`, `Label_B`
- Confirmed `Label_A` is the binary authenticity label and `Label_B` is the generator-source label
- Created a root `README.md` with a short dataset section because `README.md` was missing in the repository
- Updated `TASKS.md` with concrete next implementation tasks for dataset loading and FFT preprocessing
- Updated `DONE.md` to reflect this documentation-only phase

### Files changed
- `README.md`
- `TASKS.md`
- `DONE.md`

### Major decisions
- Keep `Rajarshi-Roy-research/Defactify_Image_Dataset` as the current candidate primary dataset for the baseline
- Keep a broad newsroom scope: images with people and faces remain in scope
- Do not redefine the product as a deepfake detector, identity verifier, or face-analysis system
- Treat the dataset as suitable for the initial binary `real` vs `ai_generated` task, but not yet fully validated for balanced newsroom coverage
- Defer model and preprocessing code until dataset follow-up checks are documented and the next phase begins

### Limitations
- The dataset license was not yet explicitly recorded from a direct license field check
- Overall dataset composition was not yet audited
- Portrait and close-up face prevalence was not measured yet
- Leakage, duplicates, and split quality were not inspected yet
- The repository already had `README-3.md`, but this phase only created and updated the required root `README.md`
- No model code, preprocessing code, or tests were implemented in this phase

### Next follow-up
- Confirm and document the dataset license
- Audit dataset composition for broad newsroom/editorial coverage
- Audit portrait / close-up face bias and decide how to handle it if overrepresented
- Implement the first reproducible dataset loader
- Implement FFT preprocessing after the dataset loading step is ready

## Documentation scope update

### Completed
- Revised `README.md` to reflect the broad newsroom scope, including images with people and faces
- Revised `TASKS.md` to add dataset composition audit work
- Added portrait / close-up face bias audit tasks and strategy options if the dataset is too portrait-heavy
- Revised `DONE.md` to record the new scope decision without changing product type
- Aligned `AGENTS.md` with the updated broad newsroom/editorial scope

### Files changed
- `AGENTS.md`
- `README.md`
- `TASKS.md`
- `DONE.md`

### Major decisions
- People and faces remain in scope as part of general newsroom/editorial imagery
- The product remains a general AI-image detector, not a deepfake detector or face-analysis system

### Limitations
- No code or dataset audit was executed in this documentation-only step
- No dataset audit has been executed yet; the new items are planning tasks only

### Next follow-up
- Align the remaining project docs with the broader newsroom scope where needed
- Perform the dataset composition and portrait-bias audits before making training-data decisions

## Foundation layer implementation

### Completed
- Implemented `training/dataset.py` as a dataset loader skeleton for `Rajarshi-Roy-research/Defactify_Image_Dataset`
- Added explicit dataset config handling for split names, dataset id, and config name
- Added dataset schema validation, `Label_A` normalization, and split summary helpers
- Implemented deterministic FFT preprocessing in `preprocessing/fft_transform.py`
- Matched the baseline FFT pipeline: resize `256x256`, grayscale, `2D FFT`, `fftshift`, log magnitude, normalization, and 3-channel repeat
- Implemented `model/predict.py` as a prediction pipeline skeleton with placeholder-safe model loading
- Added centralized threshold mapping helper for future `real` / `ai_generated` / `uncertain` verdict logic
- Implemented a minimal `app.py` Streamlit UI that accepts an uploaded image, runs preprocessing, and returns a safe placeholder result when weights are missing
- Verified the new foundation files compile with `python3 -m py_compile`

### Files changed
- `training/dataset.py`
- `preprocessing/fft_transform.py`
- `model/predict.py`
- `app.py`
- `TASKS.md`
- `DONE.md`

### Major decisions
- Use explicit placeholder inference that returns `uncertain` when model weights are not available
- Do not pretend `model/model.pth` exists or that trained inference works yet
- Keep threshold mapping logic in `model/predict.py` even before real logits are wired up
- Keep the first app iteration minimal and focused on upload, preprocessing, and safe output

### Limitations
- No training loop, evaluation code, or model architecture implementation was added in this step
- Real model loading is still a placeholder even if a weights file path exists
- The Streamlit app is a minimal foundation and does not yet include the full result-panel or diagnostics-tab UX
- No automated tests were added yet beyond a compile check
- Dataset inspection helpers exist, but the dataset composition audit itself has not been executed yet

### Next follow-up
- Add tests for dataset loading helpers, FFT preprocessing, and placeholder inference
- Build the real model-loading path after training artifacts exist
- Expand the app UI with clearer verdict copy and separate diagnostics tabs
- Run the planned dataset composition and portrait-bias audits

## Baseline model wiring

### Completed
- Added `model/resnet.py` with the baseline ResNet-50 binary classifier constructor
- Wired `model/predict.py` to build the real architecture and attempt local checkpoint loading from `model/model.pth`
- Kept explicit safe fallback behavior when weights are missing or incompatible
- Added honest inference behavior: confidence is only computed from real logits when a compatible checkpoint is loaded
- Updated `app.py` copy to reflect real architecture wiring plus placeholder-safe fallback
- Updated `README.md`, `TASKS.md`, and `DONE.md` to remove stale documentation and reflect the current runnable state
- Removed the obsolete alternate README file `README-3.md` to avoid conflicting project documentation

### Files changed
- `model/resnet.py`
- `model/predict.py`
- `app.py`
- `README.md`
- `TASKS.md`
- `DONE.md`
- `README-3.md`

### Major decisions
- Use the documented ResNet-50 architecture for inference wiring before training is implemented
- Only run real inference when a local checkpoint exists and can be loaded into the baseline architecture
- Fall back to an explicit placeholder `uncertain` result when weights are missing or incompatible
- Treat checkpoint compatibility problems as user-visible limitations rather than silently fabricating output

### Limitations
- No training script or checkpoint generation exists yet, so real inference depends on a user-provided compatible checkpoint
- The checkpoint format is assumed to be either a raw state dict or a dictionary containing `state_dict`
- No automated tests were added yet for model construction or inference behavior
- The app still uses simple result rendering and does not yet implement the full diagnostics-tab UX

### Next follow-up
- Add tests for model construction, weight loading, threshold mapping, and placeholder fallback
- Implement the first training pipeline that can produce a compatible `model/model.pth`
- Add evaluation and robustness reporting before treating predictions as meaningful for newsroom use
