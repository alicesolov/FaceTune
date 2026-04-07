# AGENTS.md

## Project overview

We are building a newsroom-oriented AI image verification tool.

Goal: determine whether an uploaded image is a real photograph or AI-generated, so that editors, journalists, and fact-checkers can decide whether the image is safe to publish in news content, articles, social posts, or visual materials.

Important scope constraint:
- Images with people and faces are in scope when they appear as part of normal newsroom and editorial content.
- We focus on broad editorial imagery: city views, buildings, nature, events, interiors, objects, portraits, documentary-style visuals, and similar news-relevant content.

Primary users:
- journalists
- editors
- fact-checkers
- newsroom staff

Core product behavior:
- user uploads an image
- system returns one of:
  - `real`
  - `ai_generated`
  - `uncertain`
- system also returns a confidence score and supporting visual diagnostics

## Current dataset direction

Planned dataset direction:
- use the Hugging Face dataset `Rajarshi-Roy-research/Defactify_Image_Dataset` as a candidate primary dataset
- treat this as the current working assumption unless the user explicitly changes the dataset choice
- if the dataset structure, labels, or license create implementation issues, document the issue and propose a concrete fallback instead of silently changing direction

Do not invent dataset facts.
If dataset details are needed for implementation, inspect the actual dataset files or documentation first.

## Product priorities

When making decisions, optimize for the following in order:

1. **Editorial reliability**
   - false positives and false negatives both matter
   - avoid overclaiming certainty
   - prefer `uncertain` over unjustified confidence

2. **Explainability**
   - every prediction should be accompanied by a confidence score and simple, editor-friendly explanation
   - the UI should help a non-technical editor understand why the system is skeptical

3. **Generalization**
   - do not optimize only for one generator family
   - design evaluation so we can see whether the detector survives new generators and recompression

4. **Practical usability**
   - fast inference
   - simple upload flow
   - clear result panel
   - useful diagnostics for newsroom workflow

## Non-goals

Unless the user explicitly asks otherwise, do **not** turn this project into:
- a face detector
- a face-specific deepfake detector
- a forensic identity verification tool
- a facial analysis system
- a watermark detector only
- a provenance system based purely on metadata
- a browser extension
- a moderation platform for all media types

This project is specifically an **image authenticity screening tool for editorial use**.

## Technical baseline

Use the current project baseline unless the user overrides it:

### Preprocessing
- resize image to `256x256`
- convert to grayscale
- compute `2D FFT`
- apply `fftshift`
- use log magnitude spectrum
- normalize
- duplicate to 3 channels for backbone compatibility

### Model
- `ResNet-50`
- binary classification head for:
  - `real`
  - `ai_generated`

### Inference labels
- `real`
- `ai_generated`
- `uncertain`

### App stack
- `Python`
- `PyTorch`
- `Streamlit`

## Expected repository structure

Prefer a structure close to this:

```text
ai-image-detector/
├── app.py
├── model/
│   ├── model.pth
│   └── predict.py
├── preprocessing/
│   ├── fft_transform.py
│   └── visualizations.py
├── training/
│   ├── train.py
│   ├── dataset.py
│   └── evaluate.py
├── data/
│   └── README.md
├── tests/
├── docs/
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

## Rules for the coding agent

### 1. Do not guess requirements
If something important is unclear, ask the user.
Examples:
- target metric
- threshold policy
- dataset split policy
- deployment target
- whether we are optimizing for recall or precision in editorial screening

### 2. Keep the user informed
For non-trivial tasks, briefly report:
- what you changed
- why you changed it
- what remains unresolved

### 3. Make small, reviewable changes
Prefer incremental commits/patches over large rewrites.

### 4. Protect the project scope
If a proposed change drifts away from editorial image authenticity detection, flag it before implementing.

### 5. Be explicit about assumptions
Whenever you make an assumption, write it down in the relevant docstring, README section, or task note.

### 6. Prefer reproducibility
Training and evaluation steps must be reproducible.
Always make random seeds, config values, and dataset paths explicit.

### 7. No fake completeness
Never claim:
- a model was trained if it was not trained
- a metric was measured if it was not computed
- a dataset was validated if it was not inspected
- a feature works if it was not tested

### 8. Respect newsroom reality
Editors need a practical decision aid, not vague ML jargon.
Outputs and UI text should be understandable by non-ML users.

## Required working style

### Before implementing a feature
Read:
- `README.md`
- `TASKS.md`
- `DONE.md`

### During implementation
Update code and docs together.

### After completing each task
Always update:
- `TASKS.md`
- `DONE.md`

`DONE.md` must contain:
- what was completed
- files changed
- major decisions
- limitations
- next follow-up if needed

`TASKS.md` must reflect the new current state, not the old one.

## Documentation requirements

Every non-trivial function must have a meaningful docstring.

Each docstring should include:
- purpose
- inputs
- outputs
- key assumptions
- failure modes if relevant

## File-by-file expectations

### `app.py`
Responsible for:
- upload flow
- size/type validation
- calling inference
- rendering verdict
- showing confidence
- rendering diagnostics tabs
- showing cautious wording for uncertain results

Do not place heavy training logic here.

### `model/predict.py`
Responsible for:
- model loading
- preprocessing invocation for inference
- logits/probabilities
- thresholding into `real`, `ai_generated`, `uncertain`

Threshold logic must be centralized and easy to edit.

### `preprocessing/fft_transform.py`
Responsible for:
- FFT preprocessing pipeline only
- deterministic transformation
- minimal side effects

### `preprocessing/visualizations.py`
Responsible for:
- FFT spectrum views
- gradient maps
- power spectrum plots
- editor-facing visual outputs

### `training/dataset.py`
Responsible for:
- dataset loading
- split handling
- transforms
- label normalization
- dataset sanity checks where appropriate

### `training/train.py`
Responsible for:
- training loop
- optimizer/scheduler setup
- checkpointing
- logging
- validation tracking
- early stopping if used

### `training/evaluate.py`
Responsible for:
- test metrics
- confusion matrix
- ROC/PR outputs if added
- robustness slices if available
- summary tables suitable for report/README

## Quality bar for implementation

A task is not complete unless:
- code runs
- imports are clean
- paths are coherent
- naming is understandable
- docstrings are added
- user-facing text is readable
- the task is reflected in `DONE.md`
- the next state is reflected in `TASKS.md`

## Evaluation principles

When evaluating the model, prioritize:
- accuracy
- precision
- recall
- F1
- confusion matrix
- calibration / confidence behavior if feasible
- robustness to compression, resize, screenshots, reposted images if feasible

Important:
A newsroom tool must not be evaluated only on ideal clean benchmark images.
If possible, include stress tests on:
- recompressed JPEGs
- screenshots
- cropped images
- resized images
- social-media-like degradation

If such evaluation is not yet implemented, explicitly state that as a limitation.

## UX principles

The app should feel safe for editorial decision-making.

Required UX behavior:
- simple upload zone
- supported formats clearly shown
- result panel with verdict and confidence
- clear warning when confidence is borderline
- explanation in plain language
- visual diagnostics in separate tabs
- no overdramatic or absolute wording

Preferred copy style:
- “Likely real photograph”
- “Likely AI-generated”
- “Uncertain — manual review recommended”

Avoid copy like:
- “This image is definitely fake”
- “100% authentic”
unless there is a deliberate product requirement and strong justification.

## Error-handling rules

Handle these cases cleanly:
- unsupported file format
- oversized upload
- corrupted image
- model weights missing
- CPU-only environment
- inference failure
- empty upload
- invalid image mode / malformed content

Errors shown to the user must be short and actionable.

## Testing expectations

Add tests where practical.

Minimum preferred coverage:
- preprocessing output shape and type
- deterministic FFT transform behavior
- prediction pipeline on a sample image
- threshold mapping to verdict labels
- basic app utility functions if separated

If full testing is not feasible, at least include smoke tests for the inference pipeline.

## Performance expectations

Optimize for practical demo and newsroom usage:
- inference should feel fast on commodity hardware
- avoid unnecessary recomputation
- model loading should be separated from per-image inference where possible
- do not block UI unnecessarily

## Security and privacy notes

Do not:
- store uploaded images permanently unless explicitly required
- expose internal file paths in UI
- log sensitive newsroom content unnecessarily
- send uploaded images to external APIs without explicit approval

Default assumption:
- local inference or controlled server-side inference only

## When proposing improvements

When suggesting changes, prefer this format:
1. problem
2. why it matters for editorial use
3. recommended change
4. implementation cost
5. expected impact
6. risk / tradeoff

This keeps project discussions practical.

## Preferred development sequence

Unless the user asks for another order, work in this sequence:
1. dataset inspection and loading
2. preprocessing pipeline
3. baseline training pipeline
4. evaluation pipeline
5. inference wrapper
6. Streamlit app
7. visual diagnostics
8. robustness testing
9. polishing and deployment

## Definition of done for MVP

MVP is done when:
- image upload works
- model inference works end-to-end
- verdict is shown as `real`, `ai_generated`, or `uncertain`
- confidence score is shown
- diagnostics are shown
- README explains setup and usage
- key limitations are documented
- project can be run by another developer without hidden steps

## If you are unsure

Do not silently improvise on critical product questions.
Ask the user first.

Especially ask before changing:
- dataset choice
- label taxonomy
- thresholds
- backbone architecture
- app framework
- scope around faces
- deployment target
- evaluation policy for newsroom false positives vs false negatives
