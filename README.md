# AI Image Detector

Newsroom-oriented image authenticity screening for editorial imagery.

The planned workflow is:
- upload an image
- classify it as `real`, `ai_generated`, or `uncertain`
- show a confidence score and supporting diagnostics

Scope note:
- images with people and faces remain in scope as part of broad newsroom/editorial coverage
- the product is still not a deepfake detector, identity verifier, or face-analysis system
- the goal remains general AI-image detection for newsroom use across scenes, events, portraits, objects, interiors, and other editorial visuals

## Dataset

Current candidate primary dataset: `Rajarshi-Roy-research/Defactify_Image_Dataset` on Hugging Face.

What was confirmed from the dataset card and raw `README.md`:
- splits: `train` 42,000, `validation` 9,000, `test` 45,000
- fields: `Caption`, `Image`, `Label_A`, `Label_B`
- `Label_A`: `0 = Real`, `1 = AI-Generated`
- `Label_B`: `0 = Real`, `1 = SD21`, `2 = SDXL`, `3 = SD3`, `4 = DALLE3`, `5 = Midjourney`
- declared scope: binary real-vs-AI classification plus generator-source identification

Current fit for this project:
- good match for the baseline `real` vs `ai_generated` training task
- useful for later generator-family evaluation via `Label_B`
- likely compatible with the broad newsroom scope, because the public examples include people and general web imagery

Current limitations of the dataset review:
- license constraints still need to be explicitly recorded in-repo after a direct license check
- overall dataset composition has not yet been audited
- portrait and close-up face prevalence has not yet been measured
- leakage and duplicate risk have not yet been inspected

## Current status

Implemented foundation:
- Hugging Face dataset loader skeleton in `training/dataset.py`
- deterministic FFT preprocessing in `preprocessing/fft_transform.py`
- baseline ResNet-50 architecture wiring in `model/resnet.py`
- inference flow in `model/predict.py` with honest fallback behavior when weights are missing
- minimal Streamlit upload app in `app.py`

Current runtime behavior:
- if `model/model.pth` exists and matches the baseline ResNet-50 binary classifier, the app can run real forward-pass inference
- if weights are missing or incompatible, the app returns a safe placeholder `uncertain` result and explains why

What is not done yet:
- no training loop or checkpoint production
- no measured model quality or confidence calibration
- no robustness evaluation or dataset composition audit yet
