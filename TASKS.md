# TASKS.md

## Current objective
Produce the first trained `model/model.pth` checkpoint and validate the end-to-end
pipeline (image → FFT → ResNet-50 → verdict) with real metrics.

## Active priorities

### 1. Pre-training dataset audit (run first)
- Run `python -m training.audit --all-splits` to audit all three splits + cross-split leakage
  - Reports class balance for train / validation / test
  - Reports within-split duplicate captions for each split
  - Reports generator (Label_B) distribution per split
  - Reports caption overlap between train↔val, train↔test, val↔test
- If `total_leaked_captions > 0`: decide whether to deduplicate before training
- Document class balance result (expected ≈ 50/50 from Defactify design)
- Note: test (45k) is larger than train (42k) — confirm this is expected
- Confirm and record the dataset license in `README.md`
- Decide on class-balanced sampling strategy if imbalance ratio > 1.5×

### 2. First training run (primary goal)
- Run `python -m training.train --epochs 10 --batch-size 32 --device auto`
- Monitor per-epoch validation F1 (now the checkpoint selection criterion)
- Confirm `model/model.pth` is written after the first epoch improvement
- Early stopping will halt automatically if val F1 plateaus for 3 epochs
- Confirm the app loads the checkpoint and shows real verdicts (not placeholder)

### 3. Post-training evaluation (required before claiming model is usable)
- Run `python -m training.evaluate --split test`
- Record accuracy, precision, recall, F1, confusion matrix in `README.md`
- Inspect per-generator slice results to check generalization
- Decide on threshold policy (current default: 0.45–0.55 = uncertain)
- If F1 < 0.70: try EfficientNet-B4 (`model/efficientnet.py`) as next backbone
  - Wire it into train.py, retrain, compare val F1 head-to-head with ResNet-50

### 4. Robustness evaluation (important for newsroom use)
- Design a small robustness test set: recompressed JPEG, cropped, resized, screenshot
- Evaluate the trained model on these stress cases
- Document findings as known limitations if performance degrades significantly

### 5. App smoke test after training
- Open `streamlit run app.py`, upload a known-real and a known-AI image
- Verify verdict badge, confidence, FFT visualization, and diagnostics tabs all work
- Confirm model status shows "Model ready" (not "Model not trained")

### 6. Dataset follow-up validation (still outstanding)
- Audit portrait / close-up face prevalence in training data
- Confirm no train/test leakage via duplicate images beyond caption matching
- Check whether test split Label_B distribution matches the train distribution

### 7. Documentation tasks
- Update `README.md` with actual training metrics once the first run completes
- Add setup instructions: `pip install -r requirements.txt`, `streamlit run app.py`
- Record the Defactify dataset license

## Open questions
- Is the dataset license permissive enough for newsroom demo and model publication?
- Should we optimize for recall (catch more AI-generated) or precision (fewer false alarms)?
- Should the uncertain band (0.45–0.55) be widened based on calibration results?
- Is `num_workers > 0` safe in the deployment environment (Linux vs Windows)?

## Definition of next milestone
The next milestone is reached when:
- `python -m training.train` runs to completion without error
- `model/model.pth` exists and the app loads it
- `python -m training.evaluate --split test` prints real metrics
- Test accuracy is above 0.70 on the held-out test split
- `pytest tests/` passes without errors
