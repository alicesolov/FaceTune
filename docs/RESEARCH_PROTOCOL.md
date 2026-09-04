# Research protocol: FFT features for real-versus-synthetic images

**Status:** amended and locked before H1-N neural runs; D0 diagnostic controls already observed.
**Scope:** still images of general scenes; no face or video claims.

The original Defactify protocol is superseded where stated by
[the geometry-control amendment](PROTOCOL_AMENDMENT_GEOMETRY_CONTROL.md). That amendment is part
of the research record: it documents why D0 results cannot be presented as a detector claim.

## Research question

Under common source-normalised rasterisation, does an FFT-magnitude representation improve a
real-versus-synthetic classifier relative to an RGB classifier of equal capacity and training
budget?  Any internal answer is exploratory; external transfer and robustness are reported
separately.

## Hypotheses

| ID | Claim under test | Comparison | Primary evidence |
| --- | --- | --- | --- |
| H1-N | FFT contains useful representation signal under a geometry-controlled rasterisation. | FFT ResNet-50 vs RGB ResNet-50 from random initialisation; radial power spectrum + logistic regression. | Exploratory grouped internal metrics, paired group ranking and cluster-bootstrap CI of FFT minus RGB. |
| H2 | A high internal score does not automatically imply generator generalization. | Freeze the model and validation threshold on Defactify; evaluate RAISE real versus each Synthbuster generator separately. | Macro average, worst-generator result, per-generator confidence intervals and generator-family relation. |
| H3 | File transformations change the observed signal; robust augmentation may reduce the drop. | FFT model with versus without train-time robust augmentation. | Same metrics under JPEG Q 95/75/50, resize and blur, reported separately. |

H2 or H3 may fail even if H1 succeeds. That is a valid result.

## Design controls

- Fix source revision, manifest hash, seed, package lock, model configuration and checkpoint.
- Use official Defactify train/validation/test only after checking whether captions or near-duplicates
  cross splits. If they do, report a grouped caption split as the primary internal result and retain
  the official split only as a benchmark replication.
- Use validation, never test, for early stopping, hyperparameters, calibration and decision threshold.
- Keep Synthbuster/RAISE locked until architecture, checkpoint rule, seed aggregation and threshold
  are fixed.
- Balance content groups during training. Each H1-N epoch samples one real and one synthetic
  sibling per leakage group, assigning the fake generator in a globally balanced uniform cycle.
  Defactify has 16k real and 80k synthetic samples, so a trivial all-synthetic classifier already
  has high plain accuracy.
- Fit metadata-only controls (dimensions, format, byte size), but do not use metadata in an image
  classifier. A strong metadata control is evidence of dataset bias, not detector quality.

## Observed dataset caveat and protocol amendment

On the leakage-resistant grouped Defactify split, the predeclared file-metadata control attained
ROC-AUC 0.981 and balanced accuracy 0.976. The dimensions and canonical PNG byte size therefore
carry a strong class shortcut. All synthetic images are square whereas the real photographs have
varied aspect ratios. Direct rectangular-to-square resizing would expose a class-correlated
resampling trace, especially to FFT features. The earlier radial result is consequently D0 only.

H1-N decodes RGB, crops a source square, applies one common 128 x 128 LANCZOS rasterisation, and
calculates FFT only after that shared raster. It never letterboxes. Neural training uses a
seeded-random square crop, whereas the fixed-feature radial baseline uses a deterministic centre
crop for every split; the two roles must not be conflated. The grouped internal test was viewed for
D0, so amended internal results are exploratory stress-test evidence rather than a confirmatory
test. The external test is required before any transfer claim.
- Repeat final selected experiments for three seeds. Report mean, standard deviation and 95% bootstrap
  intervals; do not choose the best seed.

## Models and equal budget

All H1-N neural baselines use a ResNet-50 backbone from random initialisation, 128x128 common input,
same optimizer family, epoch cap, early-stopping policy, paired group sampler and validation-only
threshold. They run for seeds 7, 17 and 42; no best seed is selected. ImageNet pretraining is
reserved for a separately labelled practical ablation because it is not an equivalent prior for an
FFT magnitude image.

1. Majority/random sanity baselines.
2. Controlled radial FFT power spectrum followed by logistic regression.
3. RGB ResNet-50 from scratch.
4. FFT-magnitude ResNet-50: grayscale, FFT2, fftshift, `log(1 + abs(FFT))`, robust per-image scale,
   repeated to three channels.

The FFT version discards phase and colour. It is a hypothesis baseline, not a declaration that this
representation is optimal. D0 legacy controls remain in the artefact directory but do not influence
H1-N selection.

## Metrics and reporting language

Primary metrics are ROC-AUC, balanced accuracy and macro-F1. PR-AUC is secondary because Defactify
has a 5:1 AI/real prevalence. Secondary metrics are per-class precision/recall, FPR at TPR 95%,
paired group-ranking accuracy, Brier score, expected calibration error, confusion matrix and
coverage-risk for abstention. FPR at TPR 95% is recomputed from the scored test ROC curve and is
descriptive; it is not the validation-selected deployment threshold. Confidence intervals and
RGB-minus-FFT comparisons resample whole `leakage_group` clusters. `accuracy` is secondary. A raw
softmax score must be called a *model score*, not a probability, until calibration is checked.

Allowed: “On the specified held-out benchmark, model X achieved metric Y.”
Not allowed: “The model authenticates images” or “a 93% score proves that an image is AI-generated.”

## Sources

- Rajarshi Roy et al., [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset).
- Corvi et al., [Synthbuster](https://doi.org/10.1109/OJSP.2023.3337714).
- Zhu et al., [GenImage](https://arxiv.org/abs/2306.08571), particularly its cross-generator and
  robustness tables as evidence that in-distribution detection can fail under shift.
