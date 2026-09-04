# Protocol amendment: source-normalised rasterisation

**Date:** 29 August 2026
**Status:** locked before running the amended neural experiments
**Scope:** a correction to the Defactify internal experiment, not a retroactive performance claim.

## Why the original internal protocol was amended

The first diagnostic controls exposed a material geometry/source confound in the prepared
Defactify corpus:

- the 16,000 real photographs have 259 observed widths, 367 observed heights and median aspect
  ratio 1.33;
- all 80,000 synthetic images are square; and
- no exact `(width, height)` pair occurs in both binary labels.

On the group-disjoint split, a logistic regression that received only canonical-PNG width, height,
aspect ratio and byte count attained ROC-AUC 0.9807 and balanced accuracy 0.9758.  The original
image transforms resized every input directly to a square.  Thus a rectangular real photo would
be anisotropically resampled while an already square synthetic image would not be.  In particular,
that operation can create a class-correlated frequency signal before an FFT is computed.

The completed radial-FFT result is retained as **D0, a diagnostic benchmark replication**.  It is
not evidence that an image detector works generally and will not be used to select an architecture
or a deployment model.  The attempted full RGB ResNet-50 run was terminated before an evaluation
result when this issue was found; it is excluded from all results tables.

## Amended internal hypothesis (H1-N)

> With a group-disjoint Defactify split and a common source-normalised rasterisation, does an
> FFT-magnitude ResNet-50 outperform an RGB ResNet-50 of equal capacity and training budget?

H1-N is deliberately narrower than a claim of universal AI-image detection.  It tests a
representation choice in one controlled corpus.

## Locked preprocessing

For every image model and every control based on image pixels:

1. Decode the image to RGB and ignore EXIF and the original file container.
2. Crop to a square without padding.  The training crop is random under the experiment seed;
   validation, test, robustness, and external evaluation use a deterministic centre crop.
3. Resize the cropped square with one documented interpolation method to 128 x 128 pixels.
   The corpus minimum short side is 128 pixels, so this step does not require upsampling.
4. Feed that same standardised raster to RGB or compute the FFT magnitude from it.  FFT never sees
   an independently resized original image.

Letterboxing is prohibited because its borders reintroduce a directly observable aspect-ratio
channel.  Image dimensions, source file size and container metadata are never supplied to the
neural models.

## Locked training comparison

- RGB ResNet-50 from random initialisation;
- FFT-magnitude ResNet-50 from random initialisation;
- radial power-spectrum logistic regression as an interpretable pixel-only baseline.

All neural runs use the same 128 x 128 rasterisation, paired group sampler, optimizer family,
learning-rate schedule, epoch cap, early stopping criterion, batch size and validation-only
threshold rule.  ImageNet pretraining is excluded from this primary representation comparison,
since it is a semantic RGB prior but has no equivalent interpretation for FFT magnitude.

The train sampler contributes one real image and one fake image from the same `leakage_group` per
group visit.  The fake generator is selected uniformly from the available synthetic siblings.
This keeps prompt/content, class, and generator contribution balanced and prevents unusually
large groups from receiving more training weight.

The three planned seeds are 7, 17 and 42.  No individual seed will be selected as the reported
winner.  Checkpoints and binary thresholds are selected only from validation predictions.

## Evaluation status and decision rules

The original Defactify grouped test set was inspected during D0.  Therefore, results of H1-N on
that same test split are labelled **exploratory internal stress-test results**, even though no
amended neural hyperparameter is selected using them.  The confirmatory evaluation is the locked
external corpus (Synthbuster synthetic images paired with a documented real-photo source), run
only after preprocessing, model family, seed aggregation, checkpoint rule and threshold rule are
frozen.

For each model and each seed, report ROC-AUC, balanced accuracy, macro-F1, per-class recall, and
FPR at TPR 95%.  Report each synthetic generator separately, followed by macro-average and
worst-generator values.  Also report paired group-ranking accuracy: the fraction of same-group
real/fake pairs for which the fake receives the higher score.  Confidence intervals and RGB-minus-
FFT comparisons must resample whole `leakage_group` clusters, not individual correlated images.

JPEG, resize and blur robustness conditions are applied only after the common rasterisation.  A
model score remains a score, not a probability, until separately calibrated on validation data.

## Interpretation rules

Any result may falsify H1-N.  A high internal score alone cannot validate a public claim or
authenticate an arbitrary image.  A web interface may serve only a selected, frozen model after
the external evaluation and must show its model card, threshold, dataset limitations and the fact
that its output is an experimental score.
