# HighRes-v1: protocol for a higher-resolution image corpus

**Status:** pre-registered acquisition protocol. The corpus is not yet materialised and this file
contains no model result.

## Why this is a separate study

The 128 x 128 Defactify H1-N work is retained as a documented low-resolution pilot. It exposed
important geometry controls, but it cannot answer whether a model benefits from fine image detail.
It must not be silently re-labelled as a high-resolution experiment simply by changing a command
line size.

HighRes-v1 is a new study with a new source, data manifest, split, preprocessing identifier and
results table. Its primary raster is **384 x 384**, obtained without upsampling from source images
whose short side is at least 512 pixels. The target is deliberately below 512: it preserves real
source detail while keeping a ResNet-50 training batch within the tested Apple unified-memory
budget.

## Research question and allowed claims

> Under a common, source-normalised 384 x 384 raster made from fixed-quality 512-pixel-or-larger
> sources, what evidence remains that an image classifier distinguishes the specified real-photo
> and generated-image distributions, both internally and on locked external tests?

This study may report performance **on named datasets and transformations**. It cannot prove the
origin of an arbitrary uploaded image or authenticate an image in the forensic sense.

## Sources and roles

| Role | Source | Permitted use |
| --- | --- | --- |
| Train and validation candidate reservoir | [CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small), revision `6c539a534c07917307c381f5af4053c6091b5278` | Build one fixed, auditable local corpus only. The release is CC BY-NC-SA 4.0, so it is for this non-commercial coursework/research use and is never added to Git or redistributed. |
| Locked external benchmark | [Synthbuster](https://zenodo.org/records/10066460) + [RAISE-1k](https://loki.disi.unitn.it/RAISE/download.html) | Open only after the HighRes-v1 architecture, validation rule, seed protocol and threshold are frozen. RAISE must never enter training. |
| Conditional descriptive benchmark | [CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval), revision `7d4a74a88d2cac93b513c0853bf92c260eaceea0` | Do not use for training or selection. Before any score, run exact- and perceptual-hash contamination checks against the materialised corpus and respect its FFHQ/COCO split restrictions. A passed check supports a clearly labelled cross-dataset result; a failed or inconclusive check excludes it. |
| Optional second independent benchmark | [GenImage official test split](https://github.com/GenImage-Dataset/GenImage) | Evaluate only after the first external benchmark is locked. Report every generator separately rather than calling the whole set “unseen”. |

The Small dataset card describes paired real and generated images, 4,803 generator models, image
resolution and generator/source metadata. Its authors also warn that many generators derive from
Stable Diffusion and that caption/source biases remain. Those are threats to validity to measure,
not properties to hide.

## Immutable inclusion gate

The metadata scan reads no image bytes. Before any selected image is downloaded, each source-catalog
must meet all of the following conditions:

| Field | Rule | Reason |
| --- | --- | --- |
| `resolution` | `width = height = 512` in the first release, or a later explicitly recorded larger square | 384-pixel training never uses an upscaled source and both labels share geometry. |
| `format`, `mode` | `PNG`, `RGB` | Prevent a container/mode shortcut from being the classifier. |
| `nsfw_flag` | Explicitly false | Keep the research corpus suitable for the coursework environment; do not infer missing flags as false. |
| `label`, `architecture`, `model_name` | Binary label is valid; real rows declare `real`; synthetic rows have a nonempty generator and non-real architecture | Preserve class provenance and reject contradictory metadata. |
| provenance | Revision-pinned shard path and row number are recorded | A row can be reproduced without relying on mutable dataset ordering. |

The selection specification will also declare a deterministic seed, class balance, maximum quota
per generator, architecture/subset and real-source strata, and the intended train/validation/test
roles. Its SHA-256 is part of the experiment identity. Quotas are intentionally **not** guessed
before the full metadata audit establishes how many valid, diverse rows exist.

The selection stage is fail-closed. It must reject an incomplete or hash-mismatched catalog, a
duplicate source locator, an inconsistent prompt-derived group, contradictory
`label`/`architecture`/`model_name` semantics, or an insufficient jointly supported source stratum
for the two classes. It must also record a deterministic rank for every eligible locator and a
pre-ranked reserve. A damaged or duplicate image may be replaced only by the next reserve item
from that frozen rank: neither manual curation nor a fresh random draw is permitted after image
bytes have been inspected.

## Acquisition and leakage audit

1. Record a source lock: repository ID, immutable revision, licence, every Parquet shard path,
   byte size, blob/LFS/Xet identifier and scan date.
2. Column-scan only metadata fields. The binary `image_data` field is excluded at the Parquet read,
   rather than removed after iteration. The source catalog stores a stable source locator,
   dimensions, label, generator/source strata and a hash of the normalised prompt; it stores no
   image data or full prompts.
3. Audit the source catalog before choosing a quota: class counts, source resolution,
   format/mode, NSFW status, real-source distribution, generator/architecture distribution,
   repeated image names and prompt groups. A partial scout is never eligible to become a research
   manifest.
4. Freeze the catalog-derived selection manifest. Only then materialise the exact selected rows.
   The materialiser must verify that each received row agrees with its pinned catalog metadata and
   that its actual decoded bytes are PNG, RGB, 512 x 512 and non-corrupt. For every accepted file
   record byte SHA-256, decoded-pixel SHA-256, perceptual hash, decoded dimensions and byte count.
   Original images and the complete local manifest remain outside Git.
5. Build the duplicate graph before assigning any HighRes-v1 partition. Its edges include source
   prompt/content groups, exact file or pixel hashes, and perceptual-hash candidates. An identical
   decoded image with conflicting labels is quarantined rather than silently assigned a preferred
   label; a perceptually close pair is a leakage boundary, not proof of identity. Exact duplicates
   within one label are reduced deterministically to one canonical instance.
6. Allocate splits by whole connected leakage components. Validation alone controls early stopping,
   hyperparameters, calibration and threshold. The internal test and all external sources remain
   untouched during those decisions.

The dataset's published `split` field is preserved as source metadata. It is not silently replaced
by a random split, and its relationship to HighRes-v1 roles will be stated in the frozen selection
manifest. If that upstream field is documented and consistent, it becomes a hard boundary; if it is
not a meaningful boundary, the new component-level split and its objective are declared explicitly,
with a cross-tab against the upstream field retained in the final audit.

## Preprocessing and compute contract

The future code path will use a new identifier, `highres_square_crop_384_v1`, not an H1-N alias.
It will EXIF-normalise, decode to RGB, take a deterministic evaluation or seeded training square
crop from the source raster, and make one common 384 x 384 LANCZOS raster before RGB or FFT
representation processing. The data loader must reject a source below the declared 512-pixel
minimum instead of upsampling it.

The device gate already measured a scratch ResNet-50 training step on this Mac. At 384 pixels,
batch 64 sustained about 43 images/s with roughly 20.4 GB Metal driver allocation; batch 80 used
about 25.6 GB with essentially the same throughput. Initial training will therefore use batch 64,
leaving headroom below the requested 25 GB budget. This is a hardware measurement, not a model
quality result.

## Evaluation discipline

- Train and compare only hypotheses declared after the materialised HighRes-v1 split is frozen;
  keep architecture, pretrained status, augmentation, budget and seeds explicit.
- Select from validation metrics only. Do not choose a checkpoint, seed, threshold, crop or
  resolution after looking at any internal test or external score.
- Report ROC-AUC, balanced accuracy, macro-F1, calibration and per-generator/per-source results;
  use group-aware confidence intervals whenever source pairing or duplicate components exist.
- Report failure and domain-shift results alongside aggregate results. A weak external result is an
  academic finding and keeps the local interface disabled.

The coursework report will place this protocol and its observed manifests/results in the data,
methodology, training and results sections. The interface is considered only after a model has a
hash-pinned external-validation record.
