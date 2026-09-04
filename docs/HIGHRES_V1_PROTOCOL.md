# HighRes-v1: protocol for a higher-resolution image corpus

**Status:** source-selection protocol for the primary study, amended after the first full source
audit. The primary HighRes-v1 corpus is not yet materialised and this file contains no
high-resolution model result.

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

### Separate Defactify exploratory audit — not HighRes-v1

The local `defactify_exploratory_native384_v2` corpus is a **separate within-Defactify sensitivity
study**, not a substitute for HighRes-v1. It contains 16,328 retained native 384 x 384 PNG crops
from 8,208 caption groups after preserving the upstream `train`/`val`/`test` roles and excluding
36 cross-role leakage components (88 rows). Its builder and launch gate verify source/output hashes,
encoded PNG/RGB mode, group isolation, and immutable sidecar evidence before an experiment.

This removes output dimensions as a direct input feature but does **not** neutralise the
label-correlated source scale: selected real sources are mainly 384–640 pixels on the short side,
whereas SD 2.1 is 768 x 768 and SD 3/SDXL are 1024 x 1024. A pre-training file-size control on the
locked upstream test role obtained ROC-AUC 0.650114 and balanced accuracy 0.595238 using no image
pixels (the dimensions are constants; canonical PNG byte size varies). This is residual corpus
bias, not detector performance. No neural model, model selection, external score, or local
interface may use this corpus as a positive result.

## Research question and allowed claims

> Under a common, source-normalised 384 x 384 raster made from fixed-quality 512-pixel-or-larger
> sources, what evidence remains that an image classifier distinguishes the specified real-photo
> and generated-image distributions, both internally and on locked external tests?

This study may report performance **on named datasets and transformations**. It cannot prove the
origin of an arbitrary uploaded image or authenticate an image in the forensic sense.

## Source decisions and roles

| Role | Source | Decision and permitted use |
| --- | --- | --- |
| Rejected broad-source candidate | [CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small), revision 6c539a534c07917307c381f5af4053c6091b5278 | The complete metadata-only audit found that the strict common 512 x 512, PNG, RGB, explicitly non-NSFW gate leaves 1,005 real rows and 228,833 generated rows. This is a class/source shortcut, not a valid general primary corpus. No model is trained on it and the gate is not silently loosened. A separate face-only cohort may be studied later only under its own protocol. |
| Conditional controlled primary candidate | [B-Free training data](https://raw.githubusercontent.com/grip-unina/B-Free/main/training_data/README.md) | If the authors' official data server becomes reachable, the planned core comparison is COCO_real_512 against SD2.1_selfconditioned, with archive checksum verification, source-ID group split and a full byte/pixel audit. The corpus is not currently materialised; no unofficial repack will be substituted. |
| Exploratory sensitivity audit, not primary training | Local pinned [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset), revision `787334f7857fa54f29027a7f09c30e895ad486ef` | Native-384 caption-matched audit only. It preserves upstream roles after component exclusion and records residual file-size bias. It may document why source-scale normalisation remains insufficient; it cannot provide a HighRes-v1 result, select a model, or enable the interface. |
| Conditional internal candidate after lineage and licence audit | [DANI](https://huggingface.co/datasets/Renyang/DANI), revision 870e29fcdc13c405fae35442899e9ba1da11691d | Metadata-only and path-only scans request no image bytes. All 540,258 rows join exactly to the pinned D-Judge mapping, and all 5,000 parents plus 25,014 captions match the checksum-locked official COCO 2017 annotations. The local non-commercial selection accepts only Flickr licence IDs 2 and 4 and prohibits raw redistribution. Training remains blocked until a frozen parent-grouped selection and byte/pixel/shortcut audit pass. |
| Locked external benchmark | [Synthbuster](https://zenodo.org/records/10066460) + [RAISE-1k](https://loki.disi.unitn.it/RAISE/download.html) | Open only after the HighRes-v1 architecture, validation rule, seed protocol and threshold are frozen. RAISE must never enter training. |
| Conditional descriptive benchmark | [CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval), revision `7d4a74a88d2cac93b513c0853bf92c260eaceea0` | Do not use for training or selection. Before any score, run exact- and perceptual-hash contamination checks against the materialised corpus and respect its FFHQ/COCO split restrictions. A passed check supports a clearly labelled cross-dataset result; a failed or inconclusive check excludes it. |
| Optional second independent benchmark | [GenImage official test split](https://github.com/GenImage-Dataset/GenImage) | Evaluate only after the first external benchmark is locked. Report every generator separately rather than calling the whole set “unseen”. |

The source decision is a gate, not an implementation detail. A large nominal resolution does not
make a usable training source: the two labels must be jointly supported in a controlled source
stratum; source-level and semantic leakage groups must be reproducible; and decoded image evidence
must later confirm that format, resolution or export pipeline has not become the classifier.

For DANI in particular, the public non-binary schema exposes only an image-level index. A separate
path-only scan recovered candidate `parent_coco_image_id` and `coco_caption_id` values from locked
basenames, then an offline audit joined all 540,258 rows exactly to a D-Judge mapping pinned at
revision 6b877a12df94ddc4f68abb54db7912dc966d17e4 (mapping SHA-256
`d21590e888794de4faa768f2a36c3f00c8088fb23330bf0b1e1addd8437999e7`). The catalogue contains
5,000 candidate parents and 25,014 candidate caption pairs; every parent and pair occurs in both
labels. Row order, category, class identifier and generator name were not used as linkage evidence.
The subsequent official COCO audit matched all 5,000 parent filenames and all 25,014 caption
ID/parent/text tuples exactly against the checksum-locked 2017 train/validation annotation archive;
all parents belong to `val2017`. The conservative licence decision permits only local
non-commercial coursework use, keeps official image licence IDs 2 and 4, excludes NoDerivs,
ShareAlike and ambiguous records, and prohibits raw redistribution. This permits a metadata-only
selection to be frozen, but a byte/pixel audit and frozen group split are still required before
training.

## Immutable inclusion gate

The metadata scan reads no image bytes. Before any selected image is downloaded, a source-catalog
must first pass the source-decision gate above and then meet all of the following conditions:

| Field | Rule | Reason |
| --- | --- | --- |
| resolution | width = height = 512 in the first release, or a later explicitly recorded larger square | 384-pixel training never uses an upscaled source and both labels share geometry. A declared source size is not treated as decoded-byte proof. |
| decoded container and mode | A later byte-level audit confirms one allowed format and RGB mode for both labels, or explicitly records a balanced conversion policy | Prevent a container or colour-mode shortcut from becoming the classifier. |
| content suitability | A documented source rule excludes unsuitable material; missing metadata are not silently treated as safe | Keep the coursework corpus within its stated scope. |
| origin metadata | Binary origin is valid; real rows have a declared real source and synthetic rows have a nonempty generator/protocol | Preserve class provenance and reject contradictory metadata. |
| leakage group | A documented, reproducible parent/source group exists before split assignment | Do not infer pairing from row order, generic category, or a coincidental index. |
| provenance | Revision-pinned shard path and row number are recorded | A row can be reproduced without relying on mutable dataset ordering. |

The selection specification will also declare a deterministic seed, class balance, maximum quota
per generator, source/protocol and real-source strata, a documented group key, and the intended
train/validation/test roles. Its SHA-256 is part of the experiment identity. Quotas are
intentionally **not** guessed before the full metadata audit establishes how many valid, diverse
rows exist.

The selection stage is fail-closed. It must reject an incomplete or hash-mismatched catalog, a
duplicate source locator, a missing or contradictory documented group, contradictory origin
semantics, or an insufficient jointly supported source stratum for the two classes. It must also
record a deterministic rank for every eligible locator and a pre-ranked reserve. A damaged or
duplicate image may be replaced only by the next reserve item from that frozen rank: neither
manual curation nor a fresh random draw is permitted after image bytes have been inspected.

## Acquisition and leakage audit

1. Record a source lock: repository ID, immutable revision, licence, every source shard or archive,
   byte size, content identifier/checksum and scan date.
2. Column-scan only explicitly declared non-binary metadata fields. The binary image field is
   excluded at the Parquet read rather than removed after iteration. The source catalog stores a
   stable locator, declared dimensions, origin/generator strata and only permitted group metadata;
   it stores no image data or prompts.
3. Audit the complete source catalog before choosing a quota: class counts, declared resolution,
   real-source and generator/protocol distributions, source-shard distribution, repeated identifiers
   and whether a documented parent group is actually available. A partial scout is never eligible
   to become a research manifest.
4. Freeze the catalog-derived selection manifest. Only then materialise the exact selected rows.
   The materialiser must verify that each received row agrees with its pinned catalog metadata and
   that its actual decoded bytes meet the declared geometry/container/mode policy and are
   non-corrupt. For every accepted file record byte SHA-256, decoded-pixel SHA-256, perceptual hash,
   decoded dimensions and byte count. Original images and the complete local manifest remain outside
   Git.
5. Build the duplicate graph before assigning any HighRes-v1 partition. Its edges include
   documented parent/source groups, exact file or pixel hashes, and perceptual-hash candidates. An
   identical decoded image with conflicting labels is quarantined rather than silently assigned a
   preferred label; a perceptually close pair is a leakage boundary, not proof of identity. Exact
   duplicates within one label are reduced deterministically to one canonical instance.
6. Allocate splits by whole connected leakage components. Validation alone controls early stopping,
   hyperparameters, calibration and threshold. The internal test and all external sources remain
   untouched during those decisions.

If a source publishes an upstream split, preserve it as metadata. It is not silently replaced by a
random split, and its relationship to HighRes-v1 roles will be stated in the frozen selection
manifest. If the upstream field is documented and consistent, it becomes a hard boundary; if it is
not meaningful, the new component-level split and its objective are declared explicitly, with a
cross-tab against the upstream field retained in the final audit.

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
