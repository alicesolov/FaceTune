# Data policy and expected manifests

Raw source files and the canonical decoded image corpus are intentionally ignored by Git. The
Defactify preparation script decodes each source image and writes RGB PNG so that EXIF and container
metadata are not available as a shortcut feature; captions, original labels and file hashes remain in
the manifest. This does **not** remove pixel-level JPEG artefacts already present after decoding.
Every experiment reads a manifest with at least:

```text
path,label,split,generator,group_id,source_id
```

- `path`: path to a decoded image, relative to the repository root or absolute;
- `label`: `0` for real and `1` for AI-generated;
- `split`: `train`, `val`, `test`, or `external`;
- `generator`: `real` for photographs, otherwise a declared generator family;
- `group_id`: a normalized-caption grouping key used by the custom split;
- `source_id`: a local row identifier. When Defactify provides no source ID, preparation uses a
  `split:index` fallback; it is not provenance evidence or a leakage link.

Optional metadata such as `caption`, `width`, `height`, `format`, `file_bytes`, `sha256` and
`phash` are retained for audit only and never passed to an image model.

## Intended sources

1. **Defactify Image Dataset** is the starting train/validation/internal-test benchmark.
   Its card reports 96,000 images: 16,000 real MS COCO images and 16,000 each from Stable Diffusion
   2.1, SDXL, SD 3, DALL-E 3 and Midjourney v6. It reports a 42k/9k/45k split. Before interpreting
   that split, `01_internal_training.ipynb` must audit normalized-caption and exact-pHash overlap.
2. **Synthbuster + RAISE-1k** is an external, locked test only. Its 9,000 synthetic images and
   1,000 real images are not to be used for threshold selection, early stopping or augmentations.
3. **CIFAKE** may be used only for a very small smoke run that proves the code executes. It is not
   representative evidence for the target task because its images are 32x32.

Raw data must retain original source and license information. The project does not assert that an
unlabelled aggregation can be commercially redistributed.
