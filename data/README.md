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

## HighRes-v1 source audits

HighRes-v1 is intentionally not a resized copy of Defactify. Source auditing is now
source-specific and writes ignored local records under `data/processed/`; no image bytes, raw
prompts or Arrow cache are committed. See [the HighRes-v1 protocol](../docs/HIGHRES_V1_PROTOCOL.md)
for the academic roles and freeze order.

The completed CommunityForensics-Small metadata audit is preserved as a rejected broad-source
decision: its strict common 512 x 512 PNG/RGB gate is overwhelmingly synthetic. Its scanner and
audit remain useful reproducibility evidence, but they must not be used to create a general
HighRes-v1 training corpus.

The DANI scanner has a separate schema and reads seven explicitly non-binary fields. The DANI
catalogue and its offline auditor both keep selection, split assignment, training and model
selection disabled because its documented public schema has no parent COCO/caption group. A
separate mapping audit must prove such a group before that status can change.

The following is the frozen historical CommunityForensics audit schema, retained for its rejected
source decision. It is not a generic HighRes-v1 materialisation contract:

```text
locator,repository_id,revision,shard_path,row_index,
image_name,label,source_width,source_height,format,mode,nsfw_flag,model_name,
architecture,subset,real_source,prompt_hash,content_group_id
```

`locator` has the revision-pinned form
`repository@revision:parquet-shard:row-index`. `prompt_hash` is an audit key, never an image
model input. Empty prompts receive a row-specific `prompt_group_id` so they do not become one false
leakage group. The historical CFS gate accepted only 512 x 512 RGB PNG rows with an explicit
non-NSFW flag; its resulting class imbalance is why it is not reused.

After the selected rows are downloaded, the final manifest adds decoded dimensions, exact file
SHA-256, perceptual hash, byte count, local path and the frozen experiment split. The split uses
leakage components rather than independent filenames and is audited again after hashes are known.

## Intended sources

1. **Defactify Image Dataset** is the historical 128 x 128 pilot benchmark.
   Its card reports 96,000 images: 16,000 real MS COCO images and 16,000 each from Stable Diffusion
   2.1, SDXL, SD 3, DALL-E 3 and Midjourney v6. It reports a 42k/9k/45k split. Before interpreting
   that split, `01_internal_training.ipynb` must audit normalized-caption and exact-pHash overlap.
2. **CommunityForensics-Small** at revision 6c539a534c07917307c381f5af4053c6091b5278 is a
   rejected broad-source candidate. Its Parquet scan excluded image_data and recorded the evidence
   for the rejection. It is research-only under its CC BY-NC-SA 4.0 release and raw data are never
   redistributed from this repository.
3. **B-Free core** is the conditional controlled primary candidate: COCO_real_512 versus
   SD2.1_selfconditioned. It may be considered only after the official archives are available,
   checksums and source-ID grouping are verified, and a byte/pixel audit is complete.
4. **DANI** at revision 870e29fcdc13c405fae35442899e9ba1da11691d is a metadata-only descriptive
   audit source. It cannot become an internal corpus until a separate revision-pinned mapping audit
   demonstrates a valid parent COCO/caption group.
5. **Synthbuster + RAISE-1k** is an external, locked test only. Its 9,000 synthetic images and
   1,000 real images are not to be used for threshold selection, early stopping or augmentations.
6. **CommunityForensics-Eval** is conditional descriptive external evidence only after exact/pHash
   contamination audit; its FFHQ/COCO split rules and prohibition on RAISE training must be
   respected. It cannot repair a model chosen on its own scores.
7. **CIFAKE** may be used only for a very small smoke run that proves the code executes. It is not
   representative evidence for the target task because its images are 32x32.

Raw data must retain original source and license information. The project does not assert that an
unlabelled aggregation can be commercially redistributed.
