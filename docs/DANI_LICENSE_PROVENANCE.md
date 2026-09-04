# DANI licence and provenance decision

**Decision:** conditional pass for local, non-commercial coursework research only.

This is a project data-governance decision, not legal advice. It permits the next metadata-only
selection step and later local training only under the restrictions below. It does not permit raw
image redistribution, commercial use, or publication of a model/artifact without a separate review.

## Locked evidence

| Layer | Locked evidence | Observed terms |
| --- | --- | --- |
| DANI dataset | `Renyang/DANI` revision `870e29fcdc13c405fae35442899e9ba1da11691d`; pinned README SHA-256 `8b9868b2f81f6144435badfaa9f23776cd8dc0504512e292f4c3f46c9ae38c51` | Dataset card declares CC BY-NC 4.0 and non-commercial research use. |
| D-Judge mapping and code | revision `6b877a12df94ddc4f68abb54db7912dc966d17e4`; pinned LICENSE SHA-256 `1a47fbc4163143b4d72ece042bb159898cc768dc6176abd1ba7b69b459e5e00b` | MIT licence for the repository materials. |
| COCO annotations | official `annotations_trainval2017.zip`, 252,907,541 bytes; MD5 `f4bbac642086de4f52a3fdda2de5fa2c`; SHA-256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` | COCO Consortium annotations are CC BY 4.0. |
| COCO website terms | website source revision `5e1c4da72464b1c6f068df0c02c91e3000ea62c4`; terms-file SHA-256 `bd019f88ee44c29b2f19c5b99888cf5bc2e7c16f57b6e52af8ed3a60462e8bdd` | COCO does not own image copyright; image use must follow the recorded Flickr licence/terms, and the user accepts responsibility. |

The offline identity audit proved that all 5,000 DANI parents are official COCO `val2017` images
and all 25,014 caption records match the official caption ID, parent ID, and text exactly. It also
retains each image's official COCO/Flickr licence ID. No caption text is emitted in audit artifacts.

## Conservative inclusion rule

The internal candidate pool accepts only official COCO image licence IDs:

- `2`: Attribution-NonCommercial 2.0;
- `4`: Attribution 2.0.

Licence IDs `1` and `5` are excluded to avoid a premature ShareAlike interpretation. IDs `3` and
`6` are excluded because they contain NoDerivs restrictions. ID `7` is excluded because “no known
copyright restrictions” is not a positive licence grant. This leaves 1,487 licence-eligible parent
images before generator-completeness checks and 1,482 parents with all five planned 1024-source
cells: one COCO real cell and four synthetic model/protocol cells.

## Enforced use conditions

1. Use is local, educational, non-commercial research for this coursework.
2. DANI, D-Judge, COCO, and the relevant Creative Commons licences must be attributed in the report.
3. Raw images, selected manifests containing local paths, caches, and audit artifacts remain ignored
   by Git and are not redistributed through the repository or the future website.
4. The website may accept a user upload and return a model score, but it must not expose training
   images. Publishing model weights or operating commercially requires a separate rights review.
5. Selection remains metadata-only. Training is still blocked until the selected image bytes pass
   geometry, format/mode, corruption, exact/perceptual duplicate, and shortcut audits.

Within those constraints, the licence/provenance gate is considered passed for building the frozen
metadata-only selection and requesting only its selected image bytes.
