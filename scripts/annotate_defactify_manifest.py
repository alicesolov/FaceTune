"""Add documented Defactify generator labels to an already prepared manifest.

This is deliberately separate from download/preparation so that a long-running image conversion
can be completed safely before enriching its CSV metadata.  It never alters image bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ai_image_detector.reproducibility import sha256_file

LABEL_B_SOURCES = {
    "0": "real",
    "1": "sd21",
    "2": "sdxl",
    "3": "sd3",
    "4": "dalle3",
    "5": "midjourney_v6",
}


def normalise_raw(value: object) -> str:
    """Keep both numeric labels and possible Hub class names interoperable."""
    text = str(value).strip().lower().replace(" ", "").replace("_", "")
    aliases = {
        "real": "0",
        "sd21": "1",
        "stable-diffusion-2.1": "1",
        "sdxl": "2",
        "sd3": "3",
        "dalle3": "4",
        "dall-e3": "4",
        "midjourneyv6": "5",
        "midjourney-v6": "5",
    }
    return aliases.get(text, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--provenance", type=Path, help="Optional preparation provenance.json to update"
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    if "raw_generator" not in frame or "label" not in frame:
        raise SystemExit("Expected raw_generator and label columns in manifest")
    codes = frame["raw_generator"].map(normalise_raw)
    unknown = sorted(set(codes) - set(LABEL_B_SOURCES))
    if unknown:
        raise SystemExit(f"Unknown Defactify Label_B values: {unknown}")
    frame["generator"] = codes.map(LABEL_B_SOURCES)
    frame["label_b_consistent"] = (codes == "0") == (frame["label"].astype(int) == 0)
    if not bool(frame["label_b_consistent"].all()):
        raise SystemExit("Label_A and Label_B disagree; refusing to write an ambiguous manifest")
    frame.to_csv(args.manifest, index=False)
    digest = sha256_file(args.manifest)
    if args.provenance:
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        provenance["manifest_sha256"] = digest
        args.provenance.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
    print(
        {
            "rows": len(frame),
            "manifest_sha256": digest,
            "generators": frame["generator"].value_counts().to_dict(),
        }
    )


if __name__ == "__main__":
    main()
