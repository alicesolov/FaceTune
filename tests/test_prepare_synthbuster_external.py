import hashlib
import json
import subprocess
import sys
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_synthbuster_external.py"


def write_pattern(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (48, 36), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 5, 24, 27), fill=(250, 250, 250))
    draw.line((0, 0, 47, 35), fill=(0, 0, 0), width=3)
    image.save(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_local_external_manifest_and_overlap_audit(tmp_path: Path) -> None:
    synthetic_root = tmp_path / "synthetic"
    synthetic_image = synthetic_root / "dalle2" / "synthetic.png"
    synthetic_image.parent.mkdir(parents=True)
    write_pattern(synthetic_image, (200, 10, 20))

    raise_root = tmp_path / "raise"
    raise_root.mkdir()
    real_image = raise_root / "real.png"
    write_pattern(real_image, (10, 30, 200))

    with Image.open(real_image) as opened:
        reference_phash = str(imagehash.phash(opened.convert("RGB")))
    reference = pd.DataFrame(
        {
            "path": [str(real_image)],
            "label": [0],
            "split": ["test"],
            "generator": ["real"],
            "group_id": ["reference-real"],
            "source_id": ["reference-real"],
            "sha256": [sha256(real_image)],
            "phash": [reference_phash],
        }
    )
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)

    output_root = tmp_path / "prepared"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--synthetic-root",
            str(synthetic_root),
            "--raise-root",
            str(raise_root),
            "--output-root",
            str(output_root),
            "--reference-manifest",
            str(reference_path),
            "--allow-partial",
        ],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert '"complete": false' in completed.stdout

    manifest = pd.read_csv(output_root / "manifest.csv")
    assert set(manifest["split"]) == {"external"}
    assert set(manifest["generator"]) == {"real", "dalle2"}
    assert set(manifest["label"]) == {0, 1}
    assert set(manifest["defactify_train_relation"]) == {
        "new_real_domain",
        "same_family_different_version",
    }
    assert manifest["pixel_sha256"].str.len().eq(64).all()

    exact = pd.read_csv(output_root / "exact_file_hash_matches.csv")
    assert len(exact) == 1
    assert exact.iloc[0]["external_source_id"] == "raise1k:real:real.png"
    candidates = pd.read_csv(output_root / "phash_near_duplicate_candidates.csv")
    assert "raise1k:real:real.png" in set(candidates["external_source_id"])

    provenance = json.loads((output_root / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["integrity"]["complete"] is False
    assert provenance["defactify_overlap_audit"]["exact_file_hash_matches"] == 1


def test_complete_counts_are_required_without_allow_partial(tmp_path: Path) -> None:
    synthetic_root = tmp_path / "synthetic"
    synthetic_image = synthetic_root / "dalle2" / "synthetic.png"
    synthetic_image.parent.mkdir(parents=True)
    write_pattern(synthetic_image, (200, 10, 20))
    raise_root = tmp_path / "raise"
    raise_root.mkdir()
    write_pattern(raise_root / "real.png", (10, 30, 200))

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--synthetic-root",
            str(synthetic_root),
            "--raise-root",
            str(raise_root),
            "--output-root",
            str(tmp_path / "prepared"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Missing Synthbuster generator directories" in completed.stderr
