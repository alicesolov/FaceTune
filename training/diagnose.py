"""False positive diagnosis for real photographs.

Runs the trained model on all Real images in a split, collects the ones
incorrectly classified as AI-generated, and reports FFT statistics and
caption patterns to identify systematic biases in the training data.

Usage:
    python -m training.diagnose
    python -m training.diagnose --split test --top-n 100 --weights-path model/model.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocessing.fft_transform import fft_preprocess
from preprocessing.visualizations import compute_power_spectrum_stats
from training.dataset import DatasetConfig, load_defactify_split

DEFAULT_WEIGHTS_PATH = Path("model/model.pth")
DEFAULT_TOP_N = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location="cpu")
    model_name = checkpoint.get("model_name", "resnet50")
    logging.info("Architecture: %s  epoch: %s", model_name, checkpoint.get("epoch"))

    if model_name == "mobilenet":
        from model.mobilenet import create_mobilenet_v3_classifier, load_mobilenet_weights
        model = create_mobilenet_v3_classifier(pretrained=False)
        load_mobilenet_weights(model, str(weights_path))
    else:
        from model.resnet import create_resnet50_binary_classifier, load_binary_classifier_weights
        model = create_resnet50_binary_classifier(pretrained=False)
        load_binary_classifier_weights(model, str(weights_path))

    return model.to(device).eval()


def _fft_stats(image) -> dict[str, float]:
    """Compute FFT power spectrum statistics for a single PIL image."""
    try:
        return compute_power_spectrum_stats(image)
    except Exception:
        return {"mean": 0.0, "std": 0.0, "center_energy": 0.0, "edge_energy": 0.0}


# ── Main diagnosis ────────────────────────────────────────────────────────────

def diagnose(
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    split: str = "test",
    top_n: int = DEFAULT_TOP_N,
    cache_dir: str | None = None,
) -> dict:
    """Collect and analyse false positive real photographs.

    Purpose:
    Identify systematic patterns in real images that the model mislabels as
    AI-generated. These patterns indicate training data gaps that can be
    addressed with hard negative mining or targeted data augmentation.

    Inputs:
    - weights_path: Trained model checkpoint.
    - split: Dataset split to scan ('test' recommended for unbiased diagnosis).
    - top_n: Number of highest-confidence false positives to surface.
    - cache_dir: Optional HuggingFace cache directory.

    Outputs:
    - Dict with aggregate statistics and top-N false positive examples,
      each including caption text, P(AI) score, and FFT spectrum statistics.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not weights_path.exists():
        raise FileNotFoundError(f"No checkpoint at '{weights_path}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(weights_path, device)

    logging.info("Loading '%s' split …", split)
    hf_ds = load_defactify_split(DatasetConfig(split=split, cache_dir=cache_dir))
    logging.info("  %d rows total", len(hf_ds))

    # Collect only real images (Label_A == 0)
    real_indices = [i for i, item in enumerate(hf_ds) if int(item["Label_A"]) == 0]
    logging.info("  %d real images", len(real_indices))

    false_positives: list[dict] = []
    true_negatives_count = 0

    logging.info("Running inference on real images …")
    for i, idx in enumerate(real_indices):
        item = hf_ds[idx]
        image = item["Image"]
        caption = str(item.get("Caption", ""))

        tensor = fft_preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        ai_prob = float(probs[1].item())

        if ai_prob >= 0.5:
            stats = _fft_stats(image)
            false_positives.append({
                "dataset_index": idx,
                "caption": caption,
                "ai_prob": round(ai_prob, 4),
                "fft_mean": round(stats["mean"], 4),
                "fft_std": round(stats["std"], 4),
                "center_energy": round(stats["center_energy"], 4),
                "edge_energy": round(stats["edge_energy"], 4),
                "center_edge_ratio": round(
                    stats["center_energy"] / stats["edge_energy"]
                    if stats["edge_energy"] > 0 else 0.0, 4
                ),
            })
        else:
            true_negatives_count += 1

        if (i + 1) % 500 == 0:
            logging.info(
                "  %d / %d  FP so far: %d",
                i + 1, len(real_indices), len(false_positives),
            )

    fpr = len(false_positives) / len(real_indices) if real_indices else 0.0
    logging.info(
        "FPR = %.1f%%  (%d / %d real images misclassified)",
        fpr * 100, len(false_positives), len(real_indices),
    )

    # Sort by confidence descending
    false_positives.sort(key=lambda x: -x["ai_prob"])

    # Caption word frequency in top-N FP
    top_fp = false_positives[:top_n]
    all_words = []
    for fp in top_fp:
        all_words.extend(fp["caption"].lower().split())
    top_words = Counter(all_words).most_common(20)

    # FFT statistic summary over all false positives
    if false_positives:
        fp_ratios = [fp["center_edge_ratio"] for fp in false_positives]
        fft_summary = {
            "mean_center_edge_ratio": round(float(np.mean(fp_ratios)), 4),
            "std_center_edge_ratio": round(float(np.std(fp_ratios)), 4),
            "min": round(float(np.min(fp_ratios)), 4),
            "max": round(float(np.max(fp_ratios)), 4),
        }
    else:
        fft_summary = {}

    # AI probability distribution of FP
    if false_positives:
        ai_probs_fp = [fp["ai_prob"] for fp in false_positives]
        prob_buckets = {
            "0.50-0.70": sum(1 for p in ai_probs_fp if 0.50 <= p < 0.70),
            "0.70-0.90": sum(1 for p in ai_probs_fp if 0.70 <= p < 0.90),
            "0.90-1.00": sum(1 for p in ai_probs_fp if p >= 0.90),
        }
    else:
        prob_buckets = {}

    logging.info("P(AI) distribution of false positives: %s", prob_buckets)
    logging.info("Top caption words in FP: %s", top_words[:10])
    logging.info("FFT center/edge ratio summary: %s", fft_summary)

    return {
        "split": split,
        "real_total": len(real_indices),
        "false_positives_total": len(false_positives),
        "fpr": round(fpr, 4),
        "prob_distribution": prob_buckets,
        "fft_center_edge_ratio": fft_summary,
        "top_caption_words": top_words,
        f"top_{top_n}_false_positives": top_fp,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose false positive real images.")
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--split", default="test",
                        help="Dataset split to diagnose (default: test)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Top-N false positives to report (default: {DEFAULT_TOP_N})")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    result = diagnose(
        weights_path=Path(args.weights_path),
        split=args.split,
        top_n=args.top_n,
        cache_dir=args.cache_dir,
    )
    # Print summary without the full FP list unless redirected
    summary = {k: v for k, v in result.items() if not k.startswith("top_")}
    print(json.dumps(summary, indent=2))
    print(f"\n(Top-{args.top_n} FP list omitted from stdout — use > file.json to save)")
