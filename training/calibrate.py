"""Threshold calibration for editorial deployment.

Sweeps the decision threshold on the validation set and finds the operating
point that maximises F1 at a given production AI-prevalence (default 5%).

Usage:
    python -m training.calibrate
    python -m training.calibrate --prevalence 0.05 --weights-path model/model.pth
    python -m training.calibrate --prevalence 0.10 --show-curve
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.resnet import create_resnet50_binary_classifier, load_binary_classifier_weights
from training.dataset import DatasetConfig, DefactifyTorchDataset, load_defactify_split
from training.train import compute_metrics

DEFAULT_WEIGHTS_PATH = Path("model/model.pth")
DEFAULT_PREVALENCE = 0.05   # 5 % AI in production editorial flow
THRESHOLD_STEPS = 99        # sweep 0.01 … 0.99


# ── Probability calibration ───────────────────────────────────────────────────

def bayes_correct(
    p_train: float,
    train_prevalence: float = 0.833,
    prod_prevalence: float = DEFAULT_PREVALENCE,
) -> float:
    """Adjust a model probability from training prevalence to production prevalence.

    Purpose:
    The model is trained on data with 83 % AI images. In production, AI images
    are 3–10 % of all submissions. Raw model probabilities are systematically
    inflated — this corrects them via Bayes' theorem.

    Inputs:
    - p_train: Raw P(AI|x) from the model.
    - train_prevalence: Fraction of AI images in training data.
    - prod_prevalence: Expected fraction of AI images in production.

    Outputs:
    - Calibrated P(AI|x) for the production prevalence.

    Key assumptions:
    - The likelihood ratio P(x|AI) / P(x|Real) is the same in train and prod.

    Failure modes:
    - Returns 0.0 if p_train is 0; returns 1.0 if p_train is 1.
    """
    if p_train <= 0.0:
        return 0.0
    if p_train >= 1.0:
        return 1.0

    # Likelihood ratio from training probabilities
    lr = (p_train / train_prevalence) / ((1.0 - p_train) / (1.0 - train_prevalence))
    # Apply production prior
    p_prod = lr * prod_prevalence / (lr * prod_prevalence + (1.0 - prod_prevalence))
    return float(p_prod)


# ── Collect raw probabilities ─────────────────────────────────────────────────

def collect_probabilities(
    weights_path: Path,
    cache_dir: str | None = None,
    split: str = "validation",
    batch_size: int = 64,
) -> tuple[list[float], list[int]]:
    """Run the trained model on a dataset split and return raw P(AI) values.

    Purpose:
    Collect raw model outputs (before thresholding) so calibrate() can sweep
    thresholds without re-running inference for each candidate.

    Inputs:
    - weights_path: Path to the .pth checkpoint.
    - cache_dir: Optional HuggingFace cache directory.
    - split: Dataset split to evaluate ('validation' or 'test').
    - batch_size: Inference batch size.

    Outputs:
    - Tuple of (ai_probs, labels) — one entry per sample.

    Failure modes:
    - Raises FileNotFoundError if the checkpoint is missing.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at '{weights_path}'. Run `python -m training.train` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    checkpoint = torch.load(weights_path, map_location="cpu")
    model_name = checkpoint.get("model_name", "resnet50")
    logging.info("Checkpoint model: %s  epoch: %s", model_name, checkpoint.get("epoch"))

    if model_name == "mobilenet":
        from model.mobilenet import create_mobilenet_v3_classifier, load_mobilenet_weights
        model = create_mobilenet_v3_classifier(pretrained=False)
        load_mobilenet_weights(model, str(weights_path))
    else:
        model = create_resnet50_binary_classifier(pretrained=False)
        load_binary_classifier_weights(model, str(weights_path))

    model.to(device).eval()

    hf_ds = load_defactify_split(DatasetConfig(split=split, cache_dir=cache_dir))
    torch_ds = DefactifyTorchDataset(hf_ds)
    loader = DataLoader(torch_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ai_probs: list[float] = []
    labels: list[int] = []

    logging.info("Running inference on '%s' split (%d rows) …", split, len(torch_ds))
    with torch.inference_mode():
        for step, (inputs, batch_labels) in enumerate(loader):
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)[:, 1]  # P(AI)
            ai_probs.extend(probs.cpu().tolist())
            labels.extend(batch_labels.tolist())
            if (step + 1) % 50 == 0:
                logging.info("  %d / %d batches", step + 1, len(loader))

    logging.info("Collected %d probabilities.", len(ai_probs))
    return ai_probs, labels


# ── Threshold sweep ───────────────────────────────────────────────────────────

def sweep_thresholds(
    ai_probs: list[float],
    labels: list[int],
    prod_prevalence: float = DEFAULT_PREVALENCE,
    train_prevalence: float = 0.833,
) -> list[dict]:
    """Sweep decision thresholds and compute metrics at each point.

    Purpose:
    Find the threshold that maximises F1 under the production prevalence.
    Metrics are computed on Bayes-corrected probabilities.

    Inputs:
    - ai_probs: Raw P(AI) from the model (training prevalence).
    - labels: Ground-truth binary labels (0=real, 1=ai_generated).
    - prod_prevalence: Target production AI prevalence for calibration.
    - train_prevalence: AI prevalence in training data.

    Outputs:
    - List of dicts sorted by threshold, each with threshold, accuracy,
      precision, recall, f1, and coverage (fraction not classified as uncertain).
    """
    calibrated = [
        bayes_correct(p, train_prevalence, prod_prevalence) for p in ai_probs
    ]

    results = []
    for step in range(1, THRESHOLD_STEPS + 1):
        threshold = step / (THRESHOLD_STEPS + 1)
        preds = [1 if p >= threshold else 0 for p in calibrated]
        metrics = compute_metrics(preds, labels)
        coverage = sum(1 for p in calibrated if p < 0.35 or p >= threshold) / len(calibrated)
        results.append({
            "threshold": round(threshold, 4),
            **{k: round(v, 4) for k, v in metrics.items()},
            "coverage": round(coverage, 4),
        })

    return results


def find_optimal_threshold(
    sweep: list[dict],
    min_precision: float = 0.70,
) -> dict:
    """Find the F1-maximising threshold subject to a minimum precision constraint.

    Purpose:
    In a newsroom context, false positives (calling a real photo AI) waste
    editor time. We require precision >= min_precision before maximising F1.

    Inputs:
    - sweep: Output of sweep_thresholds().
    - min_precision: Minimum acceptable precision for the AI class.

    Outputs:
    - The best row from sweep, or the row with highest recall if no threshold
      meets the precision constraint.
    """
    eligible = [r for r in sweep if r["precision"] >= min_precision]
    if eligible:
        return max(eligible, key=lambda r: r["f1"])
    # Fallback: highest recall (catch as many AI as possible)
    return max(sweep, key=lambda r: r["recall"])


# ── Main calibration entry point ──────────────────────────────────────────────

def calibrate(
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    prevalence: float = DEFAULT_PREVALENCE,
    cache_dir: str | None = None,
    split: str = "validation",
    show_curve: bool = False,
    min_precision: float = 0.70,
) -> dict:
    """Full calibration pipeline: collect → sweep → recommend thresholds.

    Purpose:
    Produce calibrated operating thresholds for the given production AI
    prevalence. Prints a recommendation and the full sweep for inspection.

    Inputs:
    - weights_path: Path to trained checkpoint.
    - prevalence: Expected fraction of AI images in production (0–1).
    - cache_dir: Optional HuggingFace cache directory.
    - split: Dataset split to calibrate on ('validation' recommended).
    - show_curve: If True, include the full threshold sweep in output.
    - min_precision: Minimum precision required before maximising F1.

    Outputs:
    - Dict with 'recommended_threshold', 'uncertain_lower', 'uncertain_upper',
      and 'sweep' (if show_curve=True).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ai_probs, labels = collect_probabilities(weights_path, cache_dir, split)
    sweep = sweep_thresholds(ai_probs, labels, prod_prevalence=prevalence)
    best = find_optimal_threshold(sweep, min_precision=min_precision)

    # The "uncertain" lower bound is set conservatively at half the AI threshold.
    uncertain_lower = round(best["threshold"] * 0.40, 2)
    uncertain_upper = round(best["threshold"], 2)

    logging.info("═══ Calibration results (prevalence=%.1f%%) ═══", prevalence * 100)
    logging.info(
        "  Optimal AI threshold: %.4f  (prec=%.3f  rec=%.3f  f1=%.3f  coverage=%.1f%%)",
        best["threshold"], best["precision"], best["recall"],
        best["f1"], best["coverage"] * 100,
    )
    logging.info(
        "  Recommended zones:  Real < %.2f  |  Uncertain %.2f–%.2f  |  AI > %.2f",
        uncertain_lower, uncertain_lower, uncertain_upper, uncertain_upper,
    )

    result: dict = {
        "prevalence": prevalence,
        "split": split,
        "recommended_threshold": best["threshold"],
        "uncertain_lower": uncertain_lower,
        "uncertain_upper": uncertain_upper,
        "best_metrics": best,
    }
    if show_curve:
        result["sweep"] = sweep

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate decision thresholds.")
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--prevalence", type=float, default=DEFAULT_PREVALENCE,
                        help="Production AI prevalence 0–1 (default: 0.05 = 5 %%)")
    parser.add_argument("--split", default="validation",
                        help="Dataset split to calibrate on (default: validation)")
    parser.add_argument("--min-precision", type=float, default=0.70,
                        help="Minimum AI precision before maximising F1 (default: 0.70)")
    parser.add_argument("--show-curve", action="store_true",
                        help="Include full threshold sweep in JSON output")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    result = calibrate(
        weights_path=Path(args.weights_path),
        prevalence=args.prevalence,
        cache_dir=args.cache_dir,
        split=args.split,
        show_curve=args.show_curve,
        min_precision=args.min_precision,
    )
    print(json.dumps(result, indent=2))
