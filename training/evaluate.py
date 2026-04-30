"""Evaluation pipeline for the trained AI image authenticity classifier.

Loads a saved checkpoint, runs it against the test split, and reports
accuracy, precision, recall, F1, and a confusion matrix.

Usage:
    python -m training.evaluate
    python -m training.evaluate --split test --weights-path model/model.pth
    python -m training.evaluate --split validation --cache-dir /data/hf_cache
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.resnet import create_resnet50_binary_classifier, load_binary_classifier_weights
from training.dataset import DatasetConfig, DefactifyTorchDataset, load_defactify_split
from training.train import compute_metrics

DEFAULT_WEIGHTS_PATH = Path("model/model.pth")
DEFAULT_BATCH_SIZE = 64
CLASS_NAMES = ["real", "ai_generated"]


# ── Confusion matrix ──────────────────────────────────────────────────────────

def build_confusion_matrix(
    preds: list[int], labels: list[int], num_classes: int = 2
) -> list[list[int]]:
    """Build a confusion matrix from flat prediction and label lists.

    Purpose:
    Surface per-class error patterns (false positives and false negatives)
    that aggregate accuracy metrics can hide.

    Inputs:
    - preds: Predicted class indices.
    - labels: Ground-truth class indices.
    - num_classes: Number of output classes.

    Outputs:
    - Square matrix where cm[true][pred] is the count.

    Key assumptions:
    - Class indices are contiguous from 0 to num_classes-1.

    Failure modes:
    - Silently ignores out-of-range indices.
    """
    cm = [[0] * num_classes for _ in range(num_classes)]
    for pred, label in zip(preds, labels):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            cm[label][pred] += 1
    return cm


def format_confusion_matrix(cm: list[list[int]], class_names: list[str]) -> str:
    """Return a human-readable confusion matrix string.

    Purpose:
    Produce a plain-text confusion matrix suitable for logging.

    Inputs:
    - cm: Square matrix from build_confusion_matrix.
    - class_names: Display names for each class index.

    Outputs:
    - Multi-line string with labeled rows and columns.

    Key assumptions:
    - cm dimensions match len(class_names).

    Failure modes:
    - Raises IndexError if dimensions mismatch.
    """
    col_width = max(len(n) for n in class_names) + 2
    header = "True \\ Pred".ljust(col_width) + "  ".join(
        n.center(col_width) for n in class_names
    )
    rows = [header, "-" * len(header)]
    for i, row in enumerate(cm):
        row_str = class_names[i].ljust(col_width) + "  ".join(
            str(v).center(col_width) for v in row
        )
        rows.append(row_str)
    return "\n".join(rows)


# ── Generator-slice metrics ───────────────────────────────────────────────────

def compute_per_generator_metrics(
    preds: list[int],
    labels: list[int],
    generator_labels: list[int],
) -> dict[str, dict]:
    """Compute per-generator-family metrics on the AI-generated subset.

    Purpose:
    Reveal whether the classifier generalises equally across all generator
    families (SD, DALL-E, Midjourney) or is biased toward specific ones.

    Inputs:
    - preds: Predicted class indices for the full split.
    - labels: Ground-truth Label_A values for the full split.
    - generator_labels: Label_B values for the full split.

    Outputs:
    - Mapping from generator name to a metrics dict.

    Key assumptions:
    - Label_B follows the documented 0-5 mapping.
    - Only ai_generated rows (Label_A == 1) are included in per-generator slices.

    Failure modes:
    - Returns empty dict if no AI-generated rows are present.
    """
    gen_name = {1: "SD2.1", 2: "SDXL", 3: "SD3", 4: "DALL-E 3", 5: "Midjourney"}
    slices: dict[int, tuple[list[int], list[int]]] = {}

    for pred, label, gen in zip(preds, labels, generator_labels):
        if label == 1 and gen in gen_name:
            if gen not in slices:
                slices[gen] = ([], [])
            slices[gen][0].append(pred)
            slices[gen][1].append(label)

    return {
        gen_name[gen_id]: compute_metrics(p, l)
        for gen_id, (p, l) in sorted(slices.items())
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    split: str = "test",
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device_str: str = "auto",
    cache_dir: str | None = None,
) -> dict:
    """Load the trained checkpoint and evaluate it on a dataset split.

    Purpose:
    Produce a comprehensive evaluation report including aggregate metrics,
    confusion matrix, and per-generator-family performance slice.

    Inputs:
    - split: Dataset split to evaluate on.
    - weights_path: Path to the .pth checkpoint file.
    - batch_size: Samples per evaluation batch.
    - device_str: Compute device ("auto", "cpu", or "cuda").
    - cache_dir: Optional Hugging Face dataset cache directory.

    Outputs:
    - Dictionary with metrics, confusion_matrix, and per_generator keys.

    Key assumptions:
    - The checkpoint was produced by training.train and is compatible with
      the current ResNet-50 binary classifier architecture.

    Failure modes:
    - Raises FileNotFoundError if the checkpoint does not exist.
    - Propagates dataset loading errors when the cache is empty.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logging.info("Device: %s", device)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at '{weights_path}'. "
            "Run `python -m training.train` first."
        )

    logging.info("Loading checkpoint from '%s' …", weights_path)
    model = create_resnet50_binary_classifier(pretrained=False)
    model = load_binary_classifier_weights(model, str(weights_path))
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(weights_path, map_location="cpu")
    saved_epoch = checkpoint.get("epoch", "unknown")
    saved_acc = checkpoint.get("val_accuracy", None)
    logging.info(
        "Checkpoint: epoch=%s  saved_val_accuracy=%s",
        saved_epoch,
        f"{saved_acc:.4f}" if saved_acc is not None else "n/a",
    )

    logging.info("Loading '%s' split …", split)
    hf_dataset = load_defactify_split(DatasetConfig(split=split, cache_dir=cache_dir))
    logging.info("  %d rows", len(hf_dataset))

    torch_ds = DefactifyTorchDataset(hf_dataset)
    loader = DataLoader(torch_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_gen_labels: list[int] = []

    with torch.inference_mode():
        for step, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

            if (step + 1) % 50 == 0:
                logging.info("  evaluated %d / %d batches", step + 1, len(loader))

    # Collect Label_B for per-generator slicing
    for item in hf_dataset:
        all_gen_labels.append(int(item.get("Label_B", -1)))

    metrics = compute_metrics(all_preds, all_labels)
    cm = build_confusion_matrix(all_preds, all_labels)
    per_gen = compute_per_generator_metrics(all_preds, all_labels, all_gen_labels)

    logging.info(
        "Results on '%s': acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
        split, metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
    )
    logging.info("Confusion matrix:\n%s", format_confusion_matrix(cm, CLASS_NAMES))

    if per_gen:
        logging.info("Per-generator recall on AI-generated images:")
        for gen_name, gen_m in per_gen.items():
            logging.info("  %-12s  recall=%.4f  f1=%.4f", gen_name, gen_m["recall"], gen_m["f1"])

    return {
        "split": split,
        "metrics": metrics,
        "confusion_matrix": cm,
        "per_generator": per_gen,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate the trained AI image authenticity classifier."
    )
    parser.add_argument("--split", default="test",
                        help="Dataset split: train | validation | test")
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH),
                        help=f"Checkpoint path (default: {DEFAULT_WEIGHTS_PATH})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto",
                        help="Compute device: auto | cpu | cuda")
    parser.add_argument("--cache-dir", default=None,
                        help="Hugging Face dataset cache directory")
    args = parser.parse_args()

    report = evaluate(
        split=args.split,
        weights_path=Path(args.weights_path),
        batch_size=args.batch_size,
        device_str=args.device,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, default=str))
