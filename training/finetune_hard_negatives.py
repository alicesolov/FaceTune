"""Fine-tune the current model on hard negative real photographs.

Hard negatives are real images that the current model misclassifies as
AI-generated (false positives).  Adding them to training with extra sample
weight targets specific failure modes without a full retraining cycle.

Workflow:
    Step 1 — collect FP indices (run on val to keep test unbiased):
        python -m training.diagnose --split validation --top-n 2000 --save-json fp.json

    Step 2 — fine-tune from the existing checkpoint:
        python -m training.finetune_hard_negatives --fp-json fp.json

    Step 3 — evaluate on test split (val was used for training):
        python -m training.evaluate --split test --weights-path model/model_hn.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from preprocessing.fft_transform import fft_preprocess
from training.dataset import DatasetConfig, DefactifyTorchDataset, load_defactify_split
from training.train import (
    EarlyStopping,
    _build_model,
    compute_metrics,
    save_checkpoint,
    train_one_epoch,
    DEFAULT_BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
)

DEFAULT_HARD_NEG_WEIGHT = 2.5
DEFAULT_FINETUNE_EPOCHS = 5
DEFAULT_FINETUNE_LR = 1e-5
DEFAULT_BASE_WEIGHTS = Path("model/model.pth")
DEFAULT_OUTPUT_PATH = Path("model/model_hn.pth")


class _HardNegativeDataset(torch.utils.data.Dataset):
    """Real photos (label=0) identified as false positives by diagnose.py."""

    def __init__(self, hf_dataset, indices: list[int]) -> None:
        self.dataset = hf_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[self.indices[idx]]
        return fft_preprocess(item["Image"]), 0  # Real


def _load_fp_indices(fp_json: Path) -> tuple[list[int], str]:
    """Parse diagnose.py JSON → (dataset_index list, recorded split name)."""
    with fp_json.open() as f:
        data = json.load(f)

    top_key = next(
        (k for k in data if k.startswith("top_") and "false_positive" in k),
        None,
    )
    if top_key is None:
        raise ValueError(
            f"No 'top_N_false_positives' key found in {fp_json}.\n"
            "Re-run: python -m training.diagnose --top-n 2000 --save-json fp.json"
        )

    fp_list = data[top_key]
    indices = [entry["dataset_index"] for entry in fp_list]
    recorded_split = data.get("split", "unknown")
    logging.info(
        "Loaded %d FP indices from '%s' (recorded split: %s)",
        len(indices), fp_json.name, recorded_split,
    )
    return indices, recorded_split


def _load_checkpoint_model(weights_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(weights_path, map_location="cpu")
    model_name = checkpoint.get("model_name", "resnet50")
    logging.info("Checkpoint: arch=%s  epoch=%s", model_name, checkpoint.get("epoch"))
    model = _build_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model._train_model_name = model_name
    return model.to(device)


def _make_hard_neg_sampler(
    train_labels: list[int],
    n_hard_neg: int,
    hard_neg_weight: float,
) -> WeightedRandomSampler:
    """Class-balanced sampler with extra weight on hard negatives.

    Hard negatives are real photos (class 0) that the model consistently
    misclassifies.  They receive hard_neg_weight × the normal Real weight
    so the model corrects these specific failure modes without swamping the
    overall class balance signal.
    """
    counts = Counter(train_labels)
    class_w = {cls: 1.0 / count for cls, count in counts.items()}

    weights = [class_w[lbl] for lbl in train_labels]
    base_real_w = class_w.get(0, 1.0)
    weights.extend([base_real_w * hard_neg_weight] * n_hard_neg)

    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )


def _validate_with_fpr(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], float]:
    """Validation pass that also computes FPR on the real-photo subset."""
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            loss = criterion(logits, labels.to(device))
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_preds, all_labels)

    real_total = sum(1 for l in all_labels if l == 0)
    fp_count = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fpr = fp_count / real_total if real_total > 0 else 0.0

    return running_loss / len(loader), metrics, fpr


def finetune(
    fp_json: Path,
    fp_split: str = "validation",
    weights_path: Path = DEFAULT_BASE_WEIGHTS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    epochs: int = DEFAULT_FINETUNE_EPOCHS,
    lr: float = DEFAULT_FINETUNE_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hard_neg_weight: float = DEFAULT_HARD_NEG_WEIGHT,
    device_str: str = "auto",
    cache_dir: str | None = None,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> None:
    """Fine-tune an existing checkpoint on hard negatives from diagnose.py.

    Purpose:
    Rapidly correct the specific false positive patterns identified by
    diagnosis without a full training cycle.  The original training set is
    augmented with the misclassified real photos at elevated sample weight.

    Inputs:
    - fp_json: Full JSON output of diagnose.py (requires --save-json flag).
    - fp_split: HF split the FP indices belong to ('validation' or 'test').
    - weights_path: Starting checkpoint produced by training.train.
    - output_path: Destination for the fine-tuned checkpoint.
    - epochs: Maximum fine-tuning epochs (3-5 recommended).
    - lr: Learning rate — 10x lower than full training (default 1e-5).
    - hard_neg_weight: Per-sample multiplier for hard negatives vs. normal Real.

    Note:
    The monitoring split is automatically chosen as the opposite of fp_split
    (validation → monitor on test; test → monitor on validation) so that
    hard negatives never appear in the split used to track FPR during training.

    Outputs:
    - Fine-tuned checkpoint saved at output_path (best validation F1).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.manual_seed(RANDOM_SEED)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    logging.info("Device: %s", device)

    model = _load_checkpoint_model(weights_path, device)

    fp_indices, recorded_split = _load_fp_indices(fp_json)
    if recorded_split != fp_split:
        logging.warning(
            "JSON was produced on split '%s' but --fp-split='%s'. "
            "Verify the split matches the dataset_index values.",
            recorded_split, fp_split,
        )

    logging.info("Loading '%s' split for hard negatives …", fp_split)
    fp_hf = load_defactify_split(DatasetConfig(split=fp_split, cache_dir=cache_dir))
    hard_neg_ds = _HardNegativeDataset(fp_hf, fp_indices)
    logging.info(
        "Hard negatives: %d images  (%.1f× weight vs. normal Real)",
        len(hard_neg_ds), hard_neg_weight,
    )

    logging.info("Loading train split …")
    train_hf = load_defactify_split(DatasetConfig(split="train", cache_dir=cache_dir))
    train_ds = DefactifyTorchDataset(train_hf)
    train_labels = train_ds.get_all_labels()
    logging.info("Train base: %d images", len(train_ds))

    # Use the opposite split for monitoring to avoid evaluating on images that
    # are now part of the training set (hard negatives came from fp_split).
    monitor_split = "test" if fp_split == "validation" else "validation"
    logging.info("Loading '%s' split for monitoring (fp_split='%s') …", monitor_split, fp_split)
    val_hf = load_defactify_split(DatasetConfig(split=monitor_split, cache_dir=cache_dir))
    val_ds = DefactifyTorchDataset(val_hf)

    combined_ds = ConcatDataset([train_ds, hard_neg_ds])
    sampler = _make_hard_neg_sampler(train_labels, len(hard_neg_ds), hard_neg_weight)

    train_loader = DataLoader(
        combined_ds, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logging.info(
        "Combined train: %d samples (%d base + %d hard neg)  |  val: %d",
        len(combined_ds), len(train_ds), len(hard_neg_ds), len(val_ds),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    stopper = EarlyStopping(patience=patience)

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        logging.info("── Epoch %d / %d ────────────────────────────────", epoch, epochs)

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        val_loss, val_m, fpr = _validate_with_fpr(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        logging.info(
            "  train  loss=%.4f  acc=%.4f  f1=%.4f",
            train_loss, train_m["accuracy"], train_m["f1"],
        )
        logging.info(
            "  val    loss=%.4f  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  FPR=%.1f%%  (%.0fs)",
            val_loss, val_m["accuracy"], val_m["precision"],
            val_m["recall"], val_m["f1"], fpr * 100, elapsed,
        )

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            save_checkpoint(model, epoch, val_m, output_path)

        stopper.step(val_m["f1"])
        if stopper.should_stop:
            logging.info("Early stopping at epoch %d.", epoch)
            break

    logging.info(
        "Fine-tuning complete.  Best val F1: %.4f  →  %s", best_val_f1, output_path
    )
    logging.info(
        "Monitored on '%s' split during training.  "
        "Run: python -m training.evaluate --split test --weights-path %s",
        monitor_split, output_path,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune on hard negatives produced by diagnose.py."
    )
    parser.add_argument(
        "--fp-json", required=True,
        help="JSON output from diagnose.py --save-json (contains top_N_false_positives)",
    )
    parser.add_argument(
        "--fp-split", default="validation", choices=["validation", "test"],
        help="HF split the FP indices belong to (default: validation)",
    )
    parser.add_argument(
        "--weights-path", default=str(DEFAULT_BASE_WEIGHTS),
        help=f"Starting checkpoint (default: {DEFAULT_BASE_WEIGHTS})",
    )
    parser.add_argument(
        "--output-path", default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output checkpoint (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS,
                        help=f"Max fine-tuning epochs (default: {DEFAULT_FINETUNE_EPOCHS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_FINETUNE_LR,
                        help=f"Learning rate (default: {DEFAULT_FINETUNE_LR})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--hard-neg-weight", type=float, default=DEFAULT_HARD_NEG_WEIGHT,
        help=f"Sample weight multiplier for hard negatives (default: {DEFAULT_HARD_NEG_WEIGHT})",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()

    finetune(
        fp_json=Path(args.fp_json),
        fp_split=args.fp_split,
        weights_path=Path(args.weights_path),
        output_path=Path(args.output_path),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hard_neg_weight=args.hard_neg_weight,
        device_str=args.device,
        cache_dir=args.cache_dir,
        patience=args.patience,
    )
