"""Training script for the AI image authenticity classifier.

Trains a ResNet-50 binary classifier using FFT-preprocessed images from the
Defactify dataset. Saves the best validation checkpoint to model/model.pth.

Usage:
    python -m training.train
    python -m training.train --epochs 20 --batch-size 64 --device cuda
    python -m training.train --cache-dir /data/hf_cache
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from model.resnet import create_resnet50_binary_classifier
from model.mobilenet import create_mobilenet_v3_classifier
from training.dataset import DatasetConfig, DefactifyTorchDataset, load_defactify_split

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_WEIGHTS_PATH = Path("model/model.pth")
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4
SUPPORTED_MODELS = ("resnet50", "mobilenet", "hybrid")  # hybrid = FFT+CLIP


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds: list[int], labels: list[int], generator_labels: list[int] | None = None) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 from flat label lists.

    Purpose:
    Centralize metric computation so both train and validation loops use the
    same formulas without external dependencies.

    Inputs:
    - preds: Predicted class indices.
    - labels: Ground-truth class indices.

    Outputs:
    - Dictionary with keys: accuracy, precision, recall, f1.

    Key assumptions:
    - Binary classification: positive class is 1 (ai_generated).

    Failure modes:
    - Returns 0.0 for undefined metrics when there are no positive predictions
      or no positive ground-truth labels.
    """
    if not labels:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    correct = sum(p == l for p, l in zip(preds, labels))

    accuracy = correct / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    # FPR = FP / (FP + TN) — fraction of real images misclassified as AI
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


# ── Sampler ───────────────────────────────────────────────────────────────────

def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that up-weights the minority class.

    Purpose:
    Counteract class imbalance in the training set so both real and
    ai_generated samples appear with equal expected frequency per batch.

    Inputs:
    - labels: Integer class labels for all training samples.

    Outputs:
    - WeightedRandomSampler configured for one epoch of balanced sampling.

    Key assumptions:
    - Binary labels (0 and 1). Both classes must have at least one sample.

    Failure modes:
    - Raises ZeroDivisionError if a class is entirely absent from labels.
    """
    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Epoch loops ───────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    log_every: int = 100,
) -> tuple[float, dict[str, float]]:
    """Run one full training epoch with optional mixed-precision on CUDA.

    Purpose:
    Iterate over all training batches, compute loss, backpropagate, and
    collect per-epoch metrics. Uses automatic mixed precision (AMP) on CUDA
    to reduce memory usage and speed up training.

    Inputs:
    - model: Binary classifier in training mode.
    - loader: DataLoader for the training split.
    - optimizer: Configured optimizer instance.
    - criterion: Loss function.
    - device: Compute device.
    - scaler: GradScaler for AMP; None disables mixed precision.
    - log_every: Log batch-level loss every N steps.

    Outputs:
    - Tuple of (mean_loss, metrics_dict).

    Key assumptions:
    - Model outputs two logits per sample.

    Failure modes:
    - Propagates CUDA OOM and dataloader errors.
    """
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    use_amp = scaler is not None and device.type == "cuda"

    for step, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        preds = logits.detach().argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if (step + 1) % log_every == 0:
            avg = running_loss / (step + 1)
            logging.info("  step %d/%d  loss=%.4f", step + 1, len(loader), avg)

    return running_loss / len(loader), compute_metrics(all_preds, all_labels)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Run one full validation pass without gradient computation.

    Purpose:
    Measure held-out performance after each training epoch to guide
    checkpoint selection.

    Inputs:
    - model: ResNet-50 binary classifier.
    - loader: DataLoader for the validation split.
    - criterion: Loss function.
    - device: Compute device.

    Outputs:
    - Tuple of (mean_loss, metrics_dict).

    Key assumptions:
    - Model outputs two logits per sample.

    Failure modes:
    - Propagates dataloader and device errors.
    """
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return running_loss / len(loader), compute_metrics(all_preds, all_labels)


# ── Checkpoint ────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when a metric stops improving for `patience` consecutive epochs.

    Purpose:
    Prevent overfitting and wasted compute by halting training as soon as
    held-out performance stagnates.

    Inputs:
    - patience: Epochs to wait after last improvement before stopping.
    - min_delta: Minimum absolute improvement that counts as progress.

    Outputs:
    - None. Updates `.should_stop` flag after each `.step()` call.

    Key assumptions:
    - Higher metric values are always better (e.g. F1, accuracy).

    Failure modes:
    - None.
    """

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE,
                 min_delta: float = EARLY_STOPPING_MIN_DELTA) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = -float("inf")
        self.wait: int = 0
        self.should_stop: bool = False

    def step(self, metric: float) -> None:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True


def save_checkpoint(
    model: nn.Module, epoch: int, val_metrics: dict[str, float], path: Path
) -> None:
    """Serialize the model state dict and training metadata to disk.

    Purpose:
    Persist the best-performing checkpoint so inference and evaluation can
    load real weights without retraining.

    Inputs:
    - model: Trained ResNet-50 binary classifier.
    - epoch: Epoch number for record-keeping.
    - val_metrics: Validation metric snapshot at save time.
    - path: Target file path for the checkpoint.

    Outputs:
    - None. Writes a .pth file to disk.

    Key assumptions:
    - The parent directory will be created if it does not exist.

    Failure modes:
    - Propagates OS errors if the path is not writable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": getattr(model, "_train_model_name", "resnet50"),
            "state_dict": model.state_dict(),
            "val_accuracy": val_metrics.get("accuracy", 0.0),
            "val_precision": val_metrics.get("precision", 0.0),
            "val_recall": val_metrics.get("recall", 0.0),
            "val_f1": val_metrics.get("f1", 0.0),
            "val_fpr": val_metrics.get("fpr", 1.0),
        },
        path,
    )
    logging.info(
        "  ✓ Saved checkpoint → %s  (f1=%.4f  fpr=%.4f)",
        path,
        val_metrics.get("f1", 0.0),
        val_metrics.get("fpr", 1.0),
    )


# ── Hybrid epoch loops ────────────────────────────────────────────────────────

def train_one_epoch_hybrid(
    model: nn.Module,
    loader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
    log_every: int = 100,
) -> tuple[float, dict[str, float]]:
    """Training epoch for HybridDetector — batch is (fft, clip_pv, label)."""
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    use_amp = scaler is not None and device.type == "cuda"

    for step, batch in enumerate(loader):
        fft_inputs, clip_inputs, labels = batch
        fft_inputs = fft_inputs.to(device)
        clip_inputs = clip_inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(fft_inputs, clip_inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(fft_inputs, clip_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        all_preds.extend(logits.detach().argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if (step + 1) % log_every == 0:
            logging.info("  step %d/%d  loss=%.4f", step + 1, len(loader),
                         running_loss / (step + 1))

    return running_loss / len(loader), compute_metrics(all_preds, all_labels)


def validate_hybrid(
    model: nn.Module,
    loader: "DataLoader",
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Validation pass for HybridDetector — batch is (fft, clip_pv, label)."""
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for fft_inputs, clip_inputs, labels in loader:
            fft_inputs = fft_inputs.to(device)
            clip_inputs = clip_inputs.to(device)
            labels = labels.to(device)
            logits = model(fft_inputs, clip_inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return running_loss / len(loader), compute_metrics(all_preds, all_labels)


# ── Model factory ────────────────────────────────────────────────────────────

def _build_model(model_name: str, pretrained: bool) -> nn.Module:
    """Instantiate a classifier by name."""
    if model_name == "hybrid":
        from model.hybrid import HybridDetector
        logging.info("Building HybridDetector (FFT+CLIP) …")
        model = HybridDetector()
        model._train_model_name = "hybrid"
        return model
    if model_name == "mobilenet":
        logging.info("Building MobileNetV3-Large (pretrained=%s) …", pretrained)
        return create_mobilenet_v3_classifier(pretrained=pretrained)
    logging.info("Building ResNet-50 (pretrained=%s) …", pretrained)
    return create_resnet50_binary_classifier(pretrained=pretrained)


# ── Main training entry point ─────────────────────────────────────────────────

def train(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device_str: str = "auto",
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    cache_dir: str | None = None,
    pretrained_backbone: bool = True,
    patience: int = EARLY_STOPPING_PATIENCE,
    model_name: str = "resnet50",
    max_samples: int | None = None,
    checkpoint_by: str = "f1",
    run_name: str | None = None,
    drive_dir: str | None = None,
) -> None:
    """Run the full training loop and save the best checkpoint.

    Purpose:
    Train a binary classifier on the Defactify dataset using FFT-preprocessed
    images. Applies class-balanced sampling, saves the checkpoint with the
    highest validation F1 score, and stops early when F1 stagnates.

    Inputs:
    - epochs: Maximum number of full passes over the training data.
    - batch_size: Samples per mini-batch.
    - lr: Adam learning rate.
    - device_str: Compute device ("auto", "cpu", or "cuda").
    - weights_path: Output path for the best checkpoint.
    - cache_dir: Optional local directory for the Hugging Face dataset cache.
    - pretrained_backbone: Whether to initialise from ImageNet weights.
    - patience: Early stopping patience (epochs without F1 improvement).
    - model_name: Backbone to use — "resnet50" or "mobilenet".
    - max_samples: If set, subsample train to this many rows (for quick runs).

    Outputs:
    - None. Saves checkpoint when a new best validation F1 is found.

    Key assumptions:
    - Best checkpoint selected by val F1 — more robust under class imbalance.
    - Mixed precision (AMP) is used on CUDA for faster training.
    - CrossEntropyLoss is used because the model outputs two logits per sample.
    - max_samples subsamples randomly with RANDOM_SEED; label balance is then
      handled by WeightedRandomSampler as usual.

    Failure modes:
    - Raises ImportError if datasets or torchvision are unavailable.
    - Propagates dataset download errors when cache is empty.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Experiment logger ──────────────────────────────────────────────────
    exp_logger = None
    if run_name:
        from training.experiment_logger import ExperimentLogger
        exp_logger = ExperimentLogger(run_name, drive_dir=drive_dir)
        exp_logger.save_config({
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "checkpoint_by": checkpoint_by,
            "patience": patience,
        })
        if weights_path == DEFAULT_WEIGHTS_PATH:
            weights_path = exp_logger.checkpoint_path
        logging.info("Experiment: %s  →  %s", run_name, exp_logger.exp_dir)

    torch.manual_seed(RANDOM_SEED)

    # ── Device ────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logging.info("Device: %s", device)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logging.info("Mixed precision (AMP) enabled.")

    # ── Datasets ──────────────────────────────────────────────────────────
    logging.info("Loading train split …")
    train_hf = load_defactify_split(DatasetConfig(split="train", cache_dir=cache_dir))
    logging.info("  %d rows", len(train_hf))

    logging.info("Loading validation split …")
    val_hf = load_defactify_split(DatasetConfig(split="validation", cache_dir=cache_dir))
    logging.info("  %d rows", len(val_hf))

    is_hybrid = model_name == "hybrid"

    if is_hybrid:
        from datasets.hard_negatives import DefactifyHybridDataset, make_weighted_sampler_hybrid
        try:
            from transformers import CLIPImageProcessor
            clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except ImportError:
            raise ImportError("pip install transformers  ← required for hybrid model")
        train_ds = DefactifyHybridDataset(train_hf, clip_proc)
        val_ds = DefactifyHybridDataset(val_hf, clip_proc)
    else:
        train_ds = DefactifyTorchDataset(train_hf)
        val_ds = DefactifyTorchDataset(val_hf)

    train_labels = train_ds.get_all_labels()

    # Optional subsample for fast iteration (preserves class ratios via sampler).
    if max_samples is not None and max_samples < len(train_ds):
        import torch.utils.data as tud
        rng = torch.Generator().manual_seed(RANDOM_SEED)
        indices = torch.randperm(len(train_ds), generator=rng)[:max_samples].tolist()
        train_ds = tud.Subset(train_ds, indices)
        train_labels = [train_labels[i] for i in indices]
        logging.info("  Subsampled to %d rows (--max-samples)", max_samples)

    label_counts = Counter(train_labels)
    logging.info(
        "Train label distribution: real=%d  ai_generated=%d",
        label_counts.get(0, 0),
        label_counts.get(1, 0),
    )

    sampler = make_weighted_sampler(train_labels)
    # num_workers=0 is required on Windows (no fork-based multiprocessing).
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    logging.info("Batches: train=%d  val=%d", len(train_loader), len(val_loader))

    # ── Model ─────────────────────────────────────────────────────────────
    model = _build_model(model_name, pretrained_backbone)
    model._train_model_name = model_name  # carried into checkpoint metadata
    model = model.to(device)

    # ── Optimiser / loss / scheduler ──────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Halve the LR every 3 epochs to stabilise late training.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    stopper = EarlyStopping(patience=patience)

    best_val_f1 = 0.0
    best_val_fpr = 1.0
    checkpoint_by = checkpoint_by.lower()

    _train_epoch = train_one_epoch_hybrid if is_hybrid else train_one_epoch
    _validate = validate_hybrid if is_hybrid else validate

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        logging.info("── Epoch %d / %d ────────────────────────────────", epoch, epochs)

        train_loss, train_m = _train_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler
        )
        val_loss, val_m = _validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logging.info(
            "  train  loss=%.4f  acc=%.4f  f1=%.4f  fpr=%.4f",
            train_loss, train_m["accuracy"], train_m["f1"], train_m.get("fpr", 0.0),
        )
        logging.info(
            "  val    loss=%.4f  acc=%.4f  f1=%.4f  fpr=%.4f  (%.0fs)",
            val_loss, val_m["accuracy"], val_m["f1"], val_m.get("fpr", 0.0), elapsed,
        )

        if exp_logger:
            exp_logger.log_epoch(epoch, train_m, val_m,
                                 extra={"train_loss": train_loss, "val_loss": val_loss})

        # Checkpoint selection
        improved = False
        if checkpoint_by == "fpr":
            val_fpr = val_m.get("fpr", 1.0)
            if val_fpr < best_val_fpr:
                best_val_fpr = val_fpr
                save_checkpoint(model, epoch, val_m, weights_path)
                logging.info("  ↓ New best FPR: %.4f", best_val_fpr)
                improved = True
        else:  # f1
            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                save_checkpoint(model, epoch, val_m, weights_path)
                logging.info("  ↑ New best val F1: %.4f", best_val_f1)
                improved = True

        if exp_logger:
            exp_logger.sync_to_drive()

        stopper.step(val_m["f1"])
        if stopper.should_stop:
            logging.info(
                "Early stopping at epoch %d — val F1 did not improve for %d epochs.",
                epoch, patience,
            )
            break

    logging.info(
        "Training complete. Best val F1: %.4f  Best val FPR: %.4f",
        best_val_f1, best_val_fpr,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the AI image authenticity classifier."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Mini-batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Adam learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--device", default="auto",
                        help="Compute device: auto | cpu | cuda (default: auto)")
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH),
                        help=f"Output checkpoint path (default: {DEFAULT_WEIGHTS_PATH})")
    parser.add_argument("--cache-dir", default=None,
                        help="Hugging Face dataset cache directory")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip ImageNet backbone initialisation")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE,
                        help=f"Early stopping patience in epochs (default: {EARLY_STOPPING_PATIENCE})")
    parser.add_argument("--model", default="resnet50", choices=list(SUPPORTED_MODELS),
                        help="Backbone: resnet50 (default) | mobilenet (~5× faster on CPU)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample train to N rows for quick iteration (default: all)")
    parser.add_argument("--checkpoint-by", default="f1",
                        choices=["f1", "fpr"],
                        help="Select best checkpoint by val F1 (default) or lowest FPR")
    parser.add_argument("--run-name", default=None,
                        help="Experiment name — saves to runs/<name>/ and Drive")
    parser.add_argument("--drive-dir", default=None,
                        help="Google Drive base dir for experiment sync "
                             "(default: /content/drive/MyDrive/facetune_detector_runs)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        weights_path=Path(args.weights_path),
        cache_dir=args.cache_dir,
        pretrained_backbone=not args.no_pretrained,
        patience=args.patience,
        model_name=args.model,
        max_samples=args.max_samples,
        checkpoint_by=args.checkpoint_by,
        run_name=args.run_name,
        drive_dir=args.drive_dir,
    )
