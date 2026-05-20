"""Experiment logger: saves all run artifacts to runs/<name>/ and Google Drive.

Every training/eval run writes:
  runs/<experiment_name>/
      config.yaml
      metrics.json      (updated each epoch)
      train.log
      confusion_matrix.png
      predictions.csv
      checkpoint.pt     (symlink / copy managed by caller)

If DRIVE_BASE_DIR is set, the entire run directory is mirrored to Drive after
each epoch so a Colab crash does not lose progress.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

RUNS_BASE = Path("runs")
DRIVE_BASE_DIR = "/content/drive/MyDrive/facetune_detector_runs"


class ExperimentLogger:
    """Tracks a single training run and persists all artifacts.

    Usage:
        logger = ExperimentLogger("hybrid_v1", drive_dir=DRIVE_BASE_DIR)
        logger.save_config(config_dict)
        for epoch in range(epochs):
            ...
            logger.log_epoch(epoch, train_metrics, val_metrics)
            logger.sync_to_drive()
        logger.save_confusion_matrix(cm_array, class_names)
        logger.save_predictions(rows)
    """

    def __init__(
        self,
        name: str,
        base_dir: str | Path = RUNS_BASE,
        drive_dir: str | None = None,
    ) -> None:
        self.name = name
        self.exp_dir = Path(base_dir) / name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.drive_dir: Path | None = (
            Path(drive_dir) / name if drive_dir else None
        )

        self._metrics: list[dict] = []
        self._setup_file_logger()

    # ── File logger ───────────────────────────────────────────────────────

    def _setup_file_logger(self) -> None:
        log_path = self.exp_dir / "train.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                              datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(fh)
        self.log_path = log_path

    # ── Config ────────────────────────────────────────────────────────────

    def save_config(self, config: dict) -> None:
        """Write config.yaml (falls back to JSON if PyYAML not installed)."""
        if _HAS_YAML:
            path = self.exp_dir / "config.yaml"
            with path.open("w") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            path = self.exp_dir / "config.json"
            with path.open("w") as f:
                json.dump(config, f, indent=2)
        logging.info("[ExperimentLogger] config → %s", path)

    # ── Metrics ───────────────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append epoch metrics and rewrite metrics.json."""
        row: dict[str, Any] = {
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "train": train_metrics,
            "val": val_metrics,
        }
        if extra:
            row["extra"] = extra
        self._metrics.append(row)
        path = self.exp_dir / "metrics.json"
        with path.open("w") as f:
            json.dump(self._metrics, f, indent=2)

    def best_val_metric(self, key: str = "f1") -> float:
        """Return the best validation metric seen so far."""
        if not self._metrics:
            return 0.0
        return max(m["val"].get(key, 0.0) for m in self._metrics)

    def best_val_fpr(self) -> float:
        """Return the lowest FPR seen so far (lower is better)."""
        if not self._metrics:
            return 1.0
        fprs = [m["val"].get("fpr", 1.0) for m in self._metrics]
        return min(fprs)

    # ── Confusion matrix ──────────────────────────────────────────────────

    def save_confusion_matrix(
        self,
        cm: list[list[int]] | np.ndarray,
        class_names: list[str] | None = None,
    ) -> None:
        cm = np.array(cm)
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(class_names)),
            yticks=range(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion matrix — {self.name}",
        )
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        path = self.exp_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logging.info("[ExperimentLogger] confusion_matrix → %s", path)

    # ── Predictions CSV ───────────────────────────────────────────────────

    def save_predictions(self, rows: list[dict]) -> None:
        """Write predictions.csv. Each row is a dict with consistent keys."""
        if not rows:
            return
        path = self.exp_dir / "predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logging.info("[ExperimentLogger] predictions (%d rows) → %s", len(rows), path)

    # ── Drive sync ────────────────────────────────────────────────────────

    def sync_to_drive(self) -> None:
        """Mirror run directory to Google Drive (no-op outside Colab)."""
        if self.drive_dir is None:
            return
        try:
            self.drive_dir.mkdir(parents=True, exist_ok=True)
            for src in self.exp_dir.iterdir():
                dst = self.drive_dir / src.name
                shutil.copy2(src, dst)
            logging.info("[ExperimentLogger] synced → %s", self.drive_dir)
        except Exception as exc:  # Drive not mounted, offline, etc.
            logging.warning("[ExperimentLogger] Drive sync skipped: %s", exc)

    # ── Checkpoint path helper ────────────────────────────────────────────

    @property
    def checkpoint_path(self) -> Path:
        return self.exp_dir / "checkpoint.pt"
