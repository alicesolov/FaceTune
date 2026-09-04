"""Metrics, calibration checks and confidence intervals for binary experiments."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in pairwise(edges):
        mask = (scores >= lower) & ((scores < upper) if upper < 1.0 else (scores <= upper))
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - scores[mask].mean())
    return float(error)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose the validation-only threshold that maximises balanced accuracy."""
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    values = [balanced_accuracy_score(y_true, scores >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def binary_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "n": len(y_true),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "ai_precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "ai_recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "real_recall": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "ece_15": expected_calibration_error(y_true, scores),
        "brier": float(brier_score_loss(y_true, scores)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
        fpr, tpr, _ = roc_curve(y_true, scores)
        valid = np.flatnonzero(tpr >= 0.95)
        metrics["fpr_at_tpr_95"] = float(fpr[valid[0]]) if len(valid) else float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        metrics["fpr_at_tpr_95"] = float("nan")
    return metrics


def bootstrap_interval(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    seed: int = 0,
    repeats: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI, skipping resamples that lack both classes for AUC-like metrics."""
    generator = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    values: list[float] = []
    for _ in range(repeats):
        index = generator.integers(0, len(y_true), len(y_true))
        try:
            value = metric(y_true[index], scores[index])
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


DEFAULT_CLUSTER_METRICS = (
    "roc_auc",
    "balanced_accuracy",
    "macro_f1",
    "fpr_at_tpr_95",
)


def paired_group_ranking_accuracy(
    predictions: pd.DataFrame, group_column: str = "leakage_group"
) -> dict[str, float | int]:
    """Compare fake scores to their real sibling within each content group.

    Each group receives equal weight, even when a duplicate component contains several records.
    Within a group all fake-versus-real score pairs are compared; a score tie contributes 0.5
    rather than being assigned arbitrarily to either representation or classifier.
    """
    required = {group_column, "label", "ai_score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions lack columns for group ranking: {sorted(missing)}")
    group_accuracies: list[float] = []
    correct = 0.0
    pairs = 0
    usable_groups = 0
    for _, group in predictions.groupby(group_column, sort=False):
        real_scores = group.loc[group["label"] == 0, "ai_score"].to_numpy(dtype=float)
        fake_scores = group.loc[group["label"] == 1, "ai_score"].to_numpy(dtype=float)
        if not len(real_scores) or not len(fake_scores):
            continue
        usable_groups += 1
        difference = fake_scores[:, np.newaxis] - real_scores[np.newaxis, :]
        correct += float((difference > 0).sum()) + 0.5 * float((difference == 0).sum())
        pairs += difference.size
        group_accuracies.append(
            (float((difference > 0).sum()) + 0.5 * float((difference == 0).sum()))
            / difference.size
        )
    return {
        "paired_group_ranking_accuracy": (
            float(np.mean(group_accuracies)) if group_accuracies else float("nan")
        ),
        "paired_group_ranking_micro_accuracy": correct / pairs if pairs else float("nan"),
        "paired_group_ranking_pairs": pairs,
        "paired_group_ranking_groups": usable_groups,
    }


def _cluster_indices(frame: pd.DataFrame, group_column: str) -> list[np.ndarray]:
    if group_column not in frame.columns:
        raise ValueError(f"Predictions have no {group_column!r} column")
    groups = frame[group_column].to_numpy()
    if pd.isna(groups).any():
        raise ValueError(f"Predictions contain missing {group_column!r} values")
    _, inverse = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(inverse == group) for group in range(inverse.max() + 1)]


def cluster_bootstrap_intervals(
    predictions: pd.DataFrame,
    threshold: float,
    group_column: str = "leakage_group",
    metrics: tuple[str, ...] = DEFAULT_CLUSTER_METRICS,
    seed: int = 0,
    repeats: int = 2000,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | int]]:
    """Return percentile confidence intervals by resampling full content groups.

    Individual image rows in Defactify are linked by prompt/content.  Sampling rows independently
    would understate uncertainty, so each bootstrap draw resamples `leakage_group` units.
    """
    required = {"label", "ai_score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions lack metric columns: {sorted(missing)}")
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    point_metrics = binary_metrics(
        predictions["label"].to_numpy(), predictions["ai_score"].to_numpy(), threshold
    )
    unknown = set(metrics).difference(point_metrics)
    if unknown:
        raise ValueError(f"Unknown metric keys: {sorted(unknown)}")
    clusters = _cluster_indices(predictions, group_column)
    generator = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in metrics}
    labels = predictions["label"].to_numpy(dtype=int)
    scores = predictions["ai_score"].to_numpy(dtype=float)
    for _ in range(repeats):
        selected = generator.integers(0, len(clusters), len(clusters))
        index = np.concatenate([clusters[cluster] for cluster in selected])
        sample_metrics = binary_metrics(labels[index], scores[index], threshold)
        for metric in metrics:
            value = float(sample_metrics[metric])
            if np.isfinite(value):
                values[metric].append(value)
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        distribution = values[metric]
        result[metric] = {
            "estimate": float(point_metrics[metric]),
            "lower": float(np.quantile(distribution, alpha)) if distribution else float("nan"),
            "upper": (
                float(np.quantile(distribution, 1.0 - alpha)) if distribution else float("nan")
            ),
            "bootstrap_repeats": len(distribution),
            "confidence": confidence,
        }
    return result


def paired_cluster_bootstrap_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_threshold: float,
    right_threshold: float,
    group_column: str = "leakage_group",
    metrics: tuple[str, ...] = DEFAULT_CLUSTER_METRICS,
    seed: int = 0,
    repeats: int = 2000,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | int]]:
    """Cluster-bootstrap the metric difference (left minus right) on identical images."""
    required = {"path", "label", "ai_score", group_column}
    missing_left = required.difference(left.columns)
    missing_right = required.difference(right.columns)
    if missing_left or missing_right:
        raise ValueError(
            "Both prediction tables must contain path, label, ai_score and "
            f"{group_column!r}; missing left={sorted(missing_left)}, right={sorted(missing_right)}"
        )
    left_indexed = left.set_index("path").sort_index()
    right_indexed = right.set_index("path").sort_index()
    if not left_indexed.index.is_unique or not right_indexed.index.is_unique:
        raise ValueError("Paired comparison requires one prediction per path")
    if not left_indexed.index.equals(right_indexed.index):
        raise ValueError("Paired comparison requires the same prediction paths")
    if not np.array_equal(left_indexed["label"].to_numpy(), right_indexed["label"].to_numpy()):
        raise ValueError("Paired comparison requires identical labels")
    if not np.array_equal(
        left_indexed[group_column].to_numpy(), right_indexed[group_column].to_numpy()
    ):
        raise ValueError("Paired comparison requires identical content groups")
    labels = left_indexed["label"].to_numpy(dtype=int)
    left_scores = left_indexed["ai_score"].to_numpy(dtype=float)
    right_scores = right_indexed["ai_score"].to_numpy(dtype=float)
    groups = left_indexed[[group_column]].reset_index()
    clusters = _cluster_indices(groups, group_column)
    left_point = binary_metrics(labels, left_scores, left_threshold)
    right_point = binary_metrics(labels, right_scores, right_threshold)
    unknown = set(metrics).difference(left_point).union(set(metrics).difference(right_point))
    if unknown:
        raise ValueError(f"Unknown metric keys: {sorted(unknown)}")
    generator = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(repeats):
        selected = generator.integers(0, len(clusters), len(clusters))
        index = np.concatenate([clusters[cluster] for cluster in selected])
        left_sample = binary_metrics(labels[index], left_scores[index], left_threshold)
        right_sample = binary_metrics(labels[index], right_scores[index], right_threshold)
        for metric in metrics:
            difference = float(left_sample[metric]) - float(right_sample[metric])
            if np.isfinite(difference):
                values[metric].append(difference)
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        distribution = values[metric]
        estimate = float(left_point[metric]) - float(right_point[metric])
        result[metric] = {
            "estimate": estimate,
            "lower": float(np.quantile(distribution, alpha)) if distribution else float("nan"),
            "upper": (
                float(np.quantile(distribution, 1.0 - alpha)) if distribution else float("nan")
            ),
            "bootstrap_repeats": len(distribution),
            "confidence": confidence,
        }
    return result
