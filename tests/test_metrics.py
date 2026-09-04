import numpy as np
import pandas as pd

from ai_image_detector.metrics import (
    binary_metrics,
    choose_threshold,
    cluster_bootstrap_intervals,
    expected_calibration_error,
    paired_cluster_bootstrap_difference,
    paired_group_ranking_accuracy,
)


def test_metrics_for_separable_scores_are_high() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.2, 0.8, 0.95])
    metrics = binary_metrics(labels, scores)
    assert metrics["roc_auc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert 0.0 <= expected_calibration_error(labels, scores) <= 1.0
    assert 0.2 < choose_threshold(labels, scores) <= 0.8


def test_grouped_intervals_and_paired_comparison_resample_whole_content_groups() -> None:
    predictions = pd.DataFrame(
        {
            "path": ["a-real", "a-fake", "b-real", "b-fake"],
            "leakage_group": ["a", "a", "b", "b"],
            "label": [0, 1, 0, 1],
            "ai_score": [0.1, 0.9, 0.2, 0.8],
        }
    )

    ranking = paired_group_ranking_accuracy(predictions)
    intervals = cluster_bootstrap_intervals(predictions, threshold=0.5, repeats=30, seed=3)
    weaker = predictions.assign(ai_score=[0.9, 0.1, 0.8, 0.2])
    difference = paired_cluster_bootstrap_difference(
        predictions, weaker, left_threshold=0.5, right_threshold=0.5, repeats=30, seed=3
    )

    assert ranking["paired_group_ranking_accuracy"] == 1.0
    assert ranking["paired_group_ranking_pairs"] == 2
    assert intervals["roc_auc"]["estimate"] == 1.0
    assert intervals["roc_auc"]["bootstrap_repeats"] == 30
    assert difference["roc_auc"]["estimate"] == 1.0
    assert difference["roc_auc"]["lower"] <= difference["roc_auc"]["estimate"]
