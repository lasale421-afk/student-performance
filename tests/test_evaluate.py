"""Tests for the threshold choice in evaluate.py."""

import numpy as np
import pandas as pd
import pytest

from student_risk.evaluate import choose_threshold, group_report, wilson_interval

# 6 students: 3 fail (1), 3 pass (0). Sorted by predicted probability.
Y_TRUE = np.array([0, 0, 1, 0, 1, 1])
Y_PROBA = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])


def test_choose_threshold_reaches_recall_with_best_precision():
    # Threshold 0.8 flags 2 students, both fail: recall 2/3, precision 1.
    assert choose_threshold(Y_TRUE, Y_PROBA, min_recall=0.6) == pytest.approx(0.8)
    # Recall 1 needs threshold 0.3: 4 flagged, 3 fail -> precision 0.75 (0.1 gives only 0.5).
    assert choose_threshold(Y_TRUE, Y_PROBA, min_recall=1.0) == pytest.approx(0.3)


@pytest.mark.parametrize("bad_recall", [0.0, -0.2, 1.5])
def test_choose_threshold_rejects_min_recall_outside_0_1(bad_recall):
    # A target of 0 (or below) would accept any threshold; above 1 is impossible.
    with pytest.raises(ValueError, match="min_recall must be in"):
        choose_threshold(Y_TRUE, Y_PROBA, min_recall=bad_recall)


def test_wilson_interval_known_values():
    # 15 of 20 failing students caught: Wilson 95 % interval is about [0.531, 0.888].
    lower, upper = wilson_interval(15, 20)
    assert lower == pytest.approx(0.531, abs=0.001)
    assert upper == pytest.approx(0.888, abs=0.001)
    assert 0 <= wilson_interval(0, 5)[0] and wilson_interval(5, 5)[1] <= 1  # stays in [0, 1]


def test_group_report_counts_and_recall_per_group():
    groups = pd.Series(["F", "F", "M", "M", "M", "F"])
    y_true = pd.Series(Y_TRUE)  # fails: index 2 (M), 4 (M), 5 (F)

    report = group_report(y_true, Y_PROBA, threshold=0.8, groups=groups)

    assert list(report.index) == ["F", "M"]
    assert report.loc["F", "n_students"] == 3 and report.loc["F", "n_fail"] == 1
    assert report.loc["F", "recall"] == 1.0  # its only fail (0.9) is flagged
    assert report.loc["M", "recall"] == 0.5  # 0.8 flagged, 0.3 missed
