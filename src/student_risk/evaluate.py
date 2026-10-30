"""Metrics, decision-threshold choice and threshold reports."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# The metrics used for every cross-validation in the project. Precision gets
# zero_division=0 so that a model that never predicts "fail" scores 0 instead of
# raising a warning.
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}


def choose_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float) -> float:
    """Pick the decision threshold with the best precision among those with enough recall.

    A student is flagged as "fail" when its probability is >= the threshold.

    Args:
        y_true: True 0/1 labels.
        y_proba: Predicted probability of class 1 for the same students.
        min_recall: Minimum recall wanted, in (0, 1], for example 0.80.

    Returns:
        The chosen threshold.

    Raises:
        ValueError: If min_recall is not in (0, 1].
    """
    if not 0 < min_recall <= 1:
        raise ValueError(f"min_recall must be in (0, 1], got {min_recall}.")

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # The last (precision, recall) pair is a fixed end point (1, 0) with no threshold:
    # drop it so that precision[i] and recall[i] belong to thresholds[i].
    precision, recall = precision[:-1], recall[:-1]

    # The lowest threshold flags every student (recall 1), so at least one threshold
    # always reaches a min_recall in (0, 1].
    reaches_target = recall >= min_recall
    best = np.argmax(np.where(reaches_target, precision, -1.0))
    return float(thresholds[best])


def threshold_report(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Compute the metrics and the confusion matrix at one decision threshold.

    Args:
        y_true: True 0/1 labels.
        y_proba: Predicted probability of class 1.
        threshold: Students with probability >= threshold are flagged as "fail".

    Returns:
        Dict with threshold, precision, recall, f1, share of students flagged,
        and the four confusion-matrix counts tn, fp, fn, tp.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flagged_share": float(y_pred.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def final_scores(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Compute every metric of the project for one set of predictions.

    Used once, by the evaluate command, on the test set.

    Args:
        y_true: True 0/1 labels.
        y_proba: Predicted probability of class 1.
        threshold: The decision threshold chosen during training.

    Returns:
        ROC-AUC and average precision (threshold-free), plus everything from
        threshold_report at the given threshold.
    """
    scores = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
    }
    scores.update(threshold_report(y_true, y_proba, threshold))
    return scores


def threshold_tradeoff(
    y_true: np.ndarray, y_proba: np.ndarray, min_recalls: list[float]
) -> pd.DataFrame:
    """Show what each recall target costs: threshold, precision and share of students flagged.

    Args:
        y_true: True 0/1 labels.
        y_proba: Predicted probability of class 1 (out-of-fold in this project).
        min_recalls: Recall targets to compare, for example [0.70, 0.80, 0.90].

    Returns:
        One row per target with threshold, precision, recall and flagged_share.
    """
    rows = []
    for min_recall in min_recalls:
        threshold = choose_threshold(y_true, y_proba, min_recall)
        report = threshold_report(y_true, y_proba, threshold)
        rows.append(
            {
                "min_recall": min_recall,
                "threshold": report["threshold"],
                "precision": report["precision"],
                "recall": report["recall"],
                "flagged_share": report["flagged_share"],
            }
        )
    return pd.DataFrame(rows)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion (95 % by default).

    Args:
        successes: Number of successes, e.g. failing students that were flagged.
        n: Number of trials, e.g. all failing students.
        z: Normal quantile; 1.96 gives a 95 % interval.

    Returns:
        (lower, upper) bounds, always inside [0, 1].
    """
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    # Clip: rounding can give -1e-17 instead of 0 when successes is 0 (or n).
    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))


def group_report(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float, groups: pd.Series
) -> pd.DataFrame:
    """Recall and share flagged for each group (e.g. by sex or school).

    Args:
        y_true: True 0/1 labels.
        y_proba: Predicted probability of class 1, same order as y_true.
        threshold: Decision threshold.
        groups: Group label of each student, same order as y_true.

    Returns:
        One row per group: number of students, number of fails, recall,
        precision and share flagged.
    """
    rows = {}
    for group in sorted(groups.unique()):
        mask = (groups == group).to_numpy()
        report = threshold_report(y_true[mask], np.asarray(y_proba)[mask], threshold)
        rows[group] = {
            "n_students": int(mask.sum()),
            "n_fail": int(y_true[mask].sum()),
            "recall": report["recall"],
            "precision": report["precision"],
            "flagged_share": report["flagged_share"],
        }
    return pd.DataFrame(rows).T


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """List the coefficients of a fitted linear pipeline with readable feature names.

    Args:
        model: Fitted Pipeline with steps "preprocess" and a linear "clf".

    Returns:
        One row per encoded feature, sorted by absolute coefficient (largest first),
        with the columns feature, coefficient and odds_ratio (exp of the coefficient).
    """
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["clf"].coef_[0]
    table = pd.DataFrame(
        {
            "feature": [name.split("__", 1)[1] for name in names],
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    )
    order = table["coefficient"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)
