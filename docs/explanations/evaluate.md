# `src/student_risk/evaluate.py`: explained line by line

**Role:** everything about *measuring*: the metrics used in every cross-validation (`SCORING`),
the choice of the decision threshold (`choose_threshold`), and the metrics and confusion matrix
at a given threshold (`threshold_report`). It contains no model and no data loading, so each
function can be tested with a few hand-written numbers.

---

## Imports

```python
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
```

- `numpy` is used for array operations: `argmax`, `where`, and comparisons.
- `pandas` builds the trade-off table (Phase 2b).
- `roc_auc_score` and `average_precision_score` compute the two threshold-free metrics on the
  test set (Phase 3, see `final_scores`). In cross-validation, the strings `"roc_auc"` and
  `"average_precision"` in `SCORING` call these same functions **[from prior knowledge, to
  verify]**.
- **Phase 2b change:** the `warnings` import was removed together with the unreachable fallback
  (see below).
- From `sklearn.metrics`:
  - `confusion_matrix` counts TN, FP, FN and TP.
  - `f1_score`, `precision_score` and `recall_score` are the metrics at one threshold.
  - `make_scorer` builds a custom scorer for cross-validation.
  - `precision_recall_curve` gives precision and recall for *every* possible threshold.
- The imports are inside parentheses on several lines: ruff's formatter does this when one line
  would be too long.

## `SCORING` (moved here from `models.py` in Phase 2)

```python
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}
```

- **Why it moved:** metrics belong to "evaluation". `models.py` imports it, so every model, the
  baselines and the 4 tuned models all use **the same scoring**.
- The meaning of each metric and the reason for `zero_division=0` are in `models.md`, in the
  "Phase 1" section.
- **Why precision is included** even though the GridSearchCV brief listed four metrics: precision
  is half of our goal ("recall ≥ 0.80 with the best precision"), so it is useful to see it in the
  comparison table. Adding a metric does not change which model is refitted, because `refit`
  uses average precision only.
- **Important:** recall, precision and F1 in the CV tables are computed at the **default threshold
  0.5**. ROC-AUC and AP do not depend on any threshold.

## `choose_threshold`

```python
def choose_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float) -> float:
```

- **Inputs:** the true labels, the predicted probability of "fail" for the same students, and the
  minimum recall we want (0.80, read from `config.yaml`).
- **Output:** one number, the threshold. The rule is: a student is flagged if their probability is
  ≥ the threshold.
- The docstring has a `Raises:` section. It documents that the function can raise an error, and
  in which case.

```python
    if not 0 < min_recall <= 1:
        raise ValueError(f"min_recall must be in (0, 1], got {min_recall}.")
```

- **What:** this refuses any recall target outside (0, 1]: 0 excluded, 1 included. Python allows
  the chained comparison `0 < x <= 1`, which reads like maths.
- **Why these limits:**
  - Recall is a fraction, so a target above 1 can never be reached.
  - A target of 0 or below is met by *every* threshold, so it means nothing.
  - Both can only come from a mistake, typically a typo in `config.yaml`.
- **Why an error and not a warning:** a wrong setting should stop the program immediately, with
  a message that names the bad value. This is "fail early and clearly". Silently continuing
  would produce a meaningless threshold.
- **Phase 2b change:** this replaces the earlier "fallback with a warning". The fallback could
  never trigger with a valid target (see below), so it was dead code.

```python
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
```

- **What `precision_recall_curve` does:** it sorts the distinct predicted probabilities. For each
  one, used as a threshold, it computes precision and recall when every student with probability
  ≥ that value is flagged. So it tries *every* useful threshold at once, without a hand-made loop.
- A **low threshold** flags many students: high recall (we miss few), low precision (many false
  alarms). A **high threshold** does the opposite. Choosing a threshold means choosing a point on
  this trade-off.
- It returns three arrays. `thresholds` is sorted in increasing order. `precision` and `recall`
  have **one more element** than `thresholds`.

```python
    # The last (precision, recall) pair is a fixed end point (1, 0) with no threshold:
    # drop it so that precision[i] and recall[i] belong to thresholds[i].
    precision, recall = precision[:-1], recall[:-1]
```

- scikit-learn adds a final point, precision = 1 and recall = 0 ("flag nobody"), so that the
  curve ends on the axis. That point has no threshold. We remove it so that the three arrays have
  the same length and position `i` means the same threshold in all three.
- **Without it:** the arrays would be misaligned. We could read the precision of one threshold
  and return another threshold, or go out of bounds.

```python
    # The lowest threshold flags every student (recall 1), so at least one threshold
    # always reaches a min_recall in (0, 1].
    reaches_target = recall >= min_recall
```

- This is a True/False array: which thresholds satisfy our recall goal.
- **Why at least one is always True:** the lowest threshold returned by `precision_recall_curve`
  is the smallest predicted probability. At that threshold *every* student is flagged, so
  every failing student is caught and recall = 1.0. Any target in (0, 1] is therefore
  reached at least there. The toy test confirms it: target 1.0 gives threshold 0.3, and
  threshold 0.1 also reaches recall 1. This is why no "not reachable" branch is needed.

```python
    best = np.argmax(np.where(reaches_target, precision, -1.0))
    return float(thresholds[best])
```

- `np.where(reaches_target, precision, -1.0)` keeps the precision of the valid thresholds and
  puts −1 on the others. Precision is never negative, so a −1 can never win.
- `np.argmax` gives the position of the highest precision.
- **Ties:** `argmax` returns the *first* maximum. Thresholds are increasing, so the first one is
  the lowest threshold, which has the highest recall. Among equal precisions, we therefore keep
  the one that catches the most students, which matches our priority.
- `float(...)` turns a NumPy number into a plain Python float. This is cleaner to print and to
  save.

- **Remaining limit (checked):** if `y_true` contains **no** failing student, recall is
  undefined. scikit-learn then warns "No positive class found in y_true" and the function
  returns the lowest threshold. This cannot happen with our stratified folds, which always
  contain 16 fails.
- **The real practical risk** is not an unreachable target. It is a threshold so low that it
  flags almost everyone. This is why `threshold_report` gives `flagged_share`, and why the
  trade-off table below shows it for several targets.

## `threshold_report`

```python
def threshold_report(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
```

- This turns probabilities into 0/1 decisions with the *same rule* as `choose_threshold`
  (≥ threshold).
- `np.asarray` accepts a list, a pandas Series or an array.

```python
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
```

- `confusion_matrix` returns a 2×2 table: `[[TN, FP], [FN, TP]]`. Rows are the truth (0 then 1)
  and columns are the decision.
- `.ravel()` flattens it to 4 numbers in this order.
- `labels=[0, 1]` forces a 2×2 table even if, for example, no student is predicted 1. Otherwise
  the table could shrink to 1×1, and the unpacking would crash.
- **Meaning for us:**
  - **FN** = a failing student we missed (the costly error).
  - **FP** = a student who would pass but gets tutoring anyway (the cheap error).

```python
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
```

- Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 is their harmonic mean.
- `zero_division=0` gives 0 instead of a warning when a ratio is 0/0.
- `flagged_share` is the share of students flagged (the mean of 0/1 decisions). It tells the
  school **how many tutoring places** this threshold would require.
- The result is a dict with plain Python numbers, so it becomes one row of a DataFrame and then of
  `reports/threshold_report.csv`.

## `final_scores` (Phase 3)

```python
def final_scores(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    scores = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
    }
    scores.update(threshold_report(y_true, y_proba, threshold))
    return scores
```

- **What:** all the project's metrics for one set of predictions, in one dict:
  - ROC-AUC and AP, computed from the *probabilities*, so they do not depend on the threshold;
  - everything from `threshold_report` at the saved threshold: precision, recall, F1, share
    flagged, and TN/FP/FN/TP.
- `dict.update` adds the keys of the second dict to the first.
- **Who uses it:** only the `evaluate` command, once, on the test set. The test set was not used
  to choose the model or the threshold. Both come from the saved file.
- **Why it reuses `threshold_report`:** the test metrics are computed with exactly the same rule
  (≥ threshold) and code as the out-of-fold metrics, so the two are comparable.
- **Naming:** it is not called `test_scores` on purpose. pytest collects any imported function
  whose name starts with `test_` as a test, so the name would cause confusion in the tests
  **[from prior knowledge, to verify]**.

## `threshold_tradeoff` (Phase 2b)

```python
def threshold_tradeoff(
    y_true: np.ndarray, y_proba: np.ndarray, min_recalls: list[float]
) -> pd.DataFrame:
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
```

- **What:** for each recall target (0.70, 0.80 and 0.90, from `config.yaml`), it applies *our
  own rule* (`choose_threshold`), then measures the result (`threshold_report`). It keeps the
  four numbers a decision-maker needs.
- **Why it reuses the two functions:** the table is guaranteed to follow exactly the same rule
  as the real threshold choice. There is no second implementation that could disagree.
- `rows` is a list of dicts, which pandas turns into a table with one row per target.
- **Out-of-fold only:** the Trainer calls it with the out-of-fold probabilities of the training
  set. **The test set is never used**, because looking at several thresholds on the test set and
  picking one would be "using the test set to make a choice".

### The table (our results, `reports/threshold_tradeoff.csv`, chosen model `logreg_l2`)

| recall target | threshold | precision | recall reached | students flagged |
|---|---|---|---|---|
| 0.70 | 0.171 | 0.389 | 0.725 | 28.7 % (149 of 519) |
| **0.80 (ours)** | **0.140** | **0.312** | **0.800** | **39.5 % (205 of 519)** |
| 0.90 | 0.116 | 0.248 | 0.900 | 55.9 % (290 of 519) |

**How to read it:** each extra 10 points of recall costs a lot.

- From 0.70 to 0.80, the school catches about 6 more failing students (58 to 64 of 80), but must
  flag 56 more students in total. Precision drops from 39 % to 31 %.
- From 0.80 to 0.90, it catches 8 more (64 to 72), but must flag 85 more. More than half of all
  students are flagged, and only 1 in 4 of them really fails.
- **Why the 0.70 row shows recall 0.725:** among all thresholds with recall ≥ 0.70, the one with
  the best precision happens to reach 0.725 (58 of 80). Recall moves in steps of 1/80 = 0.0125,
  and here a slightly higher recall also gave a slightly higher precision.

**How to use it:** the right row depends on the school's tutoring capacity.

- If tutors can follow about 40 % of the students, 0.80 is a reasonable balance. It is our
  stated goal, and it is the cost of missing a failing student that justifies it.
- If capacity is smaller, 0.70 is the honest alternative: fewer missed students would be
  *promised*, but the help would actually be delivered.
- The table makes this choice visible, instead of hiding it inside one number.

## `wilson_interval` (notebook phase)

```python
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    # Clip: rounding can give -1e-17 instead of 0 when successes is 0 (or n).
    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))
```

- **What:** a 95 % confidence interval for a proportion, such as a recall of 15 caught out of 20
  failing students. `z = 1.96` is the normal quantile for 95 %.
- **The formula (Wilson score interval)** **[from prior knowledge, to verify against a
  statistics reference]**:
  - centre = (p + z²/2n) / (1 + z²/n), which pulls p slightly towards 0.5;
  - half-width = z·√(p(1−p)/n + z²/4n²) / (1 + z²/n).
- **Why Wilson and not the simple "p ± 1.96·√(p(1−p)/n)" (Wald) interval:** with a small n, or a
  p near 0 or 1, the Wald interval can go below 0 or above 1, and it covers the true value less
  often than 95 %. When p = 1, its width is 0. Wilson stays inside [0, 1] and behaves well for
  small samples. Our test set has only 20 fails, and some groups have 25 **[from prior knowledge,
  to verify]**.
- **The clip (found by our test):** for 0 successes, the exact lower bound is 0, but floating-point
  rounding gave −2.8e-17. `max(0.0, …)` and `min(1.0, …)` keep the result exactly inside [0, 1].
- **Our use (checked):** test recall 15/20 gives [0.531, 0.888]. A recall of 0.75 on 20 students
  is compatible with any true recall between about 0.53 and 0.89.

## `group_report` (notebook phase)

```python
def group_report(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float, groups: pd.Series
) -> pd.DataFrame:
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
```

- **What:** the same threshold metrics, computed **separately for each group**, for example F/M or
  GP/MS. This is the fairness check of the notebook.
- `mask` is a True/False array selecting one group's students. `.to_numpy()` makes it a plain
  array, so it selects by **position** in both `y_true` (a Series) and `y_proba` (an array).
  **The three inputs must be in the same order**, which they are when they all come from the
  training set.
- **It reuses `threshold_report`**, so the rule (≥ threshold) and the metric code are exactly the
  same as everywhere else. Only the subset of students changes.
- `n_students` and `n_fail` are reported on purpose: a recall computed on 25 fails is much less
  certain than one on 400. The notebook adds a Wilson interval per group.
- `sorted(...)` gives a stable row order: F before M, GP before MS.

## `coefficient_table` (notebook phase)

```python
def coefficient_table(model: Pipeline) -> pd.DataFrame:
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
```

- **What:** one row per column that the model actually sees, after preprocessing (56 for the real
  data), with its coefficient, sorted by importance.
- `get_feature_names_out()` returns the names *after* the ColumnTransformer, in the same order as
  the coefficients. For example `numeric__age`, or `categorical__school_MS` (one-hot: original
  column, then `_`, then the category).
- `name.split("__", 1)[1]` removes the branch prefix (`numeric__`, `categorical__`), keeping a
  readable `age` or `school_MS`. `1` means split only at the first `__`.
- `coef_[0]`: for a binary logistic regression, `coef_` has shape (1, n_features). Row 0 holds
  the weights for the log-odds of class 1, "fail".
- `odds_ratio = e^coefficient` is the factor by which the odds of failing are multiplied when the
  feature increases by 1 (one standard deviation for numeric features, because they are scaled).
- Sorting by **absolute value** puts the strongest effects first, whatever their sign.
- **Works only for linear models** with a `coef_` attribute, such as our chosen `logreg_l2`. k-NN
  has no coefficients.
- `Pipeline` (imported from `sklearn.pipeline`) is only used in the type hint.
