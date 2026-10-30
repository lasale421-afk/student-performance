# `tests/test_evaluate.py`: explained line by line

**Role:** check `choose_threshold` on a tiny example where we can compute the right answer by
hand. No model and no data are needed.

For every test, "**Make it fail on purpose**" gives a change in the source code that breaks the
test. We tried each one. Undo the change afterwards.

---

## Imports and toy data

```python
"""Tests for the threshold choice in evaluate.py."""

import numpy as np
import pytest

from student_risk.evaluate import choose_threshold

# 6 students: 3 fail (1), 3 pass (0). Sorted by predicted probability.
Y_TRUE = np.array([0, 0, 1, 0, 1, 1])
Y_PROBA = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
```

- `pytest` gives us `pytest.approx`, to compare decimal numbers safely, `pytest.raises`, to check
  that an error is raised, and `pytest.mark.parametrize`, to run one test with several values.
  (Phase 2 also used `pytest.warns` for the old fallback test, now removed.)
- The toy data are constants in capital letters, shared by both test functions.
- Here is the full table you can compute by hand. "Flagged" means probability ≥ threshold.

| threshold | flagged students | TP | FP | recall | precision |
|-----------|------------------|----|----|--------|-----------|
| 0.1 | all 6 | 3 | 3 | 3/3 = 1.00 | 3/6 = 0.50 |
| 0.2 | 5 | 3 | 2 | 1.00 | 0.60 |
| 0.3 | 4 | 3 | 1 | 1.00 | 0.75 |
| 0.4 | 3 | 2 | 1 | 0.67 | 0.67 |
| 0.8 | 2 | 2 | 0 | 0.67 | 1.00 |
| 0.9 | 1 | 1 | 0 | 0.33 | 1.00 |

---

## Test 1: `test_choose_threshold_reaches_recall_with_best_precision`

**What it checks:** among the thresholds that reach the required recall, the function returns
the one with the **highest precision**. We check this for two different targets.

```python
    # Threshold 0.8 flags 2 students, both fail: recall 2/3, precision 1.
    assert choose_threshold(Y_TRUE, Y_PROBA, min_recall=0.6) == pytest.approx(0.8)
```

- Valid thresholds for recall ≥ 0.6 are 0.1, 0.2, 0.3, 0.4 and 0.8.
- The best precision among them is 1.00, at 0.8. So the answer is 0.8.
- This also checks that the function does **not** simply return the lowest threshold (0.1), which
  would also satisfy the recall goal.

```python
    # Recall 1 needs threshold 0.3: 4 flagged, 3 fail -> precision 0.75 (0.1 gives only 0.5).
    assert choose_threshold(Y_TRUE, Y_PROBA, min_recall=1.0) == pytest.approx(0.3)
```

- Valid thresholds for recall = 1.0 are 0.1, 0.2 and 0.3.
- The best precision is 0.75, at 0.3.
- This also checks the `>=` comparison: a recall of exactly 1.0 must count as reaching 1.0.
- `pytest.approx` is used because thresholds are floats.

**Make it fail on purpose:**

- In `evaluate.py`, change `recall >= min_recall` into `recall > min_recall`. With
  `min_recall=1.0`, no recall is strictly above 1, so every entry becomes −1. `argmax` then
  returns the first position, threshold 0.1, and the second assert fails.
- Or replace `precision` with `recall` inside `np.where(...)`. The function then maximises recall
  instead of precision and returns 0.1, so the first assert fails.

---

## Test 2: `test_choose_threshold_rejects_min_recall_outside_0_1` (Phase 2b)

**What it checks:** `choose_threshold` refuses a recall target outside (0, 1] with a clear
`ValueError`. It must not silently return a meaningless threshold.

```python
@pytest.mark.parametrize("bad_recall", [0.0, -0.2, 1.5])
def test_choose_threshold_rejects_min_recall_outside_0_1(bad_recall):
    # A target of 0 (or below) would accept any threshold; above 1 is impossible.
    with pytest.raises(ValueError, match="min_recall must be in"):
        choose_threshold(Y_TRUE, Y_PROBA, min_recall=bad_recall)
```

- `@pytest.mark.parametrize` runs the same test once per value: 0.0, −0.2 and 1.5. pytest
  reports them as 3 separate tests (`[0.0]`, `[-0.2]`, `[1.5]`), which is why the total test
  count grew by 2 when this test replaced the old fallback test.
- The three values are chosen on purpose:
  - **0.0** is the exact lower boundary. It must be rejected, because the interval is open
    at 0.
  - **−0.2** is clearly below the range.
  - **1.5** is above the range.
- The upper boundary 1.0 *is* valid. Test 1 already checks it: `min_recall=1.0` returns 0.3.
- `pytest.raises(ValueError, match=...)` makes the test **fail if no `ValueError` is raised**,
  or if the message does not contain "min_recall must be in". `match` is a regular expression
  searched in the message.

**Make it fail on purpose:**

- In `evaluate.py`, delete the two lines `if not 0 < min_recall <= 1:` and `raise ValueError(...)`.
  All 3 cases then fail with "DID NOT RAISE". We checked: 3 failed.
- Or change `<= 1` into `< 1`. The test still passes, but Test 1 now fails, because 1.0 is
  rejected. The two tests together pin down both boundaries.

**Why the old fallback test was removed:** Phase 2 had a fallback (warn, then return the
threshold with the highest recall) and a test for it with `min_recall=1.5`. The fallback could
never trigger for a valid target, because the lowest threshold always has recall 1. So the
fallback was replaced by this input check, and its test by this one.

---

## Test 3: `test_wilson_interval_known_values` (notebook phase)

**What it checks:**

- `wilson_interval` gives the known 95 % interval for 15 successes out of 20, our real test
  recall: about [0.531, 0.888];
- the bounds stay inside [0, 1] in the extreme cases 0/5 and 5/5.

```python
    lower, upper = wilson_interval(15, 20)
    assert lower == pytest.approx(0.531, abs=0.001)
    assert upper == pytest.approx(0.888, abs=0.001)
    assert 0 <= wilson_interval(0, 5)[0] and wilson_interval(5, 5)[1] <= 1  # stays in [0, 1]
```

- **The expected values** were computed by hand from the formula (see `evaluate.md`), with
  p = 0.75, n = 20 and z = 1.96: centre ≈ 0.710, half-width ≈ 0.178. `abs=0.001` allows for
  rounding at the third decimal.
- **The last line found a real bug:** the first version returned −2.8e-17 for 0/5 because of
  floating-point rounding. That is why the function now clips its bounds.

**Make it fail on purpose (each one checked):**

- Remove the clipping, so it returns `float(centre - half_width), float(centre + half_width)`.
  The last assert fails.
- Replace the Wilson centre with `centre = p`, which drops the pull towards 0.5. The known values
  no longer match.

## Test 4: `test_group_report_counts_and_recall_per_group` (notebook phase)

**What it checks:** `group_report` splits the students by group and computes each group's size,
number of fails and recall correctly, on a toy example we can check by hand.

```python
    groups = pd.Series(["F", "F", "M", "M", "M", "F"])
    y_true = pd.Series(Y_TRUE)  # fails: index 2 (M), 4 (M), 5 (F)

    report = group_report(y_true, Y_PROBA, threshold=0.8, groups=groups)

    assert list(report.index) == ["F", "M"]
    assert report.loc["F", "n_students"] == 3 and report.loc["F", "n_fail"] == 1
    assert report.loc["F", "recall"] == 1.0  # its only fail (0.9) is flagged
    assert report.loc["M", "recall"] == 0.5  # 0.8 flagged, 0.3 missed
```

- It reuses the 6 toy students. Their probabilities are 0.1, 0.2, 0.3, 0.4, 0.8 and 0.9, and the
  fails are at positions 2, 4 and 5.
- **By hand, with threshold 0.8:**
  - **F** (positions 0, 1, 5): 3 students, 1 fail (position 5, probability 0.9 ≥ 0.8, flagged),
    so recall 1.0.
  - **M** (positions 2, 3, 4): 2 fails. Position 4 (0.8) is flagged and position 2 (0.3) is
    missed, so recall 0.5.
- This also checks that the rows are sorted (F, then M).

**Make it fail on purpose (checked):** in `group_report`, misalign the probabilities, for example
`np.asarray(y_proba)[::-1][mask]` (reversed). Each group then gets another group's
probabilities, and the recalls are wrong. This is exactly the bug the "same order" requirement
protects against.
