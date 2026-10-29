# `tests/test_data.py`: explained line by line

**Role:** check the three data steps of `DatasetLoader`: cleaning, target creation, and the
train/test split. The fixtures `config` and `raw_df` come from `conftest.py`.

For every test, "**Make it fail on purpose**" gives a change in the source code that makes the
test fail. We tried each one. It is a good way to show the teachers that the test really checks
something. Undo the change afterwards.

---

## Imports

```python
"""Tests for DatasetLoader: cleaning, target creation and the train/test split."""

import numpy as np
import pandas as pd

from student_risk.data import DatasetLoader
```

- `numpy` gives `np.nan`, the missing value.
- `pandas` is used to build tables and to concatenate rows.
- `DatasetLoader` is the class under test.

---

## Test 1: `test_clean_strips_text_drops_missing_grade_and_duplicates`

**What it checks:** `clean()` does the three cleaning steps and renumbers the rows.

```python
    dirty = raw_df.copy()
    dirty.loc[0, "school"] = "  GP "  # extra spaces
    dirty.loc[1, "G3"] = np.nan  # no final grade
    dirty = pd.concat([dirty, dirty.iloc[[2]]])  # one duplicated row
```

We build a "dirty" table on purpose, with exactly one problem of each kind:

- row 0 has spaces around a text value;
- row 1 has no final grade;
- row 2 is appended a second time, making an exact duplicate. `iloc[[2]]` with double brackets
  keeps it as a one-row DataFrame.

```python
    cleaned = DatasetLoader(config).clean(dirty)
```

This runs the code under test.

```python
    assert cleaned.loc[0, "school"] == "GP"
    assert cleaned["G3"].notna().all()
    assert not cleaned.duplicated().any()
    assert len(cleaned) == len(raw_df) - 1  # lost the missing-grade row only
    assert list(cleaned.index) == list(range(len(cleaned)))
```

- Line 1: the spaces were removed.
- Line 2: no row without a grade remains.
- Line 3: no duplicate remains.
- Line 4: we started with 40 rows, added 1 duplicate, then removed it and the missing-grade row,
  so 39 rows remain. This proves that `clean` removed **only** the bad rows.
- Line 5: the index is 0, 1, 2... with no holes (this is `reset_index`).

**Make it fail on purpose:**

- In `data.py`, replace `column.str.strip()` with `column`. The first assert fails.
- Or delete the line `df = df.drop_duplicates()`. The duplicate check fails.

---

## Test 2: `test_build_target_and_drop_grade_columns`

**What it checks:**

- the target rule `fail = 1 if G3 < 10`, including the boundary 10;
- the name of the target;
- that G1, G2 and G3 are removed from the features. This is the anti-leakage rule.

```python
    df = pd.DataFrame(
        {"age": [16, 17, 18, 19], "G1": [1, 2, 3, 4], "G2": [1, 2, 3, 4], "G3": [0, 9, 10, 20]}
    )
```

This is a tiny hand-made table, so we can compute the expected answer in our head. The G3 values
are chosen around the boundary: 0 and 9 fail, 10 and 20 pass.

```python
    X, y = DatasetLoader(config).build_target(df)

    assert list(y) == [1, 1, 0, 0]  # 9 fails, 10 passes (boundary)
    assert y.name == "fail"
    assert list(X.columns) == ["age"]  # G1, G2 and G3 are gone
```

- The labels are exactly as expected, and 10 counts as a pass.
- The Series is named `fail`.
- Only `age` remains in X: all three grade columns were removed.

**Make it fail on purpose:**

- In `build_target`, change `<` into `<=`. The student with exactly 10 becomes a "fail", and the
  first assert fails.
- Or change `[self.config.grade_column, *self.config.leakage_columns]` into
  `[self.config.grade_column]`. G1 and G2 stay in X, and the last assert fails. This is the leak
  we want to prevent.

---

## Test 3: `test_split_is_disjoint_stratified_and_reproducible`

**What it checks:**

- no test row is in train;
- no row is lost;
- the fail rate is the same in both sets;
- the same seed gives the same split.

```python
    loader = DatasetLoader(config)
    X, y = loader.build_target(loader.clean(raw_df))

    X_train, X_test, y_train, y_test = loader.split(X, y)
```

This runs the real chain on the 40 fake rows: 30 train, 10 test.

```python
    assert set(X_train.index).isdisjoint(X_test.index)  # no test row in train
    assert len(X_train) + len(X_test) == len(X)
```

- Every row keeps its original index number after the split. If the two sets of index numbers
  share no element, then no student is in both sets.
- The sizes add up, so no student is lost or duplicated.

```python
    assert abs(y_train.mean() - y_test.mean()) < 0.1  # same fail rate in both
```

- The mean of a 0/1 column is the fail rate. Stratification keeps both rates close to the global
  37.5 % (15/40).
- **Why a tolerance of 0.1:** with 10 test rows, the rates cannot be exactly equal. Here they are
  0.367 and 0.4.

```python
    X_train_again = loader.split(X, y)[0]
    assert list(X_train.index) == list(X_train_again.index)  # same seed, same split
```

Splitting again gives exactly the same training rows, in the same order. This proves
reproducibility.

**Make it fail on purpose:**

- In `split`, replace `random_state=self.config.random_seed` with `random_state=None`. Each call
  gives a different split, and the last assert fails.
- Removing `stratify=y` also makes the test fail with our data and seed: the test set ends up
  with a different fail rate. Without stratification, the result depends on luck, which is
  exactly why we stratify.
